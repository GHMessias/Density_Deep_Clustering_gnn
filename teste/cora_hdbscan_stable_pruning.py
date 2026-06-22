from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "teste" / ".matplotlib").resolve()))

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hdbscan._hdbscan_tree import compute_stability
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, GCNConv


NEG_INF = -1.0e18


class SimpleGCNEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        embedding_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teste exploratorio no Cora com GAE simples + HDBSCAN + "
            "poda da arvore condensada para obter k grupos estaveis."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("teste/output/cora_hdbscan_stable_pruning"))
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--embedding-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_simple_gae(
    data,
    hidden_channels: int,
    embedding_channels: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, list[float], str]:
    device = resolve_device()
    model = GAE(
        SimpleGCNEncoder(
            input_channels=data.num_features,
            hidden_channels=hidden_channels,
            embedding_channels=embedding_channels,
            dropout=dropout,
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    working_data = data.clone().to(device)
    loss_history: list[float] = []

    print_banner("ETAPA 1 - GAE SIMPLES")
    print(f"Dispositivo usado: {device}")
    print(f"Epocas configuradas: {epochs}")
    print(f"Hidden channels: {hidden_channels}")
    print(f"Embedding channels: {embedding_channels}")
    print(f"Dropout: {dropout}")

    if epochs == 0:
        print("Treino desabilitado. Vou gerar embeddings com pesos aleatorios do encoder.")
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            latent_embeddings = model.encode(working_data.x, working_data.edge_index)
            loss = model.recon_loss(latent_embeddings, working_data.edge_index)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            loss_history.append(loss_value)

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(
                    f"[GAE] epoca={epoch:03d} "
                    f"recon_loss={loss_value:.6f}"
                )

    model.eval()
    with torch.no_grad():
        latent_embeddings = model.encode(working_data.x, working_data.edge_index)

    embeddings = latent_embeddings.detach().cpu().numpy()
    final_loss = loss_history[-1] if loss_history else None
    print(f"Embedding final gerado com shape: {embeddings.shape}")
    if final_loss is not None:
        print(f"Loss final registrada: {final_loss:.6f}")
    return embeddings, loss_history, str(device)


def build_condensed_tree_maps(
    raw_tree: np.ndarray,
    num_points: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, float], dict[int, float], int]:
    cluster_children: dict[int, list[int]] = defaultdict(list)
    point_children: dict[int, list[int]] = defaultdict(list)
    birth_lambda: dict[int, float] = {}
    death_lambda: dict[int, float] = defaultdict(float)

    for row in raw_tree:
        parent = int(row["parent"])
        child = int(row["child"])
        lambda_val = float(row["lambda_val"])
        child_size = int(row["child_size"])

        death_lambda[parent] = max(death_lambda[parent], lambda_val)
        if child >= num_points and child_size > 1:
            cluster_children[parent].append(child)
            birth_lambda[child] = lambda_val
        else:
            point_children[parent].append(child)

    root = int(raw_tree["parent"].min())
    birth_lambda[root] = 0.0
    return cluster_children, point_children, birth_lambda, death_lambda, root


def compute_cluster_members(
    cluster_children: dict[int, list[int]],
    point_children: dict[int, list[int]],
) -> dict[int, np.ndarray]:
    @lru_cache(maxsize=None)
    def gather(node: int) -> tuple[int, ...]:
        points = list(point_children.get(node, []))
        for child in cluster_children.get(node, []):
            points.extend(gather(child))
        return tuple(sorted(points))

    cluster_ids = sorted(set(cluster_children) | set(point_children))
    return {cluster_id: np.asarray(gather(cluster_id), dtype=np.int64) for cluster_id in cluster_ids}


def select_k_stable_clusters(
    root: int,
    target_k: int,
    cluster_children: dict[int, list[int]],
    stability: dict[int, float],
) -> tuple[list[int], float, int]:
    choices: dict[tuple[int, int], tuple[str, object]] = {}

    @lru_cache(maxsize=None)
    def solve(node: int, m: int) -> float:
        if m == 0:
            choices[(node, m)] = ("drop", None)
            return 0.0

        children = cluster_children.get(node, [])
        best_score = NEG_INF
        best_choice: tuple[str, object] | None = None

        if m == 1:
            best_score = float(stability.get(node, 0.0))
            best_choice = ("keep", None)

        if children:
            states: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
            for child in children:
                new_states: dict[int, tuple[float, list[tuple[int, int]]]] = {}
                for used_clusters, (partial_score, partial_allocs) in states.items():
                    remaining = m - used_clusters
                    for child_clusters in range(0, remaining + 1):
                        child_score = solve(child, child_clusters)
                        if child_score <= NEG_INF / 2:
                            continue
                        total_clusters = used_clusters + child_clusters
                        candidate_score = partial_score + child_score
                        previous = new_states.get(total_clusters)
                        if previous is None or candidate_score > previous[0]:
                            new_states[total_clusters] = (
                                candidate_score,
                                partial_allocs + [(child, child_clusters)],
                            )
                states = new_states

            split_state = states.get(m)
            if split_state is not None and split_state[0] > best_score:
                best_score = split_state[0]
                best_choice = ("split", split_state[1])

        if best_choice is None:
            best_choice = ("invalid", None)

        choices[(node, m)] = best_choice
        return best_score

    selected_k = target_k
    best_score = solve(root, target_k)
    if best_score <= NEG_INF / 2:
        print(
            f"[AVISO] A arvore nao conseguiu sustentar exatamente k={target_k} grupos. "
            "Vou procurar o maior valor viavel abaixo dele."
        )
        feasible_ks = [m for m in range(target_k - 1, 0, -1) if solve(root, m) > NEG_INF / 2]
        if not feasible_ks:
            raise RuntimeError("Nao encontrei nenhuma poda viavel na arvore condensada.")
        selected_k = feasible_ks[0]
        best_score = solve(root, selected_k)

    selected_nodes: list[int] = []

    def backtrack(node: int, m: int) -> None:
        action, payload = choices[(node, m)]
        if action == "keep":
            selected_nodes.append(node)
            return
        if action == "split":
            for child, child_clusters in payload:
                if child_clusters > 0:
                    backtrack(child, child_clusters)

    backtrack(root, selected_k)
    return selected_nodes, best_score, selected_k


def scatter_labels(
    ax,
    points_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            color = "#9a9a9a"
            plot_label = "noise"
        else:
            color = cmap(int(label) % 10)
            plot_label = f"cluster {int(label)}"
        ax.scatter(
            points_2d[mask, 0],
            points_2d[mask, 1],
            s=12,
            alpha=0.85,
            c=[color],
            label=plot_label,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def save_assignments_csv(
    path: Path,
    true_labels: np.ndarray,
    raw_labels: np.ndarray,
    pruned_labels: np.ndarray,
    projection: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["node_id", "true_label", "hdbscan_label", "stable_pruned_label", "pca_x", "pca_y"])
        for node_id in range(len(true_labels)):
            writer.writerow(
                [
                    node_id,
                    int(true_labels[node_id]),
                    int(raw_labels[node_id]),
                    int(pruned_labels[node_id]),
                    float(projection[node_id, 0]),
                    float(projection[node_id, 1]),
                ]
            )


def summarize_counts(labels: Iterable[int]) -> str:
    values, counts = np.unique(np.asarray(list(labels)), return_counts=True)
    pieces = [f"{int(value)}:{int(count)}" for value, count in zip(values, counts)]
    return ", ".join(pieces)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_banner("CONFIGURACAO DO TESTE")
    print(f"Seed: {args.seed}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"HDBSCAN min_cluster_size: {args.min_cluster_size}")
    print(f"HDBSCAN min_samples: {args.min_samples}")

    print_banner("ETAPA 0 - CARREGANDO O CORA")
    dataset = Planetoid(root=str(args.dataset_root), name="Cora")
    data = dataset[0]
    target_k = args.k if args.k is not None else dataset.num_classes

    print(f"Nome do dataset: {dataset.name}")
    print(f"Numero de nos: {data.num_nodes}")
    print(f"Numero de arestas (edge_index): {data.num_edges}")
    print(f"Numero de features: {dataset.num_features}")
    print(f"Numero de classes conhecidas: {dataset.num_classes}")
    print(f"k alvo para a poda estavel: {target_k}")

    embeddings, loss_history, device = train_simple_gae(
        data=data,
        hidden_channels=args.hidden_channels,
        embedding_channels=args.embedding_channels,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print_banner("ETAPA 2 - HDBSCAN NO ESPACO LATENTE")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(embeddings)
    raw_labels = clusterer.labels_.astype(np.int64)
    raw_cluster_count = int(np.sum(np.unique(raw_labels) >= 0))
    raw_noise_count = int(np.sum(raw_labels == -1))

    print(f"Clusters encontrados pelo HDBSCAN puro: {raw_cluster_count}")
    print(f"Numero de pontos marcados como ruido: {raw_noise_count}")
    print(f"Distribuicao dos rotulos do HDBSCAN: {summarize_counts(raw_labels)}")
    print(f"Cluster persistence_ reportada pela biblioteca: {clusterer.cluster_persistence_}")

    print_banner("ETAPA 3 - ARVORE CONDENSADA E ESTABILIDADE")
    raw_tree = clusterer.condensed_tree_._raw_tree
    cluster_children, point_children, birth_lambda, death_lambda, root = build_condensed_tree_maps(
        raw_tree=raw_tree,
        num_points=data.num_nodes,
    )
    member_map = compute_cluster_members(cluster_children, point_children)
    stability = {int(key): float(value) for key, value in compute_stability(raw_tree).items()}
    cluster_nodes = sorted(stability)

    print(f"Root da arvore condensada: {root}")
    print(f"Numero de linhas na condensed_tree: {len(raw_tree)}")
    print(f"Numero de clusters candidatos na arvore: {len(cluster_nodes)}")
    print("Top 10 clusters candidatos por estabilidade:")
    ranked_nodes = sorted(cluster_nodes, key=lambda node: stability[node], reverse=True)
    for node in ranked_nodes[:10]:
        members = member_map.get(node, np.array([], dtype=np.int64))
        print(
            f"  node={node:4d} "
            f"stability={stability[node]:10.4f} "
            f"birth={birth_lambda.get(node, 0.0):8.4f} "
            f"death={death_lambda.get(node, 0.0):8.4f} "
            f"size={len(members):4d} "
            f"children={len(cluster_children.get(node, []))}"
        )

    print_banner("ETAPA 4 - PODA PARA k GRUPOS ESTAVEIS")
    selected_nodes, best_score, selected_k = select_k_stable_clusters(
        root=root,
        target_k=target_k,
        cluster_children=cluster_children,
        stability=stability,
    )
    selected_nodes = sorted(selected_nodes, key=lambda node: stability.get(node, 0.0), reverse=True)

    print(f"k solicitado: {target_k}")
    print(f"k efetivamente selecionado: {selected_k}")
    print(f"Score total de estabilidade da poda: {best_score:.4f}")
    print("Clusters selecionados:")
    for index, node in enumerate(selected_nodes):
        members = member_map.get(node, np.array([], dtype=np.int64))
        print(
            f"  grupo={index:02d} "
            f"node={node:4d} "
            f"stability={stability.get(node, 0.0):10.4f} "
            f"size={len(members):4d} "
            f"birth={birth_lambda.get(node, 0.0):8.4f} "
            f"death={death_lambda.get(node, 0.0):8.4f}"
        )

    pruned_labels = np.full(data.num_nodes, -1, dtype=np.int64)
    covered = np.zeros(data.num_nodes, dtype=bool)
    for cluster_id, node in enumerate(selected_nodes):
        members = member_map.get(node, np.array([], dtype=np.int64))
        overlap = covered[members]
        if np.any(overlap):
            raise RuntimeError("A poda produziu clusters sobrepostos, o que nao deveria acontecer.")
        pruned_labels[members] = cluster_id
        covered[members] = True

    covered_ratio = float(np.mean(covered))
    noise_ratio = 1.0 - covered_ratio
    print(f"Pontos cobertos pelos k clusters estaveis: {covered.sum()} / {data.num_nodes} ({covered_ratio:.2%})")
    print(f"Pontos que ficaram como ruido apos a poda: {(~covered).sum()} / {data.num_nodes} ({noise_ratio:.2%})")
    print(f"Distribuicao dos rotulos apos a poda: {summarize_counts(pruned_labels)}")

    print_banner("ETAPA 5 - METRICAS EXPLORATORIAS")
    true_labels = data.y.detach().cpu().numpy().astype(np.int64)
    valid_mask = pruned_labels != -1
    if valid_mask.sum() > 0 and np.unique(pruned_labels[valid_mask]).size > 1:
        masked_nmi = normalized_mutual_info_score(true_labels[valid_mask], pruned_labels[valid_mask])
        masked_ari = adjusted_rand_score(true_labels[valid_mask], pruned_labels[valid_mask])
        print(f"NMI nos pontos cobertos: {masked_nmi:.4f}")
        print(f"ARI nos pontos cobertos: {masked_ari:.4f}")
    else:
        print("Nao foi possivel calcular NMI/ARI nos pontos cobertos.")

    if np.unique(raw_labels[raw_labels != -1]).size > 1:
        raw_mask = raw_labels != -1
        raw_nmi = normalized_mutual_info_score(true_labels[raw_mask], raw_labels[raw_mask])
        raw_ari = adjusted_rand_score(true_labels[raw_mask], raw_labels[raw_mask])
        print(f"NMI do HDBSCAN bruto nos pontos nao ruidosos: {raw_nmi:.4f}")
        print(f"ARI do HDBSCAN bruto nos pontos nao ruidosos: {raw_ari:.4f}")

    print_banner("ETAPA 6 - SALVANDO VISUALIZACOES E ARTEFATOS")
    projection = PCA(n_components=2, random_state=args.seed).fit_transform(embeddings)
    print(f"Projecao PCA gerada com shape: {projection.shape}")

    figure, axes = plt.subplots(1, 3, figsize=(20, 6))
    scatter_labels(axes[0], projection, true_labels, "Rotulos reais do Cora")
    scatter_labels(axes[1], projection, raw_labels, "HDBSCAN no embedding")
    scatter_labels(axes[2], projection, pruned_labels, f"Poda estavel para k={selected_k}")
    figure.tight_layout()
    projection_path = args.output_dir / "projection_overview.png"
    figure.savefig(projection_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Imagem com as projecoes salva em: {projection_path}")

    tree_figure, tree_axis = plt.subplots(1, 1, figsize=(12, 7))
    clusterer.condensed_tree_.plot(axis=tree_axis, select_clusters=False, label_clusters=False)
    tree_axis.set_title("Arvore condensada do HDBSCAN")
    tree_figure.tight_layout()
    tree_path = args.output_dir / "condensed_tree.png"
    tree_figure.savefig(tree_path, dpi=220, bbox_inches="tight")
    plt.close(tree_figure)
    print(f"Imagem da arvore condensada salva em: {tree_path}")

    csv_path = args.output_dir / "assignments.csv"
    save_assignments_csv(
        path=csv_path,
        true_labels=true_labels,
        raw_labels=raw_labels,
        pruned_labels=pruned_labels,
        projection=projection,
    )
    print(f"CSV com atribuicoes e coordenadas PCA salvo em: {csv_path}")

    summary_path = args.output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("Resumo do teste exploratorio no Cora\n")
        file.write(f"device={device}\n")
        file.write(f"epochs={args.epochs}\n")
        file.write(f"min_cluster_size={args.min_cluster_size}\n")
        file.write(f"min_samples={args.min_samples}\n")
        file.write(f"target_k={target_k}\n")
        file.write(f"selected_k={selected_k}\n")
        file.write(f"raw_cluster_count={raw_cluster_count}\n")
        file.write(f"raw_noise_count={raw_noise_count}\n")
        file.write(f"covered_points={int(covered.sum())}\n")
        file.write(f"noise_points_after_pruning={int((~covered).sum())}\n")
        file.write(f"projection_path={projection_path}\n")
        file.write(f"tree_path={tree_path}\n")
        file.write(f"csv_path={csv_path}\n")
        if loss_history:
            file.write(f"final_recon_loss={loss_history[-1]:.6f}\n")
        file.write("selected_nodes=\n")
        for index, node in enumerate(selected_nodes):
            file.write(
                f"  grupo={index:02d} node={node} "
                f"stability={stability.get(node, 0.0):.6f} "
                f"size={len(member_map.get(node, []))}\n"
            )
    print(f"Resumo textual salvo em: {summary_path}")

    print_banner("TESTE FINALIZADO")
    print("Tudo certo. O script gerou embeddings, rodou o HDBSCAN, podou a arvore e salvou os artefatos.")


if __name__ == "__main__":
    main()
