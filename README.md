# Density_Deep_Clustering_gnn

Framework de benchmark para algoritmos de clustering em grafos, com execução guiada por YAML e organização inspirada no GraphGym, mas sem depender da biblioteca.

## Objetivo

Este repositório serve para comparar algoritmos de clustering em grafos sob uma interface unificada:

- datasets carregados via `torch_geometric.datasets`;
- execução centralizada por `main.py`;
- seleção do algoritmo por registry;
- configs em `.yaml`;
- saída padronizada em `results/`.

O foco atual está em benchmarks de clustering de nós em grafos atribuídos, começando por `Cora`.

## Algoritmos Implementados

Hoje o framework já suporta:

- `kmeans_features`: `KMeans` aplicado diretamente nas features dos nós, ignorando topologia.
- `gae_kmeans_embeddings`: treino de `GAE` para obter embeddings latentes e depois `KMeans`.
- `ollivier_ricci_community`: clustering topológico com `GraphRicciCurvature` usando `ricci_community()`.
- `dgcss`: implementação própria inspirada no paper/repositório DGCSS, com encoder atencional, seed selection e loss de clustering via `KL`.

## Estratégias de Sementes do DGCSS

O algoritmo `dgcss` já aceita as seguintes estratégias registradas para seleção de sementes:

- `betweenness_centrality`
- `closeness_centrality`
- `kmeans`
- `random`

Essas estratégias vivem em `src/graph_benchmark/seed_selection/` e podem ser trocadas diretamente no YAML ou com `--set algorithm.params.seed_selector=...`.

## Organização

```text
.
├── configs/
│   ├── cora_dgcss_bc.yaml
│   ├── cora_gae_kmeans.yaml
│   ├── cora_kmeans_features.yaml
│   └── cora_ollivier_ricci_community.yaml
├── data/
│   └── raw/
├── DGCSS/
├── main.py
├── results/
├── scripts/
│   ├── run_cora_dgcss.py
│   ├── run_cora_gae_kmeans.py
│   ├── run_cora_kmeans.py
│   └── run_cora_ollivier_ricci_community.py
└── src/
    └── graph_benchmark/
        ├── clustering/
        ├── config/
        ├── datasets/
        ├── evaluation/
        ├── experiments/
        ├── models/
        ├── seed_selection/
        ├── utils/
        ├── register.py
        ├── registry.py
        └── runner.py
```

Resumo dos diretórios:

- `configs/`: cenários de benchmark em YAML.
- `data/raw/`: datasets baixados pelo `torch_geometric`.
- `results/`: métricas e atribuições de cluster geradas pelos experimentos.
- `scripts/`: wrappers opcionais para execuções rápidas.
- `src/graph_benchmark/clustering/`: algoritmos registrados.
- `src/graph_benchmark/models/`: componentes de modelagem, como `GAE` e `DGCSS`.
- `src/graph_benchmark/seed_selection/`: estratégias de seed selection do `DGCSS`.
- `src/graph_benchmark/experiments/`: pipelines reutilizáveis de execução.
- `DGCSS/`: repositório de referência clonado localmente para estudo; o framework principal não depende diretamente dele.

## Como o Framework Roda

O fluxo padrão é:

1. `main.py` lê um arquivo YAML.
2. `run.experiment` seleciona o pipeline registrado.
3. `dataset.loader` seleciona o carregador do dataset.
4. `algorithm.name` escolhe o algoritmo de clustering.
5. o experimento executa o algoritmo e salva os artefatos em `results/`.

Hoje o pipeline principal é `node_feature_clustering`, que é usado tanto para os baselines simples quanto para `GAE`, `Ricci` e `DGCSS`.

## Instalação

O ambiente precisa ter pelo menos:

- `torch`
- `torch_geometric`
- `scikit-learn`
- `pyyaml`
- `scipy`
- `networkx`

Para a baseline de Ricci, também é necessário:

- `GraphRicciCurvature`

Exemplo:

```bash
./venv/bin/pip install scikit-learn pyyaml scipy networkx GraphRicciCurvature
```

## Execução

### Entrada principal

```bash
./venv/bin/python main.py --cfg <arquivo.yaml>
```

### Baselines disponíveis

`KMeans` nas features:

```bash
./venv/bin/python main.py --cfg configs/cora_kmeans_features.yaml
```

`GAE + KMeans`:

```bash
./venv/bin/python main.py --cfg configs/cora_gae_kmeans.yaml
```

`Ollivier-Ricci community`:

```bash
./venv/bin/python main.py --cfg configs/cora_ollivier_ricci_community.yaml
```

`DGCSS` com seed selection por Betweenness Centrality:

```bash
./venv/bin/python main.py --cfg configs/cora_dgcss_bc.yaml
```

### Variando o seed selector do DGCSS

Trocar para `closeness_centrality`:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_dgcss_bc.yaml \
  --set algorithm.params.seed_selector=closeness_centrality output.dir=results/cora/dgcss_cc
```

Trocar para `kmeans`:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_dgcss_bc.yaml \
  --set algorithm.params.seed_selector=kmeans output.dir=results/cora/dgcss_kmeans
```

Trocar para `random`:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_dgcss_bc.yaml \
  --set algorithm.params.seed_selector=random output.dir=results/cora/dgcss_random
```

### Overrides úteis

Mudar seed global e diretório de saída:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_gae_kmeans.yaml \
  --set run.seed=7 output.dir=results/cora/gae_kmeans_seed7
```

Rodar um smoke test curto do DGCSS:

```bash
./venv/bin/python main.py \
  --cfg configs/cora_dgcss_bc.yaml \
  --set algorithm.params.epochs=5 output.dir=results/cora/dgcss_smoke
```

## Saídas Geradas

Cada execução salva pelo menos:

- `metrics.json`
- `assignments.csv`

Exemplos de diretórios:

- `results/cora/kmeans_features/`
- `results/cora/gae_kmeans/`
- `results/cora/ollivier_ricci_community/`
- `results/cora/dgcss_bc/`

## Estrutura dos YAMLs

Os configs seguem esta ideia:

```yaml
run:
  experiment: node_feature_clustering
  seed: 42

dataset:
  loader: planetoid
  name: Cora
  root: data/raw

algorithm:
  name: dgcss
  params:
    seed_selector: betweenness_centrality
    epochs: 400
    ...

output:
  dir: results/cora/dgcss_bc
  save_assignments: true
```

## Estado Atual do DGCSS no Framework

A implementação do `dgcss` no framework é uma versão própria inspirada no paper e no repositório clonado em `DGCSS/`, mas integrada ao padrão do projeto atual:

- encoder `GAT` de duas camadas;
- matriz de transição multi-hop;
- seleção de sementes por registry;
- perda de clustering com `KL`;
- loss total de reconstrução + clustering;
- avaliação integrada com `NMI`, `ARI`, `Purity`, `Accuracy`, `Modularity` e, quando fizer sentido, `Silhouette`.

## Observação Sobre o Diretório `DGCSS/`

O diretório `DGCSS/` foi clonado como referência para estudo do paper e da implementação original. Ele ajuda a:

- entender a arquitetura do método;
- comparar a lógica do paper com a implementação original;
- inspirar novas integrações no framework.

Mas os benchmarks do framework principal devem ser executados pelo `main.py` na raiz do projeto, não pelo `run.py` do repositório clonado.

## Próximos Passos Naturais

- adicionar novos datasets além de `Cora`;
- criar YAMLs prontos para todas as variantes do `DGCSS`;
- incluir novos seed selectors topológicos;
- adicionar outras famílias de métodos de clustering em grafos;
- versionar benchmarks e consolidar tabelas comparativas automaticamente.
