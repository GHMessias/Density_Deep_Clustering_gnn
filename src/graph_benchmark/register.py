from __future__ import annotations


def register_all() -> None:
    from graph_benchmark.clustering import arga_kmeans  # noqa: F401
    from graph_benchmark.clustering import arga_hdbscan  # noqa: F401
    from graph_benchmark.clustering import densgnn  # noqa: F401
    from graph_benchmark.clustering import densgnn2  # noqa: F401
    from graph_benchmark.clustering import dgcss  # noqa: F401
    from graph_benchmark.clustering import gae_hdbscan  # noqa: F401
    from graph_benchmark.clustering import gae_kmeans  # noqa: F401
    from graph_benchmark.clustering import kmeans  # noqa: F401
    from graph_benchmark.clustering import node2vec_kmeans  # noqa: F401
    from graph_benchmark.clustering import ricci_community  # noqa: F401
    from graph_benchmark.datasets import amazon  # noqa: F401
    from graph_benchmark.datasets import gencat_synthetic  # noqa: F401
    from graph_benchmark.datasets import planetoid  # noqa: F401
    from graph_benchmark.datasets import polblogs  # noqa: F401
    from graph_benchmark.datasets import twitch  # noqa: F401
    from graph_benchmark.experiments import node_feature_clustering  # noqa: F401
    from graph_benchmark.seed_selection import strategies  # noqa: F401
