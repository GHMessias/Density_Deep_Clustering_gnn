# Real Scenario Benchmark

Benchmark on the real `Cora` dataset, designed to mimic an
"unknown number of clusters" setting.

Shared dataset choices:
- Planetoid `Cora`
- 2708 nodes
- 5429 edges
- 1433 sparse node attributes
- 7 latent classes, but the benchmark is evaluated as if the number of
  groups were unknown to the clustering algorithm

Algorithms included:
- `ARGA + KMeans` for `k=2..10`
- `ARGA + HDBSCAN`
- `GAE + KMeans` for `k=2..10`
- `GAE + HDBSCAN`
- `DensGNN (Core)`
- `DensGNN (Border)`

Suggested execution patterns:

1. For `ARGA + KMeans` and `GAE + KMeans`, run 3 seeds:

```bash
python3 scripts/run_final_inputs.py \
  --config-dir configs/real_scenario_benchmark \
  --pattern '*kmeans*.yaml' \
  --seeds None None None
```

2. For the automatic cluster-detection algorithms, run 27 seeds:

```bash
python3 scripts/run_final_inputs.py \
  --config-dir configs/real_scenario_benchmark \
  --pattern '*hdbscan*.yaml' \
  --seeds None None None None None None None None None \
          None None None None None None None None None \
          None None None None None None None None None
```

```bash
python3 scripts/run_final_inputs.py \
  --config-dir configs/real_scenario_benchmark \
  --pattern 'real_scenario_densgnn*.yaml' \
  --seeds None None None None None None None None None \
          None None None None None None None None None \
          None None None None None None None None None
```
