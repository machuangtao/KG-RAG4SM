# WikiData5M Local In-Memory BFS Experiment

In-memory bidirectional BFS on WikiData5M using edges from `wikidata_embedding_triplet2/metadata_chunk_*.json` and labels from `data/wikidata5m_entity.txt` + `data/wikidata5m_relation.txt`.

Each run records **latency** (per-question and total), **path count**, and **triple count** in JSON metadata and a human-readable results table.

## Prerequisites

1. **Graph edges:** `wikidata_embedding_triplet2/metadata_chunk_*.json` (already in repo)
2. **Labels:** `data/wikidata5m_entity.txt`, `data/wikidata5m_relation.txt`
3. **Similarity input:** `testRes/{dataset}_similarity_results.json` (from `wikidata/cosine_similarity_search.py` or `experiment/gpu_similarity.py`)
4. **Python:** project venv with `torch`, `pandas`, `openpyxl`, etc.

## Quick start

From repo root:

```bash
# 1) Build graph cache (one-time, CPU)
python -m experiment.wikidata5m_graph --rebuild-graph --workers 8

# 2) Smoke test (5 questions, bank, depth 4)
python -m experiment.bfs_local --dataset bank --max_hops 4 --limit 5 --workers 2

# 3) Full depth sweep for one dataset (depths 1–4)
python -m experiment.run_depth_experiment --dataset bank --workers 2

# 4) Write consolidated results table (paths, triples, latency)
python -m experiment.write_full_results_txt
```

## What gets tracked

Every `testRes/{dataset}_bfs_local_paths_h{N}.json` run stores in `metadata`:

| Field | Meaning |
|-------|---------|
| `total_paths` | Total path strings retrieved across all questions |
| `total_triples` | Total edges in those paths |
| `timing.sum_question_time_seconds` | Sum of per-question BFS time (avg × N) |
| `timing.average_time_per_question_seconds` | Mean seconds per question |
| `timing.wall_clock_seconds` | Wall-clock for the parallel run |
| `timing.num_questions` | Questions timed |
| `timing.num_workers` | Thread workers used |

**Canonical summary table:** `experiment/logs/local_bfs_experiment_results.txt`  
Columns: Depth, Paths, Triples, Avg s/q, Total s — one row per dataset × depth (full dataset only).

## Datasets

`bank`, `movie`, `cms`, `emed`, `synthea` (see `experiment/config.py`).

## Long runs (screen)

```bash
# Full bank BFS depths 1–4, then refresh results txt
bash experiment/run_bank_full_and_update_txt.sh

# CMS → EMED → Synthea depths 3–4 (fixed asymmetric BFS)
screen -dmS clinical-h34 bash experiment/run_clinical_h34_fix.sh
```

Attach: `screen -r clinical-h34`

## Compare local vs remote SPARQL API

```bash
python -m experiment.compare_backends --dataset all
# → testRes/bfs_backend_comparison.csv
```

API (SPARQL) timings use `experiment/sparql_sample_estimate.py` (n=10 sample, extrapolated). Local timings use full JSON runs above.

## BFS algorithm notes

- **Asymmetric bidirectional frontier:** h3 = 2+1 hops, h4 = 2+2 hops (monotonic depth cost)
- **Path cap:** `MAX_PATHS_PER_PAIR = 5` after full frontier search (`bfs_search: full_frontier_then_cap`)
- **Benchmark mode:** `--disable-path-cap` on `run_depth_experiment` / `bfs_local` via `BENCHMARK_DISABLE_PATH_CAP`

## Outputs

| Path | Description |
|------|-------------|
| `data/wikidata5m_adjacency.pkl` | Cached in-memory graph |
| `testRes/{ds}_bfs_local_paths_h{N}.json` | Full BFS results + metadata |
| `testRes/{ds}_bfs_local_paths_h{N}.txt` | Human-readable paths |
| `experiment/logs/local_bfs_experiment_results.txt` | Consolidated paths/triples/latency table |
| `testRes/bfs_backend_comparison.csv` | Local vs SPARQL comparison |

## Hardware

- **Graph build/load:** CPU, ~4–10 GB RAM
- **BFS:** CPU threads (`--workers`, default min(8, cpu_count))
- **Similarity (optional):** CUDA via `experiment/gpu_similarity.py`
