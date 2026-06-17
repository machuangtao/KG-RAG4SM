#!/usr/bin/env python3
"""
Full WikiData5M local BFS experiment pipeline.

Stages:
  1. Build/load in-memory graph from wikidata_embedding_triplet2 metadata (CPU)
  2. Multi-GPU cosine similarity for bank/movie if needed (all CUDA devices)
  3. Local BFS depth 1-4 per dataset (CPU parallel workers)
  4. Compare vs SPARQL baselines
  5. Rank paths to Excel

Usage (from repo root):
  python -m experiment.run_pipeline --datasets bank movie cms
  python -m experiment.run_pipeline --datasets bank --limit 5 --skip-rank
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

from experiment.compare_backends import compare_dataset
from experiment.config import DATASETS, DEPTHS, EXPERIMENT_LOG_DIR, OUTPUT_DIR, ROOT
from experiment.gpu_similarity import ensure_similarity
from experiment.wikidata5m_graph import load_graph_cache

logger = logging.getLogger(__name__)


def _log_gpus():
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info("PyTorch CUDA devices: %d", n)
    for i in range(n):
        logger.info("  cuda:%d -> %s", i, torch.cuda.get_device_name(i))


def run_pipeline(
    datasets,
    limit=None,
    rebuild_graph=False,
    graph_workers=8,
    bfs_workers=None,
    skip_similarity=False,
    skip_bfs=False,
    skip_compare=False,
    skip_rank=False,
    rank_depth=4,
    max_chunks=None,
    resume=True,
    results_txt=None,
):
    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    t_pipeline = time.time()
    _log_gpus()

    # Stage 1: graph
    logger.info("=== Stage 1: Build/load WikiData5M adjacency ===")
    load_graph_cache(rebuild=rebuild_graph, workers=graph_workers, max_chunks=max_chunks)

    # Stage 2: GPU similarity
    if not skip_similarity:
        logger.info("=== Stage 2: Multi-GPU similarity search ===")
        ensure_similarity(list(datasets), limit=limit, force=limit is not None)
    else:
        logger.info("Skipping similarity stage")

    bfs_workers = bfs_workers or min(8, max(1, (os.cpu_count() or 4)))

    # Stage 3: local BFS per dataset x depth
    if not skip_bfs:
        from experiment.bfs_local import run_bfs_dataset
        from experiment.wikidata5m_graph import load_labels

        logger.info("=== Stage 3: Local BFS (workers=%d) ===", bfs_workers)
        adj, graph_meta = load_graph_cache(
            rebuild=False, workers=graph_workers, max_chunks=max_chunks
        )
        entity_labels, relation_labels = load_labels()
        for ds in datasets:
            for depth in DEPTHS:
                logger.info("BFS %s depth=%d", ds, depth)
                run_bfs_dataset(
                    dataset=ds,
                    max_hops=depth,
                    limit=limit,
                    workers=bfs_workers,
                    resume=resume,
                    rebuild_graph=False,
                    graph_workers=graph_workers,
                    max_chunks=max_chunks,
                    adj=adj,
                    graph_meta=graph_meta,
                    entity_labels=entity_labels,
                    relation_labels=relation_labels,
                )
    else:
        logger.info("Skipping BFS stage")

    # Stage 4: comparison
    if not skip_compare:
        logger.info("=== Stage 4: Compare local vs SPARQL ===")
        import csv

        all_rows = []
        for ds in datasets:
            all_rows.extend(compare_dataset(ds))
        out_csv = OUTPUT_DIR / "bfs_backend_comparison.csv"
        if all_rows:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
            logger.info("Wrote %s", out_csv)
    else:
        logger.info("Skipping compare stage")

    # Stage 5: rank paths
    if not skip_rank:
        logger.info("=== Stage 5: Path ranking to Excel ===")
        from experiment.path_ranking import rank_dataset

        for ds in datasets:
            try:
                out = rank_dataset(ds, depth=rank_depth, limit=limit)
                logger.info("Ranked %s -> %s", ds, out)
            except Exception as e:
                logger.warning("Ranking failed for %s: %s", ds, e)
    else:
        logger.info("Skipping rank stage")

    logger.info("Pipeline finished in %.1fs", time.time() - t_pipeline)

    from experiment.write_results_txt import write_results_txt
    from experiment.write_results_table import write_tables

    txt_path = write_results_txt(
        Path(results_txt) if results_txt else EXPERIMENT_LOG_DIR / "full_experiment_results.txt"
    )
    table_txt, table_md = write_tables(EXPERIMENT_LOG_DIR)
    logger.info("Results summary written to %s", txt_path)
    logger.info("Results table written to %s and %s", table_txt, table_md)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="WikiData5M local BFS full experiment")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to run (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit questions (smoke test)")
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument("--graph-workers", type=int, default=8)
    parser.add_argument("--bfs-workers", type=int, default=None)
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--skip-bfs", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-rank", action="store_true")
    parser.add_argument("--rank-depth", type=int, default=4)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Use only first N metadata chunks (smoke test; separate cache file)",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not resume BFS from checkpoints")
    parser.add_argument(
        "--results-txt",
        type=str,
        default=None,
        help="Path for consolidated results .txt (default: experiment/logs/full_experiment_results.txt)",
    )
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    run_pipeline(
        datasets=args.datasets,
        limit=args.limit,
        rebuild_graph=args.rebuild_graph,
        graph_workers=args.graph_workers,
        bfs_workers=args.bfs_workers,
        skip_similarity=args.skip_similarity,
        skip_bfs=args.skip_bfs,
        skip_compare=args.skip_compare,
        skip_rank=args.skip_rank,
        rank_depth=args.rank_depth,
        max_chunks=args.max_chunks,
        resume=not args.no_resume,
        results_txt=args.results_txt,
    )


if __name__ == "__main__":
    main()
