#!/usr/bin/env python3
"""CMS fix: fast parallel similarity + BFS + results table."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from experiment.compare_backends import compare_dataset
from experiment.config import DEPTHS, EXPERIMENT_LOG_DIR, OUTPUT_DIR, ROOT
from experiment.path_ranking import rank_dataset
from experiment.write_results_table import write_tables
from experiment.write_results_txt import write_results_txt

logger = logging.getLogger(__name__)


def _similarity_count() -> int:
    path = OUTPUT_DIR / "cms_similarity_results.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8)))
    parser.add_argument("--bfs-workers", type=int, default=min(8, max(1, (os.cpu_count() or 4))))
    parser.add_argument("--skip-similarity", action="store_true", help="Skip if cms_similarity_results.json exists")
    parser.add_argument("--resume", action="store_true", help="Resume BFS from checkpoints")
    parser.add_argument("--start-depth", type=int, default=1, help="First BFS depth to run (1-4)")
    parser.add_argument("--finish-only", action="store_true", help="Only write comparison CSV + result tables")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(EXPERIMENT_LOG_DIR / "cms_rerun.log", mode="a"),
        ],
    )

    t0 = time.time()

    if args.finish_only:
        logger.info("Finish-only: writing comparison + result tables")
    else:
        if not args.skip_similarity:
            from experiment.cms_similarity_fast import run_cms_similarity
            from wikidata.cosine_similarity_search import save_results

            logger.info("Step 1/5: CMS similarity (%d CPU workers)", args.workers)
            results = run_cms_similarity(workers=args.workers)
            save_results(results, "cms")
            logger.info("Similarity saved: %d questions", len(results))
        else:
            n = _similarity_count()
            if n < 2563:
                logger.error("Similarity incomplete (%d/2563). Run without --skip-similarity.", n)
                sys.exit(1)
            logger.info("Step 1/5: Skipping similarity (%d questions on disk)", n)

        from experiment.bfs_local import run_bfs_dataset
        from experiment.wikidata5m_graph import load_graph_cache, load_labels

        logger.info("Step 2/5: Load graph + labels")
        adj, graph_meta = load_graph_cache(rebuild=False)
        entity_labels, relation_labels = load_labels()

        depths = [d for d in DEPTHS if d >= args.start_depth]
        logger.info("Step 3/5: CMS BFS depths %s (resume=%s)", depths, args.resume)
        for depth in depths:
            logger.info("  depth=%d", depth)
            run_bfs_dataset(
                dataset="cms",
                max_hops=depth,
                workers=args.bfs_workers,
                resume=args.resume,
                adj=adj,
                graph_meta=graph_meta,
                entity_labels=entity_labels,
                relation_labels=relation_labels,
            )

    logger.info("Step 4/5: Comparison CSV")
    all_rows = []
    for ds in ("bank", "movie", "cms"):
        all_rows.extend(compare_dataset(ds))
    with open(OUTPUT_DIR / "bfs_backend_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    logger.info("Step 5/5: Rank CMS + write tables")
    try:
        rank_dataset("cms", depth=4)
    except Exception as e:
        logger.warning("Ranking failed: %s", e)

    txt_path = EXPERIMENT_LOG_DIR / "full_experiment_results.txt"
    write_results_txt(txt_path)
    txt, md = write_tables(EXPERIMENT_LOG_DIR)
    logger.info("Done in %.1fs. Results: %s | Table: %s | %s", time.time() - t0, txt_path, txt, md)


if __name__ == "__main__":
    main()
