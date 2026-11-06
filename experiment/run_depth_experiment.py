#!/usr/bin/env python3
"""Run local BFS for depths 1-4 and write depth tradeoff summary."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from experiment.bfs_local import run_bfs_dataset
from experiment.config import BENCHMARK_DISABLE_PATH_CAP, DATASETS, DEPTHS, OUTPUT_DIR, ROOT
from experiment.timing_meta import local_timing_from_meta


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Local BFS depth experiment (depths 1-4)")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument("--graph-workers", type=int, default=8)
    parser.add_argument(
        "--disable-path-cap",
        action="store_true",
        help="Disable MAX_PATHS_PER_PAIR early exit (also enabled when BENCHMARK_DISABLE_PATH_CAP=True)",
    )
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    sim = OUTPUT_DIR / f"{args.dataset}_similarity_results.json"
    if not sim.exists():
        print(f"Missing {sim}. Run gpu_similarity or run_pipeline first.", flush=True)
        sys.exit(1)

    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
    disable_path_cap = args.disable_path_cap or BENCHMARK_DISABLE_PATH_CAP

    print(f"=== Local BFS Depth Experiment: {args.dataset} ===", flush=True)
    if disable_path_cap:
        print("Path cap disabled (benchmark mode)", flush=True)
    all_meta = []
    for depth in DEPTHS:
        print(f"Running local BFS depth {depth}...", flush=True)
        t0 = time.time()
        out = run_bfs_dataset(
            dataset=args.dataset,
            max_hops=depth,
            limit=args.limit,
            workers=workers,
            rebuild_graph=args.rebuild_graph and depth == DEPTHS[0],
            graph_workers=args.graph_workers,
            disable_path_cap=disable_path_cap,
        )
        wall = time.time() - t0
        with open(out, "r", encoding="utf-8") as f:
            meta = json.load(f).get("metadata", {})
        meta["depth"] = depth
        meta["wall_time_seconds"] = wall
        all_meta.append(meta)
        timing = local_timing_from_meta(meta)
        print(
            f"  -> paths={meta.get('total_paths')} triples={meta.get('total_triples')} "
            f"sum_q={timing['sum_question_time_seconds']:.1f}s "
            f"wall={timing['wall_clock_seconds']:.1f}s",
            flush=True,
        )

    summary = OUTPUT_DIR / f"{args.dataset}_bfs_local_depth_tradeoff.txt"
    csv_path = OUTPUT_DIR / f"{args.dataset}_bfs_local_depth_tradeoff.csv"
    rows = []
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"Local BFS Depth Tradeoff: {args.dataset}\n")
        f.write("=" * 70 + "\n")
        f.write(
            f"{'Depth':<8} {'Paths':<12} {'Triples':<12} {'SumQ(s)':<12} {'Wall(s)':<12}\n"
        )
        f.write("-" * 70 + "\n")
        for m in all_meta:
            d = m.get("depth")
            paths = m.get("total_paths", 0)
            triples = m.get("total_triples", 0)
            timing = local_timing_from_meta(m)
            sum_q = timing["sum_question_time_seconds"]
            wall = timing["wall_clock_seconds"]
            f.write(f"{d:<8} {paths:<12} {triples:<12} {sum_q:<12.2f} {wall:<12.2f}\n")
            rows.append((d, paths, triples, sum_q, wall))
        f.write("=" * 70 + "\n")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dataset,depth,paths,triples,sum_question_time_seconds,wall_clock_seconds\n")
        for d, paths, triples, sum_q, wall in rows:
            f.write(f"{args.dataset},{d},{paths},{triples},{sum_q:.2f},{wall:.2f}\n")

    print(f"Summary: {summary}", flush=True)
    print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
