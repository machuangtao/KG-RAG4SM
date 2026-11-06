#!/usr/bin/env python3
"""Rank local BFS paths into Excel (wraps preprocess/bfs_path_ranking.py)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment.config import DATASETS, OUTPUT_DIR, ROOT


def rank_dataset(dataset: str, depth: int = 4, limit: int = None) -> Path:
    bfs_file = OUTPUT_DIR / f"{dataset}_bfs_local_paths_h{depth}.json"
    if not bfs_file.exists():
        raise FileNotFoundError(f"Missing BFS output: {bfs_file}")

    original = ROOT / "datasets" / "NewData" / f"test_{dataset}_q.xlsx"
    if not original.exists():
        original = ROOT / "datasets" / "reproduce" / f"test_{dataset}_q_with_paths.xlsx"
    if not original.exists():
        original = ROOT / "datasets" / "original" / f"test_{dataset}_q.xlsx"

    out = ROOT / "datasets" / "reproduce" / f"test_{dataset}_q_with_paths_local_h{depth}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "preprocess" / "bfs_path_ranking.py"),
        "--original",
        str(original),
        "--bfs_paths",
        str(bfs_file),
        "--output",
        str(out),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Rank local BFS paths to Excel")
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        out = rank_dataset(ds, depth=args.depth, limit=args.limit)
        print(f"Ranked paths saved to {out}")


if __name__ == "__main__":
    main()
