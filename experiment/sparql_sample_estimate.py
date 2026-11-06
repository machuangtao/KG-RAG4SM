#!/usr/bin/env python3
"""Run SPARQL BFS on a small sample and write extrapolated full-dataset timings."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from experiment.config import (
    DATASETS,
    DEPTHS,
    OUTPUT_DIR,
    ROOT,
    SPARQL_SAMPLE_N,
    SPARQL_SAMPLE_SEED,
)
from experiment.sample_questions import filter_similarity_subset, sample_question_ids
from experiment.sparql_results_resolver import meta_default_questions

logger = logging.getLogger(__name__)
SUMMARY_PATH = OUTPUT_DIR / "sparql_timing_estimates.json"


def _write_estimated(dataset: str, depth: int, sample_meta: dict, sample_n: int) -> Path:
    timing = sample_meta.get("timing", {})
    sample_avg = float(timing.get("average_time_per_question_seconds") or 0)
    n_full = meta_default_questions(dataset)
    est_total = sample_avg * n_full
    est = {
        "metadata": {
            "total_questions": n_full,
            "max_hops": depth,
            "total_paths": sample_meta.get("total_paths", ""),
            "total_triples": sample_meta.get("total_triples", ""),
            "timing": {
                "average_time_per_question_seconds": sample_avg,
                "total_time_seconds": est_total,
            },
            "_estimated": True,
            "_sample_n": sample_n,
            "_sample_total_sec": timing.get("total_time_seconds", 0),
            "_source": f"sample_n={sample_n} extrapolated to N={n_full}",
        },
        "results": [],
    }
    out = OUTPUT_DIR / f"{dataset}_bfs_sparql_paths_h{depth}_estimated.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(est, f, indent=2)
    logger.info(
        "Estimated %s h%d: sample_avg=%.3fs -> full N=%d tot=%.1fs",
        dataset,
        depth,
        sample_avg,
        n_full,
        est_total,
    )
    return out


def run_sample(dataset: str, depth: int, subset_path: Path, sample_n: int) -> dict:
    out_base = f"{dataset}_bfs_sparql_sample_h{depth}"
    sample_json = OUTPUT_DIR / f"{out_base}.json"
    cmd = [
        sys.executable,
        "-m",
        "modules.bfs_paths",
        "--input",
        str(subset_path.relative_to(ROOT)),
        "--output",
        out_base,
        "--max_hops",
        str(depth),
        "--limit",
        str(sample_n),
        "--batch_size",
        str(min(10, sample_n)),
    ]
    logger.info("Running SPARQL sample: %s depth=%d n=%d", dataset, depth, sample_n)
    subprocess.run(cmd, cwd=ROOT, check=True)
    with open(sample_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {})


def run_estimates(
    datasets: list[str],
    depths: tuple[int, ...] = DEPTHS,
    sample_n: int = SPARQL_SAMPLE_N,
    seed: int = SPARQL_SAMPLE_SEED,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_n": sample_n,
        "seed": seed,
        "datasets": {},
    }

    for ds in datasets:
        qids = sample_question_ids(ds, n=sample_n, seed=seed)
        subset = filter_similarity_subset(ds, qids, OUTPUT_DIR / f"{ds}_similarity_results_n{sample_n}.json")
        n_full = meta_default_questions(ds)
        ds_entry = {"n_full": n_full, "sample_ids": qids, "depths": {}}
        for depth in depths:
            sample_meta = run_sample(ds, depth, subset, len(qids))
            est_path = _write_estimated(ds, depth, sample_meta, len(qids))
            t = sample_meta.get("timing", {})
            ds_entry["depths"][str(depth)] = {
                "sample_avg_sec": t.get("average_time_per_question_seconds"),
                "sample_total_sec": t.get("total_time_seconds"),
                "estimated_avg_sec": t.get("average_time_per_question_seconds"),
                "estimated_total_sec": float(t.get("average_time_per_question_seconds") or 0) * n_full,
                "estimated_file": str(est_path),
            }
        summary["datasets"][ds] = ds_entry

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", SUMMARY_PATH)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="SPARQL sample + extrapolated API timings")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    parser.add_argument("--sample-n", type=int, default=SPARQL_SAMPLE_N)
    parser.add_argument("--seed", type=int, default=SPARQL_SAMPLE_SEED)
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    run_estimates(args.datasets, tuple(args.depths), sample_n=args.sample_n, seed=args.seed)


if __name__ == "__main__":
    main()
