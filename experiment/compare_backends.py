#!/usr/bin/env python3
"""Compare local WikiData5M BFS vs SPARQL BFS results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from experiment.config import DATASETS, DEPTHS, OUTPUT_DIR, ROOT
from experiment.sparql_results_resolver import (
    load_sparql_metadata,
    sparql_questions_with_paths,
)
from experiment.timing_meta import local_timing_from_meta


def _load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {})


def _meta_from_checkpoint(cp_path: Path, dataset: str, depth: int) -> dict:
    try:
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    results = cp.get("question_results", {})
    if not results:
        return {}
    times = cp.get("question_times", [])
    total_paths = sum(len(r.get("paths", [])) for r in results.values() if isinstance(r, dict))
    total_triples = 0
    for r in results.values():
        if not isinstance(r, dict):
            continue
        for path in r.get("paths", []):
            if isinstance(path, str):
                total_triples += max(0, len(path.split("-->")) - 1)
    total_time = sum(times) if times else 0.0
    n = len(times) if times else len(results)
    return {
        "total_paths": total_paths,
        "total_triples": total_triples,
        "total_questions": n,
        "max_hops": depth,
        "missing_entity_lookups": cp.get("missing_entity_lookups", 0),
        "timing": {
            "sum_question_time_seconds": total_time,
            "average_time_per_question_seconds": total_time / n if n else 0,
            "wall_clock_seconds": total_time,
            "num_questions": n,
        },
        "_partial": True,
        "_source": f"checkpoint:{cp_path.name} ({n} questions)",
    }


def load_local_metadata(dataset: str, depth: int) -> tuple[dict, Path | None]:
    final = OUTPUT_DIR / f"{dataset}_bfs_local_paths_h{depth}.json"
    if final.exists() and final.stat().st_size > 100:
        return _load_meta(final), final
    cp = OUTPUT_DIR / f"{dataset}_bfs_local_paths_h{depth}.json.checkpoint.json"
    if cp.exists() and cp.stat().st_size > 500:
        meta = _meta_from_checkpoint(cp, dataset, depth)
        if meta:
            return meta, None
    return {}, None


def _questions_with_paths(path: Path | None, dataset: str = "", depth: int = 0) -> int:
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        return sum(1 for r in results if r.get("paths"))
    cp = OUTPUT_DIR / f"{dataset}_bfs_local_paths_h{depth}.json.checkpoint.json"
    if dataset and cp.exists():
        try:
            with open(cp, "r", encoding="utf-8") as f:
                cpd = json.load(f)
            results = cpd.get("question_results", {})
            return sum(
                1
                for r in results.values()
                if isinstance(r, dict) and r.get("paths")
            )
        except (json.JSONDecodeError, OSError):
            pass
    return 0


def compare_dataset(dataset: str, depths=DEPTHS) -> list:
    rows = []
    for depth in depths:
        local_meta, local_path = load_local_metadata(dataset, depth)
        sparql_meta, sparql_path = load_sparql_metadata(dataset, depth)
        st = sparql_meta.get("timing", {})
        lt = local_timing_from_meta(local_meta)
        spq_q = (
            sparql_questions_with_paths(sparql_path, dataset, depth)
            if sparql_path
            else sparql_questions_with_paths(None, dataset, depth)
            or sparql_meta.get("total_questions", "")
        )
        note = ""
        if local_meta.get("_partial"):
            note = "local partial"
        rows.append(
            {
                "dataset": dataset,
                "depth": depth,
                "local_paths": local_meta.get("total_paths", ""),
                "local_triples": local_meta.get("total_triples", ""),
                "local_num_questions": lt.get("num_questions", ""),
                "local_avg_sec": lt.get("average_time_per_question_seconds", ""),
                "local_total_sec": lt.get("sum_question_time_seconds", ""),
                "local_wall_sec": lt.get("wall_clock_seconds", ""),
                "local_workers": lt.get("num_workers", ""),
                "local_questions_with_paths": _questions_with_paths(local_path, dataset, depth),
                "sparql_paths": sparql_meta.get("total_paths", ""),
                "sparql_triples": sparql_meta.get("total_triples", ""),
                "sparql_avg_sec": st.get("average_time_per_question_seconds", ""),
                "sparql_total_sec": st.get("total_time_seconds", ""),
                "sparql_questions_with_paths": spq_q,
                "local_missing_entities": local_meta.get("missing_entity_lookups", ""),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare local vs SPARQL BFS backends")
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    all_rows = []
    for ds in datasets:
        all_rows.extend(compare_dataset(ds))

    out_csv = OUTPUT_DIR / "bfs_backend_comparison.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    if not all_rows:
        print("No comparison rows produced (missing result files?)")
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    for ds in datasets:
        ds_rows = [r for r in all_rows if r["dataset"] == ds]
        ds_txt = OUTPUT_DIR / f"{ds}_bfs_backend_comparison.txt"
        with open(ds_txt, "w", encoding="utf-8") as f:
            f.write(f"BFS Backend Comparison: {ds}\n")
            f.write("=" * 90 + "\n")
            for r in ds_rows:
                f.write(
                    f"depth={r['depth']}  local_paths={r['local_paths']} sparql_paths={r['sparql_paths']}  "
                    f"local_avg_s={r['local_avg_sec']} sparql_avg_s={r['sparql_avg_sec']}\n"
                )
        print(f"Wrote {ds_txt}")

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
