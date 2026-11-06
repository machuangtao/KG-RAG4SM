#!/usr/bin/env python3
"""Sample local BFS per dataset/depth, extrapolate timings, write depth timing table."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiment.bfs_local import run_bfs_dataset
from experiment.config import (
    DATASETS,
    DEPTHS,
    EXPERIMENT_LOG_DIR,
    OUTPUT_DIR,
    ROOT,
    SPARQL_SAMPLE_N,
)
from experiment.dataset_registry import DEFAULT_QUESTION_COUNTS
from experiment.sample_questions import filter_similarity_subset, sample_question_ids
from experiment.timing_meta import local_timing_from_meta
from experiment.write_results_table import DEPTH_TIMING_TABLE_TXT, SIMPLE_COLUMNS, _fmt

SUMMARY_PATH = OUTPUT_DIR / "local_depth_timing_estimates.json"
PATHS_TIME_TABLE_TXT = "bfs_paths_time_comparison.txt"


def _load_sparql_estimates() -> dict:
    path = OUTPUT_DIR / "sparql_timing_estimates.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("datasets", {})


def _sparql_paths(ds: str, depth: int) -> int | str:
    path = OUTPUT_DIR / f"{ds}_bfs_sparql_paths_h{depth}_estimated.json"
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("metadata", {}).get("total_paths", 0) or 0)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return ""


def _sparql_row(ds: str, depth: int, spq: dict) -> tuple[float, float]:
    d = str(depth)
    depths = spq.get(ds, {}).get("depths", {})
    if d not in depths:
        return 0.0, 0.0
    entry = depths[d]
    return (
        float(entry.get("estimated_avg_sec") or 0),
        float(entry.get("estimated_total_sec") or 0),
    )


def _load_summary() -> dict:
    if SUMMARY_PATH.exists():
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def run_sample(
    datasets: tuple[str, ...],
    sample_n: int,
    seed: int,
    workers: int,
    disable_path_cap: bool,
    *,
    depths: tuple[int, ...] | None = None,
    merge: bool = True,
) -> dict:
    from experiment.wikidata5m_graph import load_graph_cache, load_labels

    adj, graph_meta = load_graph_cache(rebuild=False)
    entity_labels, relation_labels = load_labels()
    spq_all = _load_sparql_estimates()
    prior = _load_summary() if merge else {}
    out: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_n": sample_n,
        "seed": seed,
        "disable_path_cap": disable_path_cap,
        "datasets": dict(prior.get("datasets", {})),
    }

    for ds in datasets:
        n_full = DEFAULT_QUESTION_COUNTS[ds]
        qids = sample_question_ids(ds, n=sample_n, seed=seed)
        sim_subset = filter_similarity_subset(
            ds, qids, OUTPUT_DIR / f"{ds}_similarity_results_sample_n{sample_n}.json"
        )
        prior_ds = out["datasets"].get(ds, {})
        ds_entry = {
            "n_full": n_full,
            "sample_ids": qids,
            "depths": dict(prior_ds.get("depths", {})),
        }
        print(f"=== {ds} sample n={sample_n} full N={n_full} ===", flush=True)

        run_depths = depths if depths is not None else DEPTHS
        for depth in run_depths:
            sample_json = OUTPUT_DIR / f"{ds}_bfs_local_sample_h{depth}_n{sample_n}.json"
            if sample_json.exists() and sample_json.stat().st_size > 200:
                with open(sample_json, "r", encoding="utf-8") as f:
                    meta = json.load(f).get("metadata", {})
                if meta.get("timing", {}).get("sum_question_time_seconds") is not None:
                    print(f"  depth {depth} (reuse {sample_json.name})", flush=True)
                    t = local_timing_from_meta(meta)
                    est_sum = t["average_time_per_question_seconds"] * n_full
                    est_wall = t["wall_clock_seconds"] * (n_full / max(sample_n, 1))
                    spq_avg, spq_tot = _sparql_row(ds, depth, spq_all)
                    sample_paths = int(meta.get("total_paths", 0) or 0)
                    est_paths = int(round(sample_paths * n_full / max(sample_n, 1)))
                    ds_entry["depths"][str(depth)] = {
                        "sample_paths": sample_paths,
                        "estimated_paths": est_paths,
                        "sparql_paths": _sparql_paths(ds, depth),
                        "sample_triples": meta.get("total_triples", 0),
                        "sample_avg_sec": t["average_time_per_question_seconds"],
                        "sample_sum_sec": t["sum_question_time_seconds"],
                        "sample_wall_sec": t["wall_clock_seconds"],
                        "estimated_avg_sec": t["average_time_per_question_seconds"],
                        "estimated_sum_sec": est_sum,
                        "estimated_wall_sec": est_wall,
                        "sparql_avg_sec": spq_avg,
                        "sparql_total_sec": spq_tot,
                        "sample_file": str(sample_json),
                        "_reused": True,
                    }
                    continue
            print(f"  depth {depth}...", flush=True)
            out_path = run_bfs_dataset(
                dataset=ds,
                max_hops=depth,
                input_file=sim_subset,
                output_base=f"{ds}_bfs_local_sample_h{depth}_n{sample_n}",
                workers=workers,
                adj=adj,
                graph_meta=graph_meta,
                entity_labels=entity_labels,
                relation_labels=relation_labels,
                disable_path_cap=disable_path_cap,
            )
            with open(out_path, "r", encoding="utf-8") as f:
                meta = json.load(f).get("metadata", {})
            t = local_timing_from_meta(meta)
            scale = n_full / max(sample_n, 1)
            est_sum = t["average_time_per_question_seconds"] * n_full
            est_wall = t["wall_clock_seconds"] * scale
            spq_avg, spq_tot = _sparql_row(ds, depth, spq_all)
            sample_paths = int(meta.get("total_paths", 0) or 0)
            est_paths = int(round(sample_paths * n_full / max(sample_n, 1)))
            ds_entry["depths"][str(depth)] = {
                "sample_paths": sample_paths,
                "estimated_paths": est_paths,
                "sparql_paths": _sparql_paths(ds, depth),
                "sample_triples": meta.get("total_triples", 0),
                "sample_avg_sec": t["average_time_per_question_seconds"],
                "sample_sum_sec": t["sum_question_time_seconds"],
                "sample_wall_sec": t["wall_clock_seconds"],
                "estimated_avg_sec": t["average_time_per_question_seconds"],
                "estimated_sum_sec": est_sum,
                "estimated_wall_sec": est_wall,
                "sparql_avg_sec": spq_avg,
                "sparql_total_sec": spq_tot,
                "sample_file": str(out_path),
            }
            print(
                f"    sample avg={t['average_time_per_question_seconds']:.3f}s "
                f"est_sum={est_sum:.1f}s",
                flush=True,
            )
        out["datasets"][ds] = ds_entry
        SUMMARY_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    SUMMARY_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _depth_entry_from_sample(
    ds: str,
    depth: int,
    sample_n: int,
    n_full: int,
    meta: dict,
    spq_all: dict,
) -> dict:
    t = local_timing_from_meta(meta)
    est_sum = t["average_time_per_question_seconds"] * n_full
    est_wall = t["wall_clock_seconds"] * (n_full / max(sample_n, 1))
    spq_avg, spq_tot = _sparql_row(ds, depth, spq_all)
    sample_paths = int(meta.get("total_paths", 0) or 0)
    est_paths = int(round(sample_paths * n_full / max(sample_n, 1)))
    sample_json = OUTPUT_DIR / f"{ds}_bfs_local_sample_h{depth}_n{sample_n}.json"
    return {
        "sample_paths": sample_paths,
        "estimated_paths": est_paths,
        "sparql_paths": _sparql_paths(ds, depth),
        "sample_triples": meta.get("total_triples", 0),
        "sample_avg_sec": t["average_time_per_question_seconds"],
        "sample_sum_sec": t["sum_question_time_seconds"],
        "sample_wall_sec": t["wall_clock_seconds"],
        "estimated_avg_sec": t["average_time_per_question_seconds"],
        "estimated_sum_sec": est_sum,
        "estimated_wall_sec": est_wall,
        "sparql_avg_sec": spq_avg,
        "sparql_total_sec": spq_tot,
        "sample_file": str(sample_json),
    }


def rebuild_summary_from_samples(
    sample_n: int = 10,
    seed: int = 42,
    disable_path_cap: bool = True,
) -> dict:
    """Rebuild estimates JSON + table from all on-disk sample result files."""
    spq_all = _load_sparql_estimates()
    out: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_n": sample_n,
        "seed": seed,
        "disable_path_cap": disable_path_cap,
        "datasets": {},
    }
    for ds in DATASETS:
        n_full = DEFAULT_QUESTION_COUNTS[ds]
        qids = sample_question_ids(ds, n=sample_n, seed=seed)
        ds_entry = {"n_full": n_full, "sample_ids": qids, "depths": {}}
        for depth in DEPTHS:
            sample_json = OUTPUT_DIR / f"{ds}_bfs_local_sample_h{depth}_n{sample_n}.json"
            if not sample_json.exists() or sample_json.stat().st_size < 200:
                continue
            with open(sample_json, "r", encoding="utf-8") as f:
                meta = json.load(f).get("metadata", {})
            if meta.get("timing", {}).get("sum_question_time_seconds") is None:
                continue
            ds_entry["depths"][str(depth)] = _depth_entry_from_sample(
                ds, depth, sample_n, n_full, meta, spq_all
            )
        out["datasets"][ds] = ds_entry
    SUMMARY_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _results_only_table(columns: list, rows: list) -> str:
    col_sep = "  "
    header = col_sep.join(name.ljust(w) for _, name, w in columns)
    sep = col_sep.join("-" * w for _, _, w in columns)
    body = [col_sep.join(_fmt(r.get(key), w) for key, _, w in columns) for r in rows]
    return "\n".join([header, sep, *body, ""])


PATHS_TIME_COLUMNS = [
    ("dataset", "Dataset", 8),
    ("depth", "Depth", 5),
    ("sample_paths", "LocPaths", 10),
    ("sparql_paths", "APIPaths", 10),
    ("local_total_sec", "LocalSum(s)", 12),
    ("sparql_total_sec", "APITot(s)", 12),
]


def write_timing_table(summary: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        ds_entry = summary.get("datasets", {}).get(ds, {})
        for depth in DEPTHS:
            d = ds_entry.get("depths", {}).get(str(depth), {})
            rows.append(
                {
                    "dataset": ds,
                    "depth": depth,
                    "sample_paths": d.get("sample_paths", ""),
                    "sparql_paths": d.get("sparql_paths", _sparql_paths(ds, depth)),
                    "local_avg_sec": d.get("estimated_avg_sec", ""),
                    "local_total_sec": d.get("estimated_sum_sec", ""),
                    "local_wall_sec": d.get("estimated_wall_sec", ""),
                    "sparql_avg_sec": d.get("sparql_avg_sec", ""),
                    "sparql_total_sec": d.get("sparql_total_sec", ""),
                }
            )
    out_path.write_text(_results_only_table(SIMPLE_COLUMNS, rows), encoding="utf-8")
    return out_path


def write_paths_time_table(summary: dict, out_path: Path | None = None) -> Path:
    out_path = out_path or (EXPERIMENT_LOG_DIR / PATHS_TIME_TABLE_TXT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        ds_entry = summary.get("datasets", {}).get(ds, {})
        n_full = ds_entry.get("n_full", DEFAULT_QUESTION_COUNTS.get(ds, 0))
        sample_n = summary.get("sample_n", 10)
        for depth in DEPTHS:
            d = ds_entry.get("depths", {}).get(str(depth), {})
            rows.append(
                {
                    "dataset": ds,
                    "depth": depth,
                    "sample_paths": d.get("sample_paths", ""),
                    "sparql_paths": d.get("sparql_paths", _sparql_paths(ds, depth)),
                    "local_total_sec": d.get("estimated_sum_sec", ""),
                    "sparql_total_sec": d.get("sparql_total_sec", ""),
                }
            )
    out_path.write_text(_results_only_table(PATHS_TIME_COLUMNS, rows), encoding="utf-8")
    return out_path


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Local BFS sample + extrapolated depth table")
    parser.add_argument("--sample-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=None,
        help="Only run these hop depths (default: 1-4)",
    )
    parser.add_argument(
        "--disable-path-cap",
        action="store_true",
        default=True,
        help="Disable MAX_PATHS_PER_PAIR early exit (default: off cap for benchmark)",
    )
    parser.add_argument(
        "--enable-path-cap",
        action="store_true",
        help="Force MAX_PATHS_PER_PAIR=5 (use for heavy datasets to avoid OOM)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(EXPERIMENT_LOG_DIR / DEPTH_TIMING_TABLE_TXT),
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Rebuild summary/table from existing sample JSON (no BFS runs)",
    )
    args = parser.parse_args()

    disable_cap = args.disable_path_cap and not args.enable_path_cap

    if args.refresh_only:
        from experiment.write_full_results_txt import write_full_results_txt

        rebuild_summary_from_samples(
            args.sample_n, args.seed, disable_cap
        )
        table_path = write_full_results_txt()
        print(f"Wrote {SUMMARY_PATH}", flush=True)
        print(f"Wrote {table_path}", flush=True)
        return

    depth_tuple = tuple(args.depths) if args.depths else None
    summary = run_sample(
        tuple(args.datasets),
        args.sample_n,
        args.seed,
        args.workers,
        disable_cap,
        depths=depth_tuple,
    )
    summary = rebuild_summary_from_samples(
        args.sample_n, args.seed, disable_cap
    )
    from experiment.write_full_results_txt import write_full_results_txt

    table_path = write_full_results_txt()
    print(f"Wrote {SUMMARY_PATH}", flush=True)
    print(f"Wrote {table_path}", flush=True)


if __name__ == "__main__":
    main()
