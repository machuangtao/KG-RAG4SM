#!/usr/bin/env python3
"""Write the canonical full-dataset local BFS results table from testRes JSON runs."""
from __future__ import annotations

import json
import time
from pathlib import Path

from experiment.config import DATASETS, DEPTHS, GRAPH_META_CACHE, OUTPUT_DIR, RESULTS_TXT
from experiment.dataset_registry import DEFAULT_QUESTION_COUNTS
from experiment.timing_meta import local_timing_from_meta


def _read_meta_header(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size < 200:
        return None
    buf: list[str] = []
    started = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not started and '"metadata"' in line:
                started = True
                after = line.split("{", 1)
                buf.append("{" + after[1] if len(after) > 1 else "{")
                continue
            if started:
                if line.rstrip() == "  },":
                    buf.append("}\n")
                    break
                buf.append(line)
    if not buf:
        return None
    return json.loads("".join(buf))


def _full_run_meta(dataset: str, depth: int) -> dict | None:
    return _read_meta_header(OUTPUT_DIR / f"{dataset}_bfs_local_paths_h{depth}.json")


def write_full_results_txt(out_path: Path | None = None) -> Path:
    """Build one results table from full-dataset JSON only (no sample extrapolation)."""
    out_path = out_path or RESULTS_TXT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("WikiData5M Local BFS — Full Dataset Results")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source: testRes/{{dataset}}_bfs_local_paths_h{{depth}}.json")
    lines.append("")

    if GRAPH_META_CACHE.exists():
        with open(GRAPH_META_CACHE, "r", encoding="utf-8") as f:
            gmeta = json.load(f)
        lines.append(f"  Nodes: {gmeta.get('node_count', '?')}")
        lines.append(f"  Directed edges: {gmeta.get('directed_edge_count', '?')}")
        lines.append(f"  Build time (s): {gmeta.get('build_seconds', '?')}")
        lines.append("")

    for ds in DATASETS:
        n_full = DEFAULT_QUESTION_COUNTS[ds]
        lines.append("-" * 80)
        lines.append(f"Dataset: {ds.upper()}  ({n_full} questions)")
        lines.append("-" * 80)
        lines.append(f"{'Depth':<8} {'Paths':<12} {'Triples':<12} {'Avg s/q':<12} {'Total s':<12}")

        for depth in DEPTHS:
            meta = _full_run_meta(ds, depth)
            if not meta:
                lines.append(f"{depth:<8} {'(missing)':<12}")
                continue

            timing = local_timing_from_meta(meta)
            nq = timing["num_questions"]
            if nq != n_full:
                lines.append(
                    f"{depth:<8} {'(incomplete)':<12} "
                    f"{'':12} {'':12} {f'{nq}/{n_full} q':<12}"
                )
                continue

            paths = int(meta.get("total_paths", 0) or 0)
            triples = int(meta.get("total_triples", 0) or 0)
            avg = timing["average_time_per_question_seconds"]
            tot = timing["sum_question_time_seconds"]
            lines.append(
                f"{depth:<8} {paths:<12} {triples:<12} {avg:<12.3f} {tot:<12.1f}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    p = write_full_results_txt()
    print(f"Wrote {p}")
