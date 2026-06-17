#!/usr/bin/env python3
"""Write experiment comparison as formatted tables (.txt and .md)."""
from __future__ import annotations

import csv
import time
from pathlib import Path

from experiment.compare_backends import compare_dataset
from experiment.config import DATASETS, DEPTHS, EXPERIMENT_LOG_DIR, OUTPUT_DIR, SPARQL_SAMPLE_N

DEPTH_TIMING_TABLE_TXT = "bfs_local_depth_timing_table.txt"
DEPTH_TIMING_TABLE_MD = "bfs_local_depth_timing_table.md"


def _load_comparison_rows() -> list:
    csv_path = OUTPUT_DIR / "bfs_backend_comparison.csv"
    rows = []
    for ds in DATASETS:
        rows.extend(compare_dataset(ds))
    if rows:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    elif csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return rows


# Detailed columns used by write_results_txt (full comparison section).
COLUMNS = [
    ("dataset", "Dataset", 8),
    ("depth", "Depth", 6),
    ("local_num_questions", "LocN", 6),
    ("local_paths", "LocPaths", 10),
    ("local_triples", "LocTriples", 10),
    ("local_avg_sec", "LocAvgS", 10),
    ("local_total_sec", "LocSumS", 10),
    ("local_wall_sec", "LocWallS", 10),
    ("local_workers", "LocWrk", 7),
    ("local_questions_with_paths", "LocQ", 8),
    ("sparql_paths", "SpqPaths", 10),
    ("sparql_triples", "SpqTriples", 10),
    ("sparql_avg_sec", "SpqAvgS", 10),
    ("sparql_total_sec", "SpqTotS", 10),
    ("sparql_questions_with_paths", "SpqQ", 8),
    ("local_missing_entities", "MissEnt", 10),
]


SIMPLE_COLUMNS = [
    ("dataset", "Dataset", 8),
    ("depth", "Depth", 5),
    ("sample_paths", "LocPaths", 9),
    ("sparql_paths", "APIPaths", 9),
    ("local_avg_sec", "Local(s/q)", 11),
    ("local_total_sec", "LocalSum(s)", 12),
    ("local_wall_sec", "LocalWall(s)", 12),
    ("sparql_avg_sec", "API(s/q)*", 11),
    ("sparql_total_sec", "APITot(s)*", 12),
]


def _fmt(v, width: int) -> str:
    if v is None or v == "":
        return "-".rjust(width)
    try:
        f = float(v)
        if abs(f) >= 1000:
            return f"{f:.1f}".rjust(width)
        if f == int(f):
            return f"{int(f)}".rjust(width)
        return f"{f:.3f}".rjust(width)
    except (TypeError, ValueError):
        s = str(v)
        if len(s) > width:
            s = s[: width - 1] + "…"
        return s.rjust(width)


def _table_lines(rows: list) -> list:
    col_sep = "  "
    header = col_sep.join(name.ljust(w) for _, name, w in SIMPLE_COLUMNS)
    sep = col_sep.join("-" * w for _, _, w in SIMPLE_COLUMNS)
    body = [col_sep.join(_fmt(r.get(key), w) for key, _, w in SIMPLE_COLUMNS) for r in rows]
    return [header, sep, *body, ""]


def write_tables(
    out_dir: Path | None = None,
    *,
    legacy_name: str = "full_experiment_results_table.txt",
) -> tuple[Path, Path, Path, Path]:
    out_dir = out_dir or EXPERIMENT_LOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_comparison_rows()

    legacy_txt = out_dir / legacy_name
    legacy_md = out_dir / legacy_name.replace(".txt", ".md")
    depth_txt = out_dir / DEPTH_TIMING_TABLE_TXT
    depth_md = out_dir / DEPTH_TIMING_TABLE_MD

    table_body = _table_lines(rows)
    legacy_txt.write_text("\n".join(table_body), encoding="utf-8")
    depth_txt.write_text("\n".join(table_body), encoding="utf-8")

    md_lines = [
        "# WikiData5M Local BFS vs SPARQL",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| " + " | ".join(name for _, name, _ in SIMPLE_COLUMNS) + " |",
        "| " + " | ".join("---:" for _ in SIMPLE_COLUMNS) + " |",
    ]
    for r in rows:
        md_lines.append(
            "| "
            + " | ".join(
                str(r.get(key, "-")) if r.get(key, "") != "" else "-"
                for key, _, _ in SIMPLE_COLUMNS
            )
            + " |"
        )
    md_text = "\n".join(md_lines) + "\n"
    depth_md.write_text(md_text, encoding="utf-8")
    legacy_md.write_text(md_text, encoding="utf-8")

    return legacy_txt, legacy_md, depth_txt, depth_md


if __name__ == "__main__":
    t, m, dt, dm = write_tables()
    print(f"Wrote {t}\nWrote {m}\nWrote {dt}\nWrote {dm}")
