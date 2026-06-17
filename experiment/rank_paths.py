"""Rank BFS/one-hop paths to Excel for benchmark subsets."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from experiment.config import OUTPUT_DIR, ROOT


def _original_xlsx(dataset: str) -> Path:
    for p in (
        ROOT / "datasets" / "NewData" / f"test_{dataset}_q.xlsx",
        ROOT / "datasets" / "reproduce" / f"test_{dataset}_q_with_paths.xlsx",
        ROOT / "datasets" / "original" / f"test_{dataset}_q.xlsx",
    ):
        if p.exists():
            return p
    raise FileNotFoundError(f"No source Excel for {dataset}")


def _row_index_from_qid(qid: str) -> int:
    m = re.search(r"(\d+)$", qid)
    return int(m.group(1)) if m else -1


def _remap_bfs_json(bfs_json: Path, qids: List[str], out_json: Path) -> Path:
    """Remap question IDs to question_0..question_{n-1} for subset ranking."""
    with open(bfs_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data if isinstance(data, list) else [])
    by_id = {r["question_id"]: r for r in results}
    remapped = []
    for i, qid in enumerate(qids):
        if qid in by_id:
            remapped.append(
                {"question_id": f"question_{i}", "paths": by_id[qid].get("paths", [])}
            )
        else:
            remapped.append({"question_id": f"question_{i}", "paths": []})
    out = {"metadata": data.get("metadata", {}), "results": remapped}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out_json


def rank_subset(
    dataset: str,
    bfs_json: Path,
    qids: List[str],
    output_xlsx: Path,
) -> Tuple[Path, float]:
    original = _original_xlsx(dataset)
    df_full = pd.read_excel(original)
    indices = sorted({_row_index_from_qid(q) for q in qids if _row_index_from_qid(q) >= 0})
    df_sub = df_full.iloc[indices].copy().reset_index(drop=True)
    df_sub["_global_qidx"] = indices

    tmp_xlsx = OUTPUT_DIR / f"_tmp_{dataset}_subset_{len(qids)}.xlsx"
    remapped_json = OUTPUT_DIR / f"_tmp_{dataset}_bfs_remapped_{len(qids)}.json"
    df_sub.to_excel(tmp_xlsx, index=False)
    _remap_bfs_json(bfs_json, qids, remapped_json)

    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(ROOT / "preprocess" / "bfs_path_ranking.py"),
        "--original",
        str(tmp_xlsx),
        "--bfs_paths",
        str(remapped_json),
        "--output",
        str(output_xlsx),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - t0

    df_out = pd.read_excel(output_xlsx)
    if len(df_out) == len(indices):
        df_out["_global_qidx"] = indices
        df_out.to_excel(output_xlsx, index=False)

    for p in (tmp_xlsx, remapped_json):
        if p.exists():
            p.unlink(missing_ok=True)
    return output_xlsx, elapsed


def rank_for_strategy(
    dataset: str,
    strategy: str,
    qids: List[str],
    n: int = 50,
) -> Tuple[Path, float]:
    if strategy == "one_hop":
        bfs = OUTPUT_DIR / f"{dataset}_one_hop_triples_n{n}.json"
    else:
        bfs = OUTPUT_DIR / f"{dataset}_bfs_{strategy}_h4_n{n}.json"
    out = ROOT / "datasets" / "reproduce" / f"test_{dataset}_q_paths_{strategy}_n{n}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    return rank_subset(dataset, bfs, qids, out)
