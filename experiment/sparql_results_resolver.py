"""Resolve SPARQL BFS result files and tradeoff CSV fallbacks."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Tuple

from experiment.config import OUTPUT_DIR


def _load_tradeoff_rows(dataset: str) -> dict[int, dict]:
    """depth -> {paths, triples, time_seconds} from tradeoff CSV."""
    out = {}
    for name in (f"{dataset}_bfs_depth_tradeoff.csv", "bfs_depth_tradeoff.csv"):
        path = OUTPUT_DIR / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("dataset", dataset) != dataset and "dataset" in row:
                    continue
                try:
                    d = int(row["depth"])
                    out[d] = {
                        "paths": int(float(row["paths"])),
                        "triples": int(float(row["triples"])),
                        "time_seconds": float(row["time_seconds"]),
                    }
                except (KeyError, ValueError):
                    continue
        if out:
            break
    return out


def resolve_sparql_json(dataset: str, depth: int) -> Optional[Path]:
    candidates = [
        OUTPUT_DIR / f"{dataset}_bfs_sparql_paths_h{depth}_estimated.json",
        OUTPUT_DIR / f"{dataset}_bfs_sparql_paths_h{depth}.json",
        OUTPUT_DIR / f"{dataset}_bfs_sparql_paths_h{depth}_test.json",
    ]
    legacy = OUTPUT_DIR / f"{dataset}_bfs_sparql_paths.json"
    if legacy.exists():
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                meta = json.load(f).get("metadata", {})
            if meta.get("max_hops") == depth:
                candidates.append(legacy)
        except (json.JSONDecodeError, OSError):
            pass

    for p in candidates:
        if p.exists() and p.stat().st_size > 100:
            return p
    return None


def _meta_from_checkpoint(cp_path: Path, dataset: str, depth: int) -> Optional[dict]:
    """Build metadata from in-progress SPARQL checkpoint."""
    try:
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    qp = cp.get("question_paths", {})
    if not qp:
        return None
    total_paths = 0
    total_triples = 0
    for res in qp.values():
        if not isinstance(res, dict):
            continue
        paths = res.get("paths", [])
        total_paths += len(paths)
        for path in paths:
            if isinstance(path, str):
                total_triples += max(0, len(path.split("-->")) - 1)
            elif isinstance(path, list):
                total_triples += max(0, (len(path) - 1) // 2) if path and isinstance(path[0], list) else len(path) - 1
    times = cp.get("question_times", [])
    total_time = sum(times) if times else 0.0
    n = len(qp)
    n_total = meta_default_questions(dataset)
    return {
        "total_paths": total_paths,
        "total_triples": total_triples,
        "total_questions": n,
        "max_hops": depth,
        "timing": {
            "total_time_seconds": total_time,
            "average_time_per_question_seconds": total_time / n if n else 0,
        },
        "_source": f"checkpoint:{cp_path.name} ({n}/{n_total} questions)",
        "_partial": True,
    }


def load_sparql_metadata(dataset: str, depth: int) -> Tuple[dict, Optional[Path]]:
    """Return (metadata dict, source path). Uses JSON first, then checkpoint, then tradeoff CSV."""
    path = resolve_sparql_json(dataset, depth)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = dict(data.get("metadata", {}))
        meta["_source"] = str(path)
        return meta, path

    cp_path = OUTPUT_DIR / f"{dataset}_bfs_sparql_paths_h{depth}.json.checkpoint.json"
    if cp_path.exists() and cp_path.stat().st_size > 500:
        meta = _meta_from_checkpoint(cp_path, dataset, depth)
        if meta:
            return meta, None

    tradeoff = _load_tradeoff_rows(dataset)
    if depth in tradeoff:
        t = tradeoff[depth]
        n = meta_default_questions(dataset)
        avg = t["time_seconds"] / n if n else t["time_seconds"]
        meta = {
            "total_paths": t["paths"],
            "total_triples": t["triples"],
            "total_questions": n,
            "max_hops": depth,
            "timing": {
                "total_time_seconds": t["time_seconds"],
                "average_time_per_question_seconds": avg,
            },
            "_source": f"tradeoff_csv:depth={depth}",
        }
        return meta, None
    return {}, None


def meta_default_questions(dataset: str) -> int:
    defaults = {
        "bank": 146,
        "movie": 355,
        "cms": 2563,
        "emed": 8121,
        "synthea": 2963,
    }
    sim = OUTPUT_DIR / f"{dataset}_similarity_results.json"
    if sim.exists():
        try:
            with open(sim, "r", encoding="utf-8") as f:
                return len(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return defaults.get(dataset, 0)


def sparql_questions_with_paths(path: Optional[Path], dataset: str = "", depth: int = 0) -> int:
    if not path:
        cp = OUTPUT_DIR / f"{dataset}_bfs_sparql_paths_h{depth}.json.checkpoint.json"
        if dataset and cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    cpd = json.load(f)
                qp = cpd.get("question_paths", {})
                return sum(
                    1
                    for r in qp.values()
                    if isinstance(r, dict)
                    and r.get("paths")
                    and r["paths"] != ["No path found"]
                )
            except (json.JSONDecodeError, OSError):
                pass
        return 0
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    return sum(
        1
        for r in results
        if r.get("paths")
        and r["paths"] != ["No path found"]
        and (not isinstance(r["paths"], list) or any(p and p != "No path found" for p in r["paths"]))
    )
