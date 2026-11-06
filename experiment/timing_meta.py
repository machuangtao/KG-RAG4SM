"""Normalize local BFS timing metadata (new and legacy JSON)."""
from __future__ import annotations

from typing import Any, Dict, Optional


def paths_per_pair_limit(disable_path_cap: bool = False) -> Optional[int]:
    """Return MAX_PATHS_PER_PAIR, or None when the per-pair cap is disabled for benchmarks."""
    if disable_path_cap:
        return None
    from experiment.config import MAX_PATHS_PER_PAIR

    return MAX_PATHS_PER_PAIR


def local_timing_from_meta(meta: dict) -> Dict[str, Any]:
    """Extract comparable local timing fields with backward compatibility."""
    t = meta.get("timing") or {}
    n = int(t.get("num_questions") or meta.get("total_questions") or 0)
    sum_q = t.get("sum_question_time_seconds")
    avg = t.get("average_time_per_question_seconds")
    wall = t.get("wall_clock_seconds")
    if wall is None:
        wall = t.get("total_time_seconds")

    if sum_q is None and avg is not None and n:
        sum_q = float(avg) * n
    if avg is None and sum_q is not None and n:
        avg = float(sum_q) / n
    if sum_q is None:
        sum_q = 0.0
    if avg is None:
        avg = 0.0
    if wall is None:
        wall = 0.0

    workers = t.get("num_workers")
    if workers is None:
        workers = (meta.get("configuration") or {}).get("workers")

    return {
        "num_questions": n,
        "sum_question_time_seconds": float(sum_q),
        "average_time_per_question_seconds": float(avg),
        "wall_clock_seconds": float(wall),
        "num_workers": workers,
    }
