"""Sample fixed question IDs for latency benchmarks."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List

from experiment.config import BENCHMARK_N, BENCHMARK_SEED, OUTPUT_DIR


def _sort_key(qid: str) -> int:
    m = re.search(r"(\d+)$", qid)
    return int(m.group(1)) if m else 0


def load_similarity_map(dataset: str) -> dict:
    path = OUTPUT_DIR / f"{dataset}_similarity_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing similarity file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_question_ids(
    dataset: str,
    n: int = BENCHMARK_N,
    seed: int = BENCHMARK_SEED,
) -> List[str]:
    data = load_similarity_map(dataset)
    all_ids = sorted(data.keys(), key=_sort_key)
    if len(all_ids) <= n:
        return all_ids
    rng = random.Random(seed)
    picked = rng.sample(all_ids, n)
    return sorted(picked, key=_sort_key)


def filter_similarity_subset(dataset: str, qids: List[str], out_path: Path | None = None) -> Path:
    data = load_similarity_map(dataset)
    subset = {qid: data[qid] for qid in qids if qid in data}
    out_path = out_path or (OUTPUT_DIR / f"{dataset}_similarity_results_n{len(qids)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    return out_path
