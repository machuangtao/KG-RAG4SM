#!/usr/bin/env python3
"""Measure gpt-4o-mini inference latency (sequential vs concurrent batch)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from experiment.config import BENCHMARK_N, BENCHMARK_SEED, EXPERIMENT_LOG_DIR, LLM_MODEL, ROOT
from experiment.load_openai_key import ensure_openai_api_key
from experiment.sample_questions import sample_question_ids
from src.kgrag4sm import KGRAG_for_Schema_Matching

try:
    import openai
except ImportError:
    openai = None


def _row_index_from_qid(qid: str) -> int:
    m = re.search(r"(\d+)$", qid)
    return int(m.group(1)) if m else -1


def _load_prompts(
    dataset: str,
    xlsx_path: Path,
    qids: List[str],
    paths_col: str = "top 1 paths",
) -> List[dict]:
    df = pd.read_excel(xlsx_path)
    kgrag = KGRAG_for_Schema_Matching()
    system_prompt = kgrag.generate_system_prompt()
    qid_set = set(qids)
    prompts = []
    for local_i in range(len(df)):
        if "_global_qidx" in df.columns:
            global_idx = int(df.iloc[local_i]["_global_qidx"])
        else:
            global_idx = local_i
        qid = f"question_{global_idx}"
        if qid not in qid_set:
            continue
        question = df.iloc[local_i, 9] if df.shape[1] > 9 else df.iloc[local_i, 0]
        paths = df.iloc[local_i][paths_col] if paths_col in df.columns else None
        if pd.isna(paths):
            paths = None
        user_prompt = kgrag.generate_user_prompt(str(question), paths)
        prompts.append({"qid": qid, "system": system_prompt, "user": user_prompt})
    return prompts


def _call_openai(model: str, system: str, user: str) -> str:
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=256,
    )
    return resp.choices[0].message.content or ""


async def _call_openai_async(model: str, system: str, user: str, sem: asyncio.Semaphore):
    async with sem:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call_openai, model, system, user)


def run_sequential(model: str, prompts: List[dict]) -> dict:
    times: List[float] = []
    for p in tqdm(prompts, desc="LLM sequential"):
        t0 = time.perf_counter()
        _call_openai(model, p["system"], p["user"])
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times) if times else 0.0
    return {
        "mode": "sequential",
        "count": len(times),
        "average_time_per_question_seconds": avg,
        "total_time_seconds": sum(times),
        "per_question_seconds": times,
    }


async def run_concurrent_batch(model: str, prompts: List[dict], workers: int = 8) -> dict:
    sem = asyncio.Semaphore(workers)
    t0 = time.perf_counter()
    tasks = [
        _call_openai_async(model, p["system"], p["user"], sem) for p in prompts
    ]
    await asyncio.gather(*tasks)
    total = time.perf_counter() - t0
    n = len(prompts)
    return {
        "mode": "batch_concurrent",
        "workers": workers,
        "count": n,
        "total_time_seconds": total,
        "average_time_per_question_seconds": total / n if n else 0.0,
        "per_question_seconds": [total / n] * n if n else [],
    }


def measure_dataset(
    dataset: str,
    strategy: str,
    qids: List[str],
    model: str = LLM_MODEL,
    n: int = BENCHMARK_N,
    workers: int = 8,
    skip_llm: bool = False,
) -> dict:
    xlsx = ROOT / "datasets" / "reproduce" / f"test_{dataset}_q_paths_{strategy}_n{n}.xlsx"
    if not xlsx.exists():
        return {"error": f"missing {xlsx}"}

    prompts = _load_prompts(dataset, xlsx, qids)
    result = {"dataset": dataset, "strategy": strategy, "prompt_count": len(prompts)}

    ensure_openai_api_key()
    if skip_llm or not os.environ.get("OPENAI_API_KEY"):
        result["llm_skipped"] = True
        result["sequential"] = {"average_time_per_question_seconds": 0.0, "total_time_seconds": 0.0}
        result["batch"] = {"average_time_per_question_seconds": 0.0, "total_time_seconds": 0.0}
        return result

    result["sequential"] = run_sequential(model, prompts)
    result["batch"] = asyncio.run(run_concurrent_batch(model, prompts, workers=workers))
    return result


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["bank", "movie", "cms"])
    parser.add_argument("--strategies", nargs="+", default=["full_bfs", "pruned_bfs", "one_hop"])
    parser.add_argument("--limit", type=int, default=BENCHMARK_N)
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help=f"LLM model (default: {LLM_MODEL}; only gpt-4o-mini supported for benchmark)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    if args.model != LLM_MODEL:
        raise SystemExit(f"Benchmark supports only {LLM_MODEL}, got: {args.model}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    out = {}
    for ds in args.datasets:
        qids = sample_question_ids(ds, n=args.limit, seed=args.seed)
        out[ds] = {}
        for strat in args.strategies:
            print(f"LLM latency {ds} / {strat}")
            out[ds][strat] = measure_dataset(
                ds, strat, qids, model=args.model, n=args.limit, workers=args.workers,
                skip_llm=args.skip_llm,
            )

    path = EXPERIMENT_LOG_DIR / f"latency_n{args.limit}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
