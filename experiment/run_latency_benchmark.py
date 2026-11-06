#!/usr/bin/env python3
"""Run full latency benchmark (Actions 2-4) and update results txt."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from experiment.benchmark_similarity import benchmark_dataset
from experiment.config import (
    BENCHMARK_DEPTH,
    BENCHMARK_N,
    BENCHMARK_SEED,
    DATASETS,
    LLM_MODEL,
    EXPERIMENT_LOG_DIR,
    OUTPUT_DIR,
    ROOT,
)
from experiment.one_hop_neighbors import run_one_hop_benchmark
from experiment.rank_paths import rank_for_strategy
from experiment.retrieval_strategies import run_strategy_benchmark
from experiment.sample_questions import filter_similarity_subset, sample_question_ids
from experiment.load_openai_key import ensure_openai_api_key
from experiment.write_results_txt import write_results_txt

import torch

logger = logging.getLogger(__name__)


def _retrieval_json_path(dataset: str, strategy: str, n: int) -> Path:
    if strategy == "one_hop":
        return OUTPUT_DIR / f"{dataset}_one_hop_triples_n{n}.json"
    return OUTPUT_DIR / f"{dataset}_bfs_{strategy}_h{BENCHMARK_DEPTH}_n{n}.json"


def _load_existing_retrieval(dataset: str, strategy: str, n: int) -> dict | None:
    p = _retrieval_json_path(dataset, strategy, n)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f).get("metadata", {})
    return {
        "file": str(p),
        "paths": meta.get("total_paths"),
        "triples": meta.get("total_triples"),
        "avg_sec": meta.get("timing", {}).get("average_time_per_question_seconds"),
        "total_sec": meta.get("timing", {}).get("total_time_seconds"),
    }


def run_benchmark(
    datasets: list[str],
    limit: int = BENCHMARK_N,
    seed: int = BENCHMARK_SEED,
    strategies: list[str] | None = None,
    skip_llm: bool = False,
    skip_similarity_load: bool = False,
    skip_retrieval: bool = False,
) -> dict:
    strategies = strategies or ["full_bfs", "pruned_bfs", "one_hop"]
    report: dict = {
        "config": {
            "n": limit,
            "seed": seed,
            "depth": BENCHMARK_DEPTH,
            "datasets": datasets,
            "strategies": strategies,
            "model": LLM_MODEL,
        },
        "similarity": {},
        "retrieval": {},
        "ranking": {},
        "llm": {},
        "summary": [],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds in datasets:
        qids = sample_question_ids(ds, n=limit, seed=seed)
        sim_path = filter_similarity_subset(ds, qids)
        report["sampled_ids"] = report.get("sampled_ids", {})
        report["sampled_ids"][ds] = qids

        if not skip_similarity_load:
            logger.info("[%s] Similarity benchmark (%d q)", ds, len(qids))
            try:
                report["similarity"][ds] = benchmark_dataset(ds, qids, device)
            except Exception as e:
                logger.warning("Similarity failed %s: %s", ds, e)
                report["similarity"][ds] = {"error": str(e)}

        report["retrieval"][ds] = {}
        report["ranking"][ds] = {}

        for strat in strategies:
            existing = _load_existing_retrieval(ds, strat, limit) if skip_retrieval else None
            if existing:
                logger.info("[%s] Using cached retrieval %s", ds, strat)
                report["retrieval"][ds][strat] = existing
            else:
                if strat == "one_hop":
                    logger.info("[%s] 1-hop neighbors", ds)
                    p = run_one_hop_benchmark(ds, qids, similarity_path=sim_path)
                elif strat in ("full_bfs", "pruned_bfs"):
                    logger.info("[%s] %s h%d", ds, strat, BENCHMARK_DEPTH)
                    p = run_strategy_benchmark(
                        ds, qids, strat, max_hops=BENCHMARK_DEPTH, similarity_path=sim_path
                    )
                else:
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f).get("metadata", {})
                report["retrieval"][ds][strat] = {
                    "file": str(p),
                    "paths": meta.get("total_paths"),
                    "triples": meta.get("total_triples"),
                    "avg_sec": meta.get("timing", {}).get("average_time_per_question_seconds"),
                    "total_sec": meta.get("timing", {}).get("total_time_seconds"),
                }

            xlsx = ROOT / "datasets" / "reproduce" / f"test_{ds}_q_paths_{strat}_n{limit}.xlsx"
            if skip_retrieval and xlsx.exists():
                logger.info("[%s] Using cached ranking %s", ds, strat)
                report["ranking"][ds][strat] = {
                    "file": str(xlsx),
                    "total_sec": 0.0,
                    "avg_sec": 0.0,
                    "cached": True,
                }
            else:
                logger.info("[%s] Rank paths for %s", ds, strat)
                try:
                    xlsx, rank_sec = rank_for_strategy(ds, strat, qids, n=limit)
                    report["ranking"][ds][strat] = {
                        "file": str(xlsx),
                        "total_sec": rank_sec,
                        "avg_sec": rank_sec / len(qids) if qids else 0,
                    }
                except Exception as e:
                    logger.warning("Ranking failed %s/%s: %s", ds, strat, e)
                    report["ranking"][ds][strat] = {"error": str(e)}

    # LLM
    ensure_openai_api_key()
    if not skip_llm and os.environ.get("OPENAI_API_KEY"):
        from experiment.measure_llm_latency import measure_dataset

        report["llm"] = {}
        for ds in datasets:
            qids = report["sampled_ids"][ds]
            report["llm"][ds] = {}
            for strat in strategies:
                logger.info("[%s] LLM latency %s", ds, strat)
                report["llm"][ds][strat] = measure_dataset(
                    ds, strat, qids, model=LLM_MODEL, n=limit
                )
    else:
        report["llm_skipped"] = True
        logger.warning("Skipping LLM (no OPENAI_API_KEY or --skip-llm)")

    # Build summary rows
    for ds in datasets:
        for strat in strategies:
            sim_avg = report.get("similarity", {}).get(ds, {}).get(
                "average_time_per_question_seconds", 0
            )
            ret = report.get("retrieval", {}).get(ds, {}).get(strat, {})
            rank = report.get("ranking", {}).get(ds, {}).get(strat, {})
            llm_s = 0.0
            llm_b = 0.0
            if ds in report.get("llm", {}) and strat in report["llm"][ds]:
                llm_block = report["llm"][ds][strat]
                if not llm_block.get("llm_skipped"):
                    llm_s = llm_block.get("sequential", {}).get(
                        "average_time_per_question_seconds", 0
                    )
                    llm_b = llm_block.get("batch", {}).get(
                        "average_time_per_question_seconds", 0
                    )
            ret_avg = ret.get("avg_sec") or 0
            rank_avg = rank.get("avg_sec") or 0
            paths = ret.get("paths") or 0
            report["summary"].append(
                {
                    "dataset": ds,
                    "strategy": strat,
                    "sim_avg_sec": sim_avg,
                    "retrieve_avg_sec": ret_avg,
                    "rank_avg_sec": rank_avg,
                    "llm_seq_avg_sec": llm_s,
                    "llm_batch_avg_sec": llm_b,
                    "e2e_seq_avg_sec": sim_avg + ret_avg + rank_avg + llm_s,
                    "e2e_batch_avg_sec": sim_avg + ret_avg + rank_avg + llm_b,
                    "paths_total": paths,
                    "paths_per_q": paths / limit if limit else 0,
                }
            )

    # Grand mean across all dataset-strategy rows
    if report["summary"]:
        keys = [
            "sim_avg_sec",
            "retrieve_avg_sec",
            "rank_avg_sec",
            "llm_seq_avg_sec",
            "llm_batch_avg_sec",
            "e2e_seq_avg_sec",
            "e2e_batch_avg_sec",
        ]
        grand = {k: sum(r[k] for r in report["summary"]) / len(report["summary"]) for k in keys}
        grand["dataset"] = "MEAN"
        grand["strategy"] = "all"
        grand["paths_per_q"] = sum(r["paths_per_q"] for r in report["summary"]) / len(
            report["summary"]
        )
        report["summary"].append(grand)

    out_path = EXPERIMENT_LOG_DIR / f"latency_n{limit}.json"
    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["_output_path"] = str(out_path)
    return report


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Latency benchmark Actions 2-4")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=BENCHMARK_N)
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    parser.add_argument("--strategies", nargs="+", default=["full_bfs", "pruned_bfs", "one_hop"])
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true", help="Skip similarity timing (embeddings already loaded once)")
    parser.add_argument("--skip-retrieval", action="store_true", help="Use existing testRes/*_n50.json and ranked xlsx")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    t0 = time.time()
    report = run_benchmark(
        args.datasets,
        limit=args.limit,
        seed=args.seed,
        strategies=args.strategies,
        skip_llm=args.skip_llm,
        skip_similarity_load=args.skip_similarity,
        skip_retrieval=args.skip_retrieval,
    )
    txt_path = write_results_txt()
    logger.info("Benchmark done in %.1fs. Report: %s | Results: %s", time.time() - t0, report.get("_output_path"), txt_path)


if __name__ == "__main__":
    main()
