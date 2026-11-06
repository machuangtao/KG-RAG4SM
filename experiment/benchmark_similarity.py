#!/usr/bin/env python3
"""Benchmark cosine similarity latency on a question sample."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiment.config import (
    BENCHMARK_N,
    BENCHMARK_SEED,
    CHROMADB_DIR,
    ENTITY_BATCH_SIZE,
    ENTITY_EMBED_DIR,
    OUTPUT_DIR,
    PROPERTY_EMBED_DIR,
    ROOT,
    SIMILARITY_TOP_K,
)
from experiment.gpu_similarity import (
    SimilarityModel,
    cosine_search_batch,
    load_embeddings_from_npz_dir,
    load_question_embeddings,
)
from experiment.sample_questions import sample_question_ids

import torch


_shared: dict = {}


def _ensure_embeddings(device: torch.device):
    if _shared:
        return
    entity_emb, entity_meta = load_embeddings_from_npz_dir(ENTITY_EMBED_DIR)
    rel_emb, rel_meta = load_embeddings_from_npz_dir(PROPERTY_EMBED_DIR)
    _shared["entity_emb"] = entity_emb
    _shared["entity_meta"] = entity_meta
    _shared["rel_emb"] = rel_emb
    _shared["rel_meta"] = rel_meta
    _shared["device"] = device


def benchmark_dataset(dataset: str, qids: list, device: torch.device) -> dict:
    _ensure_embeddings(device)
    entity_emb = _shared["entity_emb"]
    entity_meta = _shared["entity_meta"]
    rel_emb = _shared["rel_emb"]
    rel_meta = _shared["rel_meta"]

    chroma = CHROMADB_DIR / f"chromadb_store_test_{dataset}_questions"
    collection = f"test_{dataset}_questions_collection"
    q_emb, q_meta = load_question_embeddings(chroma, collection)
    id_to_idx = {m[0]: i for i, m in enumerate(q_meta)}

    entity_t = torch.from_numpy(entity_emb).float().to(device)
    entity_t = entity_t / entity_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    rel_t = torch.from_numpy(rel_emb).float().to(device)
    rel_t = rel_t / rel_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    model = SimilarityModel().to(device)

    per_q: list[float] = []
    for qid in qids:
        if qid not in id_to_idx:
            continue
        idx = id_to_idx[qid]
        q = torch.from_numpy(q_emb[idx]).float().to(device)
        q = q / q.norm().clamp(min=1e-8)
        t0 = time.perf_counter()
        cosine_search_batch(q, entity_emb, entity_meta, device, top_k=SIMILARITY_TOP_K)
        cosine_search_batch(q, rel_emb, rel_meta, device, top_k=SIMILARITY_TOP_K)
        per_q.append(time.perf_counter() - t0)

    avg = sum(per_q) / len(per_q) if per_q else 0.0
    return {
        "dataset": dataset,
        "sample_size": len(per_q),
        "average_time_per_question_seconds": avg,
        "total_time_seconds": sum(per_q),
        "per_question_seconds": per_q,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["bank", "movie", "cms"])
    parser.add_argument("--limit", type=int, default=BENCHMARK_N)
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    args = parser.parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}
    for ds in args.datasets:
        qids = sample_question_ids(ds, n=args.limit, seed=args.seed)
        print(f"Benchmarking similarity {ds} ({len(qids)} questions)...")
        t0 = time.perf_counter()
        all_results[ds] = benchmark_dataset(ds, qids, device)
        all_results[ds]["wall_time_seconds"] = time.perf_counter() - t0

    out = OUTPUT_DIR / f"similarity_timing_n{args.limit}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
