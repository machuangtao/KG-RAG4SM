#!/usr/bin/env python3
"""Fast parallel CMS similarity using all CPU cores (shared-memory threads)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from experiment.config import OUTPUT_DIR
from wikidata.cosine_similarity_search import (
    ENTITY_EMBED_DIR,
    ENTITY_BATCH_SIZE,
    PROPERTY_EMBED_DIR,
    load_embeddings_from_npz_dir,
    load_question_embeddings_from_chromadb,
    save_results,
)

logger = logging.getLogger(__name__)

_TOPK = 10


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return x / norms


def _topk_cosine(q: np.ndarray, kg: np.ndarray, meta: list, top_k: int = _TOPK) -> list:
    best_scores = []
    best_idx = []
    for start in range(0, len(kg), ENTITY_BATCH_SIZE):
        end = min(start + ENTITY_BATCH_SIZE, len(kg))
        scores = kg[start:end] @ q
        k = min(top_k, scores.shape[0])
        idx_local = np.argpartition(-scores, k - 1)[:k]
        for j in idx_local:
            best_scores.append(float(scores[j]))
            best_idx.append(start + int(j))
    order = np.argsort([-s for s in best_scores])[:top_k]
    out = []
    for o in order:
        i = best_idx[o]
        mid, text = meta[i]
        out.append({"id": mid, "text": text, "score": best_scores[o]})
    return out


def _search_one(
    idx: int,
    q_ids: list,
    q_docs: list,
    q_norm: np.ndarray,
    ent_norm: np.ndarray,
    ent_meta: list,
    rel_norm: np.ndarray,
    rel_meta: list,
) -> tuple:
    qid = q_ids[idx]
    hits_e = _topk_cosine(q_norm[idx], ent_norm, ent_meta)
    hits_r = _topk_cosine(q_norm[idx], rel_norm, rel_meta)
    return qid, {
        "question": q_docs[idx],
        "similar_entities": hits_e,
        "similar_relations": hits_r,
    }


def run_clinical_similarity(dataset: str = "cms", workers: int | None = None) -> dict:
    workers = workers or max(1, (os.cpu_count() or 8))
    chroma = ROOT / "chromadb" / f"chromadb_store_test_{dataset}_questions"
    collection = f"test_{dataset}_questions_collection"

    logger.info("Loading %s questions from ChromaDB...", dataset)
    q_emb, q_meta = load_question_embeddings_from_chromadb(str(chroma), collection_name=collection)
    q_ids = [m[0] for m in q_meta]
    q_docs = [m[1] for m in q_meta]
    logger.info("Loaded %d questions", len(q_ids))

    logger.info("Loading entity embeddings (this takes several minutes)...")
    t0 = time.time()
    ent_emb, ent_meta = load_embeddings_from_npz_dir(ENTITY_EMBED_DIR)
    logger.info("Entities loaded: %d vectors in %.1fs", len(ent_meta), time.time() - t0)

    logger.info("Loading relation embeddings...")
    t0 = time.time()
    rel_emb, rel_meta = load_embeddings_from_npz_dir(PROPERTY_EMBED_DIR)
    logger.info("Relations loaded: %d vectors in %.1fs", len(rel_meta), time.time() - t0)

    logger.info("Normalizing embeddings...")
    q_norm = _normalize_rows(q_emb.astype(np.float32))
    ent_norm = _normalize_rows(ent_emb.astype(np.float32))
    rel_norm = _normalize_rows(rel_emb.astype(np.float32))
    del ent_emb, rel_emb, q_emb

    logger.info("Running parallel similarity: %d workers, %d questions", workers, len(q_ids))
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _search_one, i, q_ids, q_docs, q_norm, ent_norm, ent_meta, rel_norm, rel_meta
            )
            for i in range(len(q_ids))
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset}-similarity"):
            qid, row = fut.result()
            results[qid] = row

    logger.info("Similarity finished in %.1fs", time.time() - t0)
    return results


def run_cms_similarity(workers: int | None = None) -> dict:
    return run_clinical_similarity("cms", workers=workers)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cms", choices=["cms", "emed", "synthea"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8)))
    args = parser.parse_args()

    results = run_clinical_similarity(args.dataset, workers=args.workers)
    save_results(results, args.dataset)
    logger.info(
        "Saved %d questions to testRes/%s_similarity_results.json", len(results), args.dataset
    )


if __name__ == "__main__":
    main()
