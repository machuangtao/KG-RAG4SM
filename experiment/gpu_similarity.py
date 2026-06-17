#!/usr/bin/env python3
"""Multi-GPU / multi-CPU cosine similarity search for entity/relation retrieval."""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from experiment.config import (
    CHROMADB_DIR,
    CLINICAL_DATASETS,
    DATASETS,
    ENTITY_BATCH_SIZE,
    ENTITY_EMBED_DIR,
    OUTPUT_DIR,
    PROPERTY_EMBED_DIR,
    ROOT,
    SIMILARITY_TOP_K,
)

logger = logging.getLogger(__name__)


class SimilarityModel(nn.Module):
    def forward(self, question_embedding, batch_embeddings):
        return torch.matmul(question_embedding, batch_embeddings.t())


def get_device_count() -> int:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.cuda.device_count()
    return max(1, (os.cpu_count() or 4))


def _sort_embedding_files(file_paths: List[str]) -> List[str]:
    def key(p):
        base = os.path.basename(p)
        if base.startswith("embeddings_") and base.endswith(".npz"):
            try:
                return int(base.replace("embeddings_", "").replace(".npz", ""))
            except ValueError:
                return 0
        return 0

    return sorted(file_paths, key=key)


def load_embeddings_from_npz_dir(directory: Path) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Load NPZ + metadata_*.txt (same format as wikidata/cosine_similarity_search.py)."""
    embed_files = _sort_embedding_files(glob.glob(str(directory / "embeddings_*.npz")))
    if not embed_files:
        raise FileNotFoundError(f"No embeddings in {directory}")

    all_emb, all_meta = [], []
    for emb_file in tqdm(embed_files, desc=f"Loading {directory.name}"):
        emb_data = np.load(emb_file)
        emb = np.array(emb_data["embeddings"], dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        base = os.path.basename(emb_file)
        meta_file = directory / base.replace("embeddings_", "metadata_").replace(".npz", ".txt")
        meta_lines = []
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        meta_lines.append((parts[1], "\t".join(parts[2:])))

        n = min(len(emb), len(meta_lines)) if meta_lines else len(emb)
        all_emb.append(emb[:n])
        if meta_lines:
            all_meta.extend(meta_lines[:n])
        else:
            all_meta.extend([(f"item_{len(all_meta)+i}", f"item_{len(all_meta)+i}") for i in range(n)])

    logger.info("Loaded %d vectors from %s", len(all_meta), directory.name)
    return np.vstack(all_emb), all_meta


def load_question_embeddings(chroma_path: Path, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_collection(collection_name)
    data = col.get(include=["embeddings", "metadatas", "documents"])
    ids = data["ids"]
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    metas = []
    for i, qid in enumerate(ids):
        doc = (data["documents"] or [""])[i] if data["documents"] else ""
        metas.append((qid, doc))
    return embeddings, metas


def cosine_search_batch(
    q_tensor: torch.Tensor,
    kg_embeddings: np.ndarray,
    kg_metadata: List[Tuple[str, str]],
    device: torch.device,
    top_k: int = SIMILARITY_TOP_K,
) -> List[dict]:
    model = SimilarityModel().to(device)
    q = q_tensor.to(device)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    best_scores, best_meta = [], []
    n = len(kg_embeddings)
    for start in range(0, n, ENTITY_BATCH_SIZE):
        end = min(start + ENTITY_BATCH_SIZE, n)
        batch = torch.from_numpy(kg_embeddings[start:end]).float().to(device)
        batch = batch / batch.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scores = model(q, batch).squeeze(0)
        k = min(top_k, scores.numel())
        vals, idxs = torch.topk(scores, k)
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            global_idx = start + idx
            best_scores.append(v)
            best_meta.append(kg_metadata[global_idx])
    combined = sorted(zip(best_scores, best_meta), key=lambda x: x[0], reverse=True)[:top_k]
    return [{"id": m[0], "text": m[1], "score": float(s)} for s, m in combined]


def _search_one_question(
    idx: int,
    q_ids: List[str],
    q_docs: List[str],
    q_emb: np.ndarray,
    entity_emb: np.ndarray,
    entity_meta: List[Tuple[str, str]],
    relation_emb: np.ndarray,
    relation_meta: List[Tuple[str, str]],
    device: torch.device,
) -> Tuple[str, dict]:
    q_tensor = torch.from_numpy(q_emb[idx]).float()
    q_tensor = q_tensor / q_tensor.norm(dim=0).clamp(min=1e-8)
    return q_ids[idx], {
        "question": q_docs[idx],
        "similar_entities": cosine_search_batch(q_tensor, entity_emb, entity_meta, device),
        "similar_relations": cosine_search_batch(q_tensor, relation_emb, relation_meta, device),
    }


def run_search_multi_gpu(dataset_name: str, limit: int = None) -> dict:
    n_devices = get_device_count()
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    chroma_path = CHROMADB_DIR / f"chromadb_store_test_{dataset_name}_questions"
    collection = f"test_{dataset_name}_questions_collection"

    logger.info(
        "Devices: %d (%s)",
        n_devices,
        "CUDA" if use_cuda else f"CPU threads, {os.cpu_count()} cores",
    )

    logger.info("Loading question embeddings from ChromaDB...")
    q_emb, q_meta = load_question_embeddings(chroma_path, collection)
    entity_emb, entity_meta = load_embeddings_from_npz_dir(ENTITY_EMBED_DIR)
    relation_emb, relation_meta = load_embeddings_from_npz_dir(PROPERTY_EMBED_DIR)

    q_ids = [m[0] for m in q_meta]
    q_docs = [m[1] for m in q_meta]
    total = len(q_ids)
    if limit:
        total = min(total, limit)

    indices = list(range(total))
    results: Dict[str, dict] = {}

    if use_cuda:
        gpu_count = torch.cuda.device_count()
        logger.info("Multi-GPU search: %d GPUs, %d questions", gpu_count, total)
        chunks: List[List[int]] = [[] for _ in range(gpu_count)]
        for i in indices:
            chunks[i % gpu_count].append(i)

        def _gpu_worker(gpu_id: int, idxs: List[int]) -> Dict[str, dict]:
            device = torch.device(f"cuda:{gpu_id}")
            out = {}
            for i in tqdm(idxs, desc=f"cuda:{gpu_id}", leave=False):
                qid, row = _search_one_question(
                    i, q_ids, q_docs, q_emb,
                    entity_emb, entity_meta, relation_emb, relation_meta, device,
                )
                out[qid] = row
            return out

        with ThreadPoolExecutor(max_workers=gpu_count) as pool:
            futures = [pool.submit(_gpu_worker, g, chunks[g]) for g in range(gpu_count) if chunks[g]]
            for fut in as_completed(futures):
                results.update(fut.result())
    else:
        workers = max(1, (os.cpu_count() or 4))
        device = torch.device("cpu")
        logger.info("Multi-CPU search: %d threads, %d questions", workers, total)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _search_one_question,
                    i, q_ids, q_docs, q_emb,
                    entity_emb, entity_meta, relation_emb, relation_meta, device,
                )
                for i in indices
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"similarity-{dataset_name}"):
                qid, row = fut.result()
                results[qid] = row

    logger.info("Similarity done: %d questions", len(results))
    return results


def save_results(results: dict, dataset_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{dataset_name}_similarity_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved %s (%d questions)", json_path, len(results))
    return json_path


def _expected_question_count(dataset_name: str) -> int | None:
    chroma_path = CHROMADB_DIR / f"chromadb_store_test_{dataset_name}_questions"
    collection = f"test_{dataset_name}_questions_collection"
    if not chroma_path.exists():
        return None
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection(collection)
        return col.count()
    except Exception:
        return None


def ensure_similarity(datasets: List[str], limit: int = None, force: bool = False) -> None:
    for ds in datasets:
        out = OUTPUT_DIR / f"{ds}_similarity_results.json"
        expected = _expected_question_count(ds)
        if out.exists() and not force and limit is None:
            if expected is not None:
                with open(out, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if len(existing) >= expected:
                    logger.info("Similarity exists for %s (%d questions) — skip", ds, len(existing))
                    continue
                logger.warning(
                    "Similarity for %s has %d/%d questions — regenerating",
                    ds, len(existing), expected,
                )
            else:
                logger.info("Similarity exists for %s — skip", ds)
                continue
        if ds in CLINICAL_DATASETS and limit is None:
            from experiment.cms_similarity_fast import run_clinical_similarity

            results = run_clinical_similarity(ds)
        else:
            results = run_search_multi_gpu(ds, limit=limit)
        save_results(results, ds)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Multi-GPU/CPU cosine similarity search")
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    ensure_similarity(datasets, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
