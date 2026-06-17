#!/usr/bin/env python3
"""Build in-memory WikiData5M adjacency from embedding metadata chunks."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from experiment.config import (
    DATA_DIR,
    ENTITY_FILE,
    GRAPH_CACHE,
    GRAPH_META_CACHE,
    RELATION_FILE,
    ROOT,
    TRIPLET_METADATA_DIR,
    TRIPLET_METADATA_GLOB,
)

Adjacency = Dict[str, List[Tuple[str, str]]]


def load_label_map(file_path: Path, id_prefix: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not file_path.exists():
        return labels
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0].startswith(id_prefix):
                labels[parts[0]] = " ".join(parts[1:]).strip() or parts[0]
    return labels


def _chunk_index(path: Path) -> int:
    stem = path.stem  # metadata_chunk_42
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def list_metadata_chunks(max_chunks: int | None = None) -> List[Path]:
    chunks = sorted(TRIPLET_METADATA_DIR.glob(TRIPLET_METADATA_GLOB), key=_chunk_index)
    if not chunks:
        raise FileNotFoundError(f"No metadata chunks in {TRIPLET_METADATA_DIR}")
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return chunks


def _parse_chunk_file(chunk_path: str) -> Adjacency:
    adj: Adjacency = defaultdict(list)
    with open(chunk_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for rec in records:
        head = rec.get("head_id")
        rel = rec.get("relation_id")
        tail = rec.get("tail_id")
        if not head or not rel or not tail:
            continue
        adj[head].append((rel, tail))
        adj[tail].append((rel, head))
    return dict(adj)


def _merge_adjacency(partials: List[Adjacency]) -> Adjacency:
    merged: Adjacency = defaultdict(list)
    for partial in partials:
        for node, edges in partial.items():
            merged[node].extend(edges)
    return dict(merged)


def build_adjacency_parallel(workers: int = 8, max_chunks: int | None = None) -> Tuple[Adjacency, dict]:
    chunks = list_metadata_chunks(max_chunks=max_chunks)
    t0 = time.time()
    print(f"Building adjacency from {len(chunks)} chunks using {workers} workers...", flush=True)

    partials: List[Adjacency] = []
    chunk_paths = [str(p) for p in chunks]
    workers = max(1, min(workers, len(chunk_paths)))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_parse_chunk_file, p): i for i, p in enumerate(chunk_paths)}
        done = 0
        for fut in as_completed(futures):
            partials.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(chunks):
                print(f"  parsed {done}/{len(chunks)} chunks", flush=True)

    print("Merging adjacency lists...", flush=True)
    adj = _merge_adjacency(partials)
    edge_count = sum(len(v) for v in adj.values())
    meta = {
        "node_count": len(adj),
        "directed_edge_count": edge_count,
        "chunk_count": len(chunks),
        "build_seconds": time.time() - t0,
        "edge_source": str(TRIPLET_METADATA_DIR / TRIPLET_METADATA_GLOB),
    }
    print(
        f"Graph ready: {meta['node_count']:,} nodes, {meta['directed_edge_count']:,} directed edges "
        f"in {meta['build_seconds']:.1f}s",
        flush=True,
    )
    return adj, meta


def save_graph_cache(adj: Adjacency, meta: dict) -> None:
    GRAPH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_CACHE, "wb") as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(GRAPH_META_CACHE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved graph cache to {GRAPH_CACHE}", flush=True)


def load_graph_cache(
    rebuild: bool = False,
    workers: int = 8,
    max_chunks: int | None = None,
) -> Tuple[Adjacency, dict]:
    cache_path = GRAPH_CACHE
    if max_chunks is not None:
        cache_path = DATA_DIR / f"wikidata5m_adjacency_{max_chunks}chunks.pkl"

    if not rebuild and cache_path.exists():
        t0 = time.time()
        print(f"Loading graph cache from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            adj = pickle.load(f)
        meta = {}
        meta_path = GRAPH_META_CACHE if max_chunks is None else DATA_DIR / f"wikidata5m_adjacency_{max_chunks}chunks_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta["load_seconds"] = time.time() - t0
        meta.setdefault("node_count", len(adj))
        meta.setdefault("directed_edge_count", sum(len(v) for v in adj.values()))
        print(
            f"Loaded {meta['node_count']:,} nodes in {meta['load_seconds']:.1f}s",
            flush=True,
        )
        return adj, meta

    adj, meta = build_adjacency_parallel(workers=workers, max_chunks=max_chunks)
    if max_chunks is None:
        save_graph_cache(adj, meta)
    else:
        with open(cache_path, "wb") as f:
            pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta_path = DATA_DIR / f"wikidata5m_adjacency_{max_chunks}chunks_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved partial graph cache to {cache_path}", flush=True)
    return adj, meta


_LABEL_CACHE: Tuple[Dict[str, str], Dict[str, str]] | None = None


def load_labels() -> Tuple[Dict[str, str], Dict[str, str]]:
    global _LABEL_CACHE
    if _LABEL_CACHE is not None:
        return _LABEL_CACHE
    print("Loading entity/relation labels from data/...", flush=True)
    t0 = time.time()
    entity_labels = load_label_map(ENTITY_FILE, "Q")
    relation_labels = load_label_map(RELATION_FILE, "P")
    print(
        f"Labels: {len(entity_labels):,} entities, {len(relation_labels):,} relations "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    _LABEL_CACHE = (entity_labels, relation_labels)
    return _LABEL_CACHE


def main():
    parser = argparse.ArgumentParser(description="Build or load WikiData5M adjacency graph")
    parser.add_argument("--rebuild-graph", action="store_true", help="Force rebuild from metadata chunks")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for chunk parsing")
    parser.add_argument("--max-chunks", type=int, default=None, help="Parse only first N chunks (smoke test)")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    load_graph_cache(rebuild=args.rebuild_graph, workers=args.workers, max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
