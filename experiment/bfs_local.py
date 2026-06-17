#!/usr/bin/env python3
"""Local in-memory BFS path finding on WikiData5M adjacency."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from experiment.config import (
    DATASETS,
    DEFAULT_BATCH_SIZE,
    MAX_PATHS_PER_PAIR,
    OUTPUT_DIR,
    ROOT,
)
from experiment.timing_meta import paths_per_pair_limit
from experiment.wikidata5m_graph import load_graph_cache, load_labels

logger = logging.getLogger(__name__)

Edge = Tuple[str, str, str]  # (head, tail, relation)
Adjacency = Dict[str, List[Tuple[str, str]]]

_WORKER: dict = {}


def bfs_bidirectional_adj(
    adj: Adjacency,
    start: str,
    end: str,
    max_hops: int,
    max_paths: Optional[int] = MAX_PATHS_PER_PAIR,
) -> List[List[Edge]]:
    if start == end:
        return []
    if start not in adj or end not in adj:
        return []
    paths: List[List[Edge]] = []
    fwd_half = (max_hops + 1) // 2
    bwd_half = max_hops // 2

    queue_f: deque = deque([(start, [])])
    paths_f: Dict[str, List[List[Edge]]] = {start: [[]]}
    visited_f: Set[str] = {start}

    queue_b: deque = deque([(end, [])])
    paths_b: Dict[str, List[List[Edge]]] = {end: [[]]}
    visited_b: Set[str] = {end}

    def _meet(forward_path: List[Edge], backward_path: List[Edge]) -> List[Edge]:
        rev = [(t, s, p) for s, t, p in reversed(backward_path)]
        return forward_path + rev

    cap = max_paths

    while queue_f or queue_b:
        for forward in (True, False):
            if forward:
                if not queue_f:
                    continue
                node, path = queue_f.popleft()
                other = paths_b
                side_half = fwd_half
            else:
                if not queue_b:
                    continue
                node, path = queue_b.popleft()
                other = paths_f
                side_half = bwd_half

            if path and node in other:
                for op in other[node]:
                    full = _meet(path, op) if forward else _meet(op, path)
                    if 0 < len(full) <= max_hops:
                        paths.append(full)

            if len(path) >= side_half:
                continue

            for rel, nbr in adj.get(node, []):
                new_path = path + [(node, nbr, rel)]
                if forward:
                    if nbr not in visited_f:
                        visited_f.add(nbr)
                        queue_f.append((nbr, new_path))
                        paths_f.setdefault(nbr, []).append(new_path)
                else:
                    if nbr not in visited_b:
                        visited_b.add(nbr)
                        queue_b.append((nbr, new_path))
                        paths_b.setdefault(nbr, []).append(new_path)

    if cap is not None and len(paths) > cap:
        return paths[:cap]
    return paths


def format_path_edges(
    edges: List[Edge],
    entity_labels: Dict[str, str],
    relation_labels: Dict[str, str],
) -> str:
    if not edges:
        return ""
    head, tail, rel = edges[0]
    parts = [f"{entity_labels.get(head, head)} ({head})"]
    for h, t, r in edges:
        parts.append(f"--[{relation_labels.get(r, r)}] ({r})-->")
        parts.append(f"{entity_labels.get(t, t)} ({t})")
    return " ".join(parts)


def _process_question_worker(args: Tuple[str, dict, int]) -> Tuple[dict, Set[str], Set[str], float, int]:
    question_id, qdata, max_hops = args
    state = _WORKER
    adj = state["adj"]
    entity_labels = state["entity_labels"]
    relation_labels = state["relation_labels"]

    t0 = time.time()
    missing = 0
    qids: Set[str] = set()
    pids: Set[str] = set()
    formatted_paths: List[str] = []

    entities = qdata.get("similar_entities", [])[:10]
    entity_ids = [e["id"] for e in entities if e.get("id")]

    if len(entity_ids) < 2:
        return (
            {"question_id": question_id, "paths": []},
            qids,
            pids,
            time.time() - t0,
            missing,
        )

    for e1, e2 in combinations(entity_ids, 2):
        if e1 not in adj:
            missing += 1
        if e2 not in adj:
            missing += 1
        edge_paths = bfs_bidirectional_adj(
            adj, e1, e2, max_hops, max_paths=state.get("max_paths_per_pair")
        )
        for ep in edge_paths:
            for h, t, r in ep:
                qids.add(h)
                qids.add(t)
                pids.add(r)
            formatted_paths.append(format_path_edges(ep, entity_labels, relation_labels))

    elapsed = time.time() - t0
    logger.info("Question %s: %d paths in %.3fs", question_id, len(formatted_paths), elapsed)
    return (
        {"question_id": question_id, "paths": formatted_paths},
        qids,
        pids,
        elapsed,
        missing,
    )


def _init_worker_state(adj, entity_labels, relation_labels, max_paths_per_pair: Optional[int]):
    global _WORKER
    _WORKER = {
        "adj": adj,
        "entity_labels": entity_labels,
        "relation_labels": relation_labels,
        "max_paths_per_pair": max_paths_per_pair,
    }


def run_bfs_dataset(
    dataset: str,
    max_hops: int,
    input_file: Optional[Path] = None,
    output_base: Optional[str] = None,
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
    workers: int = 1,
    rebuild_graph: bool = False,
    graph_workers: int = 8,
    max_chunks: int | None = None,
    adj: Optional[Adjacency] = None,
    graph_meta: Optional[dict] = None,
    entity_labels: Optional[Dict[str, str]] = None,
    relation_labels: Optional[Dict[str, str]] = None,
    disable_path_cap: bool = False,
) -> Path:
    input_path = input_file or (OUTPUT_DIR / f"{dataset}_similarity_results.json")
    output_base = output_base or f"{dataset}_bfs_local_paths_h{max_hops}"
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    final_json = output_dir / f"{output_base}.json"
    output_txt = output_dir / f"{output_base}.txt"
    checkpoint_file = output_dir / f"{output_base}.json.checkpoint.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Similarity file not found: {input_path}")

    if adj is None or graph_meta is None:
        adj, graph_meta = load_graph_cache(
            rebuild=rebuild_graph, workers=graph_workers, max_chunks=max_chunks
        )
    if entity_labels is None or relation_labels is None:
        entity_labels, relation_labels = load_labels()

    start_from = 0
    question_results: Dict[str, dict] = {}
    question_times: List[float] = []
    all_qids: Set[str] = set()
    all_pids: Set[str] = set()
    total_missing = 0

    if resume and checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            cp = json.load(f)
        start_from = cp.get("last_batch_end", 0)
        question_results = cp.get("question_results", {})
        question_times = cp.get("question_times", [])
        all_qids = set(cp.get("all_qids", []))
        all_pids = set(cp.get("all_pids", []))
        total_missing = cp.get("missing_entity_lookups", 0)
        print(f"Resuming from question index {start_from}", flush=True)

    with open(input_path, "r", encoding="utf-8") as f:
        similarity = json.load(f)
    if isinstance(similarity, list):
        similarity = {item["question_id"]: item for item in similarity if "question_id" in item}

    items = list(similarity.items())
    if limit:
        items = items[:limit]

    overall_t0 = time.time()
    workers = max(1, workers)
    pair_cap = paths_per_pair_limit(disable_path_cap)
    _init_worker_state(adj, entity_labels, relation_labels, pair_cap)

    # Thread pool shares the in-memory graph (ProcessPool fork caused OOM with ~400MB+ adjacency).
    pool_ctx = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        for batch_start in range(start_from, len(items), batch_size):
            batch_end = min(batch_start + batch_size, len(items))
            batch = items[batch_start:batch_end]
            tasks = [(qid, qdata, max_hops) for qid, qdata in batch]

            if pool_ctx:
                futures = [pool_ctx.submit(_process_question_worker, t) for t in tasks]
                batch_out = [f.result() for f in as_completed(futures)]
            else:
                batch_out = [_process_question_worker(t) for t in tasks]

            for result, qids, pids, qtime, missing in batch_out:
                question_results[result["question_id"]] = result
                question_times.append(qtime)
                all_qids.update(qids)
                all_pids.update(pids)
                total_missing += missing

            print(f"Processed {batch_end}/{len(items)} questions", flush=True)
            cp = {
                "last_batch_end": batch_end,
                "question_results": question_results,
                "question_times": question_times,
                "all_qids": list(all_qids),
                "all_pids": list(all_pids),
                "missing_entity_lookups": total_missing,
                "max_hops": max_hops,
            }
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(cp, f, ensure_ascii=False)
    finally:
        if pool_ctx:
            pool_ctx.shutdown(wait=True)

    results = [question_results[qid] for qid, _ in items if qid in question_results]
    total_paths = sum(len(r.get("paths") or []) for r in results)
    total_triples = sum(
        max(0, (len(p.split("-->")) - 1)) for r in results for p in (r.get("paths") or [])
    )
    wall_time = time.time() - overall_t0
    sum_question_time = sum(question_times)
    n_timed = len(question_times)
    avg_time = (sum_question_time / n_timed) if n_timed else 0.0

    metadata = {
        "total_questions": len(results),
        "total_unique_entities": len(all_qids),
        "total_unique_relations": len(all_pids),
        "total_paths": total_paths,
        "total_triples": total_triples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_hops": max_hops,
        "backend": "wikidata5m_local",
        "bfs_frontier": "asymmetric",
        "bfs_search": "full_frontier_then_cap",
        "edge_source": graph_meta.get("edge_source", "wikidata_embedding_triplet2/metadata_chunk_*.json"),
        "label_source": "data/wikidata5m_entity.txt + data/wikidata5m_relation.txt",
        "graph_nodes": graph_meta.get("node_count", len(adj)),
        "graph_edges": graph_meta.get("directed_edge_count"),
        "graph_load_seconds": graph_meta.get("load_seconds") or graph_meta.get("build_seconds"),
        "missing_entity_lookups": total_missing,
        "configuration": {
            "input_file": str(input_path),
            "batch_size": batch_size,
            "workers": workers,
            "disable_path_cap": disable_path_cap,
        },
        "timing": {
            "wall_clock_seconds": wall_time,
            "sum_question_time_seconds": sum_question_time,
            "average_time_per_question_seconds": avg_time,
            "num_questions": n_timed,
            "num_workers": workers,
        },
    }

    output_data = {"metadata": metadata, "results": results}
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    with open(output_txt, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Question ID: {r['question_id']}\n")
            for p in r.get("paths") or []:
                f.write(f"{p}\n")
            if not r.get("paths"):
                f.write("No paths found.\n")
            f.write("\n" + "=" * 80 + "\n")

    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
        except OSError:
            pass

    print(f"Saved {final_json}", flush=True)
    return final_json


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="WikiData5M local in-memory BFS")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 4))),
                        help="Thread workers for BFS (shared memory; avoid ProcessPool on large graph)")
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument("--graph-workers", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    run_bfs_dataset(
        dataset=args.dataset,
        max_hops=args.max_hops,
        input_file=Path(args.input) if args.input else None,
        output_base=args.output,
        limit=args.limit,
        batch_size=args.batch_size,
        resume=args.resume,
        workers=args.workers,
        rebuild_graph=args.rebuild_graph,
        graph_workers=args.graph_workers,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
