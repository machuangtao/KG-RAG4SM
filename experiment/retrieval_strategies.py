#!/usr/bin/env python3
"""Retrieval strategy benchmarks: full BFS vs pruned BFS on a question sample."""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

from experiment.bfs_local import (
    Adjacency,
    bfs_bidirectional_adj,
    format_path_edges,
)
from experiment.config import (
    BENCHMARK_DEPTH,
    MAX_EDGES_PER_NODE,
    MAX_PATHS_PER_PAIR,
    OUTPUT_DIR,
    PRUNED_MAX_ENTITIES,
    ROOT,
    SIMILARITY_TOP_K,
)
from experiment.sample_questions import filter_similarity_subset, sample_question_ids
from experiment.wikidata5m_graph import load_graph_cache, load_labels

Edge = Tuple[str, str, str]


def _pruned_adj(adj: Adjacency, max_edges: int) -> Adjacency:
    if not max_edges:
        return adj
    return {node: edges[:max_edges] for node, edges in adj.items()}


def process_question(
    qid: str,
    qdata: dict,
    adj: Adjacency,
    entity_labels: Dict[str, str],
    relation_labels: Dict[str, str],
    max_hops: int,
    strategy: str,
    max_entities: int = SIMILARITY_TOP_K,
    max_edges_per_node: int = 0,
    relation_filter: bool = False,
) -> Tuple[dict, float, int]:
    t0 = time.perf_counter()
    missing = 0
    formatted_paths: List[str] = []

    entities = qdata.get("similar_entities", [])[:max_entities]
    entity_ids = [e["id"] for e in entities if e.get("id")]
    allowed_rels: Set[str] = set()
    if relation_filter:
        allowed_rels = {r["id"] for r in qdata.get("similar_relations", [])[:5] if r.get("id")}

    if len(entity_ids) < 2:
        return {"question_id": qid, "paths": []}, time.perf_counter() - t0, missing

    use_adj = _pruned_adj(adj, max_edges_per_node) if max_edges_per_node else adj

    for e1, e2 in combinations(entity_ids, 2):
        if e1 not in use_adj:
            missing += 1
        if e2 not in use_adj:
            missing += 1
        edge_paths = bfs_bidirectional_adj(use_adj, e1, e2, max_hops)
        for ep in edge_paths:
            if relation_filter and allowed_rels:
                if any(r not in allowed_rels for _, _, r in ep):
                    continue
            formatted_paths.append(format_path_edges(ep, entity_labels, relation_labels))

    return {"question_id": qid, "paths": formatted_paths}, time.perf_counter() - t0, missing


def run_strategy_benchmark(
    dataset: str,
    qids: List[str],
    strategy: str,
    max_hops: int = BENCHMARK_DEPTH,
    similarity_path: Path | None = None,
) -> Path:
    if strategy == "full_bfs":
        max_entities = SIMILARITY_TOP_K
        max_edges = 0
        rel_filter = False
    elif strategy == "pruned_bfs":
        max_entities = PRUNED_MAX_ENTITIES
        max_edges = MAX_EDGES_PER_NODE
        rel_filter = True
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    similarity_path = similarity_path or (
        OUTPUT_DIR / f"{dataset}_similarity_results_n{len(qids)}.json"
    )
    output_path = OUTPUT_DIR / f"{dataset}_bfs_{strategy}_h{max_hops}_n{len(qids)}.json"

    adj, graph_meta = load_graph_cache(rebuild=False)
    entity_labels, relation_labels = load_labels()

    with open(similarity_path, "r", encoding="utf-8") as f:
        similarity = json.load(f)

    results = []
    per_question_seconds: List[float] = []
    total_missing = 0
    t0_all = time.perf_counter()

    for qid in qids:
        if qid not in similarity:
            continue
        row, elapsed, missing = process_question(
            qid,
            similarity[qid],
            adj,
            entity_labels,
            relation_labels,
            max_hops,
            strategy,
            max_entities=max_entities,
            max_edges_per_node=max_edges,
            relation_filter=rel_filter,
        )
        results.append(row)
        per_question_seconds.append(elapsed)
        total_missing += missing

    total_time = time.perf_counter() - t0_all
    avg_time = sum(per_question_seconds) / len(per_question_seconds) if per_question_seconds else 0.0
    total_paths = sum(len(r.get("paths") or []) for r in results)
    total_triples = sum(
        max(0, len(p.split("-->")) - 1) for r in results for p in (r.get("paths") or [])
    )

    metadata = {
        "backend": "wikidata5m_local",
        "strategy": strategy,
        "dataset": dataset,
        "sample_size": len(results),
        "total_paths": total_paths,
        "total_triples": total_triples,
        "max_hops": max_hops,
        "missing_entity_lookups": total_missing,
        "graph_nodes": graph_meta.get("node_count"),
        "strategy_params": {
            "max_entities": max_entities,
            "max_edges_per_node": max_edges,
            "relation_filter": rel_filter,
            "max_paths_per_pair": MAX_PATHS_PER_PAIR,
        },
        "timing": {
            "total_time_seconds": total_time,
            "average_time_per_question_seconds": avg_time,
            "per_question_seconds": per_question_seconds,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, ensure_ascii=False, indent=2)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--strategy", choices=["full_bfs", "pruned_bfs"], required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hops", type=int, default=BENCHMARK_DEPTH)
    args = parser.parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    qids = sample_question_ids(args.dataset, n=args.limit, seed=args.seed)
    sim_path = filter_similarity_subset(args.dataset, qids)
    out = run_strategy_benchmark(
        args.dataset, qids, args.strategy, max_hops=args.max_hops, similarity_path=sim_path
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
