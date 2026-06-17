#!/usr/bin/env python3
"""1-hop neighbor triple export (fast approximate retrieval)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from experiment.bfs_local import Adjacency, format_path_edges
from experiment.config import (
    ONE_HOP_MAX_EDGES_PER_ENTITY,
    ONE_HOP_MAX_TRIPLES_PER_QUESTION,
    OUTPUT_DIR,
    ROOT,
    SIMILARITY_TOP_K,
)
from experiment.sample_questions import filter_similarity_subset, sample_question_ids
from experiment.wikidata5m_graph import load_graph_cache, load_labels

Edge = Tuple[str, str, str]


def collect_one_hop_paths(
    adj: Adjacency,
    entity_ids: List[str],
    entity_labels: Dict[str, str],
    relation_labels: Dict[str, str],
    allowed_relations: Set[str] | None = None,
    max_edges_per_entity: int = ONE_HOP_MAX_EDGES_PER_ENTITY,
    max_triples_per_question: int = ONE_HOP_MAX_TRIPLES_PER_QUESTION,
) -> Tuple[List[str], int]:
    formatted: List[str] = []
    missing = 0
    seen: Set[Tuple[str, str, str]] = set()

    for eid in entity_ids:
        if eid not in adj:
            missing += 1
            continue
        neighbors = adj.get(eid, [])
        if max_edges_per_entity:
            neighbors = neighbors[:max_edges_per_entity]
        for rel, nbr in neighbors:
            if allowed_relations and rel not in allowed_relations:
                continue
            key = (eid, nbr, rel)
            if key in seen:
                continue
            seen.add(key)
            path_str = format_path_edges([(eid, nbr, rel)], entity_labels, relation_labels)
            if path_str:
                formatted.append(path_str)
            if len(formatted) >= max_triples_per_question:
                return formatted, missing
    return formatted, missing


def run_one_hop_benchmark(
    dataset: str,
    qids: List[str],
    similarity_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    similarity_path = similarity_path or (
        OUTPUT_DIR / f"{dataset}_similarity_results_n{len(qids)}.json"
    )
    output_path = output_path or (OUTPUT_DIR / f"{dataset}_one_hop_triples_n{len(qids)}.json")

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
        qdata = similarity[qid]
        t0 = time.perf_counter()
        entities = qdata.get("similar_entities", [])[:SIMILARITY_TOP_K]
        entity_ids = [e["id"] for e in entities if e.get("id")]
        rels = qdata.get("similar_relations", [])[:5]
        allowed = {r["id"] for r in rels if r.get("id")}

        paths, missing = collect_one_hop_paths(
            adj,
            entity_ids,
            entity_labels,
            relation_labels,
            allowed_relations=allowed if allowed else None,
        )
        total_missing += missing
        elapsed = time.perf_counter() - t0
        per_question_seconds.append(elapsed)
        results.append({"question_id": qid, "paths": paths})

    total_time = time.perf_counter() - t0_all
    avg_time = sum(per_question_seconds) / len(per_question_seconds) if per_question_seconds else 0.0
    total_paths = sum(len(r["paths"]) for r in results)
    total_triples = total_paths

    metadata = {
        "backend": "wikidata5m_one_hop",
        "strategy": "one_hop",
        "dataset": dataset,
        "sample_size": len(results),
        "total_paths": total_paths,
        "total_triples": total_triples,
        "max_hops": 1,
        "missing_entity_lookups": total_missing,
        "graph_nodes": graph_meta.get("node_count"),
        "configuration": {
            "max_edges_per_entity": ONE_HOP_MAX_EDGES_PER_ENTITY,
            "max_triples_per_question": ONE_HOP_MAX_TRIPLES_PER_QUESTION,
        },
        "timing": {
            "total_time_seconds": total_time,
            "average_time_per_question_seconds": avg_time,
            "per_question_seconds": per_question_seconds,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, ensure_ascii=False, indent=2)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["bank", "movie", "cms"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    qids = sample_question_ids(args.dataset, n=args.limit, seed=args.seed)
    sim_path = filter_similarity_subset(args.dataset, qids)
    out = run_one_hop_benchmark(args.dataset, qids, similarity_path=sim_path)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
