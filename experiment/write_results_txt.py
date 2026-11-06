#!/usr/bin/env python3
"""Write consolidated experiment results to a .txt report."""
from __future__ import annotations

import json
import time
from pathlib import Path

from experiment.config import (
    BENCHMARK_N,
    CLINICAL_DATASETS,
    DATASETS,
    DEPTHS,
    EXPERIMENT_LOG_DIR,
    LLM_MODEL,
    OUTPUT_DIR,
    ROOT,
)
from experiment.dataset_registry import DEFAULT_QUESTION_COUNTS
from experiment.timing_meta import local_timing_from_meta
from experiment.write_results_table import COLUMNS, _fmt, _load_comparison_rows

DATASET_NQUESTIONS = dict(DEFAULT_QUESTION_COUNTS)


def _dataset_question_count(dataset: str) -> int:
    sim = OUTPUT_DIR / f"{dataset}_similarity_results.json"
    if sim.exists():
        try:
            with open(sim, "r", encoding="utf-8") as f:
                return len(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return DATASET_NQUESTIONS.get(dataset, 0)


def _read_json_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("metadata", {})


def _speed_note(dataset: str, depth: str) -> str:
    if dataset == "movie" and depth == "4":
        return "SPARQL partial~200q"
    if dataset == "cms" and depth == "4":
        return "SPARQL test n=5"
    if dataset in ("emed", "synthea") and depth == "4":
        return "SPARQL not run"
    return ""


def _append_speed_table(lines: list, cmp_rows: list) -> None:
    lines.append("")
    lines.append("Per-question avg = timing.average_time_per_question_seconds")
    lines.append("Local sum (avg×N) = timing.sum_question_time_seconds")
    lines.append("Local wall clock = timing.wall_clock_seconds (parallel; sum ≠ wall)")
    lines.append("API/Local ratio = SPARQL avg ÷ local avg  (>1 means API slower per question)")
    lines.append("")
    hdr = (
        f"{'Dataset':<8} {'Depth':<6} {'N':<6} {'LocalSum(s)':<12} {'LocalWall(s)':<12} "
        f"{'LocalAvg(s/q)':<14} {'SPARQLTot(s)':<13} {'SPARQLAvg(s/q)':<15} "
        f"{'API/Local':<10} {'Notes':<20}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in cmp_rows:
        ds = str(r.get("dataset", ""))
        depth = str(r.get("depth", ""))
        nq = int(r.get("local_num_questions") or 0) or _dataset_question_count(ds)
        loc_tot = float(r.get("local_total_sec") or 0)
        loc_wall = float(r.get("local_wall_sec") or 0)
        loc_avg = float(r.get("local_avg_sec") or 0)
        spq_tot = float(r.get("sparql_total_sec") or 0)
        spq_avg = float(r.get("sparql_avg_sec") or 0)
        if loc_avg >= 0.01 and spq_avg > 0:
            ratio = f"{spq_avg / loc_avg:.2f}x"
        elif spq_avg > 0 and loc_avg > 0:
            ratio = "API>>local"
        elif spq_avg > 0:
            ratio = "API only"
        else:
            ratio = "-"
        note = _speed_note(ds, depth)
        lines.append(
            f"{ds:<8} {depth:<6} {nq:<6} {loc_tot:<12.1f} {loc_wall:<12.1f} {loc_avg:<14.3f} "
            f"{spq_tot:<13.1f} {spq_avg:<15.3f} {ratio:<10} {note:<20}"
        )
    lines.append("")
    lines.append("Depth 4 summary:")
    h4_rows = [r for r in cmp_rows if str(r.get("depth")) == "4"]
    for r in h4_rows:
        ds = str(r.get("dataset", ""))
        loc_avg = float(r.get("local_avg_sec") or 0)
        spq_avg = float(r.get("sparql_avg_sec") or 0)
        nq = _dataset_question_count(ds)
        if loc_avg > 0 and spq_avg > 0:
            est_api_hours = spq_avg * nq / 3600
            est_loc_hours = loc_avg * nq / 3600
            lines.append(
                f"  {ds}: local ~{loc_avg:.1f}s/q ({est_loc_hours:.2f}h full); "
                f"SPARQL ~{spq_avg:.1f}s/q ({est_api_hours:.2f}h full); "
                f"API {spq_avg/loc_avg:.1f}x slower per question"
            )
        elif loc_avg > 0:
            est_loc_hours = loc_avg * nq / 3600
            lines.append(f"  {ds}: local ~{loc_avg:.1f}s/q ({est_loc_hours:.2f}h full); SPARQL: (not available)")


def write_results_txt(out_path: Path | None = None) -> Path:
    out_path = out_path or (EXPERIMENT_LOG_DIR / "full_experiment_results.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("WikiData5M Local BFS — Full Experiment Results")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Graph cache info
    meta_path = ROOT / "data" / "wikidata5m_adjacency_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            gmeta = json.load(f)
        lines.append("Graph (from wikidata_embedding_triplet2 metadata chunks)")
        lines.append(f"  Nodes: {gmeta.get('node_count', '?')}")
        lines.append(f"  Directed edges: {gmeta.get('directed_edge_count', '?')}")
        lines.append(f"  Build time (s): {gmeta.get('build_seconds', '?')}")
        lines.append("")

    # --- How results are computed ---
    lines.append("=" * 80)
    lines.append("How each stage is computed (in-memory vs remote API)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("YES — the depth tables below are IN-MEMORY local BFS only:")
    lines.append("  • Graph: adjacency dict loaded from data/wikidata5m_adjacency.pkl (~4.1M nodes, ~22M edges)")
    lines.append("  • BFS: bidirectional search on adj[entity] in RAM (experiment/bfs_local.py)")
    lines.append("  • No network calls during BFS; labels from data/wikidata5m_entity.txt + relation.txt")
    lines.append("")
    lines.append("SPARQL / API baseline (comparison table below):")
    lines.append("  • modules/bfs_paths.py → https://query.wikidata.org/sparql (async HTTP)")
    lines.append("  • Rate limit ~3 req/s; per-question pair-wise path queries + Wikidata label API")
    lines.append("  • Same similarity input (testRes/{dataset}_similarity_results.json), same top-10 entities")
    lines.append("")
    lines.append("Entity similarity (upstream of BFS, not in depth table):")
    lines.append("  • Questions: vectors stored in ChromaDB (see next section)")
    lines.append("  • KG entities/relations: all NPZ shards loaded into NumPy; brute-force cosine (not FAISS)")
    lines.append("")

    lines.append("=" * 80)
    lines.append("ChromaDB and HNSW (question embeddings)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Storage layout:")
    lines.append("  chromadb/chromadb_store_test_{bank,movie,cms,emed,synthea}_questions/")
    lines.append("  Collection: test_{dataset}_questions_collection")
    lines.append("")
    lines.append("ChromaDB default index:")
    lines.append("  • Persistent collections use an approximate nearest-neighbor index, typically HNSW")
    lines.append("    (Hierarchical Navigable Small World) on disk under the store directory.")
    lines.append("  • HNSW supports fast top-k by query vector via collection.query(n_results=k).")
    lines.append("")
    lines.append("How THIS codebase uses ChromaDB:")
    lines.append("  • Indexing: embeddings are written when question_embedding.py (or similar) populates the store.")
    lines.append("  • Retrieval in cosine_similarity_search / gpu_similarity: collection.get(include=['embeddings'])")
    lines.append("    loads ALL question vectors for the dataset (full scan in memory), not collection.query().")
    lines.append("  • Entity KG search does NOT use ChromaDB — it scans Wikidata_embed_entity/*.npz in batches.")
    lines.append("")
    lines.append("So: ChromaDB provides durable vector storage + optional HNSW for ANN queries; the experiment")
    lines.append("pipeline primarily uses .get() for questions and NumPy matmul for entity top-10.")
    lines.append("")

    # Clinical datasets (CMS, EMED, Synthea) — primary report section
    lines.append("=" * 80)
    lines.append("Clinical datasets: CMS, EMED, Synthea — local in-memory BFS (full datasets)")
    lines.append("=" * 80)
    for ds in CLINICAL_DATASETS:
        nq = _dataset_question_count(ds)
        lines.append("-" * 80)
        lines.append(f"Dataset: {ds.upper()}  ({nq} questions)")
        lines.append("-" * 80)
        lines.append(
            f"{'Depth':<8} {'Paths':<12} {'Triples':<12} {'N':<6} {'Avg s/q':<12} "
            f"{'Sum s':<12} {'Wall s':<12}"
        )
        for depth in DEPTHS:
            p = OUTPUT_DIR / f"{ds}_bfs_local_paths_h{depth}.json"
            m = _read_json_meta(p)
            if not m:
                lines.append(f"{depth:<8} {'(missing)':<12}")
                continue
            timing = local_timing_from_meta(m)
            lines.append(
                f"{depth:<8} {m.get('total_paths', 0):<12} {m.get('total_triples', 0):<12} "
                f"{timing['num_questions']:<6} "
                f"{timing['average_time_per_question_seconds']:<12.3f} "
                f"{timing['sum_question_time_seconds']:<12.1f} "
                f"{timing['wall_clock_seconds']:<12.1f}"
            )
        tradeoff = OUTPUT_DIR / f"{ds}_bfs_local_depth_tradeoff.txt"
        if tradeoff.exists():
            lines.append("")
            lines.append(f"Tradeoff file: {tradeoff}")
        lines.append("")

    # All datasets (including bank/movie)
    lines.append("=" * 80)
    lines.append("All datasets — local in-memory BFS by depth (full datasets)")
    lines.append("=" * 80)
    for ds in DATASETS:
        nq = _dataset_question_count(ds)
        lines.append("-" * 80)
        lines.append(f"Dataset: {ds.upper()}  ({nq} questions)")
        lines.append("-" * 80)
        lines.append(
            f"{'Depth':<8} {'Paths':<12} {'Triples':<12} {'N':<6} {'Avg s/q':<12} "
            f"{'Sum s':<12} {'Wall s':<12}"
        )
        for depth in DEPTHS:
            p = OUTPUT_DIR / f"{ds}_bfs_local_paths_h{depth}.json"
            m = _read_json_meta(p)
            if not m:
                lines.append(f"{depth:<8} {'(missing)':<12}")
                continue
            timing = local_timing_from_meta(m)
            lines.append(
                f"{depth:<8} {m.get('total_paths', 0):<12} {m.get('total_triples', 0):<12} "
                f"{timing['num_questions']:<6} "
                f"{timing['average_time_per_question_seconds']:<12.3f} "
                f"{timing['sum_question_time_seconds']:<12.1f} "
                f"{timing['wall_clock_seconds']:<12.1f}"
            )
        tradeoff = OUTPUT_DIR / f"{ds}_bfs_local_depth_tradeoff.txt"
        if tradeoff.exists():
            lines.append("")
            lines.append(f"Tradeoff file: {tradeoff}")
        lines.append("")

    # Speed summary: full dataset wall-clock and per-question (depth 4 focus + all depths)
    cmp_rows = _load_comparison_rows()
    lines.append("=" * 80)
    lines.append("Clinical speed: in-memory local BFS vs Wikidata SPARQL API (CMS, EMED, Synthea)")
    lines.append("=" * 80)
    clinical_rows = [r for r in cmp_rows if str(r.get("dataset")) in CLINICAL_DATASETS]
    _append_speed_table(lines, clinical_rows)
    lines.append("")
    lines.append("=" * 80)
    lines.append("Speed: in-memory local BFS vs Wikidata SPARQL API (all datasets)")
    lines.append("=" * 80)
    _append_speed_table(lines, cmp_rows)
    lines.append("")

    # Backend comparison (formatted table)
    lines.append("=" * 80)
    lines.append("Local vs SPARQL — path counts and timing (all depths)")
    lines.append("=" * 80)
    if cmp_rows:
        col_sep = " "
        header = col_sep.join(name.ljust(w) for _, name, w in COLUMNS)
        sep = col_sep.join("-" * w for _, _, w in COLUMNS)
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for r in cmp_rows:
            lines.append(col_sep.join(_fmt(r.get(key), w) for key, _, w in COLUMNS))
        lines.append("")
        lines.append(f"(CSV: {OUTPUT_DIR / 'bfs_backend_comparison.csv'})")
    else:
        lines.append("(comparison data not found)")
    lines.append("")

    # Latency benchmark (n=50, gpt-4o-mini)
    latency_path = EXPERIMENT_LOG_DIR / f"latency_n{BENCHMARK_N}.json"
    lines.append("=" * 80)
    lines.append(f"Latency benchmark (n={BENCHMARK_N}, seed=42, {LLM_MODEL})")
    lines.append("=" * 80)
    if latency_path.exists():
        with open(latency_path, "r", encoding="utf-8") as f:
            lat = json.load(f)
        cfg = lat.get("config", {})
        lines.append(
            f"Config: depth={cfg.get('depth')} datasets={cfg.get('datasets')} "
            f"strategies={cfg.get('strategies')}"
        )
        if lat.get("llm_skipped"):
            reason = lat.get("llm_skip_reason", "OPENAI_API_KEY not set or --skip-llm")
            lines.append(f"Note: LLM stage skipped ({reason})")
        lines.append("")
        hdr = (
            f"{'Dataset':<8} {'Strategy':<12} {'Sim(s)':<10} {'Retr(s)':<10} {'Rank(s)':<10} "
            f"{'LLM-seq':<10} {'LLM-batch':<10} {'E2E-seq':<10} {'E2E-batch':<10} {'Paths/q':<10}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for row in lat.get("summary", []):
            lines.append(
                f"{row.get('dataset', ''):<8} {row.get('strategy', ''):<12} "
                f"{row.get('sim_avg_sec', 0):<10.4f} {row.get('retrieve_avg_sec', 0):<10.4f} "
                f"{row.get('rank_avg_sec', 0):<10.4f} {row.get('llm_seq_avg_sec', 0):<10.4f} "
                f"{row.get('llm_batch_avg_sec', 0):<10.4f} {row.get('e2e_seq_avg_sec', 0):<10.4f} "
                f"{row.get('e2e_batch_avg_sec', 0):<10.4f} {row.get('paths_per_q', 0):<10.2f}"
            )
        lines.append("")
        lines.append(f"Full JSON: {latency_path}")
    else:
        lines.append("(latency benchmark not run yet)")
    lines.append("")

    # Output file listing
    lines.append("=" * 80)
    lines.append("Output artifacts")
    lines.append("=" * 80)
    for ds in DATASETS:
        for depth in DEPTHS:
            j = OUTPUT_DIR / f"{ds}_bfs_local_paths_h{depth}.json"
            if j.exists():
                lines.append(f"  {j}")
        xlsx = ROOT / "datasets" / "reproduce" / f"test_{ds}_q_with_paths_local_h4.xlsx"
        if xlsx.exists():
            lines.append(f"  {xlsx}")
    lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    p = write_results_txt()
    print(f"Wrote {p}")
