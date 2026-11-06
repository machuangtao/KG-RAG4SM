#!/usr/bin/env bash
# Full clinical experiment in one screen session: similarity + local BFS + report.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
LOG=experiment/logs/clinical_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) clinical pipeline start (pid=$$) ==="

wait_similarity() {
  local ds=$1
  local out="testRes/${ds}_similarity_results.json"
  while [[ ! -f "$out" ]]; do
    if pgrep -f "cms_similarity_fast --dataset ${ds}" >/dev/null; then
      echo "Waiting for ${ds} similarity (another process running)..."
      sleep 60
    else
      echo "Similarity: $ds"
      python -m experiment.cms_similarity_fast --dataset "$ds"
      return
    fi
  done
  echo "Similarity exists: $out"
}

# Synthea + EMED similarity (CMS already embedded)
wait_similarity synthea
wait_similarity emed

echo "=== Stage: load graph cache ==="
python -c "from experiment.wikidata5m_graph import load_graph_cache; load_graph_cache(rebuild=False)"

echo "=== Stage: local BFS synthea + emed (depths 1-4) ==="
python -m experiment.run_pipeline \
  --datasets synthea emed \
  --skip-similarity \
  --skip-compare \
  --skip-rank \
  --bfs-workers 8

echo "=== Stage: compare local vs SPARQL (all datasets, incl. CMS) ==="
python -m experiment.compare_backends --dataset all

echo "=== Stage: write experiment/logs/full_experiment_results.txt ==="
python -m experiment.write_results_txt
python -m experiment.write_results_table

echo "=== $(date) clinical pipeline done ==="
