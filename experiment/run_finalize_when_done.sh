#!/usr/bin/env bash
# Poll until clinical BFS finishes, then refresh full_experiment_results.txt (run inside screen).
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
LOG=experiment/logs/finalize_watcher.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) finalize watcher started ==="

mark_done() {
  grep -q "clinical pipeline done" experiment/logs/clinical_pipeline.log 2>/dev/null
}

bfs_done() {
  [[ -f testRes/synthea_bfs_local_paths_h4.json && -f testRes/emed_bfs_local_paths_h4.json ]]
}

while ! mark_done && ! bfs_done; do
  echo "$(date) waiting (clinical-bfs still running)..."
  sleep 120
done

echo "=== $(date) refreshing results ==="
python -m experiment.compare_backends --dataset all
python -m experiment.write_results_txt
python -m experiment.write_results_table
echo "=== $(date) finalize watcher done ==="
