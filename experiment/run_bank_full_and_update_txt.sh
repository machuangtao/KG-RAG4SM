#!/usr/bin/env bash
# Full bank BFS (all 146 questions, depths 1-4), then refresh results txt from all full runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python

echo "=== Bank full depth experiment started $(date -Iseconds) ==="
$PYTHON -m experiment.run_depth_experiment --dataset bank --workers 2

echo ""
echo "=== Refreshing results txt $(date -Iseconds) ==="
$PYTHON -m experiment.write_full_results_txt

echo "=== Done $(date -Iseconds) ==="
echo "Results: experiment/logs/local_bfs_experiment_results.txt"
