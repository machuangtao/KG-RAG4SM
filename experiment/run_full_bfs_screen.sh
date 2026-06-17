#!/usr/bin/env bash
# Full local BFS for datasets with incomplete runs, then refresh results txt.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python

echo "=== Full BFS + results update started $(date -Iseconds) ==="

# Bank was run with --limit 8; rerun all depths on full 146 questions.
echo ""
echo ">>> bank (146 questions, depths 1-4)"
$PYTHON -m experiment.run_depth_experiment --dataset bank --workers 2

echo ""
echo ">>> Refresh results txt from full-run JSON metadata"
$PYTHON -m experiment.write_full_results_txt

echo ""
echo "=== Done $(date -Iseconds) ==="
echo "Results: experiment/logs/local_bfs_experiment_results.txt"
