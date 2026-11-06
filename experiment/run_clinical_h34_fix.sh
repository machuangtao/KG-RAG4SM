#!/usr/bin/env bash
# Re-run CMS/EMED/Synthea depths 3+4 with fixed asymmetric full-frontier BFS.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python
WORKERS=2

for ds in cms emed synthea; do
  echo "=== $ds depth 3 $(date -Iseconds) ==="
  $PYTHON -m experiment.bfs_local --dataset "$ds" --max_hops 3 --workers "$WORKERS"
  echo "=== $ds depth 4 $(date -Iseconds) ==="
  $PYTHON -m experiment.bfs_local --dataset "$ds" --max_hops 4 --workers "$WORKERS"
  $PYTHON -m experiment.write_full_results_txt
  echo "=== $ds done $(date -Iseconds) ==="
done

echo "Clinical h3/h4 reruns complete $(date -Iseconds)"
$PYTHON -m experiment.write_full_results_txt
echo "Final results txt updated $(date -Iseconds)"
