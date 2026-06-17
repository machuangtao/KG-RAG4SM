#!/usr/bin/env bash
# n=10 fixed-BFS samples for clinical h3/h4 (feeds extrapolation until full runs finish).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python
N=10
WORKERS=2

for ds in cms emed synthea; do
  for depth in 3 4; do
    echo "=== $ds h$depth n=$N $(date -Iseconds) ==="
    $PYTHON -m experiment.bfs_local \
      --dataset "$ds" \
      --max_hops "$depth" \
      --limit "$N" \
      --workers "$WORKERS" \
      --output "${ds}_bfs_local_sample_h${depth}_n${N}_v2"
  done
done
$PYTHON -m experiment.write_full_results_txt
echo "Sample v2 + txt done $(date -Iseconds)"
