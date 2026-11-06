#!/usr/bin/env bash
# Full MOVIE depths 3+4 with fixed BFS (h4 was faster than h3 in stale May runs).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python
WORKERS=2

echo "=== MOVIE depth 3 $(date -Iseconds) ==="
$PYTHON -m experiment.bfs_local --dataset movie --max_hops 3 --workers "$WORKERS"

echo "=== MOVIE depth 4 $(date -Iseconds) ==="
$PYTHON -m experiment.bfs_local --dataset movie --max_hops 4 --workers "$WORKERS"

echo "=== Update results txt $(date -Iseconds) ==="
$PYTHON -m experiment.write_full_results_txt

echo "=== MOVIE h3/h4 done $(date -Iseconds) ==="
