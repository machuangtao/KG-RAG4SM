#!/usr/bin/env bash
# Run 4-hop local BFS on n=10 questions per dataset (one dataset at a time), then refresh tables.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=./venv/bin/python
OUT_DIR=experiment/logs
SAMPLE_N=10

mkdir -p "$OUT_DIR"

echo "=== 4-hop sample run (n=$SAMPLE_N) started $(date -Iseconds) ==="

for ds in bank movie cms emed synthea; do
  echo ""
  echo ">>> Dataset: $ds ($(date -Iseconds))"
  extra_args=()
  if [ "$ds" = "movie" ]; then
    extra_args+=(--enable-path-cap)
    echo "    (movie: path cap ON to avoid OOM)"
  else
    extra_args+=(--disable-path-cap)
  fi
  $PYTHON -m experiment.local_depth_sample_estimate \
    --datasets "$ds" \
    --depths 4 \
    --sample-n "$SAMPLE_N" \
    --workers 1 \
    "${extra_args[@]}"
done

echo ""
echo ">>> Rebuild summary + tables ($(date -Iseconds))"
$PYTHON -m experiment.local_depth_sample_estimate --refresh-only --sample-n "$SAMPLE_N"

echo "=== Done $(date -Iseconds) ==="
echo "Table: $OUT_DIR/local_bfs_experiment_results.txt"
