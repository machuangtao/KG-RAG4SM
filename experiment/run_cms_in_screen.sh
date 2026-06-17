#!/usr/bin/env bash
# Run CMS BFS completion + results tables inside screen (survives disconnect).
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

LOG=experiment/logs/cms_rerun.log
mkdir -p experiment/logs
echo "=== screen session started $(date -Iseconds) ===" | tee -a "$LOG"

python -m experiment.rerun_cms \
  --workers 32 \
  --bfs-workers 8 \
  --skip-similarity \
  --resume \
  --start-depth 3 \
  2>&1 | tee -a "$LOG"

echo "=== finished $(date -Iseconds) exit=$? ===" | tee -a "$LOG"
