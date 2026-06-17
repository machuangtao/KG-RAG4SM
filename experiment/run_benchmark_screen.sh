#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ -z "${OPENAI_API_KEY:-}" && -f openai_api.txt ]]; then
  export OPENAI_API_KEY="$(tr -d '\n\r' < openai_api.txt)"
fi
LOG=experiment/logs/latency_benchmark.log
mkdir -p experiment/logs
echo "=== started $(date -Iseconds) ===" | tee "$LOG"
# LLM: gpt-4o-mini only (see experiment/config.py LLM_MODEL)
python -m experiment.run_latency_benchmark \
  --datasets bank movie cms \
  --limit 50 --seed 42 \
  --skip-retrieval \
  2>&1 | tee -a "$LOG"
echo "=== finished $(date -Iseconds) exit=$? ===" | tee -a "$LOG"
