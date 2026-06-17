#!/usr/bin/env bash
# Run SPARQL on n=10 sample per dataset/depth, extrapolate API timings, refresh table.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
LOG=experiment/logs/sparql_estimates.log
exec > >(tee -a "$LOG") 2>&1

echo "=== SPARQL sample estimates started $(date -Iseconds) ==="
python -m experiment.sparql_sample_estimate --sample-n 10
python -m experiment.write_results_table
python -m experiment.write_results_txt
echo "=== finished $(date -Iseconds) ==="
