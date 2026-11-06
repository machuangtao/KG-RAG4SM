#!/usr/bin/env bash
# Run full WikiData5M local BFS experiment from repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-./venv/bin/python}"
exec "$PYTHON" -m experiment.run_pipeline "$@"
