#!/usr/bin/env bash
# Launch the full clinical experiment inside GNU screen.
# Usage: ./experiment/run_in_screen.sh [screen-name]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-kg-rag-clinical}"
cd "$ROOT"

if screen -ls | grep -q "[0-9]*\.${NAME}[[:space:]]"; then
  echo "Screen session '${NAME}' already exists. Attach with: screen -r ${NAME}"
  exit 1
fi

chmod +x experiment/run_clinical_pipeline.sh
screen -dmS "${NAME}" bash -c "
  cd '${ROOT}' && ./experiment/run_clinical_pipeline.sh
  echo ''
  echo 'Press Enter to close this screen window...'
  read
"
echo "Started screen session: ${NAME}"
echo "  Attach:  screen -r ${NAME}"
echo "  Log:     tail -f experiment/logs/clinical_pipeline.log"
