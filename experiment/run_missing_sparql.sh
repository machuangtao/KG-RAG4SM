#!/usr/bin/env bash
# Fill missing SPARQL cells: movie h1,h2,h4 and cms h4 (resume).
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
LOG=experiment/logs/sparql_fill.log
mkdir -p experiment/logs
exec > >(tee -a "$LOG") 2>&1

echo "=== SPARQL fill started $(date -Iseconds) ==="

for depth in 1 2 4; do
  out="testRes/movie_bfs_sparql_paths_h${depth}.json"
  if [[ -f "$out" ]] && [[ $(stat -c%s "$out") -gt 1000 ]]; then
    echo "Skip movie h${depth} (exists)"
    continue
  fi
  echo "--- movie SPARQL depth ${depth} ---"
  python -m modules.bfs_paths --dataset movie --max_hops "$depth" --output "movie_bfs_sparql_paths_h${depth}"
done

echo "--- cms SPARQL h4 (resume) ---"
python -m modules.bfs_paths --dataset cms --max_hops 4 --output cms_bfs_sparql_paths_h4 --resume

echo "=== Regenerating tables ==="
python -m experiment.compare_backends
python -m experiment.write_results_table
python -m experiment.write_results_txt

echo "=== finished $(date -Iseconds) ==="
