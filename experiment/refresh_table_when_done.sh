#!/usr/bin/env bash
# Poll until movie h4 + cms h4 SPARQL JSON exist, then refresh tables.
cd "$(dirname "$0")/.."
source venv/bin/activate
while true; do
  movie_ok=0
  cms_ok=0
  [[ -f testRes/movie_bfs_sparql_paths_h4.json ]] && [[ $(stat -c%s testRes/movie_bfs_sparql_paths_h4.json) -gt 10000 ]] && movie_ok=1
  [[ -f testRes/cms_bfs_sparql_paths_h4.json ]] && [[ $(stat -c%s testRes/cms_bfs_sparql_paths_h4.json) -gt 100000 ]] && cms_ok=1
  if [[ $movie_ok -eq 1 && $cms_ok -eq 1 ]]; then
    python -m experiment.compare_backends
    python -m experiment.write_results_table
    python -m experiment.write_results_txt
    echo "Tables refreshed $(date -Iseconds)"
    exit 0
  fi
  echo "$(date -Iseconds) waiting movie_h4=$movie_ok cms_h4=$cms_ok"
  sleep 300
done
