#!/usr/bin/env python3
"""Deprecated: use write_full_results_txt (single canonical results file)."""
from experiment.write_full_results_txt import write_full_results_txt

if __name__ == "__main__":
    p = write_full_results_txt()
    print(f"Wrote {p}")
