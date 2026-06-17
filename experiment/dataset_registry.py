"""Dataset Excel paths and question loading for the experiment pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from experiment.config import ROOT

DATASET_EXCEL: Dict[str, Path] = {
    "bank": ROOT / "datasets" / "original" / "test_bank_q.xlsx",
    "movie": ROOT / "datasets" / "original" / "test_movie_q.xlsx",
    "cms": ROOT / "datasets" / "reproduce" / "test_cms_q_with_paths.xlsx",
    "emed": ROOT / "datasets" / "reproduce" / "test_emed_q_with_paths.xlsx",
    "synthea": ROOT / "datasets" / "reproduce" / "test_synthea_q_wth_paths.xlsx",
}

DEFAULT_QUESTION_COUNTS: Dict[str, int] = {
    "bank": 146,
    "movie": 355,
    "cms": 2563,
    "emed": 8121,
    "synthea": 2963,
}


def load_question_items(dataset: str) -> List[Tuple[str, str]]:
    """Return [(question_id, question_text), ...] for a dataset."""
    path = DATASET_EXCEL.get(dataset)
    if not path or not path.exists():
        raise FileNotFoundError(f"No Excel for dataset {dataset!r}: {path}")

    df = pd.read_excel(path)
    if "question" in df.columns:
        questions = df["question"]
    elif df.shape[1] > 9:
        questions = df.iloc[:, 9]
    else:
        raise ValueError(f"{path}: no 'question' column and no column 9 fallback")

    items: List[Tuple[str, str]] = []
    for idx, q in enumerate(questions):
        if pd.isna(q):
            continue
        text = str(q).strip()
        if text:
            items.append((f"question_{idx}", text))
    return items
