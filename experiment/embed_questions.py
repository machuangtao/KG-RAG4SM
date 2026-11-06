#!/usr/bin/env python3
"""Populate ChromaDB question embeddings from dataset Excel files."""
from __future__ import annotations

import argparse
import logging
import os
import sys

from experiment.config import ROOT
from experiment.dataset_registry import DATASET_EXCEL, load_question_items

logger = logging.getLogger(__name__)


def embed_dataset(dataset: str, use_gpu: bool = True) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from modules.question_embedding import EmbeddingsGenerator

    items = load_question_items(dataset)
    if not items:
        raise RuntimeError(f"No questions loaded for {dataset}")

    gen = EmbeddingsGenerator(dataset_name=dataset, use_gpu=use_gpu)
    gen.clear_and_store_question_embeddings(items)
    n = gen.collection_questions.count()
    logger.info("Stored %d question embeddings for %s", n, dataset)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Embed dataset questions into ChromaDB")
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=sorted(DATASET_EXCEL.keys()),
        required=True,
    )
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    for ds in args.dataset:
        embed_dataset(ds, use_gpu=not args.no_gpu)


if __name__ == "__main__":
    main()
