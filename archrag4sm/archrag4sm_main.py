"""ArchRAG4SM main entry point for schema matching.

Usage (after building the ArchRAG index on Wikidata5M):

    python -m archrag4sm.archrag4sm_main \\
        --dataset cms \\
        --backbone_llm_model gpt-4o-mini \\
        --index_dir archrag4sm/index

Embeddings use RoBERTa-base (768-dim, local) by default — no API key needed.

The script mirrors the structure of kgrag4sm_main.py but retrieves KG context
dynamically from the ArchRAG index instead of pre-computed paths.
"""

import sys
import json
import logging
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModel

from src.utils import get_devices, extract_label, calculate_metrics
from src.llm import initialize_llm_model
from archrag4sm.archrag4sm_sm import ArchRAG4SM, ArchRAGRetriever


# ---------------------------------------------------------------------------
# Dataset / data-file helpers (mirrors kgrag4sm_main.py)
# ---------------------------------------------------------------------------

DATASET_FILES = {
    "cms": "datasets/reproduce/test_cms_q_with_paths.xlsx",
    "mimic": "datasets/reproduce/test_mimic_q_with_paths.xlsx",
    "synthea": "datasets/reproduce/test_synthea_q_with_paths.xlsx",
    "emed": "datasets/reproduce/test_emed_q_with_paths.xlsx",
    "bank": "datasets/reproduce/test_bank_q_with_paths.xlsx",
    "movie": "datasets/reproduce/test_imsa_q_with_paths.xlsx",
}

# Column indices in the reproduce Excel files (0-based)
COL_QUESTION = 9   # schema matching question text
COL_LABEL = 4      # ground-truth label (0 / 1)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, dataset: str, model_name: str) -> str:
    os.makedirs(os.path.join(log_dir, dataset), exist_ok=True)
    log_filename = os.path.join(
        log_dir,
        dataset,
        f"ArchRAG4SM_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
    return log_filename


# ---------------------------------------------------------------------------
# Embedding — RoBERTa-base (768-dim, mean pooling, local inference)
# ---------------------------------------------------------------------------

_roberta_cache: dict = {}

def make_roberta_embedding_func(model_name: str = "roberta-base", device: str = "cpu"):
    """Return a callable that embeds a text string using RoBERTa-base (768-dim)."""
    cache_key = (model_name, device)
    if cache_key not in _roberta_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        _roberta_cache[cache_key] = (tokenizer, model)

    tokenizer, rob_model = _roberta_cache[cache_key]

    def embed(text: str) -> list:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = rob_model(**inputs)
        # Mean pooling over token embeddings (ignore padding)
        attention_mask = inputs["attention_mask"]
        token_embs = outputs.last_hidden_state  # (1, seq_len, 768)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (token_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32).tolist()

    return embed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ArchRAG-based schema matching on a dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cms",
        choices=list(DATASET_FILES.keys()),
        help="Schema matching benchmark dataset",
    )
    parser.add_argument(
        "--backbone_llm_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM for inference (gpt-4o-mini, jellyfish-8b, jellyfish-7b, mistral-7b)",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory containing the ArchRAG index built by archrag4sm/index.sh",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/archrag4sm",
        help="Directory for log files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for local LLM inference",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=15,
        help="Number of top entities/communities to retrieve per query",
    )
    parser.add_argument(
        "--topk_e",
        type=int,
        default=10,
        help="Number of top relationships to retrieve per query",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="roberta-base",
        help="Embedding model name (default: roberta-base for 768-dim embeddings)",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for embedding model inference (default: cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N rows (useful for quick testing)",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="kg",
        choices=["it", "kg"],
        help="Prompt style: 'it' for itkgrag4sm (with relevance scoring), 'kg' for kgrag4sm (without relevance scoring)",
    )
    parser.add_argument(
        "--cache_context",
        type=str,
        default=None,
        help="Path to a JSON file for caching retrieved contexts. "
             "If the file exists, contexts are loaded from it (skipping retrieval). "
             "After retrieval, contexts are saved to this file for future runs.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    start_time = datetime.now()

    log_filename = setup_logging(args.log_dir, args.dataset, args.backbone_llm_model)
    logging.info("ArchRAG4SM — dataset=%s  model=%s", args.dataset, args.backbone_llm_model)
    logging.info("Index dir : %s", args.index_dir)
    logging.info("Embedding : %s (device=%s)", args.embedding_model, args.embedding_device)
    logging.info("Prompt style : %s", args.prompt_style)

    # Initialise LLM
    try:
        devices = get_devices(args.device)
        model = initialize_llm_model(args.backbone_llm_model, devices)
    except Exception as exc:
        logging.error("Error initialising LLM: %s", exc)
        sys.exit(1)

    # Load dataset
    data_file = DATASET_FILES.get(args.dataset)
    if not data_file:
        logging.error("Unknown dataset: %s", args.dataset)
        sys.exit(1)
    try:
        reader = pd.read_excel(data_file)
    except Exception as exc:
        logging.error("Error reading %s: %s", data_file, exc)
        sys.exit(1)

    if args.limit:
        reader = reader.iloc[: args.limit]
        logging.info("Limiting to first %d rows", args.limit)

    # Load cached contexts if available (before loading retriever)
    context_cache = {}
    cache_path = args.cache_context
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                context_cache = json.load(f)
            logging.info("Loaded %d cached contexts from %s", len(context_cache), cache_path)
        except Exception as exc:
            logging.warning("Failed to load context cache: %s", exc)

    # Check if all questions have cached contexts
    all_cached = False
    if cache_path and context_cache:
        question_keys = {str(reader.iloc[i, COL_QUESTION]) for i in range(len(reader))}
        cached_keys = set(context_cache.keys())
        all_cached = question_keys.issubset(cached_keys)
        if all_cached:
            logging.info("All %d questions have cached contexts — skipping retriever", len(question_keys))

    # Initialise ArchRAG retriever + SM wrapper (skip if all contexts cached)
    if all_cached:
        archrag4sm = ArchRAG4SM(retriever=None, prompt_style=args.prompt_style)
    else:
        embedding_func = make_roberta_embedding_func(
            model_name=args.embedding_model,
            device=args.embedding_device,
        )
        retriever = ArchRAGRetriever(
            index_dir=args.index_dir,
            embedding_func=embedding_func,
            topk=args.topk,
            topk_e=args.topk_e,
        )
        archrag4sm = ArchRAG4SM(retriever, prompt_style=args.prompt_style)

    # Inference loop
    y_true, y_pred = [], []
    cache_dirty = False

    for i in tqdm(range(len(reader)), desc="Processing questions"):
        try:
            question = reader.iloc[i, COL_QUESTION]
            ground_truth = reader.iloc[i, COL_LABEL]

            # Use cached context if available, otherwise retrieve
            question_key = str(question)
            if question_key in context_cache:
                context = context_cache[question_key]
                logging.info("\n--- Row %d (cached context) ---", i + 1)
            else:
                logging.info("\n--- Row %d (retrieving context) ---", i + 1)
                context = archrag4sm.retrieve_context(question, model)
                if cache_path and context:
                    context_cache[question_key] = context
                    cache_dirty = True

            _, _, response, _ = archrag4sm.query_for_schema_matching(
                question, model, context=context
            )
            label = extract_label(response)

            if label != -1 and not pd.isna(ground_truth):
                y_true.append(int(ground_truth))
                y_pred.append(label)

            logging.info("Question: %s", question)
            logging.info("Retrieved context: %s", context)
            logging.info("Response: %s", response)
            logging.info("Label: %s  |  GT: %s", label, ground_truth)

        except Exception as exc:
            logging.error("Error at row %d: %s", i + 1, exc)
            continue

    # Save context cache if modified
    if cache_path and cache_dirty:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(context_cache, f, ensure_ascii=False, indent=2)
            logging.info("Saved %d contexts to %s", len(context_cache), cache_path)
        except Exception as exc:
            logging.warning("Failed to save context cache: %s", exc)

    # Evaluation
    if not y_true:
        logging.warning("No valid predictions collected — check data and model.")
        sys.exit(1)

    precision, recall, f1, accuracy = calculate_metrics(y_true, y_pred)

    logging.info("\n=== Final Metrics ===")
    logging.info("Precision : %.4f", precision)
    logging.info("Recall    : %.4f", recall)
    logging.info("F1 Score  : %.4f", f1)
    logging.info("Accuracy  : %.4f", accuracy)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    logging.info("\nConfusion Matrix:")
    logging.info("  TP=%d  TN=%d  FP=%d  FN=%d", tp, tn, fp, fn)

    # Token usage summary
    tracker = archrag4sm.token_tracker
    logging.info("\n=== Token Usage ===")
    logging.info("Total tokens: %d", tracker.grand_total)
    logging.info(
        "  Community summary tokens: %d (prompt=%d, completion=%d)",
        tracker.summary_total, tracker.summary_prompt, tracker.summary_completion,
    )
    logging.info(
        "  Inference tokens: %d (prompt=%d, completion=%d)",
        tracker.inference_total, tracker.inference_prompt, tracker.inference_completion,
    )

    end_time = datetime.now()
    logging.info("\nTotal duration: %s", end_time - start_time)
    print(f"Results saved to: {log_filename}")


if __name__ == "__main__":
    main()
