import os
import json
import argparse
import pandas as pd
from tqdm import tqdm


def load_entity_aliases(entity_file: str) -> dict:
    """Returns {QID: primary_name} using the first alias on each line."""
    qid_to_name = {}
    with open(entity_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading entity aliases"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                qid_to_name[parts[0]] = parts[1]
    return qid_to_name


def load_entity_descriptions(text_file: str, max_desc_len: int = 1000) -> dict:
    """Returns {QID: truncated_description_text}."""
    qid_to_desc = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading entity descriptions"):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                qid_to_desc[parts[0]] = parts[1][:max_desc_len]
    return qid_to_desc


def load_relation_labels(relation_file: str) -> dict:
    """Returns {PID: primary_label}."""
    pid_to_label = {}
    with open(relation_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading relation labels"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pid_to_label[parts[0]] = parts[1]
    return pid_to_label


def load_triplets(triplet_file: str, max_triplets: int | None = None) -> list:
    """Returns list of (head_qid, pid, tail_qid) tuples."""
    triplets = []
    with open(triplet_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading triplets")):
            if max_triplets is not None and i >= max_triplets:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                triplets.append((parts[0], parts[1], parts[2]))
    return triplets


def load_relevant_qids_from_datasets(datasets_dir: str) -> set:
    """
    Scan reproduce/*.xlsx files for entity QIDs (columns containing Q-identifiers)
    to build a seed set for subgraph extraction.
    """
    import re

    qids = set()
    q_pattern = re.compile(r"\bQ\d+\b")
    for fname in os.listdir(datasets_dir):
        if fname.endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(datasets_dir, fname))
                for col in df.columns:
                    for val in df[col].dropna().astype(str):
                        qids.update(q_pattern.findall(val))
            except Exception:
                pass
    return qids


def expand_relevant_qids(
    seed_qids: set,
    triplets: list,
    hops: int = 1,
) -> set:
    """
    Expand seed QIDs by including their k-hop neighbours in the triplet graph.
    Returns the full set of QIDs to include.
    """
    current = set(seed_qids)
    for _ in range(hops):
        neighbours = set()
        for head, _, tail in triplets:
            if head in current:
                neighbours.add(tail)
            if tail in current:
                neighbours.add(head)
        current |= neighbours
    return current


def convert_to_archrag_format(
    entity_file: str,
    text_file: str,
    relation_file: str,
    triplet_file: str,
    output_dir: str,
    max_entities: int | None = None,
    max_triplets: int | None = None,
    relevant_qids: set | None = None,
) -> None:
    """Convert Wikidata5M files to ArchRAG entity/relationship parquet format."""
    os.makedirs(output_dir, exist_ok=True)

    qid_to_name = load_entity_aliases(entity_file)
    qid_to_desc = load_entity_descriptions(text_file)
    pid_to_label = load_relation_labels(relation_file)
    triplets = load_triplets(triplet_file, max_triplets)

    # Determine entity set
    if relevant_qids is not None:
        entity_qids = relevant_qids & set(qid_to_name.keys())
        print(f"Using {len(entity_qids)} relevant entities (from seed set)")
    elif max_entities is not None:
        entity_qids = set(list(qid_to_name.keys())[:max_entities])
        print(f"Using first {len(entity_qids)} entities (max_entities limit)")
    else:
        entity_qids = set(qid_to_name.keys())
        print(f"Using all {len(entity_qids)} entities")

    # Build entity DataFrame
    print("Building entity DataFrame...")
    entity_records = []
    qid_to_int_id: dict[str, int] = {}
    for i, qid in enumerate(tqdm(sorted(entity_qids), desc="Entities")):
        qid_to_int_id[qid] = i
        entity_records.append(
            {
                "id": qid,
                "human_readable_id": i,
                "name": qid_to_name.get(qid, qid),
                "description": qid_to_desc.get(qid, ""),
            }
        )

    entity_df = pd.DataFrame(entity_records)
    entity_path = os.path.join(output_dir, "create_final_entities.parquet")
    entity_df.to_parquet(entity_path, index=False)
    print(f"Saved {len(entity_df):,} entities → {entity_path}")

    # Build relationship DataFrame (only within the entity set)
    print("Building relationship DataFrame...")
    rel_records = []
    for i, (head_qid, pid, tail_qid) in enumerate(
        tqdm(triplets, desc="Relationships")
    ):
        if head_qid in entity_qids and tail_qid in entity_qids:
            rel_records.append(
                {
                    "human_readable_id": len(rel_records),
                    "source": qid_to_name.get(head_qid, head_qid),
                    "target": qid_to_name.get(tail_qid, tail_qid),
                    "description": pid_to_label.get(pid, pid),
                    "head_id": qid_to_int_id[head_qid],
                    "tail_id": qid_to_int_id[tail_qid],
                }
            )

    rel_df = pd.DataFrame(rel_records)
    rel_path = os.path.join(output_dir, "create_final_relationships.parquet")
    rel_df.to_parquet(rel_path, index=False)
    print(f"Saved {len(rel_df):,} relationships → {rel_path}")

    # Save QID → int mapping for downstream lookup
    mapping_path = os.path.join(output_dir, "qid_to_int_id.json")
    with open(mapping_path, "w") as f:
        json.dump(qid_to_int_id, f)
    print(f"Saved QID mapping → {mapping_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Wikidata5M files to ArchRAG entity/relationship format"
    )
    parser.add_argument(
        "--wikidata_dir",
        type=str,
        default="kbs/wikidata5m",
        help="Directory containing Wikidata5M raw files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="archrag4sm/data",
        help="Output directory for ArchRAG-format parquet files",
    )
    parser.add_argument(
        "--max_entities",
        type=int,
        default=None,
        help="Limit number of entities (None = all ~4.8M). Useful for testing.",
    )
    parser.add_argument(
        "--max_triplets",
        type=int,
        default=None,
        help="Limit number of triplets read from file (None = all).",
    )
    parser.add_argument(
        "--relevant_qids_file",
        type=str,
        default=None,
        help="JSON file with list of QIDs to include (overrides max_entities).",
    )
    parser.add_argument(
        "--auto_seed_from_datasets",
        action="store_true",
        help="Auto-extract seed QIDs from datasets/reproduce/ Excel files.",
    )
    parser.add_argument(
        "--seed_expansion_hops",
        type=int,
        default=2,
        help="Number of hops to expand seed QIDs in the KG (default: 2).",
    )
    args = parser.parse_args()

    relevant_qids = None

    if args.relevant_qids_file and os.path.exists(args.relevant_qids_file):
        with open(args.relevant_qids_file) as f:
            relevant_qids = set(json.load(f))
        print(f"Loaded {len(relevant_qids):,} QIDs from {args.relevant_qids_file}")

    elif args.auto_seed_from_datasets:
        print("Auto-extracting seed QIDs from datasets/reproduce/ ...")
        seed_qids = load_relevant_qids_from_datasets("datasets/reproduce")
        print(f"Found {len(seed_qids):,} seed QIDs from datasets")
        if seed_qids and args.seed_expansion_hops > 0:
            print(
                f"Loading triplets for {args.seed_expansion_hops}-hop expansion..."
            )
            triplets = load_triplets(
                os.path.join(args.wikidata_dir, "wikidata5m_all_triplet.txt"),
                max_triplets=args.max_triplets,
            )
            relevant_qids = expand_relevant_qids(
                seed_qids, triplets, hops=args.seed_expansion_hops
            )
            print(
                f"Expanded to {len(relevant_qids):,} QIDs "
                f"({args.seed_expansion_hops}-hop neighbourhood)"
            )
        else:
            relevant_qids = seed_qids

    convert_to_archrag_format(
        entity_file=os.path.join(args.wikidata_dir, "wikidata5m_entity.txt"),
        text_file=os.path.join(args.wikidata_dir, "wikidata5m_text.txt"),
        relation_file=os.path.join(args.wikidata_dir, "wikidata5m_relation.txt"),
        triplet_file=os.path.join(args.wikidata_dir, "wikidata5m_all_triplet.txt"),
        output_dir=args.output_dir,
        max_entities=args.max_entities,
        max_triplets=args.max_triplets,
        relevant_qids=relevant_qids,
    )


if __name__ == "__main__":
    main()
