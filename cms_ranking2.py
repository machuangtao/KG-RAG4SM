import os
import json
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import requests
import torch

#############################
# Helper Functions
#############################

def load_embeddings_from_parquet(parquet_path):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if 'embedding' not in df.columns:
        raise ValueError(f"No 'embedding' column in {parquet_path}")
    embeddings = np.stack(df['embedding'].values, axis=0)
    return embeddings

def get_sorted_chunks(directory, embed_prefix="embeddings_chunk_", meta_prefix="metadata_chunk_"):
    embed_files = [f for f in os.listdir(directory) if f.startswith(embed_prefix) and f.endswith(".npy")]
    meta_files = [f for f in os.listdir(directory) if f.startswith(meta_prefix) and f.endswith(".json")]

    def extract_index(fname, prefix):
        base = fname.replace(prefix, "")
        base = os.path.splitext(base)[0]
        return int(base)

    embed_files = sorted(embed_files, key=lambda x: extract_index(x, embed_prefix))
    meta_files = sorted(meta_files, key=lambda x: extract_index(x, meta_prefix))

    assert len(embed_files) == len(meta_files), f"Mismatch in embeddings vs metadata count in {directory}"
    return embed_files, meta_files

def load_wikidata_triplet2(directory):
    embed_files, meta_files = get_sorted_chunks(directory)
    all_embeddings = []
    all_metadata = []
    for ef, mf in tqdm(zip(embed_files, meta_files), total=len(embed_files), desc="Loading wikidata_triplet2", unit="chunk"):
        emb_path = os.path.join(directory, ef)
        meta_path = os.path.join(directory, mf)

        embeddings = np.load(emb_path)
        with open(meta_path, 'r') as f:
            metadata_list = json.load(f)

        assert len(embeddings) == len(metadata_list), f"Size mismatch in {ef} & {mf}"
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata_list)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_metadata

def load_wikidata_triplet3(directory):
    files = [
        ("snomed_parent_child_triples_embeddings.npy", "snomed_parent_child_triples_metadata.json"),
        ("umls_type_groups_triples_embeddings.npy", "umls_type_groups_triples_metadata.json")
    ]

    all_embeddings = []
    all_metadata = []
    for emb_file, meta_file in files:
        emb_path = os.path.join(directory, emb_file)
        meta_path = os.path.join(directory, meta_file)

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        embeddings = np.load(emb_path)
        with open(meta_path, "r") as f:
            metadata_list = json.load(f)

        assert len(embeddings) == len(metadata_list), f"Mismatch in embeddings vs metadata: {emb_file} & {meta_file}"

        all_embeddings.append(embeddings)
        all_metadata.extend(metadata_list)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_metadata

def normalize_embeddings(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return emb / norms

def extract_ids_from_triplet2(meta):
    return [meta.get("head_id"), meta.get("relation_id"), meta.get("tail_id")]

def extract_ids_from_triplet3(meta):
    head_id = meta.get("head", {}).get("id")
    relation_id = meta.get("relation", {}).get("id")
    tail_id = meta.get("tail", {}).get("id")
    if head_id and relation_id and tail_id and head_id.startswith("Q") and relation_id.startswith("P") and tail_id.startswith("Q"):
        return [head_id, relation_id, tail_id]
    return []

# Cache for Wikidata labels
label_cache = {}

def bulk_fetch_labels(entity_ids):
    ids_list = list(entity_ids)
    batch_size = 50  # Wikidata API limit
    for i in range(0, len(ids_list), batch_size):
        subset = ids_list[i:i+batch_size]
        query_ids = "|".join(subset)
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={query_ids}&languages=en&format=json"
        try:
            r = requests.get(url)
            if r.status_code != 200:
                print(f"Failed to fetch labels for IDs: {query_ids}")
                continue
            data = r.json()
            entities = data.get("entities", {})
            for eid, val in entities.items():
                labels = val.get("labels", {})
                if "en" in labels:
                    label_cache[eid] = labels["en"]["value"]
                else:
                    label_cache[eid] = None
        except Exception as e:
            print(f"Exception during fetching labels: {e}")
            continue

def get_english_label(entity_id):
    return label_cache.get(entity_id)

def get_english_triplet(doc_id, meta):
    """
    Returns a dictionary with either English labels or fallback to IDs if English labels are missing.
    We do NOT skip triplets if labels are missing. We'll return the best info available.
    """
    if doc_id.startswith("triplet2_"):
        head_id = meta.get("head_id")
        relation_id = meta.get("relation_id")
        tail_id = meta.get("tail_id")

        head_eng = get_english_label(head_id)
        relation_eng = get_english_label(relation_id)
        tail_eng = get_english_label(tail_id)

        # If some labels are missing, fallback to IDs
        if head_eng is None:
            head_eng = head_id
        if relation_eng is None:
            relation_eng = relation_id
        if tail_eng is None:
            tail_eng = tail_id

        return {
            "head_id": head_id,
            "head_entity": head_eng,
            "relation_id": relation_id,
            "relation": relation_eng,
            "tail_id": tail_id,
            "tail_entity": tail_eng,
            "english_triplet": f"<{head_eng}, {relation_eng}, {tail_eng}>"
        }

    elif doc_id.startswith("triplet3_"):
        head_id = meta.get("head", {}).get("id")
        relation_id = meta.get("relation", {}).get("id")
        tail_id = meta.get("tail", {}).get("id")

        if head_id and relation_id and tail_id:
            head_eng = get_english_label(head_id)
            relation_eng = get_english_label(relation_id)
            tail_eng = get_english_label(tail_id)

            # Fallback to IDs if labels missing
            if head_eng is None:
                head_eng = head_id
            if relation_eng is None:
                relation_eng = relation_id
            if tail_eng is None:
                tail_eng = tail_id

            return {
                "head_id": head_id,
                "head_entity": head_eng,
                "relation_id": relation_id,
                "relation": relation_eng,
                "tail_id": tail_id,
                "tail_entity": tail_eng,
                "english_triplet": f"<{head_eng}, {relation_eng}, {tail_eng}>"
            }
        else:
            # If IDs are not proper, just return metadata as is
            return {
                "english_triplet": "<missing IDs>"
            }

    else:
        return {
            "english_triplet": "<unrecognized doc_id>"
        }

#############################
# Main Execution
#############################

def main():
    base_dir = "/app/layers"
    cms_dir = os.path.join(base_dir, "test_embedding_full")

    # Load CMS embeddings (questions)
    print("Loading CMS embeddings...")
    cms_embeddings_path = os.path.join(cms_dir, "chroma-embeddings.parquet")
    cms_embeddings = load_embeddings_from_parquet(cms_embeddings_path)
    print(f"Loaded CMS embeddings: {cms_embeddings.shape[0]} samples.")

    # Load wikidata_triplet2 embeddings
    print("Loading wikidata_triplet2 embeddings...")
    wikidata_triplet2_dir = os.path.join(base_dir, "wikidata_embedding_triplet2")
    triplet2_embeddings, triplet2_metadata = load_wikidata_triplet2(wikidata_triplet2_dir)
    print(f"Loaded wikidata_triplet2 embeddings: {triplet2_embeddings.shape[0]} samples.")

    # Load wikidata_triplet3 embeddings
    print("Loading wikidata_triplet3 embeddings...")
    wikidata_triplet3_dir = os.path.join(base_dir, "wikidata_embedding_triplet3")
    triplet3_embeddings, triplet3_metadata = load_wikidata_triplet3(wikidata_triplet3_dir)
    print(f"Loaded wikidata_triplet3 embeddings: {triplet3_embeddings.shape[0]} samples.")

    # Normalize triplet2 and triplet3 embeddings
    print("Normalizing triplet2 embeddings...")
    triplet2_embeddings = normalize_embeddings(triplet2_embeddings)

    print("Normalizing triplet3 embeddings...")
    triplet3_embeddings = normalize_embeddings(triplet3_embeddings)

    device_ids = [0, 1, 2, 3, 4]  # Adjust GPUs as needed
    devices = [f'cuda:{i}' for i in device_ids]
    num_devices = len(devices)

    # Move triplet3 embeddings to GPUs for faster similarity search
    print("Loading triplet3 embeddings onto GPUs...")
    triplet3_tensors = []
    for dev in devices:
        triplet3_tensor = torch.tensor(triplet3_embeddings, dtype=torch.float32).to(dev)
        triplet3_tensors.append(triplet3_tensor)

    # Determine how many questions to process
    max_questions = cms_embeddings.shape[0]  # Process all questions
    print(f"Processing the first {max_questions} questions.")

    # Distribute questions across GPUs
    questions_per_gpu = (max_questions + num_devices - 1) // num_devices
    gpu_question_assignments = []
    for i in range(num_devices):
        start_q = i * questions_per_gpu
        end_q = min(start_q + questions_per_gpu, max_questions)
        if start_q < end_q:
            gpu_question_assignments.append((i, start_q, end_q))

    print(f"Distributing {max_questions} questions across {num_devices} GPUs.")

    # Initialize top-7 tracking for triplet2
    top7_triplet2_indices = {q: [] for q in range(max_questions)}
    top7_triplet2_sims = {q: [] for q in range(max_questions)}

    # Normalize CMS questions
    questions = cms_embeddings[:max_questions]
    questions = normalize_embeddings(questions).astype(np.float32)

    # Handle triplet2 on CPU in chunks
    triplet2_chunk_size = 100000
    num_triplet2 = triplet2_embeddings.shape[0]

    print("Processing triplet2 embeddings in chunks...")
    for chunk_start in tqdm(range(0, num_triplet2, triplet2_chunk_size), desc="Processing triplet2 chunks", unit="chunk"):
        chunk_end = min(chunk_start + triplet2_chunk_size, num_triplet2)
        triplet2_chunk = triplet2_embeddings[chunk_start:chunk_end].astype(np.float32)

        # Compute similarities [max_questions, chunk_size]
        sims = np.dot(questions, triplet2_chunk.T)

        for q in range(max_questions):
            sim_row = sims[q]
            if len(top7_triplet2_sims[q]) < 7:
                # Take the best needed to reach 7
                remaining = 7 - len(top7_triplet2_sims[q])
                idx_part = np.argpartition(sim_row, -remaining)[-remaining:]
                sorted_idx_part = idx_part[np.argsort(sim_row[idx_part])[::-1]]
                top7_triplet2_sims[q].extend(sim_row[sorted_idx_part].tolist())
                top7_triplet2_indices[q].extend((chunk_start + sorted_idx_part).tolist())
            else:
                # Combine and keep top7 overall
                combined_sims = np.array(top7_triplet2_sims[q] + sim_row.tolist())
                combined_indices = np.array(top7_triplet2_indices[q] + (chunk_start + np.arange(chunk_end - chunk_start)).tolist())
                top7_idx = np.argpartition(combined_sims, -7)[-7:]
                sorted_top7_idx = top7_idx[np.argsort(combined_sims[top7_idx])[::-1]]
                top7_triplet2_sims[q] = combined_sims[sorted_top7_idx].tolist()
                top7_triplet2_indices[q] = combined_indices[sorted_top7_idx].tolist()

    print("Finished processing triplet2 embeddings.")

    # Compute top3 for triplet3 embeddings per question on GPUs
    print("Computing top3 triplet3 similarities per question on GPUs...")
    top3_triplet3_indices = {q: [] for q in range(max_questions)}
    top3_triplet3_sims = {q: [] for q in range(max_questions)}

    for gpu_id, (gpu_idx, start_q, end_q) in enumerate(gpu_question_assignments):
        device = devices[gpu_idx]
        triplet3_tensor = triplet3_tensors[gpu_id]
        for q in range(start_q, end_q):
            q_embedding = questions[q].reshape(1, -1)
            q_tensor = torch.tensor(q_embedding, dtype=torch.float32).to(device)
            sim = torch.matmul(q_tensor, triplet3_tensor.T).squeeze(0).cpu().numpy()
            # top 3
            top3_idx = np.argpartition(sim, -3)[-3:]
            top3_sorted_idx = top3_idx[np.argsort(sim[top3_idx])[::-1]]
            top3_sim = sim[top3_sorted_idx]
            top3_triplet3_indices[q].extend(top3_sorted_idx.tolist())
            top3_triplet3_sims[q].extend(top3_sim.tolist())

    print("Finished computing triplet3 similarities.")

    # Combine top7 from triplet2 and top3 from triplet3
    print("Combining triplet2 and triplet3 triplets per question...")
    all_results = []
    triplet_ids_to_fetch = set()

    for q in range(max_questions):
        # Triplet2 top7
        triplet2_indices = top7_triplet2_indices[q]
        triplet2_sims_final = top7_triplet2_sims[q]
        triplet2_ids = [f"triplet2_{idx}" for idx in triplet2_indices]

        # Triplet3 top3
        triplet3_indices = top3_triplet3_indices[q]
        triplet3_sims_final = top3_triplet3_sims[q]
        triplet3_ids = [f"triplet3_{idx}" for idx in triplet3_indices]

        # Combine (total 10)
        combined_ids = triplet2_ids + triplet3_ids
        combined_sims = triplet2_sims_final + triplet3_sims_final

        # Sort by similarity descending
        sorted_indices = np.argsort(combined_sims)[::-1]
        sorted_ids = [combined_ids[i] for i in sorted_indices]
        sorted_sims = [combined_sims[i] for i in sorted_indices]

        # IMPORTANT: Do NOT trim to top10 again because we already have exactly 10 (7+3)
        # We keep all 10 results here.
        top10_ids = sorted_ids
        top10_sims = sorted_sims

        for doc_id in top10_ids:
            triplet_ids_to_fetch.add(doc_id)

        all_results.append({
            "question_index": q,
            "results": [
                {
                    "rank": rank,
                    "id": doc_id,
                    "similarity": float(sim),
                    "metadata": {}  # Will fill after label fetching
                }
                for rank, (doc_id, sim) in enumerate(zip(top10_ids, top10_sims), start=1)
            ]
        })

    print("Finished combining triplet results.")

    # Extract Q/P IDs for label fetching
    print("Extracting unique entity IDs for label fetching...")
    unique_entity_ids = set()
    for qres in all_results:
        for r in qres["results"]:
            doc_id = r["id"]
            if doc_id.startswith("triplet2_"):
                idx = int(doc_id.split("_")[1])
                meta = triplet2_metadata[idx]
                ids = extract_ids_from_triplet2(meta)
            elif doc_id.startswith("triplet3_"):
                idx = int(doc_id.split("_")[1])
                meta = triplet3_metadata[idx]
                ids = extract_ids_from_triplet3(meta)
            else:
                ids = []
            for eid in ids:
                if eid and (eid.startswith("Q") or eid.startswith("P")):
                    unique_entity_ids.add(eid)

    print(f"Total unique Q/P IDs to fetch: {len(unique_entity_ids)}")

    # Bulk fetch English labels
    print("Fetching English labels from Wikidata...")
    bulk_fetch_labels(unique_entity_ids)
    print("Finished fetching English labels.")

    # Replace IDs with English labels (no skipping if missing)
    print("Replacing triplet IDs with English labels...")
    final_results = []
    for qres in tqdm(all_results, desc="Formatting results", unit="question"):
        q_idx = qres["question_index"]
        formatted_results = []
        for r in qres["results"]:
            doc_id = r["id"]
            if doc_id.startswith("triplet2_"):
                idx = int(doc_id.split("_")[1])
                meta = triplet2_metadata[idx]
                eng_triplet = get_english_triplet(doc_id, meta)
            elif doc_id.startswith("triplet3_"):
                idx = int(doc_id.split("_")[1])
                meta = triplet3_metadata[idx]
                eng_triplet = get_english_triplet(doc_id, meta)
            else:
                eng_triplet = {"english_triplet": "<unrecognized>"}

            # Add eng_triplet info directly
            new_r = {
                "rank": r["rank"],
                "id": doc_id,
                "similarity": r["similarity"],
                "metadata": eng_triplet
            }
            formatted_results.append(new_r)

        final_results.append({
            "question_index": q_idx,
            "results": formatted_results
        })

    # Save final results to JSON
    output_json_path = os.path.join(base_dir, "test_top10_similar.json")
    with open(output_json_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"All results saved to {output_json_path}")

if __name__ == "__main__":
    main()
