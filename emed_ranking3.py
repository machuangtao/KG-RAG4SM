import os
import json
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import requests
import torch
import time

#############################
# Helper Functions
#############################

def load_embeddings_from_parquet(parquet_path):
    """
    Load embeddings from a Parquet file.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if 'embedding' not in df.columns:
        raise ValueError(f"No 'embedding' column in {parquet_path}")
    embeddings = np.stack(df['embedding'].values, axis=0)
    return embeddings

def get_sorted_chunks(directory, embed_prefix="embeddings_chunk_", meta_prefix="metadata_chunk_"):
    """
    Retrieve and sort embedding and metadata files based on their chunk indices.
    """
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
    """
    Load triplet2 embeddings and metadata from the specified directory.
    """
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
    """
    Load triplet3 embeddings and metadata from the specified directory.
    """
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

        assert len(embeddings) == len(metadata_list), \
            f"Mismatch in embeddings vs metadata: {emb_file} & {meta_file}"

        all_embeddings.append(embeddings)
        all_metadata.extend(metadata_list)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_metadata

def normalize_embeddings(emb):
    """
    Normalize embeddings to unit vectors.
    """
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return emb / norms

def extract_ids_from_triplet2(meta):
    """
    Extract QIDs and PIDs from triplet2 metadata.
    """
    return [meta.get("head_id"), meta.get("relation_id"), meta.get("tail_id")]

def extract_ids_from_triplet3(meta):
    """
    Extract QIDs and PIDs from triplet3 metadata.
    """
    head_id = meta.get("head", {}).get("id")
    relation_id = meta.get("relation", {}).get("id")
    tail_id = meta.get("tail", {}).get("id")
    # Only return them if they look like Q/P IDs
    if head_id and relation_id and tail_id and head_id.startswith("Q") and relation_id.startswith("P") and tail_id.startswith("Q"):
        return [head_id, relation_id, tail_id]
    return []

# Cache for Wikidata labels
label_cache = {}

def save_label_cache(filepath="label_cache.json"):
    """
    Save the label cache to a JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(label_cache, f, indent=2)
    print(f"Label cache saved to {filepath}")

def load_label_cache(filepath="label_cache.json"):
    """
    Load the label cache from a JSON file.
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            label_cache.update(data)
        print(f"Label cache loaded from {filepath}")
    else:
        print("No existing label cache found. Starting fresh.")

def bulk_fetch_labels(entity_ids, max_retries=3, delay=5):
    """
    Fetch English labels for a set of entity IDs in bulk using the Wikidata API.
    Implements a retry mechanism for robustness.
    """
    ids_list = list(entity_ids)
    batch_size = 50  # Wikidata API limit
    for i in range(0, len(ids_list), batch_size):
        subset = ids_list[i:i+batch_size]
        query_ids = "|".join(subset)
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={query_ids}&languages=en&format=json"
        retries = 0
        while retries < max_retries:
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    entities = data.get("entities", {})
                    for eid, val in entities.items():
                        labels = val.get("labels", {})
                        if "en" in labels:
                            label_cache[eid] = labels["en"]["value"]
                        else:
                            label_cache[eid] = "N/A"
                    break  # Success, exit retry loop
                else:
                    print(f"Failed to fetch labels for IDs: {query_ids}. Status Code: {r.status_code}")
            except Exception as e:
                print(f"Exception during fetching labels: {e}")
            retries += 1
            print(f"Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(delay)
        else:
            print(f"Failed to fetch labels for IDs: {query_ids} after {max_retries} attempts.")
            # Assign 'N/A' for all IDs in this batch
            for eid in subset:
                if eid not in label_cache:
                    label_cache[eid] = "N/A"

def get_english_label(entity_id):
    """
    Retrieve the English label for a given entity ID from the cache.
    """
    return label_cache.get(entity_id, "N/A")

def get_english_triplet(doc_id, meta):
    """
    Convert a triplet's metadata to include both QIDs/PIDs and English labels.
    Formats the triplet as:
    major depressive disorder (Q42844) --> [instance of](P31) --> class of disease (Q112193867)
    """
    if doc_id.startswith("triplet2_"):
        head_id = meta.get("head_id")
        relation_id = meta.get("relation_id")
        tail_id = meta.get("tail_id")

        if not (head_id and relation_id and tail_id):
            return None

        head_eng = get_english_label(head_id)
        relation_eng = get_english_label(relation_id)
        tail_eng = get_english_label(tail_id)

        if head_eng and relation_eng and tail_eng:
            return {
                "head_id": head_id,
                "head_entity": head_eng,
                "relation_id": relation_id,
                "relation": relation_eng,
                "tail_id": tail_id,
                "tail_entity": tail_eng,
                "english_triplet": f"{head_eng} ({head_id}) --> [{relation_eng}]({relation_id}) --> {tail_eng} ({tail_id})"
            }
        else:
            return None

    elif doc_id.startswith("triplet3_"):
        head_id = meta.get("head", {}).get("id")
        relation_id = meta.get("relation", {}).get("id")
        tail_id = meta.get("tail", {}).get("id")

        if not (head_id and relation_id and tail_id):
            return None
        if not (head_id.startswith("Q") and relation_id.startswith("P") and tail_id.startswith("Q")):
            return None

        head_eng = get_english_label(head_id)
        relation_eng = get_english_label(relation_id)
        tail_eng = get_english_label(tail_id)

        if head_eng and relation_eng and tail_eng:
            return {
                "head_id": head_id,
                "head_entity": head_eng,
                "relation_id": relation_id,
                "relation": relation_eng,
                "tail_id": tail_id,
                "tail_entity": tail_eng,
                "english_triplet": f"{head_eng} ({head_id}) --> [{relation_eng}]({relation_id}) --> {tail_eng} ({tail_id})"
            }
        else:
            return None
    else:
        return None

#############################
# Main Execution
#############################

def main():
    base_dir = "/app/layers"
    collections_base = os.path.join(base_dir, "chroma_db", "collections")
    emed_dir = os.path.join(collections_base, "emed_ques_embedding_full")
    
    # Load label cache if exists
    load_label_cache("label_cache.json")
    
    # Load emed embeddings
    print("Loading emed embeddings...")
    emed_embeddings_path = os.path.join(emed_dir, "chroma-embeddings.parquet")
    emed_embeddings = load_embeddings_from_parquet(emed_embeddings_path)
    print(f"Loaded emed embeddings: {emed_embeddings.shape[0]} samples.")
    
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
    
    # Normalize triplet2_embeddings on CPU
    print("Normalizing triplet2 embeddings...")
    triplet2_embeddings = normalize_embeddings(triplet2_embeddings)
    
    # Normalize triplet3_embeddings on CPU
    print("Normalizing triplet3 embeddings...")
    triplet3_embeddings = normalize_embeddings(triplet3_embeddings)
    
    # Convert triplet3_embeddings to torch tensors and load on GPUs
    device_ids = [0, 1, 2, 3, 4]  # 5 GPUs
    devices = [f'cuda:{i}' for i in device_ids]
    num_devices = len(devices)
    
    print("Loading triplet3 embeddings onto GPUs...")
    triplet3_tensors = []
    for dev in devices:
        triplet3_tensor = torch.tensor(triplet3_embeddings, dtype=torch.float32).to(dev)
        triplet3_tensors.append(triplet3_tensor)
    
    # Assign questions to GPUs
    max_questions = emed_embeddings.shape[0]  # Process all questions
    print(f"Processing the first {max_questions} questions.")
    
    questions_per_gpu = (max_questions + num_devices - 1) // num_devices  # Ceiling division
    gpu_question_assignments = []
    for i in range(num_devices):
        start_q = i * questions_per_gpu
        end_q = min(start_q + questions_per_gpu, max_questions)
        if start_q < end_q:
            gpu_question_assignments.append((i, start_q, end_q))
    
    print(f"Distributing {max_questions} questions across {num_devices} GPUs.")
    
    # Initialize top7 triplet2 triplet lists per question
    top7_triplet2_indices = {q: [] for q in range(max_questions)}
    top7_triplet2_sims = {q: [] for q in range(max_questions)}
    
    # Process triplet2 in smaller chunks on CPU
    triplet2_chunk_size = 100000  # Adjust based on available CPU memory
    num_triplet2 = triplet2_embeddings.shape[0]
    
    print("Processing triplet2 embeddings in chunks...")
    for chunk_start in tqdm(range(0, num_triplet2, triplet2_chunk_size), desc="Processing triplet2 chunks", unit="chunk"):
        chunk_end = min(chunk_start + triplet2_chunk_size, num_triplet2)
        current_chunk_size = chunk_end - chunk_start
        # Extract the chunk
        triplet2_chunk = triplet2_embeddings[chunk_start:chunk_end]  # Shape: [chunk_size, embedding_dim]
        triplet2_chunk = triplet2_chunk.astype(np.float32)
        # Compute similarity with all questions
        questions = emed_embeddings[:max_questions]
        questions = normalize_embeddings(questions).astype(np.float32)  # Shape: [max_questions, embedding_dim]
        # Compute dot product
        sims = np.dot(questions, triplet2_chunk.T)  # Shape: [max_questions, chunk_size]
        # For each question, keep top7 triplet2
        for q in range(max_questions):
            sim_row = sims[q]  # Shape: [chunk_size]
            top7_current = top7_triplet2_sims[q]
            top7_indices_current = top7_triplet2_indices[q]
            # If we have fewer than 7, take all
            if len(top7_current) < 7:
                remaining = 7 - len(top7_current)
                if remaining > 0:
                    idx_part = np.argpartition(sim_row, -remaining)[-remaining:]
                    sorted_idx_part = idx_part[np.argsort(sim_row[idx_part])[::-1]]
                    top7_current.extend(sim_row[sorted_idx_part].tolist())
                    top7_indices_current.extend((chunk_start + sorted_idx_part).tolist())
            else:
                # Combine existing top7 with current chunk and keep top7
                combined_sims = np.array(top7_current + sim_row.tolist())
                combined_indices = np.array(top7_indices_current + (chunk_start + np.arange(current_chunk_size)).tolist())
                top7_idx = np.argpartition(combined_sims, -7)[-7:]
                sorted_top7_idx = top7_idx[np.argsort(combined_sims[top7_idx])[::-1]]
                top7_triplet2_sims[q] = combined_sims[sorted_top7_idx].tolist()
                top7_triplet2_indices[q] = combined_indices[sorted_top7_idx].tolist()
    
    print("Finished processing triplet2 embeddings.")
    
    # Compute top3 triplet3 similarities per question on each GPU
    print("Computing top3 triplet3 similarities per question on GPUs...")
    top3_triplet3_indices = {q: [] for q in range(max_questions)}
    top3_triplet3_sims = {q: [] for q in range(max_questions)}
    
    for gpu_id, (gpu_idx, start_q, end_q) in enumerate(gpu_question_assignments):
        device = devices[gpu_idx]
        triplet3_tensor = triplet3_tensors[gpu_id]  # Already on GPU
        for q in range(start_q, end_q):
            # Get question embedding
            q_embedding = emed_embeddings[q]
            q_embedding = normalize_embeddings(q_embedding.reshape(1, -1)).astype(np.float32)  # Shape: [1, embedding_dim]
            q_tensor = torch.tensor(q_embedding, dtype=torch.float32).to(device)  # Shape: [1, embedding_dim]
            # Compute similarity with triplet3
            sim = torch.matmul(q_tensor, triplet3_tensor.T)  # Shape: [1, triplet3_size]
            sim = sim.squeeze(0).cpu().numpy()  # Shape: [triplet3_size]
            # Get top3
            top3_idx = np.argpartition(sim, -3)[-3:]
            sorted_top3_idx = top3_idx[np.argsort(sim[top3_idx])[::-1]]
            top3_sim = sim[sorted_top3_idx]
            top3_triplet3_indices[q].extend(sorted_top3_idx.tolist())
            top3_triplet3_sims[q].extend(top3_sim.tolist())
    
    print("Finished computing triplet3 similarities.")
    
    # Combine triplet2 and triplet3 triplets per question and keep top10
    print("Combining triplet2 and triplet3 triplets per question...")
    all_results = []
    triplet_ids_to_fetch = set()
    
    for q in range(max_questions):
        # Get triplet2 top7
        triplet2_indices = top7_triplet2_indices[q]
        triplet2_sims = top7_triplet2_sims[q]
        triplet2_ids = [f"triplet2_{idx}" for idx in triplet2_indices]
        # Get triplet3 top3
        triplet3_indices = top3_triplet3_indices[q]
        triplet3_sims = top3_triplet3_sims[q]
        triplet3_ids = [f"triplet3_{idx}" for idx in triplet3_indices]
        # Combine
        combined_ids = triplet2_ids + triplet3_ids
        combined_sims = triplet2_sims + triplet3_sims
        # Sort by similarity
        sorted_indices = np.argsort(combined_sims)[::-1]
        sorted_ids = [combined_ids[i] for i in sorted_indices]
        sorted_sims = [combined_sims[i] for i in sorted_indices]
        # Keep top10
        top10_ids = sorted_ids[:10]
        top10_sims = sorted_sims[:10]
        # Collect triplet IDs for label fetching
        for doc_id in top10_ids:
            triplet_ids_to_fetch.add(doc_id)
        # Append to all_results
        all_results.append({
            "question_index": q,
            "results": [
                {
                    "rank": rank,
                    "id": doc_id,
                    "similarity": float(sim),
                    "metadata": {}  # Placeholder, will fill after label fetching
                }
                for rank, (doc_id, sim) in enumerate(zip(top10_ids, top10_sims), start=1)
            ]
        })
    
    print("Finished combining triplet results.")
    
    # Extract all unique entity IDs from triplet_ids_to_fetch
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
    
    # Save the label cache for future runs
    save_label_cache("label_cache.json")
    
    # Replace triplet IDs with English labels
    print("Replacing triplet IDs with English labels...")
    final_results = []
    skipped_triplets = 0
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
                eng_triplet = None
            
            if eng_triplet is not None:
                formatted_results.append({
                    "rank": r["rank"],
                    "id": doc_id,
                    "similarity": r["similarity"],
                    "metadata": eng_triplet  # Includes QIDs/PIDs and English labels
                })
            else:
                skipped_triplets += 1  # Increment skipped count
                print(f"Skipped triplet_id: {doc_id} for question_index: {q_idx} due to missing labels.")
        final_results.append({
            "question_index": q_idx,
            "results": formatted_results
        })
    
    print(f"Skipped {skipped_triplets} triplets due to missing English labels.")
    
    # Save final results to JSON
    output_json_path = os.path.join(base_dir, "emed_top10_similar2.json")
    with open(output_json_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Filtered English-only results saved to {output_json_path}")

if __name__ == "__main__":
    main()
