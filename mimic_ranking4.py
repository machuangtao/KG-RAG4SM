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

#############################
# Main Execution
#############################

def main():
    base_dir = "/app/layers"
    synthea_dir = os.path.join(base_dir, "synthea_ques_embedding_full2")
    
    # Load synthea embeddings
    print("Loading synthea embeddings...")
    synthea_embeddings_path = os.path.join(synthea_dir, "chroma-embeddings.parquet")
    synthea_embeddings = load_embeddings_from_parquet(synthea_embeddings_path)
    total_questions = synthea_embeddings.shape[0]
    print(f"Loaded synthea embeddings: {total_questions} samples.")

    # We'll use only the first 10 questions to measure similarity time
    max_questions = 10
    max_questions = min(max_questions, total_questions)

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

    print(f"Processing the first {max_questions} question(s) for timing.")

    # Distribute the first 10 questions across 5 GPUs for triplet3 computations
    questions_per_gpu = (max_questions + num_devices - 1) // num_devices
    gpu_question_assignments = []
    for i in range(num_devices):
        start_q = i * questions_per_gpu
        end_q = min(start_q + questions_per_gpu, max_questions)
        if start_q < end_q:
            gpu_question_assignments.append((i, start_q, end_q))
    
    # Initialize top7 triplet2 triplet lists per question
    top7_triplet2_indices = {q: [] for q in range(max_questions)}
    top7_triplet2_sims = {q: [] for q in range(max_questions)}
    
    # Process triplet2 in smaller chunks on CPU
    triplet2_chunk_size = 100000
    num_triplet2 = triplet2_embeddings.shape[0]

    # Normalize the first 10 questions
    questions = synthea_embeddings[:max_questions]
    questions = normalize_embeddings(questions).astype(np.float32)

    # Start timing the similarity computations now
    similarity_start_time = time.time()

    print("Computing similarity for triplet2 (first 10 questions)...")
    for chunk_start in tqdm(range(0, num_triplet2, triplet2_chunk_size), desc="Processing triplet2 chunks", unit="chunk"):
        chunk_end = min(chunk_start + triplet2_chunk_size, num_triplet2)
        current_chunk_size = chunk_end - chunk_start
        triplet2_chunk = triplet2_embeddings[chunk_start:chunk_end].astype(np.float32)

        # Compute similarity for all first 10 questions
        sims_2d = np.dot(questions, triplet2_chunk.T)  # Shape: [10, chunk_size]

        # For each question, maintain top7
        for q in range(max_questions):
            sim_row = sims_2d[q]
            top7_current = top7_triplet2_sims[q]
            top7_indices_current = top7_triplet2_indices[q]

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

    print("Finished triplet2 similarity computation for the first 10 questions.")

    # Compute top3 triplet3 similarities per question on each GPU
    print("Computing similarity for triplet3 (first 10 questions)...")
    top3_triplet3_indices = {q: [] for q in range(max_questions)}
    top3_triplet3_sims = {q: [] for q in range(max_questions)}

    for gpu_id, (gpu_idx, start_q, end_q) in enumerate(gpu_question_assignments):
        device = devices[gpu_idx]
        triplet3_tensor = triplet3_tensors[gpu_id]  # Already on GPU
        # Extract the relevant questions for this GPU
        q_embeddings = synthea_embeddings[start_q:end_q]
        q_embeddings = normalize_embeddings(q_embeddings).astype(np.float32)
        q_tensor = torch.tensor(q_embeddings, dtype=torch.float32).to(device)  # Shape: [num_questions_on_gpu, dim]

        sim = torch.matmul(q_tensor, triplet3_tensor.T)  # Shape: [num_questions_on_gpu, triplet3_size]
        sim = sim.cpu().numpy()

        for i, q_idx in enumerate(range(start_q, end_q)):
            sim_row = sim[i]
            top3_idx = np.argpartition(sim_row, -3)[-3:]
            top3_sorted_idx = top3_idx[np.argsort(sim_row[top3_idx])[::-1]]
            top3_sim = sim_row[top3_sorted_idx]
            top3_triplet3_indices[q_idx].extend(top3_sorted_idx.tolist())
            top3_triplet3_sims[q_idx].extend(top3_sim.tolist())

    print("Finished triplet3 similarity computation for the first 10 questions.")

    # Stop timing similarity computations
    similarity_end_time = time.time()
    similarity_elapsed = similarity_end_time - similarity_start_time

    # Compute average time per question
    avg_time_per_question = similarity_elapsed / max_questions

    print(f"Time taken for similarity computations for the first {max_questions} questions: {similarity_elapsed:.2f} seconds")
    print(f"Average time per question: {avg_time_per_question:.2f} seconds")

    # Estimate total time for all questions (assuming linear scaling)
    estimated_total_time = avg_time_per_question * total_questions
    print(f"Estimated total time for all {total_questions} questions: {estimated_total_time:.2f} seconds")

if __name__ == "__main__":
    main()
