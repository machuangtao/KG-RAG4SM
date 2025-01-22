import os
import json
import logging
import traceback
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import glob
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = '/app/layers'  # Adjust if needed

# Directories
CMS_DIR = os.path.join(BASE_DIR, 'cms_ques_embedding_full')
TRIPLET2_DIR = os.path.join(BASE_DIR, 'wikidata_embedding_triplet2')
TRIPLET3_DIR = os.path.join(BASE_DIR, 'wikidata_embedding_triplet3')

# Files
CHROMA_EMBEDDINGS_PARQUET = os.path.join(CMS_DIR, 'chroma-embeddings.parquet')

# Triplet3 files
TRIPLET3_FILES = [
    ('snomed_parent_child_triples_embeddings.npy', 'snomed_parent_child_triples_metadata.json'),
    ('umls_type_groups_triples_embeddings.npy', 'umls_type_groups_triples_metadata.json')
]

JSON_OUTPUT = "cms_wikidata_similar_2.json"
TXT_OUTPUT = "cms_wikidata_similar_2.txt"

TOP_K_2 = 7
TOP_K_3 = 3
QUESTION_BATCH_SIZE = 5000  # Number of questions to process per batch

def load_triplet2_data(dir_path):
    emb_files = glob.glob(os.path.join(dir_path, 'embeddings_chunk_*.npy'))
    def extract_number(filename):
        import re
        match = re.search(r'chunk_(\d+)', filename)
        return int(match.group(1)) if match else -1
    emb_files.sort(key=extract_number)

    all_embeddings = []
    all_metadata = []

    logger.info("Loading wikidata_triplet2 embeddings and metadata...")
    for emb_file in tqdm(emb_files, desc="Loading triplet2"):
        meta_file = emb_file.replace('embeddings_', 'metadata_').replace('.npy', '.json')
        if not os.path.exists(meta_file):
            logger.error(f"Metadata file not found for {emb_file}")
            continue
        embeddings = np.load(emb_file, allow_pickle=True, mmap_mode='r')
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if len(metadata) != len(embeddings):
            logger.error(f"Mismatch in {emb_file}: embeddings({len(embeddings)}) != metadata({len(metadata)})")
            continue
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata)

    if len(all_embeddings) == 0:
        raise ValueError("No triplet2 embeddings loaded.")
    all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / norms

    return all_embeddings, all_metadata

def load_triplet3_data(dir_path):
    all_embeddings = []
    all_metadata = []
    logger.info("Loading wikidata_triplet3 embeddings and metadata...")
    for emb_name, meta_name in TRIPLET3_FILES:
        emb_file = os.path.join(dir_path, emb_name)
        meta_file = os.path.join(dir_path, meta_name)
        if not os.path.exists(emb_file) or not os.path.exists(meta_file):
            logger.error(f"Files not found for {emb_name} / {meta_name}")
            continue
        embeddings = np.load(emb_file, allow_pickle=True, mmap_mode='r')
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if len(metadata) != len(embeddings):
            logger.error(f"Mismatch in {emb_file}: embeddings({len(embeddings)}) != metadata({len(metadata)})")
            continue
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata)

    if len(all_embeddings) == 0:
        raise ValueError("No triplet3 embeddings loaded.")
    all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / norms

    return all_embeddings, all_metadata

def cosine_similarity(query, matrix):
    return np.dot(matrix, query.squeeze(0))

def find_top_k(similarities, k):
    if k >= len(similarities):
        indices = np.argsort(-similarities)
    else:
        indices = np.argpartition(-similarities, k)[:k]
        indices = indices[np.argsort(-similarities[indices])]
    return indices

def main():
    # Load triplet2 and triplet3 embeddings once
    t2_embs, t2_meta = load_triplet2_data(TRIPLET2_DIR)
    logger.info(f"Loaded {len(t2_embs)} triplet2 embeddings")

    t3_embs, t3_meta = load_triplet3_data(TRIPLET3_DIR)
    logger.info(f"Loaded {len(t3_embs)} triplet3 embeddings")

    # Open the parquet file for cms questions
    if not os.path.exists(CHROMA_EMBEDDINGS_PARQUET):
        logger.error(f"{CHROMA_EMBEDDINGS_PARQUET} not found.")
        return

    pq_file = pq.ParquetFile(CHROMA_EMBEDDINGS_PARQUET)
    # We expect columns: 'id', 'document', 'embedding'
    # 'embedding' should be a list or array of floats.

    # Prepare output files
    # For JSON, we will write a JSON array in streaming mode:
    json_file = open(JSON_OUTPUT, 'w', encoding='utf-8')
    json_file.write("[\n")

    txt_file = open(TXT_OUTPUT, 'w', encoding='utf-8')

    first_result = True
    question_count = 0

    # Read parquet in batches
    # Use iter_batches to get chunks of data
    for batch in pq_file.iter_batches(batch_size=QUESTION_BATCH_SIZE, columns=['id', 'document', 'embedding']):
        # Convert batch to pandas DataFrame for convenience
        df = batch.to_pandas()
        if df.empty:
            break

        # Extract columns
        q_ids = df['id'].tolist()
        q_docs = df['document'].tolist()
        q_embs = np.array(df['embedding'].tolist(), dtype=np.float32)

        # Normalize question embeddings
        q_norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
        q_embs = q_embs / q_norms

        # Process each question in this batch
        for i in range(len(q_ids)):
            q_id = q_ids[i]
            q_doc = q_docs[i]
            q_emb = q_embs[i:i+1, :]  # (1, D)

            # triplet2 similarity
            t2_sim = cosine_similarity(q_emb, t2_embs)
            t2_indices = find_top_k(t2_sim, TOP_K_2)
            t2_results = []
            for idx in t2_indices:
                t2_results.append({
                    "id": f"triplet2_{idx}",
                    "text": str(t2_meta[idx].get('text', '')),
                    "metadata": t2_meta[idx],
                    "distance": float(1 - t2_sim[idx])
                })

            # triplet3 similarity
            t3_sim = cosine_similarity(q_emb, t3_embs)
            t3_indices = find_top_k(t3_sim, TOP_K_3)
            t3_results = []
            for idx in t3_indices:
                t3_results.append({
                    "id": f"triplet3_{idx}",
                    "text": str(t3_meta[idx].get('text', '')),
                    "metadata": t3_meta[idx],
                    "distance": float(1 - t3_sim[idx])
                })

            combined = t2_results + t3_results
            combined.sort(key=lambda x: x['distance'])

            result_obj = {
                "question_id": q_id,
                "question": q_doc,
                "similar_items": combined
            }

            # Write to JSON
            if not first_result:
                json_file.write(",\n")
            else:
                first_result = False
            json.dump(result_obj, json_file, ensure_ascii=False, indent=2)

            # Write to TXT
            txt_file.write(f"Question ID: {q_id}\n")
            txt_file.write(f"Question: {q_doc}\n\n")
            txt_file.write("Similar Items (Top 7 from triplet2 and Top 3 from triplet3):\n")
            for item in combined:
                txt_file.write(f"- {item['id']}: {item['text']} (Distance: {item['distance']:.4f}, Metadata: {item['metadata']})\n")
            txt_file.write("\n" + "="*80 + "\n\n")

            question_count += 1
            if question_count % 100 == 0:
                gc.collect()

    # Close JSON array
    json_file.write("\n]")
    json_file.close()
    txt_file.close()

    logger.info(f"Processed {question_count} questions successfully.")
    logger.info("Similarity search completed successfully")

if __name__ == "__main__":
    main()
