import os
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# GPU setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Model initialization
model = SentenceTransformer('all-distilroberta-v1')
model.to(device)
if torch.cuda.device_count() > 1:
    print("Using DataParallel")
    model = torch.nn.DataParallel(model)

# Paths
input_file_path = "wikidata5m_entity.txt"
output_dir = "wikidata_embedding_entities"
os.makedirs(output_dir, exist_ok=True)
embeddings_file_path = os.path.join(output_dir, "wikidata_embedding_entities.npy")
metadata_file_path = os.path.join(output_dir, "wikidata_embedding_entities_metadata.txt")

def load_entities_with_metadata(file_path):
    """
    Load entities and preserve metadata (IDs and text).
    Returns a list of dictionaries containing ID and text.
    """
    entities = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    entities.append({'id': parts[0], 'text': " ".join(parts[1:])})
        print(f"Loaded {len(entities)} entities")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    return entities

def generate_embeddings(entities, model, batch_size=32):
    """
    Generate embeddings for the given entities using the provided model.
    """
    embeddings = []
    with tqdm(total=len(entities), desc="Generating Embeddings") as pbar:
        for i in range(0, len(entities), batch_size):
            batch_texts = [entity['text'] for entity in entities[i:i+batch_size]]
            with torch.cuda.amp.autocast():
                if isinstance(model, torch.nn.DataParallel):
                    batch_embeddings = model.module.encode(batch_texts, convert_to_numpy=True)
                else:
                    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
            pbar.update(len(batch_texts))
            
            # Reduce memory usage
            if i % 100 == 0:
                torch.cuda.empty_cache()
    return np.vstack(embeddings)

def save_embeddings_and_metadata(embeddings, entities, embeddings_file_path, metadata_file_path):
    """
    Save embeddings and metadata to separate files for traceability.
    """
    # Save embeddings
    np.save(embeddings_file_path, embeddings)
    print(f"Embeddings saved to {embeddings_file_path}")
    
    # Save metadata
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(f"{entity['id']}\t{entity['text']}\n")
    print(f"Metadata saved to {metadata_file_path}")

def main():
    start_time = time.time()
    print("\nLoading entities...")
    entities = load_entities_with_metadata(input_file_path)
    if not entities:
        return

    print("\nGenerating embeddings...")
    try:
        embeddings = generate_embeddings(entities, model)
        print("\nSaving embeddings and metadata...")
        save_embeddings_and_metadata(embeddings, entities, embeddings_file_path, metadata_file_path)
        print(f"\nProcess completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        torch.cuda.empty_cache()
