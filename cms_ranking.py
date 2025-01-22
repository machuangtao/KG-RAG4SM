import os
import json
import numpy as np
import torch
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import requests
import time
import chromadb  # Ensure chromadb is installed

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to display messages
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = 'all-distilroberta-v1'
TRIPLET_DATA_FILE = 'wikidata5m_all_triplet.txt'
OUTPUT_DIR = 'wikidata_embedding_triplet2'

class TripletEmbeddingCreator:
    def __init__(self,
                 model_name=MODEL_NAME,
                 triplet_file=TRIPLET_DATA_FILE,
                 output_dir=OUTPUT_DIR,
                 cache_file='label_cache.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        logger.info(f"Initialized SentenceTransformer with model: {model_name}")

        self.triplet_file = triplet_file
        self.output_dir = output_dir
        self.cache_file = cache_file

        # Initialize caches
        self.label_cache = {}
        self.load_label_cache()

        self.triplets_fetched_from_api = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize ChromaDB client
        self.chromadb_client = chromadb.Client()
        self.collection = self.chromadb_client.get_or_create_collection(name='triplet_embeddings')

    def load_label_cache(self):
        """Load label cache from a JSON file to minimize API calls."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.label_cache = json.load(f)
                logger.info(f"Loaded {len(self.label_cache)} cached labels from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache file {self.cache_file}: {e}")
                self.label_cache = {}
        else:
            logger.info("No existing cache file found. Starting with an empty cache.")
            self.label_cache = {}

    def save_label_cache(self):
        """Save the label cache to a JSON file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.label_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.label_cache)} labels to cache file {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache file {self.cache_file}: {e}")

    def batch_fetch_labels(self, ids):
        """Fetch labels for a batch of IDs from Wikidata API."""
        labels = {}
        ids_list = list(ids)
        max_ids_per_request = 50  # Wikidata API limit
        max_retries = 3

        for i in range(0, len(ids_list), max_ids_per_request):
            batch_ids = ids_list[i:i + max_ids_per_request]
            ids_str = '|'.join(batch_ids)
            url = 'https://www.wikidata.org/w/api.php'
            params = {
                'action': 'wbgetentities',
                'ids': ids_str,
                'format': 'json'
            }

            retries = 0
            backoff_time = 1  # Start with 1 second delay
            while retries < max_retries:
                try:
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    entities = data.get('entities', {})
                    for entity_id, entity_data in entities.items():
                        labels_dict = entity_data.get('labels', {})
                        if labels_dict:
                            # Get the first available label in any language
                            first_label = next(iter(labels_dict.values()))
                            label = first_label['value']
                            labels[entity_id] = label
                        else:
                            labels[entity_id] = ''  # Use empty text if no label found
                            logger.debug(f"No label found for ID {entity_id}")
                    time.sleep(0.1)  # Delay to respect API rate limits
                    break  # Successful request, exit retry loop
                except requests.exceptions.HTTPError as http_err:
                    status_code = http_err.response.status_code
                    if status_code == 429:
                        logger.warning(f"Rate limit exceeded for IDs {batch_ids}. Retrying after {backoff_time} seconds.")
                        time.sleep(backoff_time)
                        retries += 1
                        backoff_time *= 2  # Exponential backoff
                    else:
                        logger.error(f"HTTP error occurred for IDs {batch_ids}: {http_err}")
                        break  # Do not retry for other HTTP errors
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed for IDs {batch_ids}: {e}. Retrying after {backoff_time} seconds.")
                    time.sleep(backoff_time)
                    retries += 1
                    backoff_time *= 2  # Exponential backoff
            else:
                # After max retries, assign empty labels
                for entity_id in batch_ids:
                    if entity_id not in labels:
                        labels[entity_id] = ''
                        logger.warning(f"Failed to fetch label for ID {entity_id} after retries. Assigning empty text.")
                time.sleep(1)  # Delay before moving on

        return labels

    def process_triplets_in_chunks(self, chunk_size=50000, start_chunk_index=0):
        """Process triplets in chunks to handle large files."""
        chunk_index = start_chunk_index

        try:
            with open(self.triplet_file, 'r', encoding='utf-8') as f:
                # Skip lines up to start_chunk_index * chunk_size
                if start_chunk_index > 0:
                    lines_to_skip = start_chunk_index * chunk_size
                    logger.info(f"Skipping {lines_to_skip} lines to reach chunk {start_chunk_index}")
                    for _ in range(lines_to_skip):
                        next(f)
                    chunk_index = start_chunk_index

                batch_pbar = tqdm(desc="Processing batches", unit="batch")
                while True:
                    triplet_lines = []
                    ids_to_fetch = set()
                    while len(triplet_lines) < chunk_size:
                        try:
                            line = next(f)
                        except StopIteration:
                            break  # End of file
                        line = line.strip()
                        parts = line.split('\t')
                        if len(parts) == 3:
                            head_id, rel_id, tail_id = parts
                            triplet_lines.append((head_id, rel_id, tail_id))
                            if head_id not in self.label_cache:
                                ids_to_fetch.add(head_id)
                            if rel_id not in self.label_cache:
                                ids_to_fetch.add(rel_id)
                            if tail_id not in self.label_cache:
                                ids_to_fetch.add(tail_id)
                        else:
                            logger.warning(f"Invalid triplet format: {line.strip()}")
                            continue  # Invalid triplet format, skip

                    if triplet_lines:
                        # Fetch missing labels
                        if ids_to_fetch:
                            logger.info(f"Fetching {len(ids_to_fetch)} missing labels from API...")
                            fetched_labels = self.batch_fetch_labels(ids_to_fetch)
                            self.label_cache.update(fetched_labels)
                            self.save_label_cache()  # Save cache after fetching labels

                        # Now process triplets to collect valid ones
                        valid_triplets = []
                        for head_id, rel_id, tail_id in triplet_lines:
                            head_text = self.label_cache.get(head_id, '')
                            rel_text = self.label_cache.get(rel_id, '')
                            tail_text = self.label_cache.get(tail_id, '')

                            # Count how many labels are missing
                            missing_labels = 0
                            if not head_text:
                                missing_labels += 1
                            if not rel_text:
                                missing_labels += 1
                            if not tail_text:
                                missing_labels += 1

                            if missing_labels >= 2:
                                logger.debug(f"Skipping triplet due to missing labels for IDs: {head_id}, {rel_id}, {tail_id}")
                                continue  # Skip this triplet
                            else:
                                triplet = {
                                    'head': {'id': head_id, 'text': head_text},
                                    'relation': {'id': rel_id, 'text': rel_text},
                                    'tail': {'id': tail_id, 'text': tail_text}
                                }
                                valid_triplets.append(triplet)

                        if valid_triplets:
                            embeddings_file = os.path.join(self.output_dir, f'triplet_embeddings_{chunk_index}.npy')
                            metadata_file = os.path.join(self.output_dir, f'triplet_metadata_{chunk_index}.json')

                            if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
                                logger.info(f"Chunk {chunk_index} already processed. Skipping.")
                            else:
                                logger.info(f"Processing chunk {chunk_index} with {len(valid_triplets)} triplets.")
                                self.process_triplet_chunk(valid_triplets, chunk_index)

                            chunk_index += 1
                            batch_pbar.update(1)
                        else:
                            logger.info(f"No valid triplets in chunk {chunk_index}. Skipping.")
                            chunk_index += 1
                            batch_pbar.update(1)
                            continue
                    else:
                        break  # No more triplets to process

                batch_pbar.close()

        except Exception as e:
            logger.error(f"Error processing triplet data: {e}")
            raise

        self.save_label_cache()
        self.save_fetched_triplets()

    def process_triplet_chunk(self, valid_triplets, chunk_index):
        """Process a chunk of triplets."""
        embeddings = []
        metadata = []

        logger.info(f"Encoding triplets for chunk {chunk_index}...")
        # Generate embeddings
        for triplet in tqdm(valid_triplets, desc=f"Processing chunk {chunk_index}", leave=False):
            head_text = triplet['head']['text']
            relation_text = triplet['relation']['text']
            tail_text = triplet['tail']['text']

            try:
                with torch.no_grad():
                    head_emb = self.model.encode(
                        head_text,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                    rel_emb = self.model.encode(
                        relation_text,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                    tail_emb = self.model.encode(
                        tail_text,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )

                # Combine embeddings (concatenation)
                combined_emb = torch.cat([head_emb, rel_emb, tail_emb])

                # Store embedding and metadata
                embeddings.append(combined_emb.cpu().numpy())
                metadata.append(triplet)
            except Exception as e:
                logger.error(f"Error encoding triplet {triplet}: {e}")

        # Save embeddings and metadata for this chunk
        self.save_embeddings_and_metadata(embeddings, metadata, chunk_index)
        # Store embeddings in ChromaDB
        self.store_embeddings_in_chromadb(embeddings, metadata)

    def save_embeddings_and_metadata(self, embeddings, metadata, chunk_index):
        """Save embeddings and metadata to files for a specific chunk."""
        embeddings_path = os.path.join(self.output_dir, f'triplet_embeddings_{chunk_index}.npy')
        metadata_path = os.path.join(self.output_dir, f'triplet_metadata_{chunk_index}.json')

        try:
            # Save embeddings
            np.save(embeddings_path, np.array(embeddings))
            logger.info(f"Embeddings saved to {embeddings_path}")

            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings or metadata for chunk {chunk_index}: {e}")

    def store_embeddings_in_chromadb(self, embeddings, metadata):
        """Store embeddings in ChromaDB."""
        logger.info("Storing embeddings in ChromaDB...")
        try:
            ids = []
            metadatas = []
            for triplet in metadata:
                document_id = f"{triplet['head']['id']}_{triplet['relation']['id']}_{triplet['tail']['id']}"
                ids.append(document_id)
                metadatas.append(triplet)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info("Embeddings stored in ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to store embeddings in ChromaDB: {e}")

    def save_fetched_triplets(self):
        """Save the list of triplets fetched from API."""
        fetched_data = {
            'triplets_fetched_from_api': self.triplets_fetched_from_api
        }
        fetched_data_path = os.path.join(self.output_dir, 'fetched_triplets_from_api.json')
        try:
            with open(fetched_data_path, 'w', encoding='utf-8') as f:
                json.dump(fetched_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Fetched triplets saved to {fetched_data_path}")
        except Exception as e:
            logger.error(f"Failed to save fetched triplets to {fetched_data_path}: {e}")

    def process(self):
        self.process_triplets_in_chunks(chunk_size=50000, start_chunk_index=0)  # Start from chunk 0

def main():
    try:
        creator = TripletEmbeddingCreator()
        creator.process()
        logger.info("Triplet embeddings and metadata creation completed successfully")
    except Exception as e:
        logger.error(f"Error during embedding creation: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

if __name__ == "__main__":
    main()
