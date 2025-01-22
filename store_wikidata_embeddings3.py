import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import aiohttp
import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
MODEL_NAME = 'all-distilroberta-v1'  # Ensure consistency in the model name
TRIPLET_DATA_FILE = 'wikidata5m_all_triplet.txt'
OUTPUT_DIR = 'wikidata_embedding_triplet2'
CACHE_FILE = 'label_cache.json'
EMBEDDINGS_OUTPUT_FILE = 'embeddings.npy'
METADATA_OUTPUT_FILE = 'metadata.json'

class TripletEmbeddingCreator:
    def __init__(self,
                 model_name=MODEL_NAME,
                 triplet_file=TRIPLET_DATA_FILE,
                 output_dir=OUTPUT_DIR,
                 cache_file=CACHE_FILE,
                 embeddings_output_file=EMBEDDINGS_OUTPUT_FILE,
                 metadata_output_file=METADATA_OUTPUT_FILE):
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = SentenceTransformer(model_name)

        # Check for multiple GPUs and use DataParallel if available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        else:
            logger.info("Using a single GPU or CPU")

        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        logger.info(f"Initialized SentenceTransformer with model: {model_name}")

        # File paths
        self.triplet_file = triplet_file
        self.output_dir = output_dir
        self.cache_file = cache_file
        self.embeddings_output_file = embeddings_output_file
        self.metadata_output_file = metadata_output_file

        # Initialize label cache
        self.label_cache = {}
        self.load_label_cache()

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory at {self.output_dir}")

    def load_label_cache(self):
        """Load label cache from a JSON file to minimize API calls."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.label_cache = json.load(f)
                logger.info(f"Loaded {len(self.label_cache)} cached labels from {self.cache_file}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to load cache file {self.cache_file}: {e}")
                logger.warning("The cache file is corrupted and will be deleted.")
                os.remove(self.cache_file)
                self.label_cache = {}
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

    async def fetch_labels_async(self, session, batch_ids, semaphore, max_retries=3):
        """Asynchronously fetch labels for a batch of IDs from Wikidata API."""
        labels = {}
        ids_str = '|'.join(batch_ids)
        url = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbgetentities',
            'ids': ids_str,
            'format': 'json'
        }

        retries = 0
        backoff_time = 1  # Start with 1 second delay

        async with semaphore:
            while retries < max_retries:
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            entities = data.get('entities', {})
                            for entity_id in batch_ids:
                                entity_data = entities.get(entity_id)
                                if entity_data:
                                    labels_dict = entity_data.get('labels', {})
                                    if labels_dict:
                                        # Get the first available label in any language
                                        first_label = next(iter(labels_dict.values()))
                                        label = first_label['value']
                                        labels[entity_id] = label
                                    else:
                                        labels[entity_id] = ''  # Use empty text if no label found
                                        logger.debug(f"No label found for ID {entity_id}")
                                else:
                                    labels[entity_id] = ''  # Entity not found
                                    logger.debug(f"Entity data not found for ID {entity_id}")
                            logger.debug(f"Fetched labels for batch IDs: {batch_ids}")
                            return labels
                        elif response.status == 429:
                            # Rate limit exceeded
                            logger.warning(f"Rate limit exceeded for IDs {batch_ids}. Retrying after {backoff_time} seconds.")
                            await asyncio.sleep(backoff_time)
                            retries += 1
                            backoff_time *= 2  # Exponential backoff
                        else:
                            logger.error(f"HTTP error {response.status} for IDs {batch_ids}: {response.reason}")
                            break  # Do not retry for other HTTP errors
                except aiohttp.ClientError as e:
                    logger.warning(f"Request failed for IDs {batch_ids}: {e}. Retrying after {backoff_time} seconds.")
                    await asyncio.sleep(backoff_time)
                    retries += 1
                    backoff_time *= 2  # Exponential backoff

            # After max retries
            for entity_id in batch_ids:
                if entity_id not in labels:
                    labels[entity_id] = ''
                    logger.warning(f"Failed to fetch label for ID {entity_id} after retries. Assigning empty text.")
            return labels

    async def batch_fetch_labels_async(self, ids):
        """Asynchronously fetch labels for a set of IDs using aiohttp."""
        labels = {}
        ids_list = list(ids)
        max_ids_per_request = 50  # Wikidata API limit
        semaphore = asyncio.Semaphore(10)  # Throttle to 10 concurrent requests

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(0, len(ids_list), max_ids_per_request):
                batch_ids = ids_list[i:i + max_ids_per_request]
                task = asyncio.create_task(self.fetch_labels_async(session, batch_ids, semaphore))
                tasks.append(task)
            
            # Initialize a separate tqdm progress bar for fetching labels
            fetch_pbar = tqdm(total=len(tasks), desc="Fetching labels", unit="batch")
            results = []
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in fetching labels: {e}")
                fetch_pbar.update(1)
            fetch_pbar.close()

            for batch_labels in results:
                labels.update(batch_labels)
        
        return labels

    def batch_fetch_labels_wrapper(self, ids):
        """Wrapper to run the asynchronous label fetching."""
        return asyncio.run(self.batch_fetch_labels_async(ids))

    def encode_texts(self, texts, batch_size=512):
        """Encode texts using the model, handling DataParallel if necessary."""
        with torch.no_grad():
            if isinstance(self.model, torch.nn.DataParallel):
                embeddings = self.model.module.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                    batch_size=batch_size
                )
            else:
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                    batch_size=batch_size
                )
        return embeddings

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

                overall_pbar = tqdm(desc="Overall Processing", total=None, unit="chunk")
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
                            fetched_labels = self.batch_fetch_labels_wrapper(ids_to_fetch)
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
                            # Process the valid triplets to create embeddings
                            logger.info(f"Processing chunk {chunk_index} with {len(valid_triplets)} valid triplets.")
                            self.process_triplet_chunk(valid_triplets, chunk_index)
                            chunk_index += 1
                            overall_pbar.update(1)
                        else:
                            logger.info(f"No valid triplets in chunk {chunk_index}. Skipping.")
                            chunk_index += 1
                            overall_pbar.update(1)
                            continue
                    else:
                        break  # No more triplets to process

                overall_pbar.close()

        except Exception as e:
            logger.error(f"Error processing triplet data: {e}")
            logger.error(traceback.format_exc())

        self.save_label_cache()

    def process_triplet_chunk(self, valid_triplets, chunk_index):
        """Process a chunk of triplets by encoding them as combined texts and saving embeddings locally."""
        embeddings = []
        metadata = []

        logger.info(f"Encoding triplets for chunk {chunk_index}...")

        # Extract combined texts
        combined_texts = [
            f"Head: {triplet['head']['text']} | Relation: {triplet['relation']['text']} | Tail: {triplet['tail']['text']}"
            for triplet in valid_triplets
        ]

        # Determine optimal batch size based on GPU memory
        optimal_batch_size = 512  # You can experiment with different sizes

        # Initialize a tqdm progress bar for encoding
        encode_pbar = tqdm(total=len(combined_texts), desc=f"Encoding chunk {chunk_index}", unit="text")

        # Encode all combined texts in batches
        try:
            # Since SentenceTransformer's encode method processes the entire list with batch_size, 
            # we need to calculate how much each encode call processes.
            # To integrate tqdm correctly, we'll encode in sub-batches.
            for i in range(0, len(combined_texts), optimal_batch_size):
                batch_texts = combined_texts[i:i + optimal_batch_size]
                batch_embeddings = self.encode_texts(batch_texts, batch_size=optimal_batch_size)
                embeddings.append(batch_embeddings.cpu().numpy())
                encode_pbar.update(len(batch_texts))
            logger.info(f"Encoded {len(combined_texts)} triplet embeddings for chunk {chunk_index}.")
        except Exception as e:
            logger.error(f"Failed to encode embeddings for chunk {chunk_index}: {e}")
            logger.error(traceback.format_exc())
            encode_pbar.close()
            return  # Skip further processing if encoding fails

        encode_pbar.close()

        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)

        # Prepare metadata
        for triplet in valid_triplets:
            metadata_dict = {
                'head_id': triplet['head']['id'],
                'head_text': triplet['head']['text'],
                'relation_id': triplet['relation']['id'],
                'relation_text': triplet['relation']['text'],
                'tail_id': triplet['tail']['id'],
                'tail_text': triplet['tail']['text']
            }
            metadata.append(metadata_dict)

        # Save embeddings and metadata
        chunk_embeddings_file = os.path.join(self.output_dir, f'embeddings_chunk_{chunk_index}.npy')
        chunk_metadata_file = os.path.join(self.output_dir, f'metadata_chunk_{chunk_index}.json')

        try:
            np.save(chunk_embeddings_file, embeddings)
            logger.info(f"Saved embeddings to {chunk_embeddings_file}")
        except Exception as e:
            logger.error(f"Failed to save embeddings to {chunk_embeddings_file}: {e}")

        try:
            with open(chunk_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata to {chunk_metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {chunk_metadata_file}: {e}")

    def save_all_embeddings_and_metadata(self):
        """Combine all chunk embeddings and metadata into single files."""
        logger.info("Combining all chunk embeddings and metadata into single files...")

        embeddings = []
        metadata = []
        for file in os.listdir(self.output_dir):
            if file.startswith('embeddings_chunk_') and file.endswith('.npy'):
                chunk_embeddings = np.load(os.path.join(self.output_dir, file))
                embeddings.append(chunk_embeddings)
            elif file.startswith('metadata_chunk_') and file.endswith('.json'):
                with open(os.path.join(self.output_dir, file), 'r', encoding='utf-8') as f:
                    chunk_metadata = json.load(f)
                    metadata.extend(chunk_metadata)

        if embeddings:
            try:
                combined_embeddings = np.vstack(embeddings)
                np.save(os.path.join(self.output_dir, self.embeddings_output_file), combined_embeddings)
                logger.info(f"Saved combined embeddings to {self.embeddings_output_file}")
            except Exception as e:
                logger.error(f"Failed to save combined embeddings: {e}")

        if metadata:
            try:
                with open(os.path.join(self.output_dir, self.metadata_output_file), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved combined metadata to {self.metadata_output_file}")
            except Exception as e:
                logger.error(f"Failed to save combined metadata: {e}")

    def process(self):
        """Main processing function to handle triplet embedding creation."""
        self.process_triplets_in_chunks(chunk_size=50000, start_chunk_index=0)
        logger.info("All triplet embeddings processed.")
        self.save_all_embeddings_and_metadata()
        logger.info("Embeddings and metadata have been saved locally.")

def main():
    try:
        creator = TripletEmbeddingCreator()
        creator.process()
        logger.info("Triplet embeddings creation completed successfully.")
    except Exception as e:
        logger.error(f"Error during embedding creation: {e}")
        logger.error(traceback.format_exc())
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

if __name__ == "__main__":
    main()
