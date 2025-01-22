import os
import json
import time
import numpy as np
import torch
import aiohttp
import asyncio
import logging
import traceback
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration constants
MODEL_NAME = 'all-distilroberta-v1'
TRIPLET_DATA_FILE = 'wikidata5m_all_triplet.txt'
CACHE_FILE = 'label_cache.json'  # We won't actually save or load it, but we keep the name
BATCH_SIZE = 512           # For encoding
CHUNK_SIZE = 50000         # How many lines (triplets) to consider as the "first batch"
ARTIFICIAL_FETCH_DELAY = 1  # Seconds of artificial delay per request to the Wikidata API

class DebugTripletEmbedding:
    def __init__(self,
                 model_name=MODEL_NAME,
                 triplet_file=TRIPLET_DATA_FILE,
                 cache_file=CACHE_FILE,
                 chunk_size=CHUNK_SIZE):
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        else:
            logger.info("Using a single GPU or CPU")

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Initialized SentenceTransformer with model: {model_name}")

        # File paths
        self.triplet_file = triplet_file
        self.cache_file = cache_file  # We won't actually load/save it in this debug version
        self.chunk_size = chunk_size

        # Instead of truly loading a cache from disk, let's keep it in memory only
        self.label_cache = {}

    async def fetch_labels_async(self, session, batch_ids, semaphore, max_retries=3):
        """
        Asynchronously fetch labels for a batch of IDs from the Wikidata API,
        adding an artificial delay after each successful request.
        """
        labels = {}
        ids_str = '|'.join(batch_ids)
        url = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbgetentities',
            'ids': ids_str,
            'format': 'json'
        }

        retries = 0
        backoff_time = 1  # Start with 1 second for rate-limit or request failures

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
                                        # Just pick the first available label
                                        first_label = next(iter(labels_dict.values()))
                                        label = first_label['value']
                                        labels[entity_id] = label
                                    else:
                                        labels[entity_id] = ''  # No label found
                                else:
                                    labels[entity_id] = ''  # Entity missing
                            # ---- ARTIFICIAL DELAY HERE ----
                            await asyncio.sleep(ARTIFICIAL_FETCH_DELAY)
                            return labels

                        elif response.status == 429:
                            # Rate limit exceeded
                            logger.warning(f"429 Rate limit for IDs: {batch_ids} - retrying after {backoff_time}s.")
                            await asyncio.sleep(backoff_time)
                            retries += 1
                            backoff_time *= 2
                        else:
                            logger.error(f"HTTP error {response.status} for IDs {batch_ids}: {response.reason}")
                            break  # no more retries for other HTTP errors

                except aiohttp.ClientError as e:
                    logger.warning(f"Request failed for IDs {batch_ids}: {e}. Retrying after {backoff_time}s.")
                    await asyncio.sleep(backoff_time)
                    retries += 1
                    backoff_time *= 2

            # If we exit the while loop, we failed to fetch
            for entity_id in batch_ids:
                if entity_id not in labels:
                    labels[entity_id] = ''
            return labels

    async def batch_fetch_labels_async(self, ids):
        """
        Orchestrates async fetching of labels for all needed IDs,
        with a concurrency limit.
        """
        labels = {}
        ids_list = list(ids)
        max_ids_per_request = 50  # Wikidata API recommended chunk
        semaphore = asyncio.Semaphore(10)  # up to 10 concurrent requests

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(0, len(ids_list), max_ids_per_request):
                batch_ids = ids_list[i:i + max_ids_per_request]
                task = asyncio.create_task(self.fetch_labels_async(session, batch_ids, semaphore))
                tasks.append(task)

            fetch_pbar = tqdm(total=len(tasks), desc="Fetching labels", unit="batch")
            results = []
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in fetch: {e}")
                fetch_pbar.update(1)
            fetch_pbar.close()

            for batch_labels in results:
                labels.update(batch_labels)

        return labels

    def batch_fetch_labels_wrapper(self, ids):
        """Simple wrapper to run the async label fetching."""
        return asyncio.run(self.batch_fetch_labels_async(ids))

    def encode_texts(self, texts, batch_size=512):
        """Encode texts using the model, handling DataParallel if needed."""
        with torch.no_grad():
            if isinstance(self.model, torch.nn.DataParallel):
                emb = self.model.module.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                    batch_size=batch_size
                )
            else:
                emb = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                    batch_size=batch_size
                )
        return emb

    def debug_process_first_batch(self):
        """
        1) Load up to chunk_size lines (the first batch).
        2) Fetch missing labels from Wikidata (with artificial delay).
        3) Convert to combined text, encode, measure time.
        4) Print timing results (total time, average time, etc.).
        5) Do not save anything or modify any files.
        """
        # --- 1) Load the first chunk of triplets from the file ---
        triplet_lines = []
        loaded_count = 0

        # Also track all IDs for potential label fetching
        ids_to_fetch = set()

        logger.info(f"Reading up to {self.chunk_size} lines from {self.triplet_file}...")
        try:
            with open(self.triplet_file, 'r', encoding='utf-8') as f:
                for _ in range(self.chunk_size):
                    line = next(f, None)
                    if line is None:
                        break  # end of file
                    line = line.strip()
                    parts = line.split('\t')
                    if len(parts) == 3:
                        head_id, rel_id, tail_id = parts
                        triplet_lines.append((head_id, rel_id, tail_id))

                        # Track any IDs not in the cache
                        if head_id not in self.label_cache:
                            ids_to_fetch.add(head_id)
                        if rel_id not in self.label_cache:
                            ids_to_fetch.add(rel_id)
                        if tail_id not in self.label_cache:
                            ids_to_fetch.add(tail_id)

                        loaded_count += 1
                    else:
                        logger.debug(f"Invalid triplet: {line}")
        except StopIteration:
            pass
        except FileNotFoundError:
            logger.error(f"File {self.triplet_file} not found.")
            return
        except Exception as e:
            logger.error(f"Error reading triplets: {e}")
            logger.error(traceback.format_exc())
            return

        if loaded_count == 0:
            logger.info("No triplets loaded. Exiting.")
            return

        logger.info(f"Loaded {loaded_count} triplets in the first batch.")

        # --- 2) Fetch labels (async) for missing IDs, with artificial delay ---
        logger.info(f"Fetching {len(ids_to_fetch)} labels from Wikidata with artificial delay of {ARTIFICIAL_FETCH_DELAY}s per request.")
        fetch_start = time.time()
        fetched_labels = self.batch_fetch_labels_wrapper(ids_to_fetch)
        fetch_end = time.time()
        logger.info(f"Finished fetching labels. Elapsed time: {fetch_end - fetch_start:.2f} seconds.")

        # Update the label cache in memory
        self.label_cache.update(fetched_labels)

        # --- 3) Convert triplets to combined text, filter out those with <2 missing labels, then encode ---
        valid_triplets = []
        for (head_id, rel_id, tail_id) in triplet_lines:
            head_text = self.label_cache.get(head_id, '')
            rel_text  = self.label_cache.get(rel_id, '')
            tail_text = self.label_cache.get(tail_id, '')

            missing = sum(1 for x in [head_text, rel_text, tail_text] if not x)
            # If 2 or more labels are missing, skip
            if missing >= 2:
                continue
            valid_triplets.append((head_id, rel_id, tail_id))

        logger.info(f"{len(valid_triplets)} triplets are valid after filtering missing labels.")

        # Create combined texts
        combined_texts = [
            f"Head: {self.label_cache.get(hid, '')} | Relation: {self.label_cache.get(rid, '')} | Tail: {self.label_cache.get(tid, '')}"
            for (hid, rid, tid) in valid_triplets
        ]

        if not combined_texts:
            logger.info("No triplets to encode. Exiting.")
            return

        # --- Time the embedding generation for these valid triplets ---
        encode_start = time.time()
        # We'll do it in sub-batches with tqdm
        for i in tqdm(range(0, len(combined_texts), BATCH_SIZE), desc="Encoding Embeddings", unit="batch"):
            batch_texts = combined_texts[i:i + BATCH_SIZE]
            _ = self.encode_texts(batch_texts, batch_size=BATCH_SIZE)
        encode_end = time.time()

        encode_time = encode_end - encode_start
        logger.info(f"Finished encoding {len(combined_texts)} valid triplets in {encode_time:.2f} seconds.")

        # --- 4) Print timing results ---
        total_time = (fetch_end - fetch_start) + encode_time  # total from fetch + encode
        avg_time_per_triplet = total_time / loaded_count

        print("\n===== Timing Results (Debug) =====")
        print(f"Loaded triplets for first batch:    {loaded_count}")
        print(f"Valid triplets after label filter:  {len(valid_triplets)}")
        print(f"Label fetching time (this batch):   {fetch_end - fetch_start:.2f} seconds")
        print(f"Encoding time (this batch):         {encode_time:.2f} seconds")
        print(f"Total time (fetch + encode):        {total_time:.2f} seconds")
        print(f"Average time per loaded triplet:    {avg_time_per_triplet:.6f} seconds/triplet")
        print(f"Approx. total time for these {loaded_count} triplets: {avg_time_per_triplet * loaded_count:.2f} seconds.\n")

        # Not saving anything to disk.

def main():
    try:
        debugger = DebugTripletEmbedding()
        debugger.debug_process_first_batch()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

if __name__ == "__main__":
    main()
