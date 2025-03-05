#!/usr/bin/env python
import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from pathlib import Path
import pyarrow.parquet as pq
import psutil
import gc
import time
import argparse

__all__ = ["main", "SimilarityFinder", "SimilarityModel", "log_memory_usage"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Updated batch sizes as requested
QUESTION_BATCH_SIZE = 1024
ENTITY_BATCH_SIZE = 8192

TEST_MODE = False  # default; can be changed via argparse
NUM_TEST_QUESTIONS = 100

def log_memory_usage():
    process = psutil.Process()
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

class SimilarityModel(nn.Module):
    def __init__(self):
        super(SimilarityModel, self).__init__()

    def forward(self, question_embedding, batch_embeddings):
        cos_scores = torch.matmul(question_embedding, batch_embeddings.t())
        return cos_scores

class SimilarityFinder:
    def __init__(self):
        self.model = SimilarityModel()
        available_gpus = torch.cuda.device_count()
        logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs detected by torch: {available_gpus}")
        if torch.cuda.is_available() and available_gpus > 1:
            logger.info(f"Using {available_gpus} GPUs via DataParallel.")
            self.model = nn.DataParallel(self.model, device_ids=list(range(available_gpus)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        self.questions = None
        self.question_embeddings = None
        self.entity_metadata = []
        self.relation_metadata = []
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.load_metadata()
        self.load_questions_and_embeddings()
        self.load_embeddings()

    def load_embeddings(self):
        try:
            log_memory_usage()
            logger.info("Loading embeddings...")
            self.entity_embeddings = np.load("wikidata_embedding_entities/wikidata_embedding_entities.npy", mmap_mode='r')
            self.relation_embeddings = np.load("wikidata_embedding_relations/embeddings.npy", mmap_mode='r')
            logger.info(f"Loaded entity embeddings shape: {self.entity_embeddings.shape}")
            logger.info(f"Loaded relation embeddings shape: {self.relation_embeddings.shape}")
            log_memory_usage()
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    def load_metadata(self):
        try:
            logger.info("Loading metadata...")
            log_memory_usage()
            entity_metadata_path = "wikidata_embedding_entities/wikidata_embedding_entities_metadata.txt"
            with open(entity_metadata_path, 'r', encoding='utf-8') as f:
                self.entity_metadata = [line.strip().split('\t') for line in f]
            logger.info(f"Loaded {len(self.entity_metadata)} entity metadata entries")
            relation_metadata_path = "wikidata_embedding_relations/metadata.txt"
            with open(relation_metadata_path, 'r', encoding='utf-8') as f:
                self.relation_metadata = [line.strip().split('\t') for line in f]
            logger.info(f"Loaded {len(self.relation_metadata)} relation metadata entries")
            log_memory_usage()
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

    def load_questions_and_embeddings(self):
        try:
            logger.info("Loading questions and embeddings...")
            log_memory_usage()
            chroma_dir = os.path.join(os.getcwd(), 'cms_ques_embedding_full')
            parquet_files = list(Path(chroma_dir).glob('*.parquet'))
            if not parquet_files:
                raise FileNotFoundError("No parquet files found in the ChromaDB directory")
            data = pq.read_table(parquet_files[0])
            df = data.to_pandas()
            if TEST_MODE:
                logger.info(f"Test mode: processing only {NUM_TEST_QUESTIONS} questions")
                df = df.head(NUM_TEST_QUESTIONS)
            self.questions = {
                'ids': df['id'].tolist(),
                'documents': df['document'].tolist()
            }
            question_embeddings_np = np.array(df['embedding'].tolist(), dtype=np.float32)
            self.question_embeddings = torch.from_numpy(question_embeddings_np).to(torch.float32)
            self.question_embeddings = self.question_embeddings / self.question_embeddings.norm(dim=1, keepdim=True)
            logger.info(f"Loaded {len(self.questions['ids'])} questions with embeddings")
            logger.info(f"Question embeddings shape: {self.question_embeddings.shape}")
            log_memory_usage()
        except Exception as e:
            logger.error(f"Error loading questions and embeddings: {e}")
            raise

    def process_batch(self, embeddings, metadata, question_embedding_gpu, start_idx, batch_size):
        try:
            end_idx = min(start_idx + batch_size, len(metadata))
            batch_embeddings_np = embeddings[start_idx:end_idx]
            batch_embeddings = torch.tensor(batch_embeddings_np, dtype=torch.float32)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
            batch_embeddings_gpu = batch_embeddings.to(self.device)
            if question_embedding_gpu.dim() == 1:
                question_embedding_gpu = question_embedding_gpu.unsqueeze(0)
            logger.debug(f"question_embedding_gpu shape: {question_embedding_gpu.shape}")
            logger.debug(f"batch_embeddings_gpu shape: {batch_embeddings_gpu.shape}")
            cos_scores = self.model(question_embedding_gpu, batch_embeddings_gpu)
            k = min(10, cos_scores.size(1))
            top_scores, top_indices = torch.topk(cos_scores, k=k, largest=True, sorted=True, dim=1)
            top_scores = top_scores.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            results = []
            for score, idx in zip(top_scores, top_indices):
                item_idx = start_idx + idx
                if item_idx < len(metadata):
                    results.append({
                        "id": metadata[item_idx][0],
                        "text": metadata[item_idx][1],
                        "score": float(score)
                    })
            del batch_embeddings, batch_embeddings_gpu, cos_scores, top_scores, top_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return results
        except Exception as e:
            logger.error(f"Error processing batch starting at index {start_idx}: {e}")
            return []

    def process_question(self, q_idx):
        start_time = time.time()
        try:
            question_id = self.questions['ids'][q_idx]
            question_text = self.questions['documents'][q_idx]
            question_embedding = self.question_embeddings[q_idx]
            question_embedding_gpu = question_embedding.to(self.device)
            similar_entities = []
            similar_relations = []
            for start_idx in range(0, len(self.entity_metadata), ENTITY_BATCH_SIZE):
                batch_results = self.process_batch(
                    self.entity_embeddings,
                    self.entity_metadata,
                    question_embedding_gpu,
                    start_idx,
                    ENTITY_BATCH_SIZE
                )
                similar_entities.extend(batch_results)
            for start_idx in range(0, len(self.relation_metadata), ENTITY_BATCH_SIZE):
                batch_results = self.process_batch(
                    self.relation_embeddings,
                    self.relation_metadata,
                    question_embedding_gpu,
                    start_idx,
                    ENTITY_BATCH_SIZE
                )
                similar_relations.extend(batch_results)
            similar_entities.sort(key=lambda x: x['score'], reverse=True)
            similar_relations.sort(key=lambda x: x['score'], reverse=True)
            del question_embedding_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            end_time = time.time()
            logger.info(f"Time taken to process question index {q_idx} (ID: {question_id}): {end_time - start_time:.4f} seconds")
            return question_id, {
                "question": question_text,
                "similar_entities": similar_entities[:10],
                "similar_relations": similar_relations[:10]
            }
        except Exception as e:
            logger.error(f"Error processing question {q_idx}: {e}")
            end_time = time.time()
            logger.info(f"Time taken (with error) for question index {q_idx}: {end_time - start_time:.4f} seconds")
            return question_id, {
                "question": self.questions['documents'][q_idx],
                "error": str(e)
            }

    def find_similar_items(self):
        results = {}
        total_questions = len(self.questions['ids'])
        logger.info("Starting similarity search for all questions...")
        log_memory_usage()
        question_times = []
        start_time_all = time.time()
        try:
            for q_start_idx in tqdm(range(0, total_questions, QUESTION_BATCH_SIZE), desc="Processing questions"):
                q_end_idx = min(q_start_idx + QUESTION_BATCH_SIZE, total_questions)
                for q_idx in range(q_start_idx, q_end_idx):
                    q_time_start = time.time()
                    question_id, data = self.process_question(q_idx)
                    q_time_end = time.time()
                    question_times.append(q_time_end - q_time_start)
                    results[question_id] = data
                    if q_idx % 10 == 0:
                        logger.info(f"Processed {q_idx}/{total_questions} questions")
                        log_memory_usage()
                    if q_idx % 100 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in find_similar_items: {e}")
        end_time_all = time.time()
        total_time_all = end_time_all - start_time_all
        avg_time = sum(question_times) / len(question_times) if question_times else 0
        logger.info(f"Total time to process all questions: {total_time_all:.4f} seconds")
        logger.info(f"Average time per question: {avg_time:.4f} seconds")
        return results

    def save_results(self, results):
        try:
            json_path = "cms_wikidata_similar_full.json"
            if TEST_MODE:
                json_path = "cms_wikidata_similar_test.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {json_path}")
            txt_path = "cms_wikidata_similar_full.txt"
            if TEST_MODE:
                txt_path = "cms_wikidata_similar_test.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for question_id, data in results.items():
                    f.write(f"Question ID: {question_id}\n")
                    f.write(f"Question: {data['question']}\n\n")
                    if 'error' in data:
                        f.write(f"Error: {data['error']}\n")
                    else:
                        f.write("Similar Entities:\n")
                        for entity in data['similar_entities']:
                            f.write(f"- {entity['id']}: {entity['text']} (Score: {entity['score']:.4f})\n")
                        f.write("\n")
                        f.write("Similar Relations:\n")
                        for relation in data['similar_relations']:
                            f.write(f"- {relation['id']}: {relation['text']} (Score: {relation['score']:.4f})\n")
                    f.write("\n" + "="*80 + "\n\n")
            logger.info(f"Results saved to {txt_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Perform similarity search using precomputed embeddings")
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode (process fewer questions)")
    args = parser.parse_args()
    global TEST_MODE
    TEST_MODE = args.test_mode
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
        logger.info("Checking CUDA availability in main...")
        logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        logger.info("Starting script execution")
        log_memory_usage()
        finder = SimilarityFinder()
        results = finder.find_similar_items()
        finder.save_results(results)
        logger.info("Process completed successfully")
        log_memory_usage()
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
