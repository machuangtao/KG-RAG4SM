import os
# Set CUDA devices before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

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
import time  # For timing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Updated batch sizes
QUESTION_BATCH_SIZE = 256  # for processing questions in batches
ENTITY_BATCH_SIZE = 1024    # increased entity/relation batch size

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process()
    logger.info(f"Memory usage (RSS): {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

class SimilarityModel(nn.Module):
    """
    A simple model that takes a single question embedding and a batch of entity/relation embeddings
    and returns the cosine similarity scores.
    """
    def __init__(self):
        super(SimilarityModel, self).__init__()

    def forward(self, question_embedding, batch_embeddings):
        return torch.matmul(question_embedding, batch_embeddings.t())

class SimilarityFinder:
    def __init__(self):
        # Create the similarity model
        self.model = SimilarityModel()

        # Check GPUs
        available_gpus = torch.cuda.device_count()
        logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs detected by torch: {available_gpus}")

        # Use DataParallel if multiple GPUs
        if torch.cuda.is_available() and available_gpus > 1:
            logger.info(f"Using {available_gpus} GPUs via DataParallel.")
            self.model = nn.DataParallel(self.model, device_ids=list(range(available_gpus)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Initialize data containers
        self.questions = None
        self.question_embeddings = None
        self.entity_metadata = []
        self.relation_metadata = []
        self.entity_embeddings = None
        self.relation_embeddings = None
        
        # Load all data
        self.load_metadata()
        self.load_questions_and_embeddings()
        self.load_embeddings()

    def load_embeddings(self):
        """Load embeddings as memory-mapped numpy arrays on CPU."""
        logger.info("Loading embeddings...")
        log_memory_usage()
        try:
            self.entity_embeddings = np.load(
                "wikidata_embedding_entities/wikidata_embedding_entities.npy", 
                mmap_mode='r'
            )
            self.relation_embeddings = np.load(
                "wikidata_embedding_relations/embeddings.npy", 
                mmap_mode='r'
            )
            
            logger.info(f"Loaded entity embeddings shape: {self.entity_embeddings.shape}")
            logger.info(f"Loaded relation embeddings shape: {self.relation_embeddings.shape}")
            log_memory_usage()
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    def load_metadata(self):
        """Load metadata for entities and relations."""
        logger.info("Loading metadata...")
        log_memory_usage()
        try:
            # Load entity metadata
            entity_metadata_path = "wikidata_embedding_entities/wikidata_embedding_entities_metadata.txt"
            with open(entity_metadata_path, 'r', encoding='utf-8') as f:
                self.entity_metadata = [line.strip().split('\t') for line in f]
            logger.info(f"Loaded {len(self.entity_metadata)} entity metadata entries")
            
            # Load relation metadata
            relation_metadata_path = "wikidata_embedding_relations/metadata.txt"
            with open(relation_metadata_path, 'r', encoding='utf-8') as f:
                self.relation_metadata = [line.strip().split('\t') for line in f]
            logger.info(f"Loaded {len(self.relation_metadata)} relation metadata entries")
            
            log_memory_usage()
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

    def load_questions_and_embeddings(self):
        """Load questions and their embeddings from ChromaDB parquet files."""
        logger.info("Loading questions and embeddings...")
        log_memory_usage()
        try:
            chroma_dir = os.path.join(os.getcwd(), 'mimic_ques_embedding_full2')
            parquet_files = list(Path(chroma_dir).glob('*.parquet'))
            
            if not parquet_files:
                raise FileNotFoundError("No parquet files found in the ChromaDB directory")
            
            data = pq.read_table(parquet_files[0])
            df = data.to_pandas()
            
            # Only process the first 10 questions
            df = df.head(10)
            
            # Extract questions, IDs and embeddings
            self.questions = {
                'ids': df['id'].tolist(),
                'documents': df['document'].tolist()
            }
            
            question_embeddings_np = np.array(df['embedding'].tolist(), dtype=np.float32)
            
            # Convert to torch tensor and normalize on CPU first
            self.question_embeddings = torch.from_numpy(question_embeddings_np)
            self.question_embeddings = self.question_embeddings / self.question_embeddings.norm(dim=1, keepdim=True)

            logger.info(f"Loaded {len(self.questions['ids'])} questions (first 10 only) with embeddings")
            logger.info(f"Question embeddings shape: {self.question_embeddings.shape}")
            log_memory_usage()
        except Exception as e:
            logger.error(f"Error loading questions and embeddings: {e}")
            raise

    @torch.no_grad()
    def process_batch(self, embeddings, metadata, question_embedding_gpu, start_idx, batch_size):
        """Process a batch of embeddings for similarity search using the model."""
        end_idx = min(start_idx + batch_size, len(metadata))
        batch_embeddings_np = embeddings[start_idx:end_idx]

        batch_embeddings = torch.tensor(batch_embeddings_np, dtype=torch.float32)
        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
        batch_embeddings_gpu = batch_embeddings.to(self.device)

        if question_embedding_gpu.dim() == 1:
            question_embedding_gpu = question_embedding_gpu.unsqueeze(0)

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

        # Cleanup
        del batch_embeddings, batch_embeddings_gpu, cos_scores, top_scores, top_indices
        return results

    def process_question(self, q_idx):
        """Process a single question to find top similar entities and relations."""
        question_start_time = time.time()

        question_id = self.questions['ids'][q_idx]
        question_text = self.questions['documents'][q_idx]
        question_embedding = self.question_embeddings[q_idx].to(self.device)

        similar_entities = []
        similar_relations = []

        # Process entities
        for start_idx in range(0, len(self.entity_metadata), ENTITY_BATCH_SIZE):
            batch_results = self.process_batch(
                self.entity_embeddings,
                self.entity_metadata,
                question_embedding,
                start_idx,
                ENTITY_BATCH_SIZE
            )
            similar_entities.extend(batch_results)

        # Process relations
        for start_idx in range(0, len(self.relation_metadata), ENTITY_BATCH_SIZE):
            batch_results = self.process_batch(
                self.relation_embeddings,
                self.relation_metadata,
                question_embedding,
                start_idx,
                ENTITY_BATCH_SIZE
            )
            similar_relations.extend(batch_results)

        # Sort and get top 10
        similar_entities.sort(key=lambda x: x['score'], reverse=True)
        similar_relations.sort(key=lambda x: x['score'], reverse=True)

        del question_embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        question_end_time = time.time()
        question_time = question_end_time - question_start_time

        return question_id, {
            "question": question_text,
            "similar_entities": similar_entities[:10],
            "similar_relations": similar_relations[:10],
            "processing_time": question_time
        }

    def find_similar_items(self):
        """Find similar entities and relations for each question."""
        results = {}
        total_questions = len(self.questions['ids'])
        
        logger.info("Starting similarity search for the first 10 questions...")
        log_memory_usage()

        question_times = []
        total_start_time = time.time()

        # Since we only have up to 10 questions, one batch is enough
        # But we keep the loop for consistency
        for q_start_idx in tqdm(range(0, total_questions, QUESTION_BATCH_SIZE), desc="Processing question batches"):
            q_end_idx = min(q_start_idx + QUESTION_BATCH_SIZE, total_questions)

            for q_idx in tqdm(range(q_start_idx, q_end_idx), desc="Processing individual questions", leave=False):
                try:
                    question_id, data = self.process_question(q_idx)
                    results[question_id] = data
                    question_times.append(data["processing_time"])
                except Exception as e:
                    logger.error(f"Error processing question {q_idx}: {e}")
                    question_id = self.questions['ids'][q_idx]
                    question_text = self.questions['documents'][q_idx]
                    results[question_id] = {
                        "question": question_text,
                        "error": str(e)
                    }

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        avg_time_per_question = total_time / total_questions if total_questions > 0 else 0

        # Print timing results
        logger.info("======== Timing Information (First 10 Questions) ========")
        logger.info(f"Total processing time for the first 10 questions: {total_time:.2f} seconds")
        logger.info(f"Average processing time per question (first 10): {avg_time_per_question:.4f} seconds")

        if question_times:
            avg_question_time = sum(question_times) / len(question_times)
            logger.info(f"Average individual question processing time (based on recorded per-question times): {avg_question_time:.4f} seconds")

        return results

    def save_results(self, results):
        """Save results of the first 10 questions to JSON and TXT files."""
        try:
            json_path = "mimic_wikidata_similar_first10.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {json_path}")
            
            txt_path = "mimic_wikidata_similar_first10.txt"
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

                        if 'processing_time' in data:
                            f.write(f"\nProcessing time for this question: {data['processing_time']:.4f} seconds\n")

                    f.write("\n" + "="*80 + "\n\n")
            
            logger.info(f"Results saved to {txt_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def main():
    try:
        logger.info("Starting script execution for the first 10 questions")
        log_memory_usage()
        
        finder = SimilarityFinder()
        results = finder.find_similar_items()
        finder.save_results(results)
        
        logger.info("Process completed successfully for the first 10 questions")
        log_memory_usage()
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
