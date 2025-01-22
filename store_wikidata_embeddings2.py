import os
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

        try:
            # Model initialization
            self.model = SentenceTransformer('all-distilroberta-v1', device=self.device)
            if torch.cuda.device_count() > 1:
                logger.info("Using DataParallel")
                self.model = torch.nn.DataParallel(self.model)
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def load_relations(self, file_path):
        """
        Load relations and preserve metadata (IDs and text).
        Returns a list of dictionaries containing ID and text.
        """
        relations = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) > 1 and parts[0].startswith('P'):
                        relations.append({'id': parts[0], 'text': " ".join(parts[1:])})
            logger.info(f"Loaded {len(relations)} relations")
        except FileNotFoundError:
            logger.error(f"File {file_path} not found.")
            raise
        return relations

    def generate_embeddings(self, relations, batch_size=32):
        """
        Generate embeddings for the given relations using the provided model.
        """
        embeddings = []
        with tqdm(total=len(relations), desc="Generating Embeddings") as pbar:
            for i in range(0, len(relations), batch_size):
                batch_texts = [relation['text'] for relation in relations[i:i+batch_size]]
                try:
                    with torch.no_grad():
                        if isinstance(self.model, torch.nn.DataParallel):
                            batch_embeddings = self.model.module.encode(
                                batch_texts,
                                convert_to_numpy=True,
                                show_progress_bar=False
                            )
                        else:
                            batch_embeddings = self.model.encode(
                                batch_texts,
                                convert_to_numpy=True,
                                show_progress_bar=False
                            )
                    embeddings.append(batch_embeddings)
                    pbar.update(len(batch_texts))

                    # Reduce memory usage
                    if i % 1000 == 0:
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error in batch {i}: {e}")
                    continue
        return np.vstack(embeddings)

    def save_embeddings(self, embeddings, relations, output_dir):
        """
        Save embeddings and metadata to separate files for traceability.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save embeddings
        embeddings_file = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_file, embeddings)
        logger.info(f"Embeddings saved to {embeddings_file}")

        # Save metadata
        metadata_file = os.path.join(output_dir, "metadata.txt")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for relation in relations:
                f.write(f"{relation['id']}\t{relation['text']}\n")
        logger.info(f"Metadata saved to {metadata_file}")

def main():
    try:
        generator = EmbeddingGenerator()

        input_path = "wikidata5m_relation.txt"
        output_dir = "wikidata_embedding_relations"

        # Load relations
        relations = generator.load_relations(input_path)
        if not relations:
            logger.error("No relations loaded. Exiting.")
            return

        # Generate embeddings
        embeddings = generator.generate_embeddings(relations)
        logger.info(f"Generated embeddings for {len(relations)} relations")

        # Save embeddings and metadata
        generator.save_embeddings(embeddings, relations, output_dir)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
        logger.info(f"Process completed in {time.time() - start_time:.2f} seconds")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        torch.cuda.empty_cache()
