import os
import json
import chromadb
from chromadb.config import Settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import datetime
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("similarity_finder.log", mode='w')  # Overwrite log file on each run
    ]
)
logger = logging.getLogger(__name__)

class SimilarityFinder:
    def __init__(self, entity_mapping_path='entity_mapping.json', property_mapping_path='property_mapping.json'):
        """
        Initializes the SimilarityFinder by loading ChromaDB clients and mapping files.

        Parameters:
            - entity_mapping_path (str): Path to the entity mapping JSON file.
            - property_mapping_path (str): Path to the property mapping JSON file.
        """
        logger.info("Initializing SimilarityFinder...")
        self.initialize_chromadb_clients()
        self.entity_mapping = self.load_mapping(entity_mapping_path, 'Entity')
        self.property_mapping = self.load_mapping(property_mapping_path, 'Property')
        self.load_and_preprocess_embeddings()

    def initialize_chromadb_clients(self):
        """
        Initializes ChromaDB clients for questions, entities, and relationships.
        """
        logger.info("Initializing ChromaDB clients...")
        base_dir = os.getcwd()
        try:
            # Initialize ChromaDB clients
            self.client_questions = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=os.path.join(base_dir, 'cms_ques_embedding_full')
            ))
            self.client_entities = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=os.path.join(base_dir, 'chromadb_store_wikidata_entities')
            ))
            self.client_relationships = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=os.path.join(base_dir, 'chromadb_store_wikidata_relationships')
            ))

            # Get collections
            self.collection_questions = self.client_questions.get_collection("sentence_embedding_cms_questions_full")
            self.collection_entities = self.client_entities.get_collection("wikidata_entities")
            self.collection_relationships = self.client_relationships.get_collection("wikidata_relationships")
            logger.info("ChromaDB collections loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB clients: {e}")
            traceback.print_exc()
            raise e

    def load_mapping(self, filepath, mapping_type='Entity'):
        """
        Loads a mapping from a JSON file.

        Parameters:
            - filepath (str): Path to the mapping JSON file.
            - mapping_type (str): Type of mapping ('Entity' or 'Property') for logging purposes.

        Returns:
            - dict: The loaded mapping dictionary.
        """
        logger.info(f"Loading {mapping_type} mapping from {filepath}...")
        if not os.path.exists(filepath):
            logger.error(f"{mapping_type} mapping file not found at {filepath}")
            raise FileNotFoundError(f"{mapping_type} mapping file not found at {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        logger.info(f"Loaded {mapping_type} mapping with {len(mapping)} entries.")
        return mapping

    def load_and_preprocess_embeddings(self):
        """
        Loads and preprocesses embeddings from ChromaDB collections.
        """
        try:
            # Load embeddings from collections
            logger.info("Loading embeddings from collections...")
            self.question_embeddings, self.question_ids, self.question_texts = self._load_embeddings(self.collection_questions, "questions")
            self.entity_embeddings, self.entity_ids, self.entity_texts = self._load_embeddings(self.collection_entities, "entities")
            self.relationship_embeddings, self.relationship_ids, self.relationship_texts = self._load_embeddings(self.collection_relationships, "relationships")

            # Print dimensions for debugging
            logger.info(f"Question embeddings dimension: {self.question_embeddings.shape}")
            logger.info(f"Entity embeddings dimension: {self.entity_embeddings.shape}")
            logger.info(f"Relationship embeddings dimension: {self.relationship_embeddings.shape}")

            # Normalize embeddings
            self.question_embeddings = self.normalize_embeddings(self.question_embeddings)
            self.entity_embeddings = self.normalize_embeddings(self.entity_embeddings)
            self.relationship_embeddings = self.normalize_embeddings(self.relationship_embeddings)
            logger.info("Embeddings loaded and normalized successfully.")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            traceback.print_exc()
            raise e

    def _load_embeddings(self, collection, collection_name):
        """
        Helper function to load embeddings from a ChromaDB collection.

        Parameters:
            - collection: The ChromaDB collection to load.
            - collection_name (str): Name of the collection for logging.

        Returns:
            - tuple: (embeddings, ids, texts)
        """
        logger.info(f"Loading embeddings from collection '{collection_name}'...")
        try:
            data = collection.get()
            if not data['ids']:
                logger.warning(f"No data found in the '{collection_name}' collection.")
                return np.array([]), [], []
            embeddings = np.array(data['embeddings']).astype('float32')
            ids = data['ids']
            texts = [meta.get('text', '') for meta in data['metadatas']]
            logger.info(f"Loaded {len(ids)} embeddings from '{collection_name}' collection.")
            return embeddings, ids, texts
        except Exception as e:
            logger.error(f"Error loading '{collection_name}' collection: {e}")
            return np.array([]), [], []

    def normalize_embeddings(self, embeddings):
        """
        Normalizes embeddings to unit vectors.

        Parameters:
            - embeddings (np.ndarray): The embeddings to normalize.

        Returns:
            - np.ndarray: The normalized embeddings.
        """
        if embeddings.size == 0:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def find_similar_items(self, top_k=10, batch_size=8):
        """
        Processes all questions in batches to find similar entities and relationships.

        Parameters:
            - top_k (int): Number of top similar items to retrieve.
            - batch_size (int): Number of questions to process in each batch.

        Returns:
            - list: A list of question results containing similar entities and relationships.
        """
        try:
            if self.question_embeddings.size == 0:
                logger.warning("No question embeddings to process.")
                return []
            total_questions = len(self.question_ids)
            results = []

            logger.info(f"Processing {total_questions} questions in batches of {batch_size}.")
            for batch_start in tqdm(range(0, total_questions, batch_size), desc="Processing batches"):
                batch_end = min(batch_start + batch_size, total_questions)
                batch_embeddings = self.question_embeddings[batch_start:batch_end]
                batch_ids = self.question_ids[batch_start:batch_end]
                batch_texts = self.question_texts[batch_start:batch_end]

                # Similar Entities
                entity_similarities = cosine_similarity(batch_embeddings, self.entity_embeddings)
                # Similar Relationships
                relationship_similarities = cosine_similarity(batch_embeddings, self.relationship_embeddings)

                for idx in range(batch_end - batch_start):
                    question_id = batch_ids[idx]
                    question_text = batch_texts[idx]
                    # Process entities
                    top_entity_indices = np.argsort(-entity_similarities[idx])[:top_k]
                    similar_entities = [
                        {
                            'rank': rank + 1,
                            'wikidata_id': self.entity_ids[i],
                            'text': self.entity_texts[i],
                            'similarity_score': float(entity_similarities[idx][i])  # Convert to Python float
                        }
                        for rank, i in enumerate(top_entity_indices)
                    ]
                    # Process relationships
                    top_relationship_indices = np.argsort(-relationship_similarities[idx])[:top_k]
                    similar_relationships = [
                        {
                            'rank': rank + 1,
                            'property_id': self.relationship_ids[i],
                            'text': self.relationship_texts[i],
                            'similarity_score': float(relationship_similarities[idx][i])  # Convert to Python float
                        }
                        for rank, i in enumerate(top_relationship_indices)
                    ]
                    results.append({
                        'question_id': question_id,
                        'question_text': question_text,
                        'similar_items': {
                            'entities': similar_entities,
                            'relationships': similar_relationships
                        }
                    })
            return results
        except Exception as e:
            logger.error(f"Error in find_similar_items: {e}")
            traceback.print_exc()
            return []

    def save_results(self, results, output_path='results.json'):
        """
        Saves the similarity results to a JSON file.

        Parameters:
            - results (list): The list of similarity results to save.
            - output_path (str): File path for saving results.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}.")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            traceback.print_exc()

def main():
    """
    The main execution function.
    """
    try:
        finder = SimilarityFinder(entity_mapping_path='entity_mapping.json', property_mapping_path='property_mapping.json')
        results = finder.find_similar_items(top_k=10, batch_size=8)
        finder.save_results(results, output_path='similar_items.json')
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
