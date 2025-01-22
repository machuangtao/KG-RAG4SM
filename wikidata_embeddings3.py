import os
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import requests
import time
import traceback

class EmbeddingsGenerator:
    def __init__(self):
        # Initialize ChromaDB for embeddings
        self.initialize_chromadb()
        # Initialize caches for labels
        self.entity_label_cache = {}
        self.relationship_label_cache = {}

    def initialize_chromadb(self):
        print("Initializing ChromaDB for embeddings...")

        # Get the current directory
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")

        # Adjust paths to avoid duplicate 'layers'
        if os.path.basename(current_dir) == 'layers':
            base_dir = current_dir
        else:
            base_dir = os.path.join(current_dir, 'layers')

        # Path for val_emed_questions embeddings
        persist_directory_questions = os.path.join(
            base_dir, 'chromadb_store_val_emed_questions'
        )
        print(f"Questions embeddings directory: {persist_directory_questions}")

        if not os.path.exists(persist_directory_questions):
            print(f"Questions embeddings directory not found: {persist_directory_questions}")
            raise FileNotFoundError(f"Directory not found: {persist_directory_questions}")

        # Initialize ChromaDB client for questions
        self.client_questions = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_questions
        ))

        # Get the questions collection
        try:
            collection_names = [col.name for col in self.client_questions.list_collections()]
            if "val_emed_questions_collection" in collection_names:
                self.collection_questions = self.client_questions.get_collection(
                    name="val_emed_questions_collection"
                )
                print("Questions collection loaded successfully.")
            else:
                print("Questions collection 'val_emed_questions_collection' does not exist.")
                raise Exception("Questions collection not found.")
        except Exception as e:
            print(f"Error retrieving questions collection: {e}")
            raise e

        # Path for wikidata_entities embeddings
        persist_directory_entities = os.path.join(
            base_dir, 'chromadb_store_wikidata_entities'
        )
        print(f"Entities embeddings directory: {persist_directory_entities}")

        if not os.path.exists(persist_directory_entities):
            print(f"Entities embeddings directory not found: {persist_directory_entities}")
            raise FileNotFoundError(f"Directory not found: {persist_directory_entities}")

        # Initialize ChromaDB client for entities
        self.client_entities = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_entities
        ))

        # Load the entities collection
        try:
            collection_names = [col.name for col in self.client_entities.list_collections()]
            if "wikidata_entities" in collection_names:
                self.collection_entities = self.client_entities.get_collection(
                    name="wikidata_entities"
                )
                print("Entities collection loaded successfully.")
            else:
                print("Entities collection 'wikidata_entities' does not exist.")
                raise Exception("Entities collection not found.")
        except Exception as e:
            print(f"Error retrieving entities collection: {e}")
            raise e

        # Path for wikidata_relationships embeddings
        persist_directory_relationships = os.path.join(
            base_dir, 'chromadb_store_wikidata_relationships'
        )
        print(f"Relationships embeddings directory: {persist_directory_relationships}")

        if not os.path.exists(persist_directory_relationships):
            print(f"Relationships embeddings directory not found: {persist_directory_relationships}")
            raise FileNotFoundError(f"Directory not found: {persist_directory_relationships}")

        # Initialize ChromaDB client for relationships
        self.client_relationships = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_relationships
        ))

        # Load the relationships collection
        try:
            collection_names = [col.name for col in self.client_relationships.list_collections()]
            if "wikidata_relationships" in collection_names:
                self.collection_relationships = self.client_relationships.get_collection(
                    name="wikidata_relationships"
                )
                print("Relationships collection loaded successfully.")
            else:
                print("Relationships collection 'wikidata_relationships' does not exist.")
                raise Exception("Relationships collection not found.")
        except Exception as e:
            print(f"Error retrieving relationships collection: {e}")
            raise e

    def get_entity_label(self, entity_id):
        # Check cache first
        if entity_id in self.entity_label_cache:
            return self.entity_label_cache[entity_id]
        # Fetch entity label from Wikidata
        try:
            url = (
                'https://www.wikidata.org/w/api.php?action=wbgetentities'
                f'&ids={entity_id}&format=json&props=labels&languages=en'
            )
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                entities = data.get('entities', {})
                entity_data = entities.get(entity_id, {})
                label_data = entity_data.get('labels', {})
                label = label_data.get('en', {}).get('value', '')
                if not label and label_data:
                    label = next(iter(label_data.values())).get('value', '')
                # Cache the result
                self.entity_label_cache[entity_id] = label
                return label
            else:
                print(
                    f"Failed to retrieve label for entity {entity_id}. "
                    f"HTTP status code: {response.status_code}"
                )
                return ''
        except Exception as e:
            print(f"Exception occurred while fetching label for entity {entity_id}: {e}")
            return ''

    def get_relationship_label(self, relationship_id):
        # Check cache first
        if relationship_id in self.relationship_label_cache:
            return self.relationship_label_cache[relationship_id]
        # Fetch relationship label from Wikidata
        try:
            url = (
                'https://www.wikidata.org/w/api.php?action=wbgetentities'
                f'&ids={relationship_id}&format=json&props=labels&languages=en'
            )
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                entities = data.get('entities', {})
                relationship_data = entities.get(relationship_id, {})
                label_data = relationship_data.get('labels', {})
                label = label_data.get('en', {}).get('value', '')
                if not label and label_data:
                    label = next(iter(label_data.values())).get('value', '')
                # Cache the result
                self.relationship_label_cache[relationship_id] = label
                return label
            else:
                print(
                    f"Failed to retrieve label for relationship {relationship_id}. "
                    f"HTTP status code: {response.status_code}"
                )
                return ''
        except Exception as e:
            print(f"Exception occurred while fetching label for relationship {relationship_id}: {e}")
            return ''

    def close(self):
        # Ensure data is persisted before closing
        self.client_questions.persist()
        self.client_entities.persist()
        self.client_relationships.persist()

def main():
    print("Script is starting...")
    total_start_time = time.time()
    try:
        print("Initializing embeddings and loading collections...")
        embeddings_generator = EmbeddingsGenerator()
        print("Embeddings initialized successfully.")

        # Retrieve all question IDs
        question_data = embeddings_generator.collection_questions.get()
        if 'ids' in question_data and question_data['ids']:
            question_ids = question_data['ids']
            print(f"Total number of questions: {len(question_ids)}")
        else:
            print("No question IDs found in the collection.")
            return

        # Initialize a dictionary to store results
        results = {}

        # Open the file to save the results
        with open('emed_val_results.txt', 'w', encoding='utf-8') as result_file:
            # Process each question
            for question_id in tqdm(question_ids, desc="Processing questions"):
                # Retrieve question embedding and metadata
                question_data = embeddings_generator.collection_questions.get(
                    ids=[question_id]
                )
                if question_data and 'embeddings' in question_data and len(question_data['embeddings']) > 0:
                    question_embedding = np.array(question_data['embeddings'][0])
                    question_text = question_data['metadatas'][0]['text']

                    # Normalize the query embedding for cosine similarity
                    question_embedding = question_embedding / np.linalg.norm(question_embedding) if np.linalg.norm(question_embedding) != 0 else question_embedding

                    # Query the entities collection
                    n_results = 10  # Number of top similar entities
                    try:
                        similar_entities = embeddings_generator.collection_entities.query(
                            query_embeddings=[question_embedding.tolist()],
                            n_results=n_results
                        )
                    except Exception as e:
                        print(f"Error querying entities collection: {e}")
                        similar_entities = None

                    # Collect similar entities data
                    similar_entities_data = []
                    if similar_entities and 'ids' in similar_entities and similar_entities['ids']:
                        for sim_id, distance in zip(
                            similar_entities['ids'][0],
                            similar_entities['distances'][0]
                        ):
                            entity_id = sim_id
                            label = embeddings_generator.get_entity_label(entity_id)
                            similarity_score = 1 - distance / 2  # Cosine distance is between 0 and 2
                            similar_entities_data.append({
                                'id': entity_id,
                                'label': label,
                                'similarity_score': similarity_score
                            })
                    else:
                        print(f"No similar entities found for question {question_id}")
                        similar_entities_data = []

                    # Query the relationships collection
                    try:
                        similar_relationships = embeddings_generator.collection_relationships.query(
                            query_embeddings=[question_embedding.tolist()],
                            n_results=n_results
                        )
                    except Exception as e:
                        print(f"Error querying relationships collection: {e}")
                        similar_relationships = None

                    # Collect similar relationships data
                    similar_relationships_data = []
                    if similar_relationships and 'ids' in similar_relationships and similar_relationships['ids']:
                        for sim_id, distance in zip(
                            similar_relationships['ids'][0],
                            similar_relationships['distances'][0]
                        ):
                            relationship_id = sim_id
                            label = embeddings_generator.get_relationship_label(relationship_id)
                            similarity_score = 1 - distance / 2  # Cosine distance is between 0 and 2
                            similar_relationships_data.append({
                                'id': relationship_id,
                                'label': label,
                                'similarity_score': similarity_score
                            })
                    else:
                        print(f"No similar relationships found for question {question_id}")
                        similar_relationships_data = []

                    # Save results for the current question
                    results[question_id] = {
                        'question_text': question_text,
                        'similar_entities': similar_entities_data,
                        'similar_relationships': similar_relationships_data
                    }

                    # Write top 10 similar entities and relationships to the file and print to terminal
                    output_str = f"\nQuestion ID: {question_id}\nQuestion Text: {question_text}\n"

                    output_str += "Top 10 Similar Entities:\n"
                    if similar_entities_data:
                        for idx, entity in enumerate(similar_entities_data, 1):
                            output_str += f"{idx}. {entity['label']} ({entity['id']}) - Similarity Score: {entity['similarity_score']:.4f}\n"
                    else:
                        output_str += "No similar entities found.\n"

                    output_str += "Top 10 Similar Relationships:\n"
                    if similar_relationships_data:
                        for idx, relationship in enumerate(similar_relationships_data, 1):
                            output_str += f"{idx}. {relationship['label']} ({relationship['id']}) - Similarity Score: {relationship['similarity_score']:.4f}\n"
                    else:
                        output_str += "No similar relationships found.\n"

                    result_file.write(output_str)
                    print(output_str)
                else:
                    print(f"No embedding found for question {question_id}.")

        # Save the results to a JSON file
        with open('emed_val_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()
    finally:
        if 'embeddings_generator' in locals():
            embeddings_generator.close()
        total_end_time = time.time()
        print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
