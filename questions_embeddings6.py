import os
import shutil
import numpy as np
import pandas as pd
import nltk
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import requests
import time
import torch
from transformers import AutoTokenizer, AutoModel
import csv
import traceback
import json

class EmbeddingsGenerator:
    def __init__(self):
        # Initialize RoBERTa tokenizer and model
        print("Initializing RoBERTa tokenizer and model for embeddings...")
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = AutoModel.from_pretrained('roberta-base')

        # Set the model to evaluation mode
        self.model.eval()

        # Download NLTK data files if not already present
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

        # Initialize ChromaDB for embeddings
        self.initialize_chromadb()

        # Flag to check if embedding shape has been printed
        self.embedding_shape_printed = False

    def initialize_chromadb(self):
        print("Initializing ChromaDB for embeddings...")

        # Handle cases where __file__ may not be defined
        if '__file__' in globals():
            current_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            current_dir = os.getcwd()

        # Questions
        persist_directory_questions = os.path.join(
            current_dir, 'chromadb_store_questions'
        )

        # Delete the persist directory if it exists to avoid dimension mismatch
        if os.path.exists(persist_directory_questions):
            shutil.rmtree(persist_directory_questions)
            print("Deleted existing persist directory for questions to avoid dimension mismatch.")

        os.makedirs(persist_directory_questions, exist_ok=True)
        self.client_questions = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_questions
        ))

        # Create a new collection with 'cosine' distance function using metadata
        self.collection_questions = self.client_questions.create_collection(
            name="questions_collection",
            metadata={'hnsw:space': 'cosine'}
        )
        print("Created new 'questions_collection' with cosine similarity.")

        # Entities
        persist_directory_entities = os.path.join(
            current_dir, 'chromadb_store_wikidata_entities'
        )

        # Delete the persist directory if it exists
        if os.path.exists(persist_directory_entities):
            shutil.rmtree(persist_directory_entities)
            print("Deleted existing persist directory for entities to avoid dimension mismatch.")

        os.makedirs(persist_directory_entities, exist_ok=True)
        self.client_entities = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_entities
        ))

        # Create a new collection with 'cosine' distance function using metadata
        self.collection_entities = self.client_entities.create_collection(
            name="wikidata_entities",
            metadata={'hnsw:space': 'cosine'}
        )
        print("Created new 'wikidata_entities' collection with cosine similarity.")

    def store_question_embeddings(self, items):
        # items: list of tuples (question_id, question_text)
        batch_size = 8  # Adjust as needed based on your system's memory

        start_time = time.time()
        for start_idx in tqdm(
            range(0, len(items), batch_size),
            desc="Storing question embeddings"
        ):
            end_idx = min(start_idx + batch_size, len(items))
            batch_items = items[start_idx:end_idx]
            batch_texts = [text for _, text in batch_items]
            batch_ids = [str(question_id) for question_id, _ in batch_items]

            # Generate embeddings for the batch
            batch_embeddings = self.generate_embeddings(batch_texts)

            # Normalize embeddings for cosine similarity
            batch_embeddings = [emb / np.linalg.norm(emb) if np.linalg.norm(emb) != 0 else emb for emb in batch_embeddings]

            # Prepare metadata
            batch_metadatas = [{'text': text} for _, text in batch_items]

            # Add to ChromaDB
            try:
                self.collection_questions.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
            except Exception as e:
                print(f"Error adding question embeddings to ChromaDB: {e}")
                continue

        # Rebuild the index after adding embeddings
        self.collection_questions.create_index()

        end_time = time.time()
        print(f"Finished storing question embeddings in ChromaDB. Time taken: {end_time - start_time:.2f} seconds.")
        print(f"Total questions stored: {self.collection_questions.count()}")

    def store_wikidata_embeddings(self, items):
        # items: list of tuples (entity_id, entity_text)
        batch_size = 8  # Adjust as needed based on your system's memory

        collection = self.collection_entities

        start_time = time.time()
        for start_idx in tqdm(
            range(0, len(items), batch_size),
            desc="Storing entity embeddings"
        ):
            end_idx = min(start_idx + batch_size, len(items))
            batch_items = items[start_idx:end_idx]
            batch_texts = [text for _, text in batch_items]
            batch_ids = [str(entity_id) for entity_id, _ in batch_items]

            # Generate embeddings for the batch
            batch_embeddings = self.generate_embeddings(batch_texts)

            # Normalize embeddings for cosine similarity
            batch_embeddings = [emb / np.linalg.norm(emb) if np.linalg.norm(emb) != 0 else emb for emb in batch_embeddings]

            # Prepare metadata
            batch_metadatas = [{'text': text} for _, text in batch_items]

            # Add to ChromaDB
            try:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
            except Exception as e:
                print(f"Error adding embeddings to ChromaDB: {e}")
                continue

        # Rebuild the index after adding embeddings
        collection.create_index()

        end_time = time.time()
        print(f"Finished storing embeddings in ChromaDB. Time taken: {end_time - start_time:.2f} seconds.")
        print(f"Total entities stored: {collection.count()}")

    def generate_embeddings(self, texts):
        # Tokenize and encode the texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get the embeddings from the last hidden state
            embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]

            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask  # Shape: [batch_size, hidden_size]

            # Convert to list of embeddings
            embeddings_list = mean_pooled.cpu().numpy()

        # Print the shape of the embeddings for debugging (only once)
        if not self.embedding_shape_printed:
            print(f"Generated embeddings of shape: {mean_pooled.shape}")
            self.embedding_shape_printed = True

        return embeddings_list.tolist()

    def close(self):
        # Ensure data is persisted before closing
        self.client_questions.persist()
        self.client_entities.persist()

    def get_entity_full_text(self, entity_id):
        # Fetch labels and descriptions for the entity
        url = (
            'https://www.wikidata.org/w/api.php?action=wbgetentities'
            f'&ids={entity_id}&format=json&languages=en&props=labels|descriptions'
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            entities = data.get('entities', {})
            entity_data = entities.get(entity_id, {})
            label_data = entity_data.get('labels', {})
            description_data = entity_data.get('descriptions', {})
            label = label_data.get('en', {}).get('value', '')
            if not label and label_data:
                label = next(iter(label_data.values())).get('value', '')
            description = description_data.get('en', {}).get('value', '')
            if not description and description_data:
                description = next(iter(description_data.values())).get('value', '')
            text = f"{label}. {description}.".strip()
            return text
        else:
            print(
                f"Failed to retrieve data for entity {entity_id}. "
                f"HTTP status code: {response.status_code}"
            )
            return ''

    def get_entity_label(self, entity_id):
        # Fetch entity label
        url = (
            'https://www.wikidata.org/w/api.php?action=wbgetentities'
            f'&ids={entity_id}&format=json&props=labels&languages=en'
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            entities = data.get('entities', {})
            entity_data = entities.get(entity_id, {})
            label_data = entity_data.get('labels', {})
            label = label_data.get('en', {}).get('value', '')
            if not label and label_data:
                label = next(iter(label_data.values())).get('value', '')
            time.sleep(0.05)  # Be polite to the API
            return label
        else:
            print(
                f"Failed to retrieve label for entity {entity_id}. "
                f"HTTP status code: {response.status_code}"
            )
            return ''

def main():
    print("Script is starting...")
    total_start_time = time.time()
    try:
        print("Initializing embeddings...")
        embeddings_generator = EmbeddingsGenerator()
        print("Embeddings initialized successfully.")

        # Read data from the CSV file
        csv_file_path = 'data.csv'  # Update with your CSV file path
        df = pd.read_csv(csv_file_path)

        # Verify that the required columns are present
        required_columns = ['question', 'wikidata entities']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV file must contain the columns: {required_columns}")
            return

        # Extract questions and Wikidata entities
        questions_series = df['question']
        wikidata_entities_series = df['wikidata entities']

        # Prepare items to store: list of tuples (question_id, question_text)
        question_items = []
        for idx, question in enumerate(questions_series):
            question_id = f"question_{idx}"
            question_text = str(question)
            question_items.append((question_id, question_text))

        # Store question embeddings in ChromaDB
        print("Storing question embeddings...")
        start_time = time.time()
        embeddings_generator.store_question_embeddings(question_items)
        end_time = time.time()
        print(f"Question embeddings stored successfully in ChromaDB. Time taken: {end_time - start_time:.2f} seconds.\n")

        # Collect all unique Wikidata entity IDs
        print("Collecting unique Wikidata entity IDs...")
        start_time = time.time()
        entity_ids = set()
        for entity_list in wikidata_entities_series:
            if pd.notna(entity_list):
                ids = [eid.strip() for eid in entity_list.split(',') if eid.strip().startswith('Q')]
                entity_ids.update(ids)
            else:
                continue

        end_time = time.time()
        print(f"Collected {len(entity_ids)} unique entity IDs. Time taken: {end_time - start_time:.2f} seconds.\n")

        # Check if entity_ids is empty
        if not entity_ids:
            print("No entity IDs were collected. Please check the CSV data.")
            return

        # Fetch and store Wikidata entity embeddings
        print("Fetching and storing Wikidata entity embeddings...")
        start_time = time.time()
        entity_items = []
        for entity_id in tqdm(entity_ids, desc="Processing entities"):
            text = embeddings_generator.get_entity_full_text(entity_id)
            if not text:
                text = entity_id  # Use ID if text is empty
            entity_items.append((entity_id, text))
        embeddings_generator.store_wikidata_embeddings(entity_items)
        end_time = time.time()
        print(f"Wikidata entity embeddings stored successfully. Time taken: {end_time - start_time:.2f} seconds.\n")

        # Querying embeddings and saving results
        print("Querying embeddings and saving results...")
        start_time = time.time()

        # Collect data for BFS script
        question_similar_data = {}

        # Now, for each question, retrieve its embedding and find top similar entities
        for idx, (question_id, question_text) in enumerate(tqdm(question_items, desc="Processing questions")):
            # Retrieve question embedding
            result_data = embeddings_generator.collection_questions.get(ids=[question_id])
            if result_data and 'embeddings' in result_data and len(result_data['embeddings']) > 0:
                question_embedding = np.array(result_data['embeddings'][0])

                # Normalize the query embedding for cosine similarity
                question_embedding = question_embedding / np.linalg.norm(question_embedding) if np.linalg.norm(question_embedding) != 0 else question_embedding

                # Query the entities collection
                n_results = 10  # Number of top similar items

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
                if similar_entities and 'ids' in similar_entities and len(similar_entities['ids']) > 0:
                    for sim_id, distance in zip(
                        similar_entities['ids'][0],
                        similar_entities['distances'][0]
                    ):
                        entity_id = sim_id
                        label = embeddings_generator.get_entity_label(entity_id)
                        similarity_score = 1 - distance  # Since distance is between 0 and 2 for cosine
                        similar_entities_data.append({
                            'id': entity_id,
                            'label': label,
                            'similarity_score': similarity_score
                        })

                # Save data for BFS script
                question_similar_data[question_id] = {
                    'question_text': question_text,
                    'similar_entities': similar_entities_data
                }
            else:
                print(f"No embedding found for {question_id}.")

        # Save the similar entities data to a JSON file
        with open('question_similar_data.json', 'w', encoding='utf-8') as f:
            json.dump(question_similar_data, f, indent=4)

        end_time = time.time()
        print(f"Finished querying embeddings and saving results. Time taken: {end_time - start_time:.2f} seconds.\n")

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
