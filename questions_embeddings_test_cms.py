import os
import numpy as np
import pandas as pd
import nltk
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import requests
import time
import json
import traceback
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingsGenerator:
    def __init__(self):
        # Initialize RoBERTa tokenizer and model
        print("Initializing RoBERTa tokenizer and model for embeddings...")
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = AutoModel.from_pretrained('roberta-base')
        self.model.eval()

        # Download NLTK data files if not already present
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

        # Initialize ChromaDB for embeddings
        self.embedding_shape_printed = False  
        self.initialize_chromadb()

    def initialize_chromadb(self):
        print("Initializing ChromaDB for embeddings...")

        current_dir = os.getcwd()

        # Paths for the ChromaDB directories
        persist_directory_questions = os.path.join(
            current_dir, 'chromadb_store_test_cms_questions'
        )
        persist_directory_entities = os.path.join(
            current_dir, 'chromadb_store_wikidata_entities_test_cms'
        )
        persist_directory_relationships = os.path.join(
            current_dir, 'chromadb_store_wikidata_relationships_test_cms'
        )

        # Ensure directories exist
        os.makedirs(persist_directory_questions, exist_ok=True)
        os.makedirs(persist_directory_entities, exist_ok=True)
        os.makedirs(persist_directory_relationships, exist_ok=True)

        # Initialize ChromaDB clients
        self.client_questions = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_questions
        ))
        self.client_entities = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_entities
        ))
        self.client_relationships = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_relationships
        ))

        # Delete existing collections if they exist to avoid dimension mismatches
        try:
            self.client_questions.delete_collection(name="test_cms_questions_collection")
            print("Deleted existing 'test_cms_questions_collection' to avoid dimension mismatches.")
        except Exception:
            pass

        try:
            self.client_entities.delete_collection(name="wikidata_entities_test_cms")
            print("Deleted existing 'wikidata_entities_test_cms' collection to avoid dimension mismatches.")
        except Exception:
            pass

        try:
            self.client_relationships.delete_collection(name="wikidata_relationships_test_cms")
            print("Deleted existing 'wikidata_relationships_test_cms' collection to avoid dimension mismatches.")
        except Exception:
            pass

        # Create new collections
        self.collection_questions = self.client_questions.create_collection(
            name="test_cms_questions_collection"
        )
        print("Created new 'test_cms_questions_collection'.")

        self.collection_entities = self.client_entities.create_collection(
            name="wikidata_entities_test_cms"
        )
        print("Created new 'wikidata_entities_test_cms' collection.")

        self.collection_relationships = self.client_relationships.create_collection(
            name="wikidata_relationships_test_cms"
        )
        print("Created new 'wikidata_relationships_test_cms' collection.")

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
            embeddings = outputs.last_hidden_state  

            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask  # Shape: [batch_size, hidden_size]

            # Normalize the embeddings
            mean_pooled = mean_pooled / mean_pooled.norm(dim=1, keepdim=True)

            # Convert to list of embeddings
            embeddings_list = mean_pooled.cpu().numpy()

        # Print the shape of embeddings once for verification
        if not self.embedding_shape_printed:
            print(f"Embedding shape: {embeddings_list.shape}")
            self.embedding_shape_printed = True

        return embeddings_list.tolist()

    def store_question_embeddings(self, items):
        batch_size = 8  

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

            # Prepare metadata
            batch_metadatas = [{'text': text} for _, text in batch_items]

            # Add to ChromaDB
            self.collection_questions.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts
            )

        end_time = time.time()
        print(f"Finished storing question embeddings in ChromaDB. Time taken: {end_time - start_time:.2f} seconds.")
        print(f"Total questions stored: {self.collection_questions.count()}")

    def store_wikidata_embeddings(self, items, is_entity=True):
        collection = self.collection_entities if is_entity else self.collection_relationships

        batch_size = 8

        start_time = time.time()
        for start_idx in tqdm(
            range(0, len(items), batch_size),
            desc=f"Storing {'entity' if is_entity else 'property'} embeddings"
        ):
            end_idx = min(start_idx + batch_size, len(items))
            batch_items = items[start_idx:end_idx]
            batch_texts = [text for _, text in batch_items]
            batch_ids = [str(entity_id) for entity_id, _ in batch_items]

            # Generate embeddings for the batch
            batch_embeddings = self.generate_embeddings(batch_texts)

            # Prepare metadata
            batch_metadatas = [{'text': text} for _, text in batch_items]

            # Add to ChromaDB
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts
            )

        end_time = time.time()
        entity_type = 'entities' if is_entity else 'properties'
        print(f"Finished storing embeddings in ChromaDB. Time taken: {end_time - start_time:.2f} seconds.")
        print(f"Total {entity_type} stored: {collection.count()}")

    def close(self):
        # Ensure data is persisted before closing
        self.client_questions.persist()
        self.client_entities.persist()
        self.client_relationships.persist()

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

    def get_property_full_text(self, prop_id):
        # Fetch labels and descriptions for the property
        url = (
            'https://www.wikidata.org/w/api.php?action=wbgetentities'
            f'&ids={prop_id}&format=json&languages=en&props=labels|descriptions'
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            entities = data.get('entities', {})
            prop_data = entities.get(prop_id, {})
            label_data = prop_data.get('labels', {})
            description_data = prop_data.get('descriptions', {})
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
                f"Failed to retrieve data for property {prop_id}. "
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
            time.sleep(0.01)  # Be polite to the API
            return label
        else:
            print(
                f"Failed to retrieve label for entity {entity_id}. "
                f"HTTP status code: {response.status_code}"
            )
            return ''

    def get_property_label(self, prop_id):
        # Fetch property label
        url = (
            'https://www.wikidata.org/w/api.php?action=wbgetentities'
            f'&ids={prop_id}&format=json&props=labels&languages=en'
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            entities = data.get('entities', {})
            prop_data = entities.get(prop_id, {})
            label_data = prop_data.get('labels', {})
            label = label_data.get('en', {}).get('value', '')
            if not label and label_data:
                label = next(iter(label_data.values())).get('value', '')
            time.sleep(0.01)  # Be polite to the API
            return label
        else:
            print(
                f"Failed to retrieve label for property {prop_id}. "
                f"HTTP status code: {response.status_code}"
            )
            return ''

def get_wikidata_properties(limit=1000):
    # Fetch a list of property IDs from Wikidata
    print("Fetching Wikidata property IDs...")
    start_time = time.time()
    properties = []
    apcontinue = ''
    while len(properties) < limit:
        url = (
            'https://www.wikidata.org/w/api.php?action=query&list=allpages'
            f'&apnamespace=120&format=json&aplimit=500{apcontinue}'
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            pages = data.get('query', {}).get('allpages', [])
            for page in pages:
                title = page.get('title', '')
                if title.startswith('Property:'):
                    prop_id = title[len('Property:'):]
                    properties.append(prop_id)
            if 'continue' in data:
                apcontinue = '&apcontinue=' + data['continue']['apcontinue']
            else:
                break
        else:
            print(f"Failed to fetch properties. HTTP status code: {response.status_code}")
            break
    properties = properties[:limit]
    end_time = time.time()
    print(f"Finished fetching {len(properties)} property IDs. Time taken: {end_time - start_time:.2f} seconds.")
    return properties

def get_wikidata_entities(limit=5000):
    # Fetch a list of entity IDs from Wikidata
    print("Fetching additional Wikidata entity IDs...")
    start_time = time.time()
    entities = []
    apcontinue = ''
    while len(entities) < limit:
        url = (
            'https://www.wikidata.org/w/api.php?action=query&list=allpages'
            f'&apnamespace=0&format=json&aplimit=500{apcontinue}'
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            pages = data.get('query', {}).get('allpages', [])
            for page in pages:
                title = page.get('title', '')
                if title.startswith('Q'):
                    entity_id = title
                    entities.append(entity_id)
            if 'continue' in data:
                apcontinue = '&apcontinue=' + data['continue']['apcontinue']
            else:
                break
        else:
            print(f"Failed to fetch entities. HTTP status code: {response.status_code}")
            break
    entities = entities[:limit]
    end_time = time.time()
    print(f"Finished fetching {len(entities)} entity IDs. Time taken: {end_time - start_time:.2f} seconds.")
    return entities

def main():
    print("Script is starting...")
    total_start_time = time.time()
    try:
        print("Initializing embeddings...")
        embeddings_generator = EmbeddingsGenerator()
        print("Embeddings initialized successfully.")

        # Path to your test_cms_q.xlsx file
        excel_file_path = '/app/datafinal/test_cms_q.xlsx'

        # Read data from the Excel file
        print(f"Reading data from {excel_file_path}...")
        df = pd.read_excel(excel_file_path)

        # Verify that the 'question' column is present
        if 'question' not in df.columns:
            print("The Excel file must contain a 'question' column.")
            return

        # Extract questions and any associated entities if available
        questions = df['question']
        wikidata_entities = df.get('wikidata entities', pd.Series([]))

        # Prepare items to store: list of tuples (question_id, question_text)
        question_items = []
        for idx, question in enumerate(questions):
            question_id = f"question_{idx}"
            question_items.append((question_id, str(question)))

        # Store question embeddings in ChromaDB
        print("Storing question embeddings...")
        start_time = time.time()
        embeddings_generator.store_question_embeddings(question_items)
        end_time = time.time()
        print(f"Question embeddings stored successfully in ChromaDB. Time taken: {end_time - start_time:.2f} seconds.\n")

        # Collect all unique Wikidata entity IDs from the dataset
        print("Collecting unique Wikidata entity IDs from the dataset...")
        start_time = time.time()
        entity_ids = set()
        for entity_list in wikidata_entities:
            if isinstance(entity_list, str):
                ids = [eid.strip() for eid in entity_list.split(',') if eid.strip().startswith('Q')]
                entity_ids.update(ids)
            else:
                continue

        # Fetch additional entities from Wikidata
        additional_entities = get_wikidata_entities(limit=5000)
        entity_ids.update(additional_entities)

        end_time = time.time()
        print(f"Collected {len(entity_ids)} unique entity IDs. Time taken: {end_time - start_time:.2f} seconds.\n")

        # Check if entity_ids is empty
        if not entity_ids:
            print("No entity IDs were collected. Please check the dataset.")
            return

        # Collect properties (relationships) from Wikidata
        property_ids = get_wikidata_properties(limit=1000)

        # Fetch and store Wikidata entity embeddings
        print("Fetching and storing Wikidata entity embeddings...")
        start_time = time.time()
        entity_items = []
        for entity_id in tqdm(entity_ids, desc="Processing entities"):
            text = embeddings_generator.get_entity_full_text(entity_id)
            if not text:
                text = entity_id  # Use ID if text is empty
            entity_items.append((entity_id, text))
        embeddings_generator.store_wikidata_embeddings(entity_items, is_entity=True)
        end_time = time.time()
        print(f"Wikidata entity embeddings stored successfully. Time taken: {end_time - start_time:.2f} seconds.\n")

        # Fetch and store Wikidata property embeddings
        print("Fetching and storing Wikidata property embeddings...")
        start_time = time.time()
        property_items = []
        for prop_id in tqdm(property_ids, desc="Processing properties"):
            text = embeddings_generator.get_property_full_text(prop_id)
            if not text:
                text = prop_id  # Use ID if text is empty
            property_items.append((prop_id, text))
        embeddings_generator.store_wikidata_embeddings(property_items, is_entity=False)
        end_time = time.time()
        print(f"Wikidata property embeddings stored successfully. Time taken: {end_time - start_time:.2f} seconds.\n")

        # Print the total counts for debugging
        total_entities = embeddings_generator.collection_entities.count()
        total_relationships = embeddings_generator.collection_relationships.count()
        print(f"Total entities in collection: {total_entities}")
        print(f"Total relationships in collection: {total_relationships}\n")

        # Querying embeddings and writing results
        print("Querying embeddings and writing results to JSON file...")
        start_time = time.time()

        results = []

        # For each question, retrieve its embedding and find top similar entities and relationships
        for idx, question in enumerate(tqdm(questions, desc="Processing questions")):
            question_id = f"question_{idx}"

            # Retrieve question embedding
            result_data = embeddings_generator.collection_questions.get(ids=[question_id])
            if result_data and 'embeddings' in result_data and len(result_data['embeddings']) > 0:
                question_embedding = np.array(result_data['embeddings'][0])

                # Normalize the question embedding
                question_embedding = question_embedding / np.linalg.norm(question_embedding)

                # Query the entities collection
                n_results = 10  

                try:
                    similar_entities = embeddings_generator.collection_entities.query(
                        query_embeddings=[question_embedding.tolist()],
                        n_results=n_results
                    )
                except Exception as e:
                    print(f"Error querying entities collection: {e}")
                    similar_entities = None

                # Query the relationships collection
                try:
                    similar_relationships = embeddings_generator.collection_relationships.query(
                        query_embeddings=[question_embedding.tolist()],
                        n_results=n_results
                    )
                except Exception as e:
                    print(f"Error querying relationships collection: {e}")
                    similar_relationships = None

                # Build the result dictionary
                result_entry = {
                    'question_id': question_id,
                    'question_text': question,
                    'similar_entities': [],
                    'similar_relationships': []
                }

                # Add similar entities
                if similar_entities and 'ids' in similar_entities and len(similar_entities['ids']) > 0:
                    for sim_id, distance in zip(
                        similar_entities['ids'][0],
                        similar_entities['distances'][0]
                    ):
                        similarity = 1 - distance  # Cosine similarity
                        label = embeddings_generator.get_entity_label(sim_id)
                        result_entry['similar_entities'].append({
                            'entity_id': sim_id,
                            'label': label,
                            'similarity': float(similarity)
                        })
                else:
                    print(f"No similar entities found for question {question_id}.")

                # Add similar relationships
                if similar_relationships and 'ids' in similar_relationships and len(similar_relationships['ids']) > 0:
                    for sim_id, distance in zip(
                        similar_relationships['ids'][0],
                        similar_relationships['distances'][0]
                    ):
                        similarity = 1 - distance  # Cosine similarity
                        label = embeddings_generator.get_property_label(sim_id)
                        result_entry['similar_relationships'].append({
                            'property_id': sim_id,
                            'label': label,
                            'similarity': float(similarity)
                        })
                else:
                    print(f"No similar relationships found for question {question_id}.")

                results.append(result_entry)
            else:
                print(f"No embedding found for {question_id}.")

        # Write results to JSON file
        with open('result_test_cms.json', 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)

        end_time = time.time()
        print(f"Finished querying embeddings and writing results. Time taken: {end_time - start_time:.2f} seconds.\n")

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
