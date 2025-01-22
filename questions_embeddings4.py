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
from io import StringIO
import traceback

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

        # Create a new collection with 'ip' (inner product) for dot product similarity
        self.collection_questions = self.client_questions.create_collection(
            name="questions_collection",
            metadata={'hnsw:space': 'ip'}
        )
        print("Created new 'questions_collection' with dot product similarity.")

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

        # Create a new collection with 'ip' (inner product) for dot product similarity
        self.collection_entities = self.client_entities.create_collection(
            name="wikidata_entities",
            metadata={'hnsw:space': 'ip'}
        )
        print("Created new 'wikidata_entities' collection with dot product similarity.")

        # Relationships
        persist_directory_relationships = os.path.join(
            current_dir, 'chromadb_store_wikidata_relationships'
        )

        # Delete the persist directory if it exists
        if os.path.exists(persist_directory_relationships):
            shutil.rmtree(persist_directory_relationships)
            print("Deleted existing persist directory for relationships to avoid dimension mismatch.")

        os.makedirs(persist_directory_relationships, exist_ok=True)
        self.client_relationships = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_relationships
        ))

        # Create a new collection with 'ip' (inner product) for dot product similarity
        self.collection_relationships = self.client_relationships.create_collection(
            name="wikidata_relationships",
            metadata={'hnsw:space': 'ip'}
        )
        print("Created new 'wikidata_relationships' collection with dot product similarity.")

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

            # Do NOT normalize embeddings for dot product similarity
            # batch_embeddings = [emb / np.linalg.norm(emb) for emb in batch_embeddings]

            # Prepare metadata
            batch_metadatas = [{'text': text} for _, text in batch_items]

            # Add to ChromaDB
            self.collection_questions.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts
            )

        # Rebuild the index after adding embeddings
        self.collection_questions.create_index()

        end_time = time.time()
        print(f"Finished storing question embeddings in ChromaDB. Time taken: {end_time - start_time:.2f} seconds.")
        print(f"Total questions stored: {self.collection_questions.count()}")

        # Retrieve and print some examples
        print("\nRetrieving examples from ChromaDB for questions...")
        try:
            results = self.collection_questions.get(ids=None, limit=5)
            for i in range(len(results['ids'])):
                print(f"ID: {results['ids'][i]}")
                print(f"Text: {results['documents'][i]}")
                print(f"Metadata: {results['metadatas'][i]}\n")
        except Exception as e:
            print(f"Error retrieving examples from ChromaDB: {e}")

    def store_wikidata_embeddings(self, items, is_entity=True):
        # items: list of tuples (entity_id, entity_text)
        batch_size = 8  # Adjust as needed based on your system's memory

        collection = self.collection_entities if is_entity else self.collection_relationships

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

            # Do NOT normalize embeddings for dot product similarity
            # batch_embeddings = [emb / np.linalg.norm(emb) for emb in batch_embeddings]

            # Prepare metadata
            batch_metadatas = [{'text': text} for _, text in batch_items]

            # Add to ChromaDB
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts
            )

        # Rebuild the index after adding embeddings
        collection.create_index()

        end_time = time.time()
        print(f"Finished storing embeddings in ChromaDB. Time taken: {end_time - start_time:.2f} seconds.")
        print(f"Total {'entities' if is_entity else 'properties'} stored: {collection.count()}")

        # Retrieve and print some examples
        print(f"\nRetrieving examples from ChromaDB for {'entities' if is_entity else 'properties'}...")
        try:
            results = collection.get(ids=None, limit=5)
            for i in range(len(results['ids'])):
                print(f"ID: {results['ids'][i]}")
                print(f"Text: {results['documents'][i]}")
                print(f"Metadata: {results['metadatas'][i]}\n")
        except Exception as e:
            print(f"Error retrieving examples from ChromaDB: {e}")

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
            time.sleep(0.05)  # Be polite to the API
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
            time.sleep(0.05)  # Be polite to the API
            return label
        else:
            print(
                f"Failed to retrieve label for property {prop_id}. "
                f"HTTP status code: {response.status_code}"
            )
            return ''

def get_wikidata_properties(limit=100):
    # Fetch a list of property IDs from Wikidata
    print("Fetching Wikidata property IDs...")
    start_time = time.time()
    url = (
        'https://www.wikidata.org/w/api.php?action=query&list=allpages'
        f'&apnamespace=120&format=json&aplimit={limit}'
    )
    response = requests.get(url)
    properties = []
    if response.status_code == 200:
        data = response.json()
        pages = data.get('query', {}).get('allpages', [])
        for page in pages:
            title = page.get('title', '')
            if title.startswith('Property:'):
                prop_id = title[len('Property:'):]
                properties.append(prop_id)
    else:
        print(f"Failed to fetch properties. HTTP status code: {response.status_code}")
    end_time = time.time()
    print(f"Finished fetching property IDs. Time taken: {end_time - start_time:.2f} seconds.")
    return properties

def get_wikidata_entities(limit=1000):
    # Fetch a list of entity IDs from Wikidata
    print("Fetching additional Wikidata entity IDs...")
    start_time = time.time()
    url = (
        'https://www.wikidata.org/w/api.php?action=query&list=allpages'
        f'&apnamespace=0&format=json&aplimit={limit}'
    )
    response = requests.get(url)
    entities = []
    if response.status_code == 200:
        data = response.json()
        pages = data.get('query', {}).get('allpages', [])
        for page in pages:
            title = page.get('title', '')
            if title.startswith('Q'):
                entity_id = title
                entities.append(entity_id)
    else:
        print(f"Failed to fetch entities. HTTP status code: {response.status_code}")
    end_time = time.time()
    print(f"Finished fetching entity IDs. Time taken: {end_time - start_time:.2f} seconds.")
    return entities

def main():
    print("Script is starting...")
    total_start_time = time.time()
    try:
        print("Initializing embeddings...")
        embeddings_generator = EmbeddingsGenerator()
        print("Embeddings initialized successfully.")

        # Read data from a CSV file to handle complex data correctly
        data = """omop,table,des1,des2,label,d1,d2,d3,d4,question,wikidata entities
person-person_id,beneficiarysummary-desynpuf_id,"the person domain contains records that uniquely identify each patient in the source data who is time at-risk to have clinical observations recorded within the source systems.;a unique identifier for each person.","beneficiarysummary pertains to a synthetic medicare beneficiary ;beneficiary code",1,"the person domain contains records that uniquely identify each patient in the source data who is time at-risk to have clinical observations recorded within the source systems.","a unique identifier for each person.","beneficiarysummary pertains to a synthetic medicare beneficiary ","beneficiary code","Attribute 1 person-person_id and its description 1 the person domain contains records that uniquely identify each patient in the source data who is time at-risk to have clinical observations recorded within the source systems.;a unique identifier for each person.
Attribute 2 beneficiarysummary-desynpuf_id and its description 2 beneficiarysummary pertains to a synthetic medicare beneficiary ;beneficiary code
Do attribute 1 and attribute 2 are semantically matched with each other?","Q181600, Q19847637, Q2596417"
person-person_id,beneficiarysummary-bene_birth_dt,"the person domain contains records that uniquely identify each patient in the source data who is time at-risk to have clinical observations recorded within the source systems.;a unique identifier for each person.","beneficiarysummary pertains to a synthetic medicare beneficiary ;date of birth",0,"the person domain contains records that uniquely identify each patient in the source data who is time at-risk to have clinical observations recorded within the source systems.","a unique identifier for each person.","beneficiarysummary pertains to a synthetic medicare beneficiary ","date of birth","Attribute 1 person-person_id and its description 1 the person domain contains records that uniquely identify each patient in the source data who is time at-risk to have clinical observations recorded within the source systems.;a unique identifier for each person.
Attribute 2 beneficiarysummary-bene_birth_dt and its description 2 beneficiarysummary pertains to a synthetic medicare beneficiary ;date of birth
Do attribute 1 and attribute 2 are semantically matched with each other?","Q181600, Q19847637, Q59670535, Q2389905"
observation-observation_date,clinical-events-event_date,"the observation date.","the date of the clinical event",1,"the observation date.","the observation date","clinical event date",,"Attribute 1 observation-observation_date and its description 1 the observation date.
Attribute 2 clinical-events-event_date and its description 2 the date of the clinical event
Do attribute 1 and attribute 2 are semantically matched with each other?","Q28912, Q11023"
condition-condition_concept_id,claims-diagnosis_code,"the condition concept identifier.","diagnosis code from claims data",1,"the condition concept identifier.","condition concept identifier","diagnosis code",,"Attribute 1 condition-condition_concept_id and its description 1 the condition concept identifier.
Attribute 2 claims-diagnosis_code and its description 2 diagnosis code from claims data
Do attribute 1 and attribute 2 are semantically matched with each other?","Q12136, Q5282129"
drug-drug_exposure_start_date,prescriptions-start_date,"the date when the drug exposure started.","start date of the prescription",1,"the date when the drug exposure started.","drug exposure start date","prescription start date",,"Attribute 1 drug-drug_exposure_start_date and its description 1 the date when the drug exposure started.
Attribute 2 prescriptions-start_date and its description 2 start date of the prescription
Do attribute 1 and attribute 2 are semantically matched with each other?","Q12140, Q8453"
visit-visit_occurrence_id,encounters-encounter_id,"the visit occurrence identifier.","unique identifier for the encounter",1,"the visit occurrence identifier.","visit occurrence identifier","encounter identifier",,"Attribute 1 visit-visit_occurrence_id and its description 1 the visit occurrence identifier.
Attribute 2 encounters-encounter_id and its description 2 unique identifier for the encounter
Do attribute 1 and attribute 2 are semantically matched with each other?","Q189538, Q2304467"
procedure-procedure_concept_id,operations-operation_code,"the procedure concept identifier.","code representing the operation performed",1,"the procedure concept identifier.","procedure concept identifier","operation code",,"Attribute 1 procedure-procedure_concept_id and its description 1 the procedure concept identifier.
Attribute 2 operations-operation_code and its description 2 code representing the operation performed
Do attribute 1 and attribute 2 are semantically matched with each other?","Q179278, Q1432866"
measurement-measurement_value,lab-tests-test_result_value,"the value of the measurement.","result value from lab tests",1,"the value of the measurement.","measurement value","lab test result value",,"Attribute 1 measurement-measurement_value and its description 1 the value of the measurement.
Attribute 2 lab-tests-test_result_value and its description 2 result value from lab tests
Do attribute 1 and attribute 2 are semantically matched with each other?","Q79184, Q732577"
device-device_exposure_id,medical-devices-device_id,"the device exposure identifier.","unique identifier for the medical device",1,"the device exposure identifier.","device exposure identifier","medical device identifier",,"Attribute 1 device-device_exposure_id and its description 1 the device exposure identifier.
Attribute 2 medical-devices-device_id and its description 2 unique identifier for the medical device
Do attribute 1 and attribute 2 are semantically matched with each other?","Q1183543, Q21191270"
care_site-care_site_id,hospital-care_site_id,"the care site identifier.","unique identifier for the hospital or care site",1,"the care site identifier.","care site identifier","hospital identifier",,"Attribute 1 care_site-care_site_id and its description 1 the care site identifier.
Attribute 2 hospital-care_site_id and its description 2 unique identifier for the hospital or care site
Do attribute 1 and attribute 2 are semantically matched with each other?","Q16917, Q133492"
provider-provider_id,practitioners-practitioner_id,"the provider identifier.","unique identifier for the healthcare practitioner",1,"the provider identifier.","provider identifier","practitioner identifier",,"Attribute 1 provider-provider_id and its description 1 the provider identifier.
Attribute 2 practitioners-practitioner_id and its description 2 unique identifier for the healthcare practitioner
Do attribute 1 and attribute 2 are semantically matched with each other?","Q2588354, Q1206022"
"""

        # Read the data into a pandas DataFrame with proper quoting
        df = pd.read_csv(
            StringIO(data),
            sep=",",
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            engine='python'
        )

        # Verify that the 'question' column is present
        print("DataFrame columns:", df.columns.tolist())

        questions = df['question']
        wikidata_entities = df['wikidata entities']

        # Prepare items to store: list of tuples (question_id, question_text)
        question_items = []
        for idx, question in enumerate(questions):
            question_id = f"question_{idx}"
            question_items.append((question_id, question))

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
        for entity_list in wikidata_entities:
            if isinstance(entity_list, str):
                ids = [eid.strip() for eid in entity_list.split(',') if eid.strip().startswith('Q')]
                entity_ids.update(ids)
            else:
                continue

        # Fetch additional entities from Wikidata
        additional_entities = get_wikidata_entities(limit=1000)
        entity_ids.update(additional_entities)

        end_time = time.time()
        print(f"Collected {len(entity_ids)} unique entity IDs. Time taken: {end_time - start_time:.2f} seconds.\n")

        # Check if entity_ids is empty
        if not entity_ids:
            print("No entity IDs were collected. Please check the CSV data.")
            return

        # Collect properties (relationships) from Wikidata
        property_ids = get_wikidata_properties(limit=100)

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

        # Querying embeddings and printing results
        print("Querying embeddings and printing results...")
        start_time = time.time()

        # Open the file to save the results
        with open('result.txt', 'w', encoding='utf-8') as result_file:

            # Now, for each question, retrieve its embedding and find top similar entities and relationships
            for idx, question in enumerate(tqdm(questions, desc="Processing questions")):
                question_id = f"question_{idx}"
                # Print and save the question details
                output_str = f"\nQuestion ID: {question_id}\nQuestion Text: {question}\n\n"
                print(output_str)
                result_file.write(output_str)

                # Retrieve question embedding
                result_data = embeddings_generator.collection_questions.get(ids=[question_id])
                if result_data and 'embeddings' in result_data and len(result_data['embeddings']) > 0:
                    question_embedding = np.array(result_data['embeddings'][0])

                    # Do NOT normalize query embedding for dot product similarity
                    # question_embedding = question_embedding / np.linalg.norm(question_embedding)

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

                    # Query the relationships collection
                    try:
                        similar_relationships = embeddings_generator.collection_relationships.query(
                            query_embeddings=[question_embedding.tolist()],
                            n_results=n_results
                        )
                    except Exception as e:
                        print(f"Error querying relationships collection: {e}")
                        similar_relationships = None

                    # Print and save similar entities
                    output_str = "Top similar Wikidata entities:\n"
                    print(output_str)
                    result_file.write(output_str)
                    if similar_entities and 'ids' in similar_entities and len(similar_entities['ids']) > 0:
                        for sim_id, distance in zip(
                            similar_entities['ids'][0],
                            similar_entities['distances'][0]
                        ):
                            entity_id = sim_id
                            label = embeddings_generator.get_entity_label(entity_id)
                            output_str = f"  ID: {entity_id}\n  Label: {label}\n  Similarity Score (Dot Product): {distance}\n\n"
                            print(output_str)
                            result_file.write(output_str)
                    else:
                        output_str = "No similar entities found.\n"
                        print(output_str)
                        result_file.write(output_str)

                    # Print and save similar relationships
                    output_str = "Top similar Wikidata relationships:\n"
                    print(output_str)
                    result_file.write(output_str)
                    if similar_relationships and 'ids' in similar_relationships and len(similar_relationships['ids']) > 0:
                        for sim_id, distance in zip(
                            similar_relationships['ids'][0],
                            similar_relationships['distances'][0]
                        ):
                            prop_id = sim_id
                            label = embeddings_generator.get_property_label(prop_id)
                            output_str = f"  ID: {prop_id}\n  Label: {label}\n  Similarity Score (Dot Product): {distance}\n\n"
                            print(output_str)
                            result_file.write(output_str)
                    else:
                        output_str = "No similar relationships found.\n"
                        print(output_str)
                        result_file.write(output_str)
                else:
                    output_str = f"No embedding found for {question_id}.\n"
                    print(output_str)
                    result_file.write(output_str)

        end_time = time.time()
        print(f"Finished querying embeddings and printing results. Time taken: {end_time - start_time:.2f} seconds.\n")

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
