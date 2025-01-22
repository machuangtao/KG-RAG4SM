import os
import numpy as np
import requests
import time
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class QuestionEmbeddings:
    def __init__(self):
        # Initialize Sentence Transformer model
        self.model = SentenceTransformer('all-distilroberta-v1')

        # Initialize ChromaDB for questions
        self.initialize_chromadb()

    def initialize_chromadb(self):
        print("Initializing ChromaDB for question embeddings...")
        current_dir = os.getcwd()
        persist_directory = os.path.join(
            current_dir, 'chromadb_store_questions'
        )

        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))

        # Check if the collection exists
        collections = self.client.list_collections()
        collection_names = [c.name for c in collections]
        if "questions_collection" in collection_names:
            print("Collection 'questions_collection' exists.")
            self.collection = self.client.get_collection(
                name="questions_collection"
            )
        else:
            raise Exception("Questions collection not found. Please run the initial embedding script.")

    def get_question_embedding(self, question_id):
        result_data = self.collection.get(ids=[question_id])
        if result_data and 'embeddings' in result_data and len(result_data['embeddings']) > 0:
            question_embedding = np.array(result_data['embeddings'][0])
            return question_embedding
        else:
            print(f"No embedding found for {question_id}.")
            return None

    def close(self):
        # Ensure data is persisted before closing
        self.client.persist()

class WikidataEmbeddings:
    def __init__(self):
        # Initialize ChromaDB for Wikidata embeddings
        self.initialize_chromadb()

    def initialize_chromadb(self):
        print("Initializing ChromaDB for Wikidata embeddings...")
        current_dir = os.getcwd()

        # Entities
        persist_directory_entities = os.path.join(
            current_dir, 'chromadb_store_wikidata_entities'
        )
        os.makedirs(persist_directory_entities, exist_ok=True)
        self.client_entities = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_entities
        ))
        collections_entities = self.client_entities.list_collections()
        collection_names_entities = [c.name for c in collections_entities]
        if "wikidata_entities" in collection_names_entities:
            print("Collection 'wikidata_entities' exists.")
            self.collection_entities = self.client_entities.get_collection(
                name="wikidata_entities"
            )
        else:
            raise Exception("Entities collection not found. Please run the initial embedding script.")

        # Relationships
        persist_directory_relationships = os.path.join(
            current_dir, 'chromadb_store_wikidata_relationships'
        )
        os.makedirs(persist_directory_relationships, exist_ok=True)
        self.client_relationships = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory_relationships
        ))
        collections_relationships = self.client_relationships.list_collections()
        collection_names_relationships = [c.name for c in collections_relationships]
        if "wikidata_relationships" in collection_names_relationships:
            print("Collection 'wikidata_relationships' exists.")
            self.collection_relationships = self.client_relationships.get_collection(
                name="wikidata_relationships"
            )
        else:
            raise Exception("Relationships collection not found. Please run the initial embedding script.")

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
            return label
        else:
            print(
                f"Failed to retrieve label for property {prop_id}. "
                f"HTTP status code: {response.status_code}"
            )
            return ''

    def close(self):
        # Ensure data is persisted before closing
        self.client_entities.persist()
        self.client_relationships.persist()

class WikidataGraphExplorer:
    def __init__(self):
        self.session = requests.Session()
        self.cache = {}
        self.sleep_time = 0.1  # Sleep to be polite to the API

    def get_neighbors(self, entity_id):
        if entity_id in self.cache:
            return self.cache[entity_id]

        url = (
            'https://www.wikidata.org/w/api.php?action=wbgetentities'
            f'&ids={entity_id}&format=json&languages=en&props=claims'
        )
        response = self.session.get(url)
        if response.status_code == 200:
            data = response.json()
            entities = data.get('entities', {})
            entity_data = entities.get(entity_id, {})
            claims = entity_data.get('claims', {})
            neighbors = []
            for prop, claim_list in claims.items():
                for claim in claim_list:
                    mainsnak = claim.get('mainsnak', {})
                    datavalue = mainsnak.get('datavalue', {})
                    if not datavalue:
                        continue
                    value = datavalue.get('value', {})
                    if datavalue.get('type') == 'wikibase-entityid':
                        value_id = value.get('id', '')
                        neighbors.append((prop, value_id))
            self.cache[entity_id] = neighbors
            time.sleep(self.sleep_time)
            return neighbors
        else:
            print(
                f"Failed to retrieve data for entity {entity_id}. "
                f"HTTP status code: {response.status_code}"
            )
            return []

    def bfs(self, start_entity_id, max_nodes=50):
        visited = set()
        queue = [(start_entity_id, None, None)]  # (entity_id, parent_id, property_id)
        subgraph = []

        while queue and len(visited) < max_nodes:
            current_entity_id, parent_id, property_id = queue.pop(0)
            if current_entity_id in visited:
                continue
            visited.add(current_entity_id)
            subgraph.append((parent_id, property_id, current_entity_id))
            neighbors = self.get_neighbors(current_entity_id)
            for prop_id, neighbor_id in neighbors:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, current_entity_id, prop_id))
            if len(visited) >= max_nodes:
                break
        return subgraph

    def get_labels(self, ids):
        ids = list(set(ids))
        labels = {}
        batch_size = 50  # Max IDs per request
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            ids_str = '|'.join(batch_ids)
            url = (
                'https://www.wikidata.org/w/api.php?action=wbgetentities'
                f'&ids={ids_str}&format=json&languages=en&props=labels'
            )
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                entities = data.get('entities', {})
                for entity_id, entity_data in entities.items():
                    label_data = entity_data.get('labels', {})
                    label = label_data.get('en', {}).get('value', '')
                    if not label and label_data:
                        label = next(iter(label_data.values())).get('value', '')
                    labels[entity_id] = label
                time.sleep(self.sleep_time)
            else:
                print(
                    f"Failed to retrieve labels for IDs {batch_ids}. "
                    f"HTTP status code: {response.status_code}"
                )
        return labels

    def close(self):
        self.session.close()

def main():
    question_embedder = QuestionEmbeddings()
    wikidata_embedder = WikidataEmbeddings()
    graph_explorer = WikidataGraphExplorer()

    try:
        # Assuming you have already stored question embeddings
        # Let's get the list of question IDs
        all_questions = question_embedder.collection.get()
        question_ids = all_questions['ids']
        question_texts = all_questions['documents']

        # Open a file to save the results
        with open('bfs_path_with_topk_entity2_results.txt', 'w', encoding='utf-8') as result_file:
            def write_and_print(message):
                result_file.write(message + '\n')
                print(message)

            for question_id, question_text in zip(question_ids, question_texts):
                write_and_print(f"\nQuestion ID: {question_id}")
                write_and_print(f"Question Text: {question_text}\n")

                # Get question embedding
                question_embedding = question_embedder.get_question_embedding(question_id)
                if question_embedding is None:
                    write_and_print("No embedding found for this question.")
                    continue

                # Find the top 10 similar entities
                n_results = 10  # Number of top similar items

                similar_entities = wikidata_embedder.collection_entities.query(
                    query_embeddings=[question_embedding.tolist()],
                    n_results=n_results
                )

                # Find the top 10 similar relationships
                similar_relationships = wikidata_embedder.collection_relationships.query(
                    query_embeddings=[question_embedding.tolist()],
                    n_results=n_results
                )

                # Print top 10 similar entities
                write_and_print("Top 10 Similar Entities:")
                top_entity_ids = set()
                if similar_entities and 'ids' in similar_entities and len(similar_entities['ids']) > 0:
                    for sim_id, distance in zip(
                        similar_entities['ids'][0],
                        similar_entities['distances'][0]
                    ):
                        top_entity_ids.add(sim_id)
                        label = wikidata_embedder.get_entity_label(sim_id)
                        write_and_print(f"  ID: {sim_id}, Label: {label}, Distance: {distance}")
                else:
                    write_and_print("No similar entities found.")

                # Print top 10 similar relationships
                write_and_print("\nTop 10 Similar Relationships:")
                top_relationship_ids = set()
                if similar_relationships and 'ids' in similar_relationships and len(similar_relationships['ids']) > 0:
                    for sim_id, distance in zip(
                        similar_relationships['ids'][0],
                        similar_relationships['distances'][0]
                    ):
                        top_relationship_ids.add(sim_id)
                        label = wikidata_embedder.get_property_label(sim_id)
                        write_and_print(f"  ID: {sim_id}, Label: {label}, Distance: {distance}")
                else:
                    write_and_print("No similar relationships found.")

                # Get the most similar entity as starting point
                if len(top_entity_ids) == 0:
                    write_and_print("\nNo similar entities found for this question.")
                    continue

                most_similar_entity_id = similar_entities['ids'][0][0]
                starting_entity_label = wikidata_embedder.get_entity_label(most_similar_entity_id)

                write_and_print(f"\nStarting Entity ID: {most_similar_entity_id}")
                write_and_print(f"Starting Entity Label: {starting_entity_label}\n")

                # Perform BFS from the starting entity
                subgraph = graph_explorer.bfs(most_similar_entity_id, max_nodes=50)

                # Print BFS subgraph
                write_and_print("BFS Subgraph:")
                entity_ids_in_bfs = set()
                property_ids_in_bfs = set()
                for parent_id, prop_id, entity_id in subgraph:
                    if parent_id is None:
                        # Starting node
                        entity_label = wikidata_embedder.get_entity_label(entity_id)
                        write_and_print(f"  [{entity_id}] {entity_label}")
                    else:
                        parent_label = wikidata_embedder.get_entity_label(parent_id)
                        entity_label = wikidata_embedder.get_entity_label(entity_id)
                        prop_label = wikidata_embedder.get_property_label(prop_id)
                        write_and_print(f"  [{parent_id}] {parent_label} --[{prop_id}] {prop_label}--> [{entity_id}] {entity_label}")
                    entity_ids_in_bfs.update([parent_id, entity_id])
                    if prop_id:
                        property_ids_in_bfs.add(prop_id)

                # Refine the subgraph to retain only nodes and edges that are in top entities or relationships
                refined_subgraph = []
                for parent_id, prop_id, entity_id in subgraph:
                    if (entity_id in top_entity_ids or parent_id in top_entity_ids) and (prop_id in top_relationship_ids):
                        refined_subgraph.append((parent_id, prop_id, entity_id))
                    elif entity_id in top_entity_ids and prop_id is None:
                        # Include starting node
                        refined_subgraph.append((parent_id, prop_id, entity_id))

                if not refined_subgraph:
                    write_and_print("\nNo paths in BFS subgraph match the top entities and relationships.")
                    continue

                # Collect all entity and property IDs to get labels
                entity_ids = set()
                property_ids = set()
                for parent_id, prop_id, entity_id in refined_subgraph:
                    if entity_id:
                        entity_ids.add(entity_id)
                    if parent_id:
                        entity_ids.add(parent_id)
                    if prop_id:
                        property_ids.add(prop_id)

                # Get labels
                entity_labels = graph_explorer.get_labels(entity_ids)
                property_labels = graph_explorer.get_labels(property_ids)

                # Print refined subgraph context
                write_and_print("\nRefined Subgraph Context:")
                for parent_id, prop_id, entity_id in refined_subgraph:
                    if parent_id is None:
                        # Starting node
                        entity_label = entity_labels.get(entity_id, entity_id)
                        write_and_print(f"  [{entity_id}] {entity_label}")
                    else:
                        parent_label = entity_labels.get(parent_id, parent_id)
                        entity_label = entity_labels.get(entity_id, entity_id)
                        prop_label = property_labels.get(prop_id, prop_id)
                        write_and_print(f"  [{parent_id}] {parent_label} --[{prop_id}] {prop_label}--> [{entity_id}] {entity_label}")

                write_and_print("")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        question_embedder.close()
        wikidata_embedder.close()
        graph_explorer.close()

if __name__ == "__main__":
    main()
