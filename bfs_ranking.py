import os
import json
import requests
import torch
from chromadb import Client
from chromadb.config import Settings
from tqdm import tqdm
import csv
import numpy as np

class EmbeddingsGenerator:
    def __init__(self):
        # Initialize caches
        self.entity_label_cache = {}
        self.predicate_label_cache = {}
        # Initialize ChromaDB client
        self.client_questions = self.initialize_chromadb("chromadb_store_questions")
        self.collection_questions = self.client_questions.get_collection("questions_collection")
        
    def initialize_chromadb(self, persist_directory):
        return Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))

    def get_question_embedding(self, question_id):
        # Fetch the question embedding from ChromaDB
        result_data = self.collection_questions.get(ids=[question_id])
        if result_data and 'embeddings' in result_data and len(result_data['embeddings']) > 0:
            return np.array(result_data['embeddings'][0])
        return None

    def get_entity_label(self, entity_id):
        # Check cache first
        if entity_id in self.entity_label_cache:
            return self.entity_label_cache[entity_id]
        # Fetch entity label
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

    def get_predicate_label(self, predicate_id):
        # Check cache first
        if predicate_id in self.predicate_label_cache:
            return self.predicate_label_cache[predicate_id]
        # Fetch predicate label
        try:
            url = (
                'https://www.wikidata.org/w/api.php?action=wbgetentities'
                f'&ids={predicate_id}&format=json&props=labels&languages=en'
            )
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                entities = data.get('entities', {})
                entity_data = entities.get(predicate_id, {})
                label_data = entity_data.get('labels', {})
                label = label_data.get('en', {}).get('value', '')
                if not label and label_data:
                    label = next(iter(label_data.values())).get('value', '')
                # Cache the result
                self.predicate_label_cache[predicate_id] = label
                return label
            else:
                print(
                    f"Failed to retrieve label for predicate {predicate_id}. "
                    f"HTTP status code: {response.status_code}"
                )
                return ''
        except Exception as e:
            print(f"Exception occurred while fetching label for predicate {predicate_id}: {e}")
            return ''

def main():
    # Load bfs_results.json
    if not os.path.exists('bfs_results.json'):
        print("Error: 'bfs_results.json' file not found.")
        return

    with open('bfs_results.json', 'r', encoding='utf-8') as f:
        bfs_results = json.load(f)

    # Collect all unique predicate IDs used in the paths
    predicate_ids_set = set()
    for question_id, paths in bfs_results.items():
        for path_data in paths:
            path = path_data['path']
            for element in path:
                if 'predicate_id' in element:
                    predicate_ids_set.add(element['predicate_id'])
    predicate_ids = list(predicate_ids_set)
    print(f"Total unique predicates in paths: {len(predicate_ids)}")

    # Initialize EmbeddingsGenerator
    embeddings_generator = EmbeddingsGenerator()

    # Load question_similar_data.json to get question texts
    if not os.path.exists('question_similar_data.json'):
        print("Error: 'question_similar_data.json' file not found.")
        return

    with open('question_similar_data.json', 'r', encoding='utf-8') as f:
        question_similar_data = json.load(f)

    pruned_bfs_results = {}
    top_k = 10  # Adjust this value as needed

    # Prepare CSV file
    csv_filename = 'pruned_bfs_results.csv'
    fieldnames = ['question_id', 'question_text', 'path_labels', 'score']
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for question_id in tqdm(bfs_results.keys(), desc="Processing questions"):
        # Get question text and embedding
        question_text = question_similar_data.get(question_id, {}).get('question_text', '')
        question_embedding = embeddings_generator.get_question_embedding(question_id)

        if question_embedding is None:
            print(f"No embedding found for {question_id}.")
            continue

        # Normalize question embedding for cosine similarity
        question_embedding = question_embedding / np.linalg.norm(question_embedding) if np.linalg.norm(question_embedding) != 0 else question_embedding

        # Rank paths by score
        paths = bfs_results[question_id]
        path_scores = []
        for path_data in paths:
            path = path_data['path']
            predicates_in_path = [element['predicate_id'] for element in path if 'predicate_id' in element]

            # Compute score: count of matching predicates
            score = sum(1 for pid in predicates_in_path if pid in predicate_ids_set)
            path_scores.append((score, path_data))

        # Rank paths by score
        ranked_paths = sorted(path_scores, key=lambda x: x[0], reverse=True)

        # Keep top 2 paths
        if ranked_paths:
            # If only one path, keep that one
            if len(ranked_paths) == 1:
                score, path_data = ranked_paths[0]
                pruned_bfs_results[question_id] = [path_data]

                # Write to CSV
                path_labels = path_data['path_labels']
                writer.writerow({
                    'question_id': question_id,
                    'question_text': question_text,
                    'path_labels': path_labels,
                    'score': str(score)
                })
            else:
                # Keep top 2 paths
                top_paths = ranked_paths[:2]
                pruned_bfs_results[question_id] = [path_data for score, path_data in top_paths]

                # Concatenate path_labels and scores
                path_labels_list = []
                scores_list = []
                for score, path_data in top_paths:
                    path_labels_list.append(path_data['path_labels'])
                    scores_list.append(str(score))
                # Join paths with '|'
                path_labels_combined = ' | '.join(path_labels_list)
                scores_combined = ' | '.join(scores_list)

                # Write to CSV
                writer.writerow({
                    'question_id': question_id,
                    'question_text': question_text,
                    'path_labels': path_labels_combined,
                    'score': scores_combined
                })
        else:
            print(f"No paths available for question {question_id}")

    # Close CSV file
    csv_file.close()

    # Save pruned BFS results to a JSON file
    with open('pruned_bfs_results.json', 'w', encoding='utf-8') as f:
        json.dump(pruned_bfs_results, f, indent=4)
    print("Pruned BFS results saved to 'pruned_bfs_results.json'")
    print(f"CSV file '{csv_filename}' has been created.")

if __name__ == '__main__':
    main()
