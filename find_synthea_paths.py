import os
import json
import pandas as pd
import requests
import time
from collections import deque
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class EmbeddingsGenerator:
    def __init__(self):
        self.entity_label_cache = {}
        self.predicate_label_cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EmbeddingsGenerator/1.0',
        })

    def get_entity_label(self, entity_id):
        if entity_id in self.entity_label_cache:
            return self.entity_label_cache[entity_id]
        try:
            url = (
                f'https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&format=json&props=labels&languages=en'
            )
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                label_data = data.get('entities', {}).get(entity_id, {}).get('labels', {}).get('en', {})
                label = label_data.get('value', '')
                self.entity_label_cache[entity_id] = label
                return label
            else:
                print(f"Failed to fetch label for {entity_id}: {response.status_code}")
                return ''
        except Exception as e:
            print(f"Exception fetching label for {entity_id}: {e}")
            return ''

    def get_predicate_label(self, predicate_id):
        if predicate_id in self.predicate_label_cache:
            return self.predicate_label_cache[predicate_id]
        try:
            url = (
                f'https://www.wikidata.org/w/api.php?action=wbgetentities&ids={predicate_id}&format=json&props=labels&languages=en'
            )
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                label_data = data.get('entities', {}).get(predicate_id, {}).get('labels', {}).get('en', {})
                label = label_data.get('value', '')
                self.predicate_label_cache[predicate_id] = label
                return label
            else:
                print(f"Failed to fetch label for {predicate_id}: {response.status_code}")
                return ''
        except Exception as e:
            print(f"Exception fetching label for {predicate_id}: {e}")
            return ''

    def close(self):
        pass  # No need to persist anything as we're using in-memory caching

class BFS_Wikidata:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BFS_Wikidata/1.0',
            'Accept': 'application/sparql-results+json',
        })
        self.neighbors_cache = {}

    def get_neighbors(self, entity_id):
        if entity_id in self.neighbors_cache:
            return self.neighbors_cache[entity_id]

        query = f'''
        PREFIX wd: <http://www.wikidata.org/entity/>
        SELECT ?neighbor ?predicate ?direction WHERE {{
          {{
            wd:{entity_id} ?predicate ?neighbor .
            FILTER(isIRI(?neighbor))
            BIND("outgoing" AS ?direction)
          }}
          UNION
          {{
            ?neighbor ?predicate wd:{entity_id} .
            FILTER(isIRI(?neighbor))
            BIND("incoming" AS ?direction)
          }}
        }}
        LIMIT 1000
        '''
        url = 'https://query.wikidata.org/sparql'
        params = {'query': query}
        headers = {'Accept': 'application/sparql-results+json'}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(1)  # Delay to respect rate limits
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', {}).get('bindings', [])
                    neighbors = []
                    for result in results:
                        neighbor_uri = result['neighbor']['value']
                        predicate_uri = result['predicate']['value']
                        direction = result['direction']['value']
                        neighbor_id = neighbor_uri.split('/')[-1]
                        predicate_id = predicate_uri.split('/')[-1]
                        if neighbor_id.startswith('Q') and predicate_id.startswith('P'):
                            neighbors.append((neighbor_id, predicate_id, direction))
                    self.neighbors_cache[entity_id] = neighbors
                    return neighbors
                else:
                    print(f"SPARQL query failed for {entity_id}: {response.status_code}")
                    time.sleep(5)
                    continue
            except Exception as e:
                print(f"Error querying neighbors for {entity_id}: {e}")
                time.sleep(5)
                continue
        return []

    def bfs_all_paths(self, start_id, end_id, max_depth=2, max_paths_per_pair=5):
        queue = deque([(start_id, [{'entity_id': start_id}])])
        paths_found = []
        visited = set()
        while queue and len(paths_found) < max_paths_per_pair:
            current_id, path = queue.popleft()
            depth = (len(path) - 1) // 2
            if depth >= max_depth:
                continue
            neighbors = self.get_neighbors(current_id)
            for neighbor_id, predicate_id, direction in neighbors:
                new_path = path + [{'predicate_id': predicate_id, 'direction': direction}, {'entity_id': neighbor_id}]
                if neighbor_id == end_id:
                    paths_found.append(new_path)
                    if len(paths_found) >= max_paths_per_pair:
                        break
                else:
                    if depth + 1 < max_depth and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, new_path))
        return paths_found

def process_bfs_task(task, embeddings_generator, bfs_wikidata):
    question_id, question_text, entity1, entity2 = task
    try:
        paths = bfs_wikidata.bfs_all_paths(entity1, entity2, max_depth=2, max_paths_per_pair=5)
        if paths:
            path_strs = []
            for path in paths:
                path_str = ""
                for idx in range(0, len(path), 2):
                    entity = path[idx]
                    label = embeddings_generator.get_entity_label(entity['entity_id'])
                    path_str += f"{label} ({entity['entity_id']})"
                    if idx + 1 < len(path):
                        predicate = path[idx + 1]
                        direction = predicate['direction']
                        pred_label = embeddings_generator.get_predicate_label(predicate['predicate_id'])
                        if direction == 'outgoing':
                            path_str += f" --[{pred_label}]--> "
                        else:
                            path_str += f" <--[{pred_label}]-- "
                path_strs.append(path_str)
            # Combine paths for this entity pair with '|'
            combined_paths = ' | '.join(path_strs)
            return {
                'question_id': question_id,
                'paths': path_strs,
                'path_labels': combined_paths,
                'output_str': f"Question ID: {question_id}\nPaths:\n" + '\n'.join(path_strs) + '\n\n'
            }
        else:
            return None
    except Exception as e:
        print(f"Error processing BFS task for {entity1}, {entity2}: {e}")
        return None

def main():
    embeddings_generator = EmbeddingsGenerator()
    bfs_wikidata = BFS_Wikidata()
    output_path = 'test_synthea_q_with_paths.xlsx'
    text_output_path = 'paths_synthea_output.txt'

    try:
        # Read data from 'test_synthea_q.xlsx'
        df = pd.read_excel('test_synthea_q.xlsx')

        # Load the JSON file with similar entities
        with open('result_test_synthea.json', 'r') as f:
            question_similar_data = json.load(f)

        # Create a mapping from question_id to similar entities
        question_entities_map = {}
        for item in question_similar_data:
            question_id = item['question_id']
            entities = item.get('entities', [])[:10]  # Top 10 similar entities
            entity_ids = [entity['id'] for entity in entities]
            question_entities_map[question_id] = entity_ids

        tasks = []
        for idx, row in df.iterrows():
            question_id = f"question_{idx}"
            # Adjust the column name if it's different in your data
            question_text = row['question']
            entity_ids = question_entities_map.get(question_id, [])
            if len(entity_ids) < 2:
                continue
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    tasks.append((question_id, question_text, entity_ids[i], entity_ids[j]))

        # Use a lower number of workers (5) to reduce API rate-limit risks
        max_workers = 5

        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(text_output_path, 'w', encoding='utf-8') as txt_file:
            futures = {executor.submit(process_bfs_task, task, embeddings_generator, bfs_wikidata): task for task in tasks}
            # Initialize paths_dict to collect paths per question
            paths_dict = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                result = future.result()
                if result:
                    question_id = result['question_id']
                    output_str = result['output_str']
                    txt_file.write(output_str)
                    path_labels = result['path_labels']
                    if question_id in paths_dict:
                        paths_dict[question_id] += ' | ' + path_labels
                    else:
                        paths_dict[question_id] = path_labels

        # Add 'Paths' column to df
        df['Paths'] = df.apply(lambda row: paths_dict.get(f"question_{row.name}", ''), axis=1)
        df.to_excel(output_path, index=False)
        print(f"Results saved to {output_path} and {text_output_path}")
    finally:
        embeddings_generator.close()

if __name__ == "__main__":
    main()
