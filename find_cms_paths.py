import os
import json
import requests
import time
from collections import deque
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import itertools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'wikidata_bfs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WikidataAPI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research Project/1.0'
        })
        self.label_cache = {}

    def get_label(self, entity_id, max_retries=3):
        if entity_id in self.label_cache:
            return self.label_cache[entity_id]

        for attempt in range(max_retries):
            try:
                time.sleep(1)
                url = 'https://www.wikidata.org/w/api.php'
                params = {
                    'action': 'wbgetentities',
                    'ids': entity_id,
                    'format': 'json',
                    'props': 'labels',
                    'languages': 'en'
                }
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                label = (
                    data.get('entities', {})
                    .get(entity_id, {})
                    .get('labels', {})
                    .get('en', {})
                    .get('value', entity_id)
                )
                
                if label:
                    self.label_cache[entity_id] = label
                    return label
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {entity_id}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        return entity_id

class BFSWikidata:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research Project BFS/1.0',
            'Accept': 'application/sparql-results+json'
        })
        self.neighbors_cache = {}
        self.failed_entities = set()

    def get_neighbors(self, entity_id, max_retries=3):
        if entity_id in self.failed_entities:
            return []
            
        if entity_id in self.neighbors_cache:
            return self.neighbors_cache[entity_id]

        query = """
        SELECT DISTINCT ?neighbor ?predicate ?direction WHERE {
          {
            wd:%s ?predicate ?neighbor .
            FILTER(STRSTARTS(STR(?neighbor), "http://www.wikidata.org/entity/Q"))
            BIND("outgoing" AS ?direction)
          }
          UNION
          {
            ?neighbor ?predicate wd:%s .
            FILTER(STRSTARTS(STR(?neighbor), "http://www.wikidata.org/entity/Q"))
            BIND("incoming" AS ?direction)
          }
        }
        LIMIT 100
        """ % (entity_id, entity_id)

        for attempt in range(max_retries):
            try:
                time.sleep(1)
                response = self.session.get(
                    'https://query.wikidata.org/sparql',
                    params={'query': query, 'format': 'json'},
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                for result in data.get('results', {}).get('bindings', []):
                    neighbor_uri = result['neighbor']['value']
                    predicate_uri = result['predicate']['value']
                    direction = result['direction']['value']
                    
                    neighbor_id = neighbor_uri.split('/')[-1]
                    predicate_id = predicate_uri.split('/')[-1]
                    
                    if neighbor_id.startswith('Q') and predicate_id.startswith('P'):
                        results.append((neighbor_id, predicate_id, direction))
                
                self.neighbors_cache[entity_id] = results
                return results
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for neighbors of {entity_id}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        self.failed_entities.add(entity_id)
        return []

    def find_path(self, start_id, end_id, max_depth=2):
        if start_id in self.failed_entities or end_id in self.failed_entities:
            return None
            
        queue = deque([(start_id, [{'entity': start_id}])])
        visited = {start_id}
        
        while queue:
            current_id, path = queue.popleft()
            current_depth = (len(path) - 1) // 2
            
            if current_depth >= max_depth:
                continue
            
            for neighbor_id, predicate_id, direction in self.get_neighbors(current_id):
                if neighbor_id == end_id:
                    return path + [
                        {'predicate': predicate_id, 'direction': direction},
                        {'entity': neighbor_id}
                    ]
                    
                if neighbor_id not in visited and current_depth < max_depth - 1:
                    visited.add(neighbor_id)
                    new_path = path + [
                        {'predicate': predicate_id, 'direction': direction},
                        {'entity': neighbor_id}
                    ]
                    queue.append((neighbor_id, new_path))
            
            time.sleep(0.1)
        
        return None

def process_entity_pair(start_entity, end_entity, wikidata_api, bfs):
    """Process a single entity pair to find path"""
    try:
        path = bfs.find_path(start_entity['entity_id'], end_entity['entity_id'])
        
        if not path:
            return None
            
        # Format path with labels
        path_str = []
        for item in path:
            if 'entity' in item:
                entity_id = item['entity']
                label = wikidata_api.get_label(entity_id)
                path_str.append(f"{label} ({entity_id})")
            else:
                predicate_id = item['predicate']
                direction = item['direction']
                label = wikidata_api.get_label(predicate_id)
                arrow = "-->" if direction == "outgoing" else "<--"
                path_str.append(f"{arrow} [{label}] {arrow}")
        
        return {
            'start_entity': {
                'id': start_entity['entity_id'],
                'label': start_entity['label'],
                'similarity': start_entity['similarity']
            },
            'end_entity': {
                'id': end_entity['entity_id'],
                'label': end_entity['label'],
                'similarity': end_entity['similarity']
            },
            'path': ' '.join(path_str)
        }
        
    except Exception as e:
        logger.error(f"Error processing entity pair {start_entity['entity_id']} - {end_entity['entity_id']}: {str(e)}")
        return None

def process_question_task(question, wikidata_api, bfs):
    """Process all entity pairs for a question"""
    try:
        # Get all similar entities
        entities = question['similar_entities']
        paths = []
        
        # Generate all possible pairs of entities
        entity_pairs = list(itertools.combinations(entities, 2))
        
        # Process each pair
        for start_entity, end_entity in entity_pairs:
            path_result = process_entity_pair(start_entity, end_entity, wikidata_api, bfs)
            if path_result:
                paths.append(path_result)
        
        if paths:  # Only return result if paths were found
            return {
                'question_id': question['question_id'],
                'question_text': question['question_text'],
                'paths': paths
            }
            
    except Exception as e:
        logger.error(f"Error processing question {question.get('question_id')}: {str(e)}")
    
    return None

def main():
    try:
        # Load questions data
        with open('result_sentence_embedding_cms_top100.json', 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        logger.info(f"Processing {len(questions_data)} questions")
        
        # Initialize API and BFS
        wikidata_api = WikidataAPI()
        bfs = BFSWikidata()
        
        results = []
        
        # Process questions with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            # Submit tasks for each question
            for question in questions_data:
                future = executor.submit(process_question_task, question, wikidata_api, bfs)
                futures.append(future)
            
            # Process completed tasks
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
                result = future.result()
                if result:
                    results.append(result)
                    
                    # Save intermediate results
                    if len(results) % 10 == 0:
                        with open(f'wikidata_paths_intermediate_{len(results)}.json', 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save final results
        with open('wikidata_paths_final.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save summary
        summary = {
            'total_questions': len(questions_data),
            'questions_with_paths': len(results),
            'total_paths': sum(len(q['paths']) for q in results)
        }
        with open('wikidata_paths_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Processing completed. Found paths for {len(results)} questions.")
        logger.info(f"Total paths found: {summary['total_paths']}")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
