import json
import requests
import time
import logging
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikidataAPI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PathFinder/1.0 (pythonbot)',
            'Accept': 'application/sparql-results+json'
        })
        self.label_cache = {}
        self.initialize_chromadb()

    def initialize_chromadb(self):
        base_dir = os.getcwd()
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=os.path.join(base_dir, 'chromadb_store_wikidata_entities')
        ))
        self.collection = self.client.get_collection("wikidata_entities")
        collection_data = self.collection.get()
        self.id_map = {}
        for id_, metadata in zip(collection_data['ids'], collection_data['metadatas']):
            if metadata and 'text' in metadata:
                text = metadata['text'].split('.')[0].strip()  # Take text before first period
                self.id_map[text] = id_
        logger.info(f"Loaded {len(self.id_map)} entity mappings")

    def get_label(self, entity_id, max_retries=3):
        if entity_id in self.label_cache:
            return self.label_cache[entity_id]

        for attempt in range(max_retries):
            try:
                time.sleep(1)
                response = self.session.get(
                    'https://www.wikidata.org/w/api.php',
                    params={
                        'action': 'wbgetentities',
                        'ids': entity_id,
                        'format': 'json',
                        'props': 'labels',
                        'languages': 'en'
                    },
                    timeout=30
                )
                data = response.json()
                label = data.get('entities', {}).get(entity_id, {}).get('labels', {}).get('en', {}).get('value', entity_id)
                self.label_cache[entity_id] = label
                return label
            except Exception as e:
                logger.error(f"Label fetch attempt {attempt + 1} failed for {entity_id}: {e}")
                if attempt == max_retries - 1:
                    return entity_id
                time.sleep(2 ** attempt)

    def find_paths(self, start_id, end_id):
        paths = []
        
        # Direct paths query
        direct_query = f"""
        SELECT DISTINCT ?pred WHERE {{
            {{ wd:{start_id} ?pred wd:{end_id} }}
            UNION
            {{ wd:{end_id} ?pred wd:{start_id} }}
            FILTER(STRSTARTS(STR(?pred), "http://www.wikidata.org/prop/direct/"))
        }}
        """

        # Execute direct paths query
        try:
            time.sleep(1)
            response = self.session.get(
                'https://query.wikidata.org/sparql',
                params={'query': direct_query},
                timeout=30
            )
            
            if response.status_code == 200:
                for result in response.json()['results']['bindings']:
                    pred_id = result['pred']['value'].split('/')[-1]
                    paths.append({
                        'type': 'direct',
                        'predicate': pred_id
                    })
                    logger.info(f"Found direct path between {start_id} and {end_id}")
        except Exception as e:
            logger.error(f"Direct path query failed: {e}")

        # Two-hop paths query
        two_hop_query = f"""
        SELECT DISTINCT ?mid ?p1 ?p2 WHERE {{
            {{ 
                wd:{start_id} ?p1 ?mid .
                ?mid ?p2 wd:{end_id} .
            }} UNION {{
                wd:{end_id} ?p1 ?mid .
                ?mid ?p2 wd:{start_id} .
            }}
            FILTER(?mid != wd:{start_id} && ?mid != wd:{end_id})
            FILTER(STRSTARTS(STR(?mid), "http://www.wikidata.org/entity/"))
            FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
            FILTER(STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))
        }}
        LIMIT 5
        """

        try:
            time.sleep(1)
            response = self.session.get(
                'https://query.wikidata.org/sparql',
                params={'query': two_hop_query},
                timeout=30
            )
            
            if response.status_code == 200:
                for result in response.json()['results']['bindings']:
                    mid_id = result['mid']['value'].split('/')[-1]
                    p1_id = result['p1']['value'].split('/')[-1]
                    p2_id = result['p2']['value'].split('/')[-1]
                    paths.append({
                        'type': '2-hop',
                        'middle': mid_id,
                        'predicates': [p1_id, p2_id]
                    })
                    logger.info(f"Found 2-hop path between {start_id} and {end_id} via {mid_id}")
        except Exception as e:
            logger.error(f"2-hop path query failed: {e}")

        return paths

    def format_path(self, path_data, start_id, end_id):
        try:
            if path_data['type'] == 'direct':
                pred_label = self.get_label(path_data['predicate'])
                start_label = self.get_label(start_id)
                end_label = self.get_label(end_id)
                return {
                    'path_text': f"{start_label} ({start_id}) --> [{pred_label}] --> {end_label} ({end_id})",
                    'entities': [start_id, end_id],
                    'predicates': [path_data['predicate']]
                }
            else:
                mid_id = path_data['middle']
                pred1_id, pred2_id = path_data['predicates']
                
                start_label = self.get_label(start_id)
                mid_label = self.get_label(mid_id)
                end_label = self.get_label(end_id)
                pred1_label = self.get_label(pred1_id)
                pred2_label = self.get_label(pred2_id)
                
                return {
                    'path_text': f"{start_label} ({start_id}) --> [{pred1_label}] --> {mid_label} ({mid_id}) --> [{pred2_label}] --> {end_label} ({end_id})",
                    'entities': [start_id, mid_id, end_id],
                    'predicates': [pred1_id, pred2_id]
                }
        except Exception as e:
            logger.error(f"Error formatting path: {e}")
            return None

def process_question(api, question):
    entities = []
    for entity in question['similar_items']['entities']:
        if 'metadata' in entity and 'text' in entity['metadata']:
            text = entity['metadata']['text'].split('.')[0].strip()
            if text in api.id_map:
                entity_id = api.id_map[text]
                entities.append(entity_id)
                logger.info(f"Found entity: {text} -> {entity_id}")

    if len(entities) < 2:
        logger.info(f"Not enough entities found for question {question['question_id']}")
        return None

    paths_found = []
    logger.info(f"Processing {len(entities)} entities for question {question['question_id']}")

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            entity1, entity2 = entities[i], entities[j]
            logger.info(f"Finding paths between {entity1} and {entity2}")
            
            try:
                paths = api.find_paths(entity1, entity2)
                if paths:
                    formatted_paths = []
                    for path in paths:
                        formatted = api.format_path(path, entity1, entity2)
                        if formatted:
                            formatted_paths.append(formatted)
                    
                    if formatted_paths:
                        paths_found.append({
                            'entity_pair': [entity1, entity2],
                            'paths': formatted_paths,
                            'path_count': len(formatted_paths)
                        })
                        logger.info(f"Found {len(formatted_paths)} valid paths")
            except Exception as e:
                logger.error(f"Error processing paths for {entity1}-{entity2}: {e}")
                continue

    if paths_found:
        result = {
            'question_id': question['question_id'],
            'question_text': question['question_text'],
            'total_paths_found': sum(p['path_count'] for p in paths_found),
            'paths': paths_found
        }
        # Save intermediate result
        save_intermediate_result(result)
        return result
    return None

def save_intermediate_result(result):
    filename = f"paths_intermediate_{result['question_id']}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving intermediate result: {e}")

def main():
    try:
        logger.info("Loading data...")
        with open('cms_wikidata_similar.json', 'r') as f:
            data = json.load(f)

        api = WikidataAPI()
        results = []

        for question in tqdm(data['results'], desc="Processing questions"):
            try:
                result = process_question(api, question)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing question {question['question_id']}: {e}")
                continue

        output = {
            'metadata': {
                'total_questions_processed': len(results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'max_hops': 2,
                'questions_with_paths': len(results)
            },
            'results': results
        }

        with open('cms_wikidata_path.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully processed {len(results)} questions with paths")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
