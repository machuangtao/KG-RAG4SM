import json
import requests
import time
import logging
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikidataAPI:
    def __init__(self, max_workers=10):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PathFinder/1.0 (pythonbot)',
            'Accept': 'application/sparql-results+json'
        })
        self.label_cache = {}
        self.path_cache = {}
        self.max_workers = max_workers
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
                text = metadata['text'].split('.')[0].strip()
                self.id_map[text] = id_
        logger.info(f"Loaded {len(self.id_map)} entity mappings")

    def get_label(self, entity_id):
        if entity_id in self.label_cache:
            return self.label_cache[entity_id]

        try:
            response = self.session.get(
                'https://www.wikidata.org/w/api.php',
                params={
                    'action': 'wbgetentities',
                    'ids': entity_id,
                    'format': 'json',
                    'props': 'labels',
                    'languages': 'en'
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            label = data.get('entities', {}).get(entity_id, {}).get('labels', {}).get('en', {}).get('value', entity_id)
            self.label_cache[entity_id] = label
            return label
        except Exception as e:
            logger.error(f"Failed to fetch label for {entity_id}: {e}")
            return entity_id

    def find_paths(self, start_id, end_id):
        cache_key = f"{start_id}-{end_id}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

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
        try:
            response = self.session.get(
                'https://query.wikidata.org/sparql',
                params={'query': direct_query},
                timeout=20
            )
            response.raise_for_status()
            for result in response.json()['results']['bindings']:
                pred_id = result['pred']['value'].split('/')[-1]
                paths.append({
                    'type': 'direct',
                    'predicate': pred_id
                })
        except Exception as e:
            logger.error(f"Direct path query failed for {start_id}-{end_id}: {e}")

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
            response = self.session.get(
                'https://query.wikidata.org/sparql',
                params={'query': two_hop_query},
                timeout=20
            )
            response.raise_for_status()
            for result in response.json()['results']['bindings']:
                mid_id = result['mid']['value'].split('/')[-1]
                p1_id = result['p1']['value'].split('/')[-1]
                p2_id = result['p2']['value'].split('/')[-1]
                paths.append({
                    'type': '2-hop',
                    'middle': mid_id,
                    'predicates': [p1_id, p2_id]
                })
        except Exception as e:
            logger.error(f"2-hop path query failed for {start_id}-{end_id}: {e}")

        self.path_cache[cache_key] = paths
        return paths

    def format_path(self, path_data, start_id, end_id):
        try:
            if path_data['type'] == 'direct':
                pred_label = self.get_label(path_data['predicate'])
                start_label = self.get_label(start_id)
                end_label = self.get_label(end_id)
                return f"{start_label} ({start_id}) --> [{pred_label}] --> {end_label} ({end_id})"
            else:
                mid_id = path_data['middle']
                pred1_id, pred2_id = path_data['predicates']
                start_label = self.get_label(start_id)
                mid_label = self.get_label(mid_id)
                end_label = self.get_label(end_id)
                pred1_label = self.get_label(pred1_id)
                pred2_label = self.get_label(pred2_id)
                return f"{start_label} ({start_id}) --> [{pred1_label}] --> {mid_label} ({mid_id}) --> [{pred2_label}] --> {end_label} ({end_id})"
        except Exception as e:
            logger.error(f"Error formatting path: {e}")
            return None

def process_question(api, question):
    entities = []
    for entity in question['similar_items']['entities']:
        if 'metadata' in entity and 'text' in entity['metadata']:
            text = entity['metadata']['text'].split('.')[0].strip()
            if text in api.id_map:
                entities.append(api.id_map[text])

    if len(entities) < 2:
        return None

    paths_found = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(api.find_paths, entities[i], entities[j]): (i, j)
            for i in range(len(entities))
            for j in range(i + 1, len(entities))
        }
        for future in as_completed(futures):
            i, j = futures[future]
            paths = future.result()
            if paths:
                for path in paths:
                    formatted = api.format_path(path, entities[i], entities[j])
                    if formatted:
                        paths_found.append(formatted)

    if paths_found:
        return {
            'question_id': question['question_id'],
            'question_text': question['question_text'],
            'paths': paths_found
        }
    return None

def main():
    logger.info("Loading data...")
    with open('emed_wikidata_similar_full.json', 'r') as f:
        data = json.load(f)

    api = WikidataAPI()
    results = []

    for question in tqdm(data['results'], desc="Processing questions"):
        result = process_question(api, question)
        if result:
            results.append(result)

    output = {
        'metadata': {
            'total_questions_processed': len(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': results
    }

    with open('emed_wikidata_full_path.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Processing completed. Results saved to emed_wikidata_full_path.json")

if __name__ == "__main__":
    main()
