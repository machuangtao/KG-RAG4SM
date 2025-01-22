import os
import json
import asyncio
import aiohttp
import logging
import re
from itertools import combinations
import time
from tqdm.asyncio import tqdm as tqdm_asyncio
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wikidata_debug.log'),
        # Uncomment the following line to see logs in the console
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.propagate = False

class AsyncRateLimiter:
    def __init__(self, calls_per_second=3):
        # We'll allow 3 concurrent requests
        self.semaphore = asyncio.Semaphore(3)
        self.interval = 1.0 / calls_per_second
        self.last_call = defaultdict(float)
    
    async def acquire(self, key):
        await self.semaphore.acquire()
        now = time.time()
        elapsed = now - self.last_call[key]
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        self.last_call[key] = time.time()
    
    def release(self):
        self.semaphore.release()

class WikidataPathFinder:
    def __init__(self, sparql_endpoint='https://query.wikidata.org/sparql', max_hops=3):
        self.sparql_endpoint = sparql_endpoint
        self.rate_limiter = AsyncRateLimiter(calls_per_second=3)
        self.headers = {
            'User-Agent': 'WikidataPathFinder/1.0 (your_email@example.com)'
        }
        self.max_hops = max_hops
        self.cache = {}

    def normalize_path(self, path):
        """
        Normalize the path by sorting it, ignoring direction, to prevent duplicates.
        """
        return tuple(path)

    def construct_sparql_query_direct(self, start_qid, end_qid):
        return f"""
        SELECT DISTINCT ?prop WHERE {{
            {{
                wd:{start_qid} ?prop wd:{end_qid} .
            }} UNION {{
                wd:{end_qid} ?prop wd:{start_qid} .
            }} UNION {{
                ?intermediate ?prop wd:{start_qid} .
                wd:{end_qid} ?prop ?intermediate .
            }}
            FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
        }}
        LIMIT 100
        """

    def construct_sparql_query_two_hops(self, start_qid, end_qid):
        return f"""
        SELECT DISTINCT ?p1 ?mid ?p2 WHERE {{
            {{
                wd:{start_qid} ?p1 ?mid .
                ?mid ?p2 wd:{end_qid} .
            }} UNION {{
                wd:{end_qid} ?p1 ?mid .
                ?mid ?p2 wd:{start_qid} .
            }} UNION {{
                ?mid ?p1 wd:{start_qid} .
                ?mid ?p2 wd:{end_qid} .
            }}
            FILTER(?mid != wd:{start_qid} && ?mid != wd:{end_qid})
            FILTER(isIRI(?mid) && STRSTARTS(STR(?mid), "http://www.wikidata.org/entity/Q"))
            FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
            FILTER(STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))
        }}
        LIMIT 100
        """

    def construct_sparql_query_three_hops(self, start_qid, end_qid):
        return f"""
        SELECT DISTINCT ?p1 ?mid1 ?p2 ?mid2 ?p3 WHERE {{
            {{
                wd:{start_qid} ?p1 ?mid1 .
                ?mid1 ?p2 ?mid2 .
                ?mid2 ?p3 wd:{end_qid} .
            }} UNION {{
                wd:{end_qid} ?p1 ?mid1 .
                ?mid1 ?p2 ?mid2 .
                ?mid2 ?p3 wd:{start_qid} .
            }} UNION {{
                ?mid1 ?p1 wd:{start_qid} .
                ?mid2 ?p2 ?mid1 .
                wd:{end_qid} ?p3 ?mid2 .
            }}
            FILTER(?mid1 != wd:{start_qid} && ?mid1 != wd:{end_qid} &&
                   ?mid2 != wd:{start_qid} && ?mid2 != wd:{end_qid})
            FILTER(isIRI(?mid1) && STRSTARTS(STR(?mid1), "http://www.wikidata.org/entity/Q"))
            FILTER(isIRI(?mid2) && STRSTARTS(STR(?mid2), "http://www.wikidata.org/entity/Q"))
            FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
            FILTER(STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))
            FILTER(STRSTARTS(STR(?p3), "http://www.wikidata.org/prop/direct/"))
        }}
        LIMIT 100
        """

    async def execute_sparql_query(self, query, session):
        cache_key = hash(query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            await self.rate_limiter.acquire('sparql')
            params = {'query': query, 'format': 'json'}
            async with session.get(self.sparql_endpoint, params=params, headers=self.headers, timeout=60) as response:
                if response.status == 429:
                    await asyncio.sleep(5)
                    return await self.execute_sparql_query(query, session)
                response.raise_for_status()
                data = await response.json()
                results = data.get('results', {}).get('bindings', [])
                self.cache[cache_key] = results
                return results
        except Exception as e:
            logger.error(f"Query error: {e}")
            return []
        finally:
            self.rate_limiter.release()

    async def find_paths(self, start_qid, end_qid, session):
        paths = set()  # Use a set to store unique paths
        
        # Direct connections (ignoring direction)
        if self.max_hops >= 1:
            query = self.construct_sparql_query_direct(start_qid, end_qid)
            results = await self.execute_sparql_query(query, session)
            for result in results:
                prop_url = result['prop']['value']
                prop_match = re.match(r'.*?/prop/direct/(P\d+)$', prop_url)
                if prop_match:
                    prop = prop_match.group(1)
                    path = [
                        ('entity', start_qid),
                        ('predicate', prop),
                        ('entity', end_qid)
                    ]
                    paths.add(self.normalize_path(path))

        # Two-hop connections (ignoring direction)
        if self.max_hops >= 2:
            query = self.construct_sparql_query_two_hops(start_qid, end_qid)
            results = await self.execute_sparql_query(query, session)
            for result in results:
                p1_url = result['p1']['value']
                p2_url = result['p2']['value']
                mid_url = result['mid']['value']

                # Extract IDs
                p1_match = re.match(r'.*?/prop/direct/(P\d+)$', p1_url)
                p2_match = re.match(r'.*?/prop/direct/(P\d+)$', p2_url)
                mid_match = re.match(r'.*?/entity/(Q\d+)$', mid_url)

                if p1_match and p2_match and mid_match:
                    p1 = p1_match.group(1)
                    p2 = p2_match.group(1)
                    mid = mid_match.group(1)
                    path = [
                        ('entity', start_qid),
                        ('predicate', p1),
                        ('entity', mid),
                        ('predicate', p2),
                        ('entity', end_qid)
                    ]
                    paths.add(self.normalize_path(path))

        # Three-hop connections (ignoring direction)
        if self.max_hops >= 3:
            query = self.construct_sparql_query_three_hops(start_qid, end_qid)
            results = await self.execute_sparql_query(query, session)
            for result in results:
                p1_url = result['p1']['value']
                p2_url = result['p2']['value']
                p3_url = result['p3']['value']
                mid1_url = result['mid1']['value']
                mid2_url = result['mid2']['value']

                # Extract IDs
                p1_match = re.match(r'.*?/prop/direct/(P\d+)$', p1_url)
                p2_match = re.match(r'.*?/prop/direct/(P\d+)$', p2_url)
                p3_match = re.match(r'.*?/prop/direct/(P\d+)$', p3_url)
                mid1_match = re.match(r'.*?/entity/(Q\d+)$', mid1_url)
                mid2_match = re.match(r'.*?/entity/(Q\d+)$', mid2_url)

                if p1_match and p2_match and p3_match and mid1_match and mid2_match:
                    p1 = p1_match.group(1)
                    p2 = p2_match.group(1)
                    p3 = p3_match.group(1)
                    mid1 = mid1_match.group(1)
                    mid2 = mid2_match.group(1)
                    path = [
                        ('entity', start_qid),
                        ('predicate', p1),
                        ('entity', mid1),
                        ('predicate', p2),
                        ('entity', mid2),
                        ('predicate', p3),
                        ('entity', end_qid)
                    ]
                    paths.add(self.normalize_path(path))

        return list(paths)
    
def normalize_path(path):
    """
    Normalize paths to ensure unique representation.
    Convert paths to tuples, sort intermediates, and ignore directionality.
    """
    normalized_path = []
    for element in path:
        if element[0] == 'entity':
            normalized_path.append(('entity', element[1]))
        elif element[0] == 'predicate':
            normalized_path.append(('predicate', element[1]))
    return tuple(normalized_path)

async def fetch_labels(ids, session, rate_limiter):
    labels = {}
    endpoint_url = 'https://www.wikidata.org/w/api.php'
    headers = {'User-Agent': 'WikidataPathFinder/1.0 (your_email@example.com)'}

    # Remove any invalid IDs
    valid_ids = [id for id in ids if re.match(r'^[QP]\d+$', id)]

    # Process in batches of 50
    batch_size = 50
    for i in range(0, len(valid_ids), batch_size):
        batch = valid_ids[i:i+batch_size]
        try:
            await rate_limiter.acquire('labels')
            params = {
                'action': 'wbgetentities',
                'ids': '|'.join(batch),
                'format': 'json',
                'languages': 'en',
                'props': 'labels'
            }
            async with session.get(endpoint_url, params=params, headers=headers) as response:
                data = await response.json()
                for entity_id, entity_data in data.get('entities', {}).items():
                    label = entity_data.get('labels', {}).get('en', {}).get('value')
                    if label:
                        labels[entity_id] = label
        except Exception as e:
            logger.error(f"Error fetching labels: {e}")
        finally:
            rate_limiter.release()
            await asyncio.sleep(0.1)  # Small delay between batches

    return labels

async def format_path_with_labels(path, entity_labels, property_labels):
    path_text_parts = []
    for i, (comp_type, comp_id) in enumerate(path):
        if comp_type == 'entity':
            label = entity_labels.get(comp_id, comp_id)
            path_text_parts.append(f"{label} ({comp_id})")
        elif comp_type == 'predicate':
            label = property_labels.get(comp_id, comp_id)
            path_text_parts.append(f"--[{label}] ({comp_id})-->")
        else:
            path_text_parts.append(comp_id)
    return ' '.join(path_text_parts)

async def process_question(question_id, similar_entities, pathfinder, session):
    """
    Process a single question: find paths among the top similar entities.
    Returns:
      result: { 'question_id': ..., 'paths': [...] }
      qids_in_paths: set of Q-ids found
      pids_in_paths: set of P-ids found
      elapsed_time: time taken to process this question
    """
    start_time = time.time()  # Track start time for this question

    try:
        paths = []
        qids_in_paths = set()
        pids_in_paths = set()

        entity_ids = [entity['id'] for entity in similar_entities[:10] if 'id' in entity]
        if len(entity_ids) < 2:
            logger.warning(f"Question {question_id}: insufficient entities")
            end_time = time.time()
            return (
                {'question_id': question_id, 'paths': []},
                qids_in_paths,
                pids_in_paths,
                end_time - start_time
            )

        entity_pairs = list(combinations(entity_ids, 2))
        for pair in entity_pairs:
            entity1, entity2 = pair
            pair_paths = await pathfinder.find_paths(entity1, entity2, session)
            if pair_paths:
                for path in pair_paths:
                    for comp_type, comp_id in path:
                        if comp_type == 'entity':
                            qids_in_paths.add(comp_id)
                        elif comp_type == 'predicate':
                            pids_in_paths.add(comp_id)
                    paths.append(path)

        # Always store the paths, even if unformatted
        result = {
            'question_id': question_id,
            'paths': paths  # Store unformatted paths
        }

        end_time = time.time()
        single_question_time = end_time - start_time
        logger.info(f"Time taken for question {question_id}: {single_question_time:.4f} seconds")

        return result, qids_in_paths, pids_in_paths, single_question_time

    except Exception as e:
        logger.error(f"Error processing question {question_id}: {e}")
        end_time = time.time()
        return (
            {'question_id': question_id, 'paths': []},
            set(),
            set(),
            end_time - start_time
        )

async def save_final_output(output_file, results, metadata):
    output_data = {
        'metadata': metadata,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

async def main():
    input_file = "cms_wikidata_similar_full.json"
    output_text_file = "cms_wikidata_paths_final_full.txt"
    final_output_file = "cms_wikidata_paths_final_full.json"
    max_questions = None  # Process all questions
    max_hops = 3

    # Clear previous results
    open(output_text_file, 'w').close()

    with open(input_file, 'r', encoding='utf-8') as f:
        similarity_data = json.load(f)

    # Ensure similarity_data is a dictionary
    if isinstance(similarity_data, list):
        similarity_data = {item['question_id']: item for item in similarity_data if 'question_id' in item}

    if max_questions:
        similarity_data = dict(list(similarity_data.items())[:max_questions])

    pathfinder = WikidataPathFinder(max_hops=max_hops)
    rate_limiter = AsyncRateLimiter(calls_per_second=3)
    
    # Measure total time for all questions
    overall_start_time = time.time()

    async with aiohttp.ClientSession() as session:
        results = []
        all_qids = set()
        all_pids = set()
        question_paths = {}  # To store raw/unformatted paths
        question_times = []  # To track time per question

        # First pass: collect all IDs and paths
        tasks = []
        for question_id, question_data in similarity_data.items():
            task = process_question(
                question_id,
                question_data.get('similar_entities', []),
                pathfinder,
                session
            )
            tasks.append(task)
        
        # Use tqdm_asyncio.gather for progress bar
        # This returns a list of tuples: (result, qids, pids, single_question_time)
        results_list = await tqdm_asyncio.gather(*tasks, desc="Processing questions")

        # Gather all results
        for (result, qids, pids, single_time) in results_list:
            question_id = result['question_id']
            all_qids.update(qids)
            all_pids.update(pids)
            question_paths[question_id] = result  # Store unformatted paths
            question_times.append(single_time)

        # Fetch labels
        logger.info("Fetching labels for entities and properties...")
        entity_labels = await fetch_labels(all_qids, session, rate_limiter)
        property_labels = await fetch_labels(all_pids, session, rate_limiter)

        # Second pass: update with formatted paths
        logger.info("Formatting paths with labels...")
        for question_id, result in question_paths.items():
            paths = result.get('paths', [])
            formatted_paths = []
            if not paths:
                formatted_paths.append("No path found")
            else:
                for path in paths:
                    formatted_path = await format_path_with_labels(path, entity_labels, property_labels)
                    formatted_paths.append(formatted_path)
            result['paths'] = formatted_paths
            results.append(result)

            # Save formatted paths to TXT file
            with open(output_text_file, 'a', encoding='utf-8') as txt_f:
                txt_f.write(f"Question ID: {question_id}\n")
                if formatted_paths:
                    for path in formatted_paths:
                        txt_f.write(f"{path}\n")
                else:
                    txt_f.write("No paths found.\n")
                txt_f.write("\n" + "="*80 + "\n")

        # Calculate overall time
        overall_end_time = time.time()
        total_time_for_all = overall_end_time - overall_start_time
        avg_time_per_question = (sum(question_times) / len(question_times)) if question_times else 0.0

        logger.info(f"Total time to process all questions: {total_time_for_all:.4f} seconds")
        logger.info(f"Average time per question: {avg_time_per_question:.4f} seconds")

        # Save final output with metadata
        metadata = {
            'total_questions': len(results),
            'total_unique_entities': len(all_qids),
            'total_unique_relations': len(all_pids),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'max_hops': max_hops,
            'configuration': {
                'input_file': input_file,
                'rate_limit': 3,
                'batch_size': 50
            },
            'timing': {
                'total_time_seconds': total_time_for_all,
                'average_time_per_question_seconds': avg_time_per_question
            }
        }
        
        await save_final_output(final_output_file, results, metadata)
        logger.info(f"Final results saved to {final_output_file}")

if __name__ == "__main__":
    asyncio.run(main())
