import os
import json
import chromadb
from chromadb.config import Settings
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re

class SimilarityFinder:
    def __init__(self):
        self.initialize_chromadb_clients()
        
    def initialize_chromadb_clients(self):
        base_dir = os.getcwd()
        
        # Initialize clients
        self.client_questions = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=os.path.join(base_dir, 'cms_ques_embedding')
        ))
        self.client_entities = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=os.path.join(base_dir, 'chromadb_store_wikidata_entities')
        ))
        self.client_relationships = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=os.path.join(base_dir, 'chromadb_store_wikidata_relationships')
        ))
        
        # Get collections
        self.collection_questions = self.client_questions.get_collection("sentence_embedding_cms_questions_top10")
        self.collection_entities = self.client_entities.get_collection("wikidata_entities")
        self.collection_relationships = self.client_relationships.get_collection("wikidata_relationships")

    def extract_wikidata_id(self, metadata):
        if isinstance(metadata, dict):
            # Check if wikidata_id is directly available
            if 'wikidata_id' in metadata:
                return metadata['wikidata_id']
            # Try to extract from text
            if 'text' in metadata:
                match = re.search(r'Q\d+', metadata['text'])
                if match:
                    return match.group()
            # Try to extract from id field
            if 'id' in metadata and isinstance(metadata['id'], str):
                match = re.search(r'Q\d+', metadata['id'])
                if match:
                    return match.group()
        return None

    def get_top_similar(self, query_embedding, target_embeddings, metadata_list, top_k=10):
        similarities = cosine_similarity([query_embedding], target_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_items = []
        for idx, index in enumerate(top_indices, 1):
            metadata = metadata_list[index]
            score = float(similarities[index])
            wikidata_id = self.extract_wikidata_id(metadata)
            
            similar_items.append({
                'rank': idx,
                'wikidata_id': wikidata_id,
                'metadata': metadata,
                'similarity_score': score
            })
            
        return similar_items

    def find_similar_items(self):
        print("Fetching data from collections...")
        questions_data = self.collection_questions.get()
        entities_data = self.collection_entities.get()
        relationships_data = self.collection_relationships.get()
        
        entities_embeddings = np.array(entities_data['embeddings'])
        relationships_embeddings = np.array(relationships_data['embeddings'])
        
        results = []
        for idx, (q_id, q_embedding, question_text) in enumerate(
            zip(questions_data['ids'], questions_data['embeddings'], questions_data['documents']),
            start=1
        ):
            print(f"\nProcessing question {idx}/10")
            similar_entities = self.get_top_similar(
                q_embedding,
                entities_embeddings,
                entities_data['metadatas']
            )
            similar_relationships = self.get_top_similar(
                q_embedding,
                relationships_embeddings,
                relationships_data['metadatas']
            )
            
            # Print found entities for verification
            found_entities = [e['wikidata_id'] for e in similar_entities if e['wikidata_id']]
            print(f"Found {len(found_entities)} entities with Wikidata IDs: {found_entities}")
            
            question_result = {
                'question_number': idx,
                'question_id': q_id,
                'question_text': question_text,
                'similar_items': {
                    'entities': similar_entities,
                    'relationships': similar_relationships
                }
            }
            results.append(question_result)
            
        return results

    def save_results(self, results):
        output = {
            'metadata': {
                'total_questions': len(results),
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            },
            'results': results
        }
        
        # Save JSON
        with open('cms_wikidata_similar.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        # Save readable text summary
        with open('cms_wikidata_similar.txt', 'w', encoding='utf-8') as f:
            f.write(f"Total Questions Processed: {len(results)}\n\n")
            
            for question in results:
                f.write(f"Question #{question['question_number']}\n")
                f.write(f"ID: {question['question_id']}\n")
                f.write(f"Text: {question['question_text']}\n\n")
                
                f.write("Similar Entities:\n")
                for entity in question['similar_items']['entities']:
                    f.write(f"{entity['rank']}. ")
                    if entity['wikidata_id']:
                        f.write(f"[{entity['wikidata_id']}] ")
                    f.write(f"{entity['metadata'].get('text', 'No text available')}")
                    f.write(f" (Score: {entity['similarity_score']:.4f})\n")
                
                f.write("\nSimilar Relationships:\n")
                for rel in question['similar_items']['relationships']:
                    f.write(f"{rel['rank']}. ")
                    if rel['wikidata_id']:
                        f.write(f"[{rel['wikidata_id']}] ")
                    f.write(f"{rel['metadata'].get('text', 'No text available')}")
                    f.write(f" (Score: {rel['similarity_score']:.4f})\n")
                
                f.write("\n" + "="*80 + "\n\n")

def main():
    try:
        finder = SimilarityFinder()
        results = finder.find_similar_items()
        finder.save_results(results)
        print("\nSimilarity results processed and saved successfully")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
