import numpy as np
import json
import os
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

class WikidataEmbeddings:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        embeddings_path = os.path.join(current_dir, '..', 'embeddings', 'wikidata_translation_v1_vectors.npy')
        vocab_json_path = os.path.join(current_dir, '..', 'embeddings', 'wikidata_translation_v1_names.json')

        try:
            print("Loading Wikidata embeddings with memory mapping...")
            self.embeddings = np.load(embeddings_path, mmap_mode='r')
            self.embedding_dim = self.embeddings.shape[1]
            print(f"Finished loading Wikidata embeddings. Embedding dimension: {self.embedding_dim}")
            print(f"Total number of embeddings: {len(self.embeddings)}")

            print("Loading vocabulary mapping...")
            with open(vocab_json_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            print("Separating entities and relationships...")
            self.entity_token_to_index = {}
            self.relation_token_to_index = {}
            self.index_to_entity_token = {}
            self.index_to_relation_token = {}

            if isinstance(vocab_data, list):
                vocab_items = enumerate(vocab_data)
            elif isinstance(vocab_data, dict):
                vocab_items = ((int(idx), token) for idx, token in vocab_data.items())
            else:
                raise ValueError("Unsupported format in vocab file.")

            for idx, token in tqdm(vocab_items, total=len(vocab_data), desc="Processing tokens"):
                if token.startswith('<http://www.wikidata.org/entity/'):
                    self.entity_token_to_index[token] = idx
                    self.index_to_entity_token[idx] = token
                elif token.startswith('<http://www.wikidata.org/prop/direct/'):
                    self.relation_token_to_index[token] = idx
                    self.index_to_relation_token[idx] = token

            print(f"Number of entities: {len(self.entity_token_to_index)}")
            print(f"Number of relationships: {len(self.relation_token_to_index)}")

            # Initialize ChromaDB clients and collections for entities and relationships
            print("Initializing ChromaDB...")
            persist_directory_entities = os.path.join(current_dir, '..', 'chromadb_store_entities')
            persist_directory_relationships = os.path.join(current_dir, '..', 'chromadb_store_relationships')

            self.client_entities = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory_entities
            ))

            self.client_relationships = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory_relationships
            ))

            # Entities collection
            collections_entities = self.client_entities.list_collections()
            collection_names_entities = [c.name for c in collections_entities]
            if "wikidata_entities" in collection_names_entities:
                self.collection_entities = self.client_entities.get_collection(name="wikidata_entities")
                print("Collection 'wikidata_entities' exists.")
            else:
                print("Collection 'wikidata_entities' does not exist. Creating a new collection.")
                self.collection_entities = self.client_entities.create_collection(name="wikidata_entities")

            # Relationships collection
            collections_relationships = self.client_relationships.list_collections()
            collection_names_relationships = [c.name for c in collections_relationships]
            if "wikidata_relationships" in collection_names_relationships:
                self.collection_relationships = self.client_relationships.get_collection(name="wikidata_relationships")
                print("Collection 'wikidata_relationships' exists.")
            else:
                print("Collection 'wikidata_relationships' does not exist. Creating a new collection.")
                self.collection_relationships = self.client_relationships.create_collection(name="wikidata_relationships")

            # Add embeddings to ChromaDB if the collections are empty
            try:
                collection_count_entities = self.collection_entities.count()
            except Exception as e:
                print(f"Error getting entities collection count: {e}")
                collection_count_entities = 0

            try:
                collection_count_relationships = self.collection_relationships.count()
            except Exception as e:
                print(f"Error getting relationships collection count: {e}")
                collection_count_relationships = 0

            print(f"Entities collection count: {collection_count_entities}")
            print(f"Relationships collection count: {collection_count_relationships}")

            if collection_count_entities == 0 or collection_count_relationships == 0:
                print("Storing embeddings in ChromaDB...")

                # Set max_embeddings to a reasonable number for testing
                max_embeddings = 100000  # Adjust as needed

                # Get indices for entities and relationships
                entity_indices = list(self.entity_token_to_index.values())
                relation_indices = list(self.relation_token_to_index.values())

                # Limit the number of embeddings to max_embeddings
                entity_indices = entity_indices[:max_embeddings]
                relation_indices = relation_indices[:max_embeddings]

                # Store entity embeddings
                print("Storing entity embeddings...")
                batch_size = 1000  # Adjust as needed
                for start_idx in tqdm(range(0, len(entity_indices), batch_size), desc="Storing entity embeddings"):
                    end_idx = min(start_idx + batch_size, len(entity_indices))
                    batch_indices = entity_indices[start_idx:end_idx]
                    batch_embeddings = [self.embeddings[idx].tolist() for idx in batch_indices]
                    batch_ids = [str(idx) for idx in batch_indices]
                    batch_metadatas = [{
                        "token": self.index_to_entity_token.get(idx, "Unknown"),
                        "index": idx
                    } for idx in batch_indices]

                    self.collection_entities.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )

                    # Debug: Print examples from the first batch
                    if start_idx == 0:
                        print("\nExamples of entity embeddings being stored:")
                        for i in range(min(5, len(batch_metadatas))):
                            print(f"ID: {batch_ids[i]}, Metadata: {batch_metadatas[i]}")

                # Store relationship embeddings
                print("Storing relationship embeddings...")
                for start_idx in tqdm(range(0, len(relation_indices), batch_size), desc="Storing relationship embeddings"):
                    end_idx = min(start_idx + batch_size, len(relation_indices))
                    batch_indices = relation_indices[start_idx:end_idx]
                    batch_embeddings = [self.embeddings[idx].tolist() for idx in batch_indices]
                    batch_ids = [str(idx) for idx in batch_indices]
                    batch_metadatas = [{
                        "token": self.index_to_relation_token.get(idx, "Unknown"),
                        "index": idx
                    } for idx in batch_indices]

                    self.collection_relationships.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )

                    # Debug: Print examples from the first batch
                    if start_idx == 0:
                        print("\nExamples of relationship embeddings being stored:")
                        for i in range(min(5, len(batch_metadatas))):
                            print(f"ID: {batch_ids[i]}, Metadata: {batch_metadatas[i]}")

                print("Finished storing embeddings in ChromaDB.")

                # After storing, retrieve and print examples from ChromaDB
                print("\nRetrieving examples from ChromaDB collections...")

                # Entities
                print("\nExamples of entities stored in ChromaDB:")
                try:
                    results = self.collection_entities.get(limit=5)
                    for i in range(len(results['ids'])):
                        print(f"ID: {results['ids'][i]}, Metadata: {results['metadatas'][i]}")
                except Exception as e:
                    print(f"Error retrieving entities: {e}")

                # Relationships
                print("\nExamples of relationships stored in ChromaDB:")
                try:
                    results = self.collection_relationships.get(limit=5)
                    for i in range(len(results['ids'])):
                        print(f"ID: {results['ids'][i]}, Metadata: {results['metadatas'][i]}")
                except Exception as e:
                    print(f"Error retrieving relationships: {e}")

            else:
                print("Embeddings already exist in ChromaDB.")

                # Retrieve and print examples from existing ChromaDB collections
                print("\nRetrieving examples from existing ChromaDB collections...")

                # Entities
                print("\nExamples of entities stored in ChromaDB:")
                try:
                    results = self.collection_entities.get(limit=5)
                    for i in range(len(results['ids'])):
                        print(f"ID: {results['ids'][i]}, Metadata: {results['metadatas'][i]}")
                except Exception as e:
                    print(f"Error retrieving entities: {e}")

                # Relationships
                print("\nExamples of relationships stored in ChromaDB:")
                try:
                    results = self.collection_relationships.get(limit=5)
                    for i in range(len(results['ids'])):
                        print(f"ID: {results['ids'][i]}, Metadata: {results['metadatas'][i]}")
                except Exception as e:
                    print(f"Error retrieving relationships: {e}")

        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close(self):
        # Ensure data is persisted before closing
        self.client_entities.persist()
        self.client_relationships.persist()
        # No explicit close method in chromadb Client as of current versions

def main():
    try:
        wikidata_embeddings = WikidataEmbeddings()
        print("\nWikidata embeddings have been successfully stored in ChromaDB.")

        # Additional debugging in main if needed
        # Examples have already been printed in the __init__ method

    except Exception as e:
        print(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'wikidata_embeddings' in locals():
            wikidata_embeddings.close()

if __name__ == "__main__":
    main()
    
