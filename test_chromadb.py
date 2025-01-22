import os
import chromadb
from chromadb.config import Settings

def test_chromadb():
    persist_directory = '/app/layers/chroma_db_test'
    os.makedirs(persist_directory, exist_ok=True)
    
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    ))
    
    try:
        # Create a new collection
        collection = client.create_collection(name="test_collection", embedding_function=None)
        print("Collection 'test_collection' created successfully.")
        
        # Add a single embedding
        collection.add(
            documents=["Test Document"],
            embeddings=[[0.1] * 2304],  # Example embedding with 2304 dimensions
            ids=["test_id"]
        )
        print("Embedding added successfully.")
        
        # Persist the database
        client.persist()
        print("ChromaDB persisted successfully.")
        
        # Re-initialize the client to verify persistence
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        collections = client.list_collections()
        print(f"Existing collections after re-initialization: {[col.name for col in collections]}")
        
        # Retrieve the collection and verify the embedding
        retrieved_collection = client.get_collection(name="test_collection")
        results = retrieved_collection.query(query_texts=["Test Document"], n_results=1)
        print(f"Query results: {results}")
        
    except Exception as e:
        print(f"Error during ChromaDB test: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    test_chromadb()
