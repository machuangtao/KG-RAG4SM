import os
import numpy as np
import pandas as pd
import nltk
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import time
import traceback
from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.embedding_shape_printed = False
        self.initialize_chromadb()

    def initialize_chromadb(self):
        persist_directory = os.path.join(os.getcwd(), 'cms_ques_embedding')
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))

        # For ChromaDB 0.3.0, we need to handle collection creation differently
        try:
            self.collection = self.client.create_collection(
                name="sentence_embedding_cms_questions_top10",
                metadata={"description": "Top 10 CMS questions embeddings"}
            )
        except ValueError:
            # Collection already exists
            self.collection = self.client.get_collection(
                name="sentence_embedding_cms_questions_top10"
            )

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        embeddings_list = embeddings.cpu().numpy()
        
        if not self.embedding_shape_printed:
            print(f"Embedding shape: {embeddings_list.shape}")
            self.embedding_shape_printed = True
            
        return embeddings_list.tolist()

    def store_question_embeddings(self, items):
        batch_size = 5  # Process in smaller batches to avoid memory issues
        total_items = len(items)
        
        for i in range(0, total_items, batch_size):
            batch_items = items[i:i + batch_size]
            
            # Extract batched data
            batch_ids = [str(question_id) for question_id, _ in batch_items]
            batch_texts = [text for _, text in batch_items]
            
            try:
                # Generate embeddings for batch
                batch_embeddings = self.generate_embeddings(batch_texts)
                
                # Prepare metadata
                batch_metadatas = [{"text": text} for text in batch_texts]
                
                # Add to collection using ChromaDB 0.3.0 compatible method
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                print(f"Successfully added batch {i//batch_size + 1} ({len(batch_ids)} items)")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                traceback.print_exc()
                continue

        # Persist after all batches are processed
        try:
            self.client.persist()
            print("Successfully persisted all embeddings to disk")
        except Exception as e:
            print(f"Error persisting to disk: {str(e)}")

    def close(self):
        try:
            self.client.persist()
        except Exception as e:
            print(f"Error during close: {str(e)}")

def main():
    start_time = time.time()
    
    try:
        # Initialize the embeddings generator
        embeddings_generator = EmbeddingsGenerator()
        
        # Read the Excel file
        excel_file_path = '/app/datafinal/test_cms_q.xlsx'
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Excel file not found at {excel_file_path}")
            
        df = pd.read_excel(excel_file_path)
        if 'question' not in df.columns:
            raise ValueError("Excel file must contain a 'question' column")
        
        # Process top 10 questions
        questions = df['question'][:10]
        question_items = [(f"question_{idx}", str(question).strip()) 
                         for idx, question in enumerate(questions)]
        
        # Store embeddings
        embeddings_generator.store_question_embeddings(question_items)
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()
        
    finally:
        if 'embeddings_generator' in locals():
            embeddings_generator.close()
        
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()