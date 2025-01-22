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
    def __init__(self, model_name='all-distilroberta-v1'):
        """
        Initializes the EmbeddingsGenerator with the specified SentenceTransformer model.
        """
        print(f"Initializing EmbeddingsGenerator with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.embedding_shape_printed = False
        self.initialize_chromadb()

    def initialize_chromadb(self):
        """
        Initializes ChromaDB clients and manages the embedding collection.
        """
        print("Initializing ChromaDB...")
        persist_directory = os.path.join(os.getcwd(), 'mimic_ques_embedding_full2')
        print(f"Persist directory: {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))

        collection_name = "sentence_embedding_mimic_questions_full2"

        # Check if the collection exists
        existing_collections = self.client.list_collections()
        print(f"Existing collections: {[col.name for col in existing_collections]}")
        if any(col.name == collection_name for col in existing_collections):
            try:
                self.client.delete_collection(name=collection_name)
                print(f"Deleted existing collection '{collection_name}'.")
            except Exception as e:
                print(f"Error deleting existing collection: {str(e)}")
                traceback.print_exc()

        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "mimic questions embeddings using all-distilroberta-v1"}
            )
            print(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            traceback.print_exc()

    def generate_embeddings(self, texts):
        """
        Generates normalized embeddings for a list of texts.
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        embeddings_list = embeddings.cpu().numpy()
        if not self.embedding_shape_printed:
            print(f"Embedding shape (expected 768): {embeddings_list.shape}")
            self.embedding_shape_printed = True
        return embeddings_list.tolist()

    def store_question_embeddings(self, items, batch_size=8):
        """
        Stores question embeddings in ChromaDB in batches.
        Measures and prints total and average processing time at the end.
        """
        total_items = len(items)
        print(f"Total questions to process: {total_items}")
        
        # Start timing
        start_time = time.time()

        for i in tqdm(range(0, total_items, batch_size), desc="Processing batches"):
            batch_items = items[i:i + batch_size]
            batch_ids = [str(question_id) for question_id, _ in batch_items]
            batch_texts = [text for _, text in batch_items]

            try:
                batch_embeddings = self.generate_embeddings(batch_texts)
                print(f"Batch {i//batch_size + 1} embeddings shape: {np.array(batch_embeddings).shape}")
                batch_metadatas = [{"text": text} for text in batch_texts]

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

        # Persist after all additions
        try:
            self.client.persist()
            print("All embeddings persisted to disk.")
        except Exception as e:
            print(f"Error persisting to disk: {str(e)}")
            traceback.print_exc()

        # End timing
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_question = total_time / total_items if total_items > 0 else 0
        
        # Print timing information
        print("======== Timing Information ========")
        print(f"Total processing time for all questions: {total_time:.2f} seconds")
        print(f"Average processing time per question: {avg_time_per_question:.4f} seconds")

    def close(self):
        """
        Closes the ChromaDB client by persisting any remaining data.
        """
        try:
            self.client.persist()
            print("ChromaDB client closed and data persisted.")
        except Exception as e:
            print(f"Error during close: {str(e)}")
            traceback.print_exc()


def main():
    start_time = time.time()
    try:
        embeddings_generator = EmbeddingsGenerator(model_name='all-distilroberta-v1')
        excel_file_path = '/app/datafinal/test_mimic_q.xlsx'
        if not os.path.exists(excel_file_path):
            print(f"Excel file not found: {excel_file_path}")
            raise FileNotFoundError(f"Excel file not found at {excel_file_path}")
        print(f"Excel file found: {excel_file_path}")

        df = pd.read_excel(excel_file_path)
        print(f"Excel columns: {df.columns.tolist()}")
        if 'question' not in df.columns:
            raise ValueError("Excel file must contain a 'question' column")

        questions = df['question']
        question_items = [
            (f"question_{idx}", str(question).strip())
            for idx, question in enumerate(questions)
            if isinstance(question, str) and question.strip()
        ]
        embeddings_generator.store_question_embeddings(question_items)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()
    finally:
        if 'embeddings_generator' in locals():
            embeddings_generator.close()
        print(f"Total script execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
