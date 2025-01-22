import os
import json
import numpy as np
import torch
import logging
import csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from io import StringIO

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = 'all-distilroberta-v1'
OUTPUT_DIR = 'wikidata_embedding_triplet3'

class TripletEmbeddingCreator:
    def __init__(self,
                 model_name=MODEL_NAME,
                 output_dir=OUTPUT_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded pretrained SentenceTransformer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise e

        # Check for multiple GPUs and use DataParallel if available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        else:
            logger.info("Using a single GPU")

        self.model.to(self.device)
        logger.info(f"Initialized SentenceTransformer with model: {model_name}")

        # Ensure output directory is inside 'layers' folder
        self.output_dir = os.path.join(os.getcwd(), output_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory at '{self.output_dir}'")
        else:
            logger.info(f"Output directory already exists at '{self.output_dir}'")

    def encode_texts(self, texts):
        """Encode texts using the model, handling DataParallel if necessary."""
        with torch.no_grad():
            if isinstance(self.model, torch.nn.DataParallel):
                embeddings = self.model.module.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device
                )
            else:
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device
                )
        return embeddings

    def clean_file(self, dataset_file):
        """Read the dataset file, remove NULL bytes, and return cleaned content."""
        try:
            # Read the file in binary mode
            with open(dataset_file, 'rb') as f:
                content = f.read()
            original_size = len(content)
            logger.debug(f"Original size of '{dataset_file}': {original_size} bytes")

            # Check for NULL bytes
            if b'\x00' in content:
                logger.warning(f"NULL bytes found in '{dataset_file}'. Removing them.")
                # Remove NULL bytes
                cleaned_content = content.replace(b'\x00', b'')
                cleaned_size = len(cleaned_content)
                logger.debug(f"Size after removing NULL bytes: {cleaned_size} bytes")
            else:
                logger.info(f"No NULL bytes found in '{dataset_file}'. No cleaning needed.")
                cleaned_content = content
                cleaned_size = original_size

            if cleaned_size == 0:
                logger.error(f"All content removed from '{dataset_file}' after cleaning.")
                return None

            # Decode the cleaned binary content to string
            try:
                decoded_content = cleaned_content.decode('utf-8')
            except UnicodeDecodeError as e:
                logger.error(f"Unicode decode error for '{dataset_file}': {e}")
                return None

            # Log a preview of the cleaned content
            preview_length = 500  # Number of characters to preview
            preview = decoded_content[:preview_length].replace('\n', '\\n').replace('\r', '\\r')
            logger.debug(f"First {preview_length} characters of cleaned content for '{dataset_file}':\n{preview}")

            return decoded_content
        except Exception as e:
            logger.error(f"Error cleaning dataset file '{dataset_file}': {e}")
            return None

    def process_dataset(self, dataset_file, dataset_name):
        """Process a dataset CSV file and create embeddings."""
        try:
            # Clean the file to remove NULL bytes
            cleaned_content = self.clean_file(dataset_file)
            if cleaned_content is None:
                logger.error(f"Failed to clean dataset '{dataset_file}'. Skipping.")
                return

            if not cleaned_content.strip():
                logger.error(f"Cleaned content of '{dataset_file}' is empty. Skipping.")
                return

            # Use StringIO to simulate a file-like object for pandas
            data = StringIO(cleaned_content)
            try:
                # **Updated delimiter from '\t' to ','**
                df = pd.read_csv(
                    data,
                    sep=',',  # Changed from '\t' to ','
                    engine='python',
                    quoting=csv.QUOTE_MINIMAL,
                    on_bad_lines='skip'  # Skip bad lines instead of raising errors
                )
                logger.info(f"Loaded dataset '{dataset_file}' with {len(df)} triplets.")
            except pd.errors.EmptyDataError:
                logger.error(f"No columns to parse from file '{dataset_file}'. It may be empty after cleaning.")
                return
            except pd.errors.ParserError as e:
                logger.error(f"Pandas parser error for '{dataset_file}': {e}")
                return
            except Exception as e:
                logger.error(f"Unexpected error while loading '{dataset_file}': {e}")
                return

        except Exception as e:
            logger.error(f"Error processing dataset '{dataset_file}': {e}")
            return

        # Check if required columns are present
        required_columns = {'subject', 'predicate', 'object'}
        if not required_columns.issubset(df.columns):
            logger.error(f"Dataset '{dataset_file}' does not contain the required columns: {required_columns}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return

        # Collect triplets
        triplets = []
        for index, row in df.iterrows():
            subject = str(row['subject']).strip()
            predicate = str(row['predicate']).strip()
            obj = str(row['object']).strip()
            triplet = {
                'head': {'id': f"{dataset_name}_subject_{index}", 'text': subject},
                'relation': {'id': f"{dataset_name}_predicate_{index}", 'text': predicate},
                'tail': {'id': f"{dataset_name}_object_{index}", 'text': obj}
            }
            triplets.append(triplet)

        logger.info(f"Processing {len(triplets)} triplets from dataset '{dataset_name}'.")

        if triplets:
            # Log the first 5 triplets for verification
            logger.debug(f"First 5 triplets from '{dataset_name}': {triplets[:5]}")
            self.process_triplets(triplets, dataset_name)
        else:
            logger.warning(f"No triplets to process in dataset '{dataset_name}'.")

    def process_triplets(self, triplets, dataset_name):
        """Process triplets and create embeddings."""
        embeddings = []
        metadata = []

        logger.info(f"Encoding triplets for dataset '{dataset_name}'...")

        # Process embeddings in batches to improve GPU utilization
        batch_size = 64  # Adjust batch size based on your GPU memory

        texts = []
        triplet_ids = []

        for triplet in triplets:
            head_text = triplet['head']['text']
            relation_text = triplet['relation']['text']
            tail_text = triplet['tail']['text']

            combined_text = f"{head_text} {relation_text} {tail_text}"
            texts.append(combined_text)
            metadata.append(triplet)

        # Encode texts in batches
        total_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding embeddings for dataset '{dataset_name}'"):
            batch_texts = texts[i:i+batch_size]
            try:
                batch_embeddings = self.encode_texts(batch_texts)
                total_embeddings.extend(batch_embeddings.cpu())
            except Exception as e:
                logger.error(f"Error encoding batch starting at index {i} for dataset '{dataset_name}': {e}")
                continue

        # Reconstruct triplet embeddings
        embedding_dim = total_embeddings[0].shape[0] if total_embeddings else 0
        for idx, emb in enumerate(total_embeddings):
            if emb.shape[0] != embedding_dim:
                logger.warning(f"Embedding at index {idx} has unexpected shape {emb.shape}. Skipping.")
                continue
            embeddings.append(emb.numpy())

        # Save embeddings and metadata for this dataset
        self.save_embeddings_and_metadata(embeddings, metadata, dataset_name)

    def save_embeddings_and_metadata(self, embeddings, metadata, dataset_name):
        """Save embeddings and metadata to files for a dataset."""
        embeddings_path = os.path.join(self.output_dir, f'{dataset_name}_embeddings.npy')
        metadata_path = os.path.join(self.output_dir, f'{dataset_name}_metadata.json')

        try:
            if embeddings:
                # Save embeddings
                np.save(embeddings_path, np.array(embeddings))
                logger.info(f"Embeddings saved to {embeddings_path}")
            else:
                logger.warning(f"No embeddings to save for dataset '{dataset_name}'.")

            if metadata:
                # Save metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                logger.info(f"Metadata saved to {metadata_path}")
            else:
                logger.warning(f"No metadata to save for dataset '{dataset_name}'.")
        except Exception as e:
            logger.error(f"Failed to save embeddings or metadata for dataset '{dataset_name}': {e}")

    def process(self):
        # Define the datasets to process
        datasets = [
            {'file': 'umls_type_groups_triples.csv', 'name': 'umls_type_groups_triples'},
            {'file': 'snomed_parent_child_triples.csv', 'name': 'snomed_parent_child_triples'}
        ]

        for dataset in datasets:
            dataset_file = dataset['file']
            dataset_name = dataset['name']

            # Adjust path to point to '/app/datasets/'
            dataset_path = os.path.join('/app/datasets', dataset_file)

            # Check if the file exists
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset file '{dataset_path}' does not exist.")
                continue

            self.process_dataset(dataset_path, dataset_name)

def main():
    try:
        creator = TripletEmbeddingCreator()
        creator.process()
        logger.info("Triplet embeddings and metadata creation completed successfully")
    except Exception as e:
        logger.error(f"Error during embedding creation: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

if __name__ == "__main__":
    main()
