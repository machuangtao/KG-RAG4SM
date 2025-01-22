import os
import shutil
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

#############################
# Helper Functions
#############################

def copy_chroma_db(source_dir, target_dir):
    """
    Copy an existing Chroma DB directory (with chroma-collections.parquet,
    chroma-embeddings.parquet, and index) into target_dir.
    Show a progress bar if there are multiple files.
    """
    os.makedirs(target_dir, exist_ok=True)
    files = os.listdir(source_dir)
    with tqdm(total=len(files), desc=f"Copying from {source_dir}", unit="file") as pbar:
        for filename in files:
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy(src_path, dst_path)
            pbar.update(1)

    print(f"Copied Chroma DB from {source_dir} to {target_dir}")

def create_client_for_collection(collection_dir):
    """
    Create a Chroma client pointing to a directory.
    """
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=collection_dir
        )
    )

def get_sorted_chunks(directory, embed_prefix="embeddings_chunk_", meta_prefix="metadata_chunk_"):
    """
    For wikidata_embedding_triplet2 directory, return two sorted lists of embeddings and metadata files.
    """
    embed_files = [f for f in os.listdir(directory) if f.startswith(embed_prefix) and f.endswith(".npy")]
    meta_files = [f for f in os.listdir(directory) if f.startswith(meta_prefix) and f.endswith(".json")]

    def extract_index(fname, prefix):
        base = fname.replace(prefix, "")
        base = os.path.splitext(base)[0]
        return int(base)

    embed_files = sorted(embed_files, key=lambda x: extract_index(x, embed_prefix))
    meta_files = sorted(meta_files, key=lambda x: extract_index(x, meta_prefix))

    assert len(embed_files) == len(meta_files), f"Mismatch in embeddings vs metadata count in {directory}"
    return embed_files, meta_files

def flatten_metadata_values(metadata):
    """
    Convert any dictionary or list metadata values into JSON strings so that all values are strings, ints, or floats.
    """
    new_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, dict) or isinstance(v, list):
            new_metadata[k] = json.dumps(v)
        else:
            new_metadata[k] = v
    return new_metadata

def add_wikidata_triplet2_data(source_dir, target_dir):
    """
    Create a new Chroma DB at target_dir and add embeddings from wikidata_embedding_triplet2.
    """
    os.makedirs(target_dir, exist_ok=True)
    client = create_client_for_collection(target_dir)
    collection = client.create_collection("wikidata_triplet2")

    embed_files, meta_files = get_sorted_chunks(source_dir)

    # Use tqdm to show progress for the embedding chunks
    for i, (ef, mf) in enumerate(tqdm(zip(embed_files, meta_files),
                                      total=len(embed_files),
                                      desc="Uploading wikidata_triplet2 chunks",
                                      unit="chunk")):
        embed_path = os.path.join(source_dir, ef)
        meta_path = os.path.join(source_dir, mf)

        embeddings = np.load(embed_path)
        with open(meta_path, 'r') as f:
            metadata_list = json.load(f)

        assert len(embeddings) == len(metadata_list), f"Size mismatch in {ef} & {mf}"

        ids = [f"wikidata2_{i}_{j}" for j in range(len(embeddings))]
        embeddings_list = embeddings.tolist()

        # Flatten metadata to ensure compatibility
        metadata_list = [flatten_metadata_values(m) for m in metadata_list]

        collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadata_list,
            documents=[""] * len(embeddings)
        )

    client.persist()
    print(f"Completed uploading wikidata_triplet2 into {target_dir}.")

def add_wikidata_triplet3_data(source_dir, target_dir):
    """
    Create a new Chroma DB at target_dir and add embeddings from wikidata_embedding_triplet3.
    If the collection already exists, get it instead of creating a new one.
    """
    os.makedirs(target_dir, exist_ok=True)
    client = create_client_for_collection(target_dir)
    try:
        collection = client.create_collection("wikidata_triplet3")
    except Exception as e:
        # If the collection already exists, just get it
        if "already exists" in str(e):
            collection = client.get_collection("wikidata_triplet3")
        else:
            raise e

    files = [
        ("snomed_parent_child_triples_embeddings.npy", "snomed_parent_child_triples_metadata.json"),
        ("umls_type_groups_triples_embeddings.npy", "umls_type_groups_triples_metadata.json")
    ]

    # Show progress for uploading these two sets of embeddings
    for embed_file, meta_file in tqdm(files, desc="Uploading wikidata_triplet3 files", unit="file"):
        embed_path = os.path.join(source_dir, embed_file)
        meta_path = os.path.join(source_dir, meta_file)

        embeddings = np.load(embed_path)
        with open(meta_path, "r") as f:
            metadata_list = json.load(f)

        assert len(embeddings) == len(metadata_list), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_list)} metadata entries in {embed_file}"

        # Flatten metadata to ensure compatibility
        metadata_list = [flatten_metadata_values(m) for m in metadata_list]

        ids = [f"wikidata3_{embed_file}_{j}" for j in range(len(embeddings))]
        embeddings_list = embeddings.tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadata_list,
            documents=[""] * len(embeddings)
        )

    client.persist()
    print(f"Completed uploading wikidata_triplet3 into {target_dir}.")

#############################
# Main Execution
#############################

base_dir = "/app/layers"
collections_base = os.path.join(base_dir, "chroma_db", "collections")

# Only handle wikidata_triplet3 in this version:
wikidata_triplet3_source = os.path.join(base_dir, "wikidata_embedding_triplet3")
wikidata_triplet3_target = os.path.join(collections_base, "wikidata_triplet3")
add_wikidata_triplet3_data(wikidata_triplet3_source, wikidata_triplet3_target)

print("Successfully uploaded wikidata_triplet3 embeddings to ChromaDB.")
