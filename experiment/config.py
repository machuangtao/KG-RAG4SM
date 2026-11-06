"""Paths and constants for the WikiData5M local BFS experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = ROOT / "experiment"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "testRes"
EXPERIMENT_LOG_DIR = EXPERIMENT_DIR / "logs"
# Single canonical human-readable results table (full-dataset local BFS).
RESULTS_TXT = EXPERIMENT_LOG_DIR / "local_bfs_experiment_results.txt"

ENTITY_FILE = DATA_DIR / "wikidata5m_entity.txt"
RELATION_FILE = DATA_DIR / "wikidata5m_relation.txt"
GRAPH_CACHE = DATA_DIR / "wikidata5m_adjacency.pkl"
GRAPH_META_CACHE = DATA_DIR / "wikidata5m_adjacency_meta.json"

TRIPLET_METADATA_DIR = ROOT / "wikidata_embedding_triplet2"
TRIPLET_METADATA_GLOB = "metadata_chunk_*.json"

ENTITY_EMBED_DIR = ROOT / "Wikidata_embed_entity"
PROPERTY_EMBED_DIR = ROOT / "Wikidata_embed_property"
CHROMADB_DIR = ROOT / "chromadb"

DATASETS = ("bank", "movie", "cms", "emed", "synthea")
# Clinical benchmarks reported together (CMS, EMED, Synthea)
CLINICAL_DATASETS = ("cms", "emed", "synthea")
DEPTHS = (1, 2, 3, 4)

MAX_PATHS_PER_PAIR = 5
# When True, depth-cost benchmarks disable per-pair early exit (production default stays capped).
BENCHMARK_DISABLE_PATH_CAP = False
DEFAULT_BATCH_SIZE = 50
ENTITY_BATCH_SIZE = 8192

SIMILARITY_TOP_K = 10

BENCHMARK_N = 50
BENCHMARK_SEED = 42
SPARQL_SAMPLE_N = 10
SPARQL_SAMPLE_SEED = 42
BENCHMARK_DEPTH = 4
MAX_EDGES_PER_NODE = 50
PRUNED_MAX_ENTITIES = 6
ONE_HOP_MAX_EDGES_PER_ENTITY = 50
ONE_HOP_MAX_TRIPLES_PER_QUESTION = 200

# Latency benchmark LLM (fixed — do not use other models)
LLM_MODEL = "gpt-4o-mini"
