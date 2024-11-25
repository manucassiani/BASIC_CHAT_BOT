import os
from pathlib import Path

# Base directory path config
BASE_DIRECTORY = os.path.dirname(Path(__file__).parent)

# Knowledge Base
PATH_KNOWLEDGE_BASE = os.path.join(BASE_DIRECTORY, "knowledge_base")

# Config ETL #
PATH_LOCAL_STORAGE = os.path.join(
    BASE_DIRECTORY, "pi_agent_core", "index_generation", "storage"
)
PATH_LOCAL_STORAGE_TRANSFORMED_DATA = os.path.join(
    BASE_DIRECTORY, "pi_agent_core", "index_generation", "storage", "transformed_data"
)
PATH_LOCAL_STORAGE_READING_DATA_JOBLIB_FILE = os.path.join(
    BASE_DIRECTORY,
    "pi_agent_core",
    "index_generation",
    "storage",
    "transformed_data",
    "llama_index_documents.joblib",
)
PATH_LOCAL_STORAGE_CHUNKED_DATA_JOBLIB_FILE = os.path.join(
    BASE_DIRECTORY,
    "pi_agent_core",
    "index_generation",
    "storage",
    "transformed_data",
    "llama_index_chunked_documents.joblib",
)
PATH_LOCAL_STORAGE_VECTOR_STORE = os.path.join(
    BASE_DIRECTORY, "pi_agent_core", "index_generation", "storage", "vector_store"
)
INDEX_PATH = os.path.join(
    BASE_DIRECTORY, "pi_agent_core", "index_generation", "storage", "vector_store"
)

# Agent config
PI_AGENT_CONFIG = os.path.join(BASE_DIRECTORY, "config", "pi_agent_config.yml")

# Vector Store config
VECTOR_STORE = "chroma"
CHROMA_PERSISTENT_CLIENT_PATH = os.path.join(
    BASE_DIRECTORY, "pi_agent_core", "index_generation", "storage", "chroma_collection"
)
CHROMA_COLLECTION_NAME = "pi_collection"
