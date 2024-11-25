import os
import logging
import joblib

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.ingestion import IngestionPipeline

from config.config import (
    PATH_KNOWLEDGE_BASE,
    PATH_LOCAL_STORAGE,
    PATH_LOCAL_STORAGE_TRANSFORMED_DATA,
    PATH_LOCAL_STORAGE_READING_DATA_JOBLIB_FILE,
    PATH_LOCAL_STORAGE_CHUNKED_DATA_JOBLIB_FILE,
    PATH_LOCAL_STORAGE_VECTOR_STORE,
    PI_AGENT_CONFIG,
    VECTOR_STORE,
)
from pi_agent_core.index_generation.transformation_logic import (
    get_transformation_context,
)
from pi_agent_core.index_generation.vector_store_logic import get_storage_context
from pi_agent_core.helpers.utils import load_config_file, delete_tmp_files


def extraction() -> None:
    """Reads raw data from the specified knowledge base directory and saves it in a temporary location.

    Process:
        - Reads data using SimpleDirectoryReader.
        - Creates a temporary folder to store the raw data.
        - Saves the raw documents using joblib for later use.
    """
    # Reading raw data
    reader = SimpleDirectoryReader(input_dir=PATH_KNOWLEDGE_BASE)
    documents = reader.load_data()

    # Create temporary transformed_data folder
    os.makedirs(PATH_LOCAL_STORAGE_TRANSFORMED_DATA, exist_ok=True)
    joblib.dump(documents, PATH_LOCAL_STORAGE_READING_DATA_JOBLIB_FILE)

    logging.info("--- Finish extraction process. Next step transformation process ---")


def transformation(documents: list) -> None:
    """Transforms the raw documents by applying a series of transformations.

    Args:
        documents (list): List of raw documents to transform.

    Process:
        - Loads transformation context from configuration.
        - Applies transformations using an IngestionPipeline.
        - Saves the transformed documents in a temporary location.
    """
    # Load model config
    agent_params = load_config_file(PI_AGENT_CONFIG)

    # Get transformation context
    transformation_context = get_transformation_context(agent_params)

    # Generate nodes from base documents
    pipeline = IngestionPipeline(transformations=transformation_context)
    transformed_documents = pipeline.run(documents=documents, model_params=agent_params)

    # Save the transformed documents
    joblib.dump(transformed_documents, PATH_LOCAL_STORAGE_CHUNKED_DATA_JOBLIB_FILE)

    logging.info(
        "--- Finish transformation process. Next step vectorization process ---"
    )


def vectorization(documents: list[Document]) -> None:
    """Converts the transformed documents into a vectorized format and stores the resulting index.

    Args:
        documents (list[Document]): List of transformed documents to vectorize.

    Process:
        - Loads the service and storage contexts.
        - Creates a VectorStoreIndex from the documents.
        - Saves the vectorized index to a temporary local directory.
    """

    # Get the service and storage contexts.
    logging.info("getting service context")
    service_context = Settings
    logging.info("getting storage context")
    storage_context = get_storage_context(vector_store=VECTOR_STORE)

    # Create a VectorStoreIndex from the documents using the specified contexts
    logging.info("creating VectorStoreIndex")
    vector_store_index = VectorStoreIndex(
        nodes=documents,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True,
    )
    # Create temporary vector_store folder
    os.makedirs(PATH_LOCAL_STORAGE_VECTOR_STORE, exist_ok=True)
    # Persist the index to the temporary local directory "vector_store".
    vector_store_index.storage_context.persist(
        persist_dir=PATH_LOCAL_STORAGE_VECTOR_STORE
    )

    logging.info("--- Finish vectorization process. Next step load process ---")


def create_index_from_knowleadge_base() -> None:
    """Orchestrates the entire process of creating an index from the knowledge base.

    Process:
        - Deletes temporary files from previous runs.
        - Executes the extraction, transformation, and vectorization steps sequentially.
        - Saves the vectorized index in the specified storage location.
    """
    delete_tmp_files(directory=PATH_LOCAL_STORAGE)
    extraction()
    documents = joblib.load(PATH_LOCAL_STORAGE_READING_DATA_JOBLIB_FILE)
    transformation(documents=documents)
    transformed_documents = joblib.load(PATH_LOCAL_STORAGE_CHUNKED_DATA_JOBLIB_FILE)
    vectorization(transformed_documents)
