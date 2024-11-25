import faiss
import logging
import chromadb

from config.config import CHROMA_PERSISTENT_CLIENT_PATH, CHROMA_COLLECTION_NAME

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_storage_context(vector_store: str = "simple") -> StorageContext:
    """Creates a storage context based on the specified vector store type.

    Args:
        vector_store (str): The type of vector store to use. Options are:
            - "faiss": Uses FAISS for similarity search.
            - "chroma": Uses Chroma for persistent storage.
            - "simple" (default): Uses the default simple vector store.

    Returns:
        StorageContext: A storage context configured with the chosen vector store.
    """
    if vector_store == "faiss":
        logging.info("Vector store choosen: FAISS")
        # hardcoding embedding dimensionality to config dict
        d = 3072
        faiss_index = faiss.IndexFlatIP(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    elif vector_store == "chroma":
        logging.info("Vector store choosen: CHROMA")
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSISTENT_CLIENT_PATH)
        chroma_collection = chroma_client.create_collection(CHROMA_COLLECTION_NAME)
        # set up ChromaVectorStore
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    else:
        logging.info("Vector store choosen: SIMPLE")
        storage_context = StorageContext.from_defaults()

    return storage_context
