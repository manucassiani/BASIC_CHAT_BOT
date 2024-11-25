import chromadb

from config.config import CHROMA_PERSISTENT_CLIENT_PATH, CHROMA_COLLECTION_NAME

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex


# Agregar Singleton
class IndexManagment:
    """Manages the creation, loading, and maintenance of vector-based indexes.
    Supports multiple vector stores, including Simple, Faiss, and ChromaDB,
    for flexible integration with different storage backends.
    """

    def __init__(self):
        self.global_indexes = None

    def load_index(self, index_path: str, vector_store: str = "faiss") -> BaseIndex:
        """Loads an index from a specified storage backend into memory.
        If the index is already loaded, it returns the existing global index.

        Args:
            index_path (str): The directory path where the index is persisted.
            vector_store (str): The type of vector store to use. Options include:
                - "simple": Default simple index storage.
                - "faiss": Faiss-based index for fast similarity search.
                - "chroma": ChromaDB for persistent vector storage.
        Returns:
            BaseIndex: The loaded index object.
        """
        if self.global_indexes is None:
            match vector_store:
                case "simple":
                    global_base_index = self._build_index_simple(index_path=index_path)
                case "faiss":
                    global_base_index = self._build_index_faiss(index_path=index_path)
                case "chroma":
                    global_base_index = self._build_index_chroma()

            self.global_indexes = global_base_index

            return self.global_indexes

    def _build_index_simple(self, index_path: str) -> BaseIndex:
        """Build a simple index using the default llama-index storage context.

        Args:
            index_path (str): The directory path where the index is persisted.

        Returns:
            BaseIndex: The loaded index.
        """
        storage_context = StorageContext.from_defaults(persist_dir=index_path)

        index = load_index_from_storage(storage_context=storage_context)

        return index

    def _build_index_faiss(self, index_path: str) -> BaseIndex:
        """Build an index using Faiss as the vector store.

        Args:
            index_path (str): The directory path where the index is persisted.

        Returns:
            BaseIndex: The loaded index.
        """
        vector_store = FaissVectorStore.from_persist_dir(index_path)
        storage_context = StorageContext.from_defaults(
            persist_dir=index_path, vector_store=vector_store
        )

        index = load_index_from_storage(storage_context=storage_context)

        return index

    def _build_index_chroma(self) -> BaseIndex:
        """Build an index using ChromaDB as the vector store.

        Returns:
            BaseIndex: The loaded index.
        """

        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSISTENT_CLIENT_PATH)
        chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=Settings.embed_model
        )

        return index
