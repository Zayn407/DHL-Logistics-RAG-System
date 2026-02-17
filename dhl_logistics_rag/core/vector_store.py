"""Vector store management."""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreStats:
    """Statistics about the vector store."""
    collection_name: str
    document_count: int
    storage_type: str


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(
        self,
        collection_name: str = "dhl_knowledge",
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._vector_store: Optional[Chroma] = None
        self._client = None

    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_directory)
                logger.info(f"Created persistent ChromaDB client at {self.persist_directory}")
            else:
                self._client = chromadb.Client()
                logger.info("Created in-memory ChromaDB client")
        return self._client

    def has_existing_data(self) -> bool:
        """Check if vector store already has data."""
        try:
            client = self._get_client()
            collection = client.get_collection(self.collection_name)
            count = collection.count()
            if count > 0:
                logger.info(f"Found existing collection '{self.collection_name}' with {count} documents")
                return True
        except Exception:
            pass
        return False

    def load_existing(self, embeddings) -> Chroma:
        """Load existing vector store from disk."""
        client = self._get_client()
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            client=client
        )
        count = self._vector_store._collection.count()
        logger.info(f"Loaded existing vector store: {count} documents")
        return self._vector_store

    def create_from_documents(
        self,
        documents: List[Document],
        embeddings
    ) -> Chroma:
        """Create vector store from documents.

        Args:
            documents: List of documents to index
            embeddings: Embedding service/object

        Returns:
            Chroma vector store
        """
        client = self._get_client()

        # Delete existing collection if it exists
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass

        logger.info(f"Creating vector store with {len(documents)} documents...")

        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=self.collection_name,
            client=client
        )

        logger.info(f"Vector store created: {self.collection_name}")
        return self._vector_store

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = 10
    ) -> VectorStoreRetriever:
        """Get a retriever from the vector store.

        Args:
            search_type: Type of search ('similarity', 'mmr')
            k: Number of documents to retrieve

        Returns:
            VectorStoreRetriever
        """
        if self._vector_store is None:
            raise ValueError("Vector store not initialized. Call create_from_documents first.")

        retriever = self._vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        logger.info(f"Created retriever: {search_type}, k={k}")
        return retriever

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """Perform similarity search.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of similar documents
        """
        if self._vector_store is None:
            raise ValueError("Vector store not initialized")

        return self._vector_store.similarity_search(query, k=k)

    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics."""
        if self._vector_store is None:
            return VectorStoreStats(
                collection_name=self.collection_name,
                document_count=0,
                storage_type="not_initialized"
            )

        return VectorStoreStats(
            collection_name=self.collection_name,
            document_count=self._vector_store._collection.count(),
            storage_type="in_memory" if self.persist_directory is None else "persistent"
        )

    @property
    def vector_store(self) -> Optional[Chroma]:
        """Get the underlying vector store."""
        return self._vector_store
