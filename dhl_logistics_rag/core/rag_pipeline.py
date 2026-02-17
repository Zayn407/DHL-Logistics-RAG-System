"""Main RAG pipeline orchestration."""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from langchain_core.documents import Document

from ..common.config import Config
from ..common.constants import PDF_SOURCES
from .document_loader import DocumentLoader
from .chunker import ChunkerFactory, BaseChunker
from .embedder import EmbeddingService
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics about the RAG pipeline."""
    documents_loaded: int
    chunks_created: int
    embedding_dimensions: int
    vector_store_size: int


class RAGPipeline:
    """Orchestrates the complete RAG pipeline.

    Pipeline stages:
    1. Load documents from PDFs
    2. Chunk documents (semantic or fixed-size)
    3. Generate embeddings
    4. Store in vector database
    5. Create retriever
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        # Initialize components
        self.loader = DocumentLoader()
        self.embedding_service = EmbeddingService(
            provider=self.config.embedding.provider,
            model=self.config.embedding.model,
            base_url=self.config.embedding.base_url
        )
        self.vector_store_manager = VectorStoreManager(
            collection_name=self.config.chroma.collection_name,
            persist_directory=self.config.chroma.persist_directory
        )

        self._chunker: Optional[BaseChunker] = None
        self._documents: List[Document] = []
        self._chunks: List[Document] = []
        self._initialized = False

    def _create_chunker(self) -> BaseChunker:
        """Create chunker based on configuration."""
        return ChunkerFactory.create(
            method=self.config.chunking.method,
            embeddings=self.embedding_service.get_langchain_embeddings(),
            threshold_type=self.config.chunking.threshold_type,
            threshold_amount=self.config.chunking.threshold_amount
        )

    def load_documents(
        self,
        pdf_files: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """Stage 1: Load documents from PDFs.

        Args:
            pdf_files: Optional dict of source_name -> file_path.
                      Uses default PDF_SOURCES if not provided.
        """
        sources = pdf_files or PDF_SOURCES
        self._documents = self.loader.load_multiple_pdfs(sources)
        logger.info(f"Loaded {len(self._documents)} documents")
        return self._documents

    def chunk_documents(
        self,
        documents: Optional[List[Document]] = None
    ) -> List[Document]:
        """Stage 2: Chunk documents.

        Args:
            documents: Documents to chunk. Uses loaded documents if not provided.
        """
        docs = documents or self._documents

        if not docs:
            raise ValueError("No documents to chunk. Call load_documents first.")

        if self._chunker is None:
            self._chunker = self._create_chunker()

        self._chunks = self._chunker.chunk(docs)
        logger.info(f"Created {len(self._chunks)} chunks")
        return self._chunks

    def create_vector_store(
        self,
        chunks: Optional[List[Document]] = None
    ):
        """Stage 3 & 4: Create embeddings and store in vector database.

        Args:
            chunks: Chunks to index. Uses created chunks if not provided.
        """
        docs = chunks or self._chunks

        if not docs:
            raise ValueError("No chunks to index. Call chunk_documents first.")

        self.vector_store_manager.create_from_documents(
            documents=docs,
            embeddings=self.embedding_service.get_langchain_embeddings()
        )

        self._initialized = True
        logger.info("Vector store created successfully")

    def get_retriever(self):
        """Stage 5: Get retriever for querying.

        Returns:
            VectorStoreRetriever
        """
        if not self._initialized:
            raise ValueError("Pipeline not initialized. Run the pipeline first.")

        return self.vector_store_manager.get_retriever(
            search_type=self.config.retriever.search_type,
            k=self.config.retriever.k
        )

    def run(
        self,
        pdf_files: Optional[Dict[str, str]] = None,
        force_rebuild: bool = False
    ):
        """Run the complete pipeline.

        Args:
            pdf_files: Optional dict of source_name -> file_path
            force_rebuild: Force rebuild even if data exists

        Returns:
            VectorStoreRetriever
        """
        logger.info("Starting RAG pipeline...")

        # Check if we can reuse existing vector store
        if not force_rebuild and self.vector_store_manager.has_existing_data():
            logger.info("Found existing vector store data, skipping rebuild")
            self.vector_store_manager.load_existing(
                self.embedding_service.get_langchain_embeddings()
            )
            self._initialized = True
            retriever = self.get_retriever()
            logger.info("RAG pipeline loaded from existing data")
            return retriever

        # Full pipeline: Load → Chunk → Embed → Store
        logger.info("Building vector store from scratch...")

        # Stage 1: Load
        self.load_documents(pdf_files)

        # Stage 2: Chunk
        self.chunk_documents()

        # Stage 3 & 4: Embed and Store
        self.create_vector_store()

        # Stage 5: Create retriever
        retriever = self.get_retriever()

        logger.info("RAG pipeline completed successfully")
        return retriever

    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        embedding_dims = 0
        if self._initialized:
            test_result = self.embedding_service.test_connection()
            embedding_dims = test_result.dimensions

        vs_stats = self.vector_store_manager.get_stats()

        return PipelineStats(
            documents_loaded=len(self._documents),
            chunks_created=len(self._chunks),
            embedding_dimensions=embedding_dims,
            vector_store_size=vs_stats.document_count
        )
