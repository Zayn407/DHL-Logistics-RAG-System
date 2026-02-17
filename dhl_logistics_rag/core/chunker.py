"""Document chunking strategies."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkStats:
    """Statistics about chunking results."""
    total_chunks: int
    min_size: int
    max_size: int
    avg_size: float
    median_size: int


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        pass

    def get_stats(self, chunks: List[Document]) -> ChunkStats:
        """Calculate statistics for chunks."""
        sizes = [len(chunk.page_content) for chunk in chunks]
        sorted_sizes = sorted(sizes)

        return ChunkStats(
            total_chunks=len(chunks),
            min_size=min(sizes) if sizes else 0,
            max_size=max(sizes) if sizes else 0,
            avg_size=sum(sizes) / len(sizes) if sizes else 0,
            median_size=sorted_sizes[len(sorted_sizes) // 2] if sizes else 0
        )


class FixedSizeChunker(BaseChunker):
    """Fixed-size character-based chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self.separators
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents using fixed-size chunking."""
        chunks = self._splitter.split_documents(documents)
        logger.info(f"Fixed-size chunking: {len(documents)} docs → {len(chunks)} chunks")
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic similarity-based chunking."""

    def __init__(
        self,
        embeddings,
        threshold_type: str = "percentile",
        threshold_amount: int = 85,
        fallback_chunker: Optional[FixedSizeChunker] = None
    ):
        self.embeddings = embeddings
        self.threshold_type = threshold_type
        self.threshold_amount = threshold_amount
        self.fallback_chunker = fallback_chunker or FixedSizeChunker()

        # Lazy import to avoid circular dependency
        from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker

        self._splitter = LangChainSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=threshold_type,
            breakpoint_threshold_amount=threshold_amount
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents using semantic chunking with fallback."""
        chunks = []

        for doc in documents:
            try:
                doc_chunks = self._splitter.create_documents(
                    texts=[doc.page_content],
                    metadatas=[doc.metadata]
                )
                chunks.extend(doc_chunks)
            except Exception as e:
                logger.warning(f"Semantic chunking failed for {doc.metadata.get('source_name')}: {e}")
                logger.info("Falling back to fixed-size chunking")
                fallback_chunks = self.fallback_chunker.chunk([doc])
                chunks.extend(fallback_chunks)

        logger.info(f"Semantic chunking: {len(documents)} docs → {len(chunks)} chunks")
        return chunks


class ChunkerFactory:
    """Factory for creating chunker instances."""

    @staticmethod
    def create(
        method: str,
        embeddings=None,
        **kwargs
    ) -> BaseChunker:
        """Create a chunker based on method type.

        Args:
            method: 'semantic' or 'fixed'
            embeddings: Required for semantic chunking
            **kwargs: Additional arguments for the chunker
        """
        if method == "semantic":
            if embeddings is None:
                raise ValueError("Embeddings required for semantic chunking")
            return SemanticChunker(embeddings=embeddings, **kwargs)
        elif method == "fixed":
            return FixedSizeChunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
