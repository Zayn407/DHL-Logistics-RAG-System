"""Tests for chunking strategies."""

import pytest
from langchain_core.documents import Document

from dhl_logistics_rag.core.chunker import FixedSizeChunker, ChunkerFactory


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def setup_method(self):
        self.chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)

    def test_chunk_single_document(self):
        """Test chunking a single document."""
        doc = Document(
            page_content="A" * 250,
            metadata={"source": "test"}
        )
        chunks = self.chunker.chunk([doc])

        assert len(chunks) > 1
        assert all(len(c.page_content) <= 100 for c in chunks)

    def test_preserve_metadata(self):
        """Test that metadata is preserved."""
        doc = Document(
            page_content="Test content " * 50,
            metadata={"source": "test", "page": 1}
        )
        chunks = self.chunker.chunk([doc])

        for chunk in chunks:
            assert chunk.metadata.get("source") == "test"

    def test_empty_document_list(self):
        """Test chunking empty list."""
        chunks = self.chunker.chunk([])
        assert len(chunks) == 0

    def test_get_stats(self):
        """Test chunk statistics."""
        docs = [Document(page_content="A" * 200)]
        chunks = self.chunker.chunk(docs)
        stats = self.chunker.get_stats(chunks)

        assert stats.total_chunks == len(chunks)
        assert stats.min_size > 0
        assert stats.max_size <= 100


class TestChunkerFactory:
    """Tests for ChunkerFactory."""

    def test_create_fixed_chunker(self):
        """Test creating fixed-size chunker."""
        chunker = ChunkerFactory.create("fixed", chunk_size=500)
        assert isinstance(chunker, FixedSizeChunker)

    def test_invalid_method(self):
        """Test invalid chunking method."""
        with pytest.raises(ValueError):
            ChunkerFactory.create("invalid_method")

    def test_semantic_without_embeddings(self):
        """Test semantic chunker without embeddings raises error."""
        with pytest.raises(ValueError):
            ChunkerFactory.create("semantic")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
