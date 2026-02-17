"""Embedding service for vector generation."""

import logging
from typing import List, Optional, Any
from dataclasses import dataclass

from ..common.llm_factory import EmbeddingFactory

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    text: str
    vector: List[float]
    dimensions: int


class EmbeddingService:
    """Service for generating text embeddings.

    Supports multiple providers: ollama, openai
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._embeddings: Optional[Any] = None

    @property
    def embeddings(self) -> Any:
        """Lazy initialization of embeddings."""
        if self._embeddings is None:
            self._embeddings = EmbeddingFactory.create(
                provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key
            )
            logger.info(f"Initialized embeddings: provider={self.provider}, model={self.model}")
        return self._embeddings

    @classmethod
    def from_config(cls, config) -> "EmbeddingService":
        """Create from Config object."""
        return cls(
            provider=config.embedding.provider,
            model=config.embedding.model,
            base_url=config.embedding.base_url
        )

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        return self.embeddings.embed_documents(texts)

    def test_connection(self) -> EmbeddingResult:
        """Test embedding generation with a sample text."""
        test_text = "Test embedding generation"
        vector = self.embed_query(test_text)

        result = EmbeddingResult(
            text=test_text,
            vector=vector,
            dimensions=len(vector)
        )

        logger.info(f"Embedding test successful: {result.dimensions} dimensions")
        return result

    def get_langchain_embeddings(self) -> Any:
        """Get the underlying LangChain embeddings object."""
        return self.embeddings
