"""Configuration management for the RAG system."""

import os
from dataclasses import dataclass
from typing import Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration (provider-agnostic)."""
    provider: str = "ollama"  # ollama, openai, anthropic
    model: str = "mistral"
    temperature: float = 0.3
    base_url: str = "http://localhost:11434"  # For Ollama


@dataclass
class EmbeddingConfig:
    """Embedding configuration (provider-agnostic)."""
    provider: str = "ollama"  # ollama, openai
    model: Optional[str] = None  # Uses provider default if None
    base_url: str = "http://localhost:11434"  # For Ollama


@dataclass
class ChromaConfig:
    host: str = "localhost"
    port: int = 8000
    persist_directory: str = "./chroma_db"
    collection_name: str = "dhl_knowledge"


@dataclass
class ChunkingConfig:
    method: str = "semantic"
    threshold_type: str = "percentile"
    threshold_amount: int = 85
    fallback_chunk_size: int = 1000
    fallback_chunk_overlap: int = 200


@dataclass
class RetrieverConfig:
    search_type: str = "similarity"
    k: int = 10


# Keep for backwards compatibility
@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    temperature: float = 0.3


class Config:
    """Central configuration manager."""

    _instance: Optional["Config"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Provider-agnostic LLM config
        self.llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            model=os.getenv("LLM_MODEL", "mistral"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

        # Provider-agnostic Embedding config
        self.embedding = EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "ollama")),
            model=os.getenv("EMBEDDING_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

        # Keep for backwards compatibility
        self.ollama = OllamaConfig(
            base_url=self.llm.base_url,
            model=self.llm.model,
            temperature=self.llm.temperature
        )

        self.chroma = ChromaConfig(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            collection_name="dhl_knowledge"
        )

        self.chunking = ChunkingConfig(
            threshold_amount=int(os.getenv("CHUNK_THRESHOLD", "85"))
        )

        self.retriever = RetrieverConfig(
            k=int(os.getenv("RETRIEVER_K", "10"))
        )

        self._initialized = True

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        instance = cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if "llm" in data:
            instance.llm = LLMConfig(**data["llm"])
        if "embedding" in data:
            instance.embedding = EmbeddingConfig(**data["embedding"])
        if "chroma" in data:
            instance.chroma = ChromaConfig(**data["chroma"])
        if "chunking" in data:
            instance.chunking = ChunkingConfig(**data["chunking"])
        if "retriever" in data:
            instance.retriever = RetrieverConfig(**data["retriever"])

        return instance

    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None

    def __repr__(self):
        return f"Config(llm={self.llm}, embedding={self.embedding})"
