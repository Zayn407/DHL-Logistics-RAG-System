"""LLM and Embedding provider factory."""

import os
import logging
from typing import Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create(
        provider: str,
        model: str,
        temperature: float = 0.3,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create an LLM instance based on provider.

        Args:
            provider: LLM provider ('ollama', 'openai', 'anthropic')
            model: Model name
            temperature: Sampling temperature
            base_url: Base URL (for Ollama)
            api_key: API key (for cloud providers)
            **kwargs: Additional provider-specific arguments

        Returns:
            LLM instance
        """
        provider = provider.lower()

        if provider == Provider.OLLAMA.value:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model,
                base_url=base_url or "http://localhost:11434",
                temperature=temperature,
                **kwargs
            )

        elif provider == Provider.OPENAI.value:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                **kwargs
            )

        elif provider == Provider.ANTHROPIC.value:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                **kwargs
            )

        else:
            raise ValueError(f"Unknown LLM provider: {provider}. "
                           f"Supported: {[p.value for p in Provider]}")


class EmbeddingFactory:
    """Factory for creating Embedding instances."""

    # Default models for each provider
    DEFAULT_MODELS = {
        "ollama": "mistral",
        "openai": "text-embedding-3-small",
        "anthropic": "text-embedding-3-small",  # Use OpenAI for Anthropic
    }

    @staticmethod
    def create(
        provider: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create an Embedding instance based on provider.

        Args:
            provider: Embedding provider ('ollama', 'openai')
            model: Model name (uses default if not provided)
            base_url: Base URL (for Ollama)
            api_key: API key (for cloud providers)
            **kwargs: Additional provider-specific arguments

        Returns:
            Embedding instance
        """
        provider = provider.lower()
        model = model or EmbeddingFactory.DEFAULT_MODELS.get(provider)

        if provider == Provider.OLLAMA.value:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=model,
                base_url=base_url or "http://localhost:11434",
                **kwargs
            )

        elif provider in [Provider.OPENAI.value, Provider.ANTHROPIC.value]:
            # Anthropic doesn't have embeddings, use OpenAI
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                **kwargs
            )

        else:
            raise ValueError(f"Unknown embedding provider: {provider}")


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Any:
    """Convenience function to get LLM from environment.

    Uses environment variables if not provided:
    - LLM_PROVIDER (default: ollama)
    - LLM_MODEL (default: mistral)
    """
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    model = model or os.getenv("LLM_MODEL", "mistral")

    logger.info(f"Creating LLM: provider={provider}, model={model}")
    return LLMFactory.create(provider=provider, model=model, **kwargs)


def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Any:
    """Convenience function to get Embeddings from environment.

    Uses environment variables if not provided:
    - EMBEDDING_PROVIDER (default: same as LLM_PROVIDER)
    - EMBEDDING_MODEL (default: provider-specific)
    """
    provider = provider or os.getenv("EMBEDDING_PROVIDER") or os.getenv("LLM_PROVIDER", "ollama")
    model = model or os.getenv("EMBEDDING_MODEL")

    logger.info(f"Creating Embeddings: provider={provider}, model={model}")
    return EmbeddingFactory.create(provider=provider, model=model, **kwargs)
