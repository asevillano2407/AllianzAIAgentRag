"""Embedding provider contracts."""

from allianz_claims_rag_agent.embeddings.base import EmbeddingProvider
from allianz_claims_rag_agent.embeddings.ollama import OllamaEmbeddingProvider

__all__ = ["EmbeddingProvider", "OllamaEmbeddingProvider"]
