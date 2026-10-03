"""Composition helpers for optional two-stage retrieval."""

from allianz_claims_rag_agent.embeddings import EmbeddingProvider
from allianz_claims_rag_agent.retrieval.base import VectorStore
from allianz_claims_rag_agent.retrieval.cross_encoder import LocalCrossEncoderReranker
from allianz_claims_rag_agent.retrieval.reranking import (
    CandidateRetriever,
    RerankingRetriever,
)
from allianz_claims_rag_agent.retrieval.services import SemanticRetriever


def build_evidence_retriever(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    *,
    reranker_model: str | None = None,
    candidate_k: int = 12,
    reranker_batch_size: int = 4,
) -> CandidateRetriever:
    """Build semantic retrieval with optional local cross-encoder reranking."""
    semantic_retriever = SemanticRetriever(embedding_provider, vector_store)
    if reranker_model is None:
        return semantic_retriever
    return RerankingRetriever(
        candidate_retriever=semantic_retriever,
        reranker=LocalCrossEncoderReranker(
            model_name=reranker_model,
            batch_size=reranker_batch_size,
        ),
        candidate_k=candidate_k,
    )
