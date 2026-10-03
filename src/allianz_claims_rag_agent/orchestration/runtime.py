"""Reusable local runtime assembly for CLI and user interfaces."""

from pathlib import Path
from typing import Any

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.embeddings import OllamaEmbeddingProvider
from allianz_claims_rag_agent.generation import AnswerGenerator, OllamaStructuredLlm
from allianz_claims_rag_agent.orchestration.graph import (
    AgentDependencies,
    build_claims_graph,
)
from allianz_claims_rag_agent.orchestration.routing import DeterministicQueryRouter
from allianz_claims_rag_agent.retrieval import (
    QdrantVectorStore,
    build_evidence_retriever,
    collection_name_for_model,
)


def run_local_agent(
    query: str,
    *,
    settings: Settings | None = None,
    rerank: bool = False,
    embedding_model: str | None = None,
    llm_model: str | None = None,
    qdrant_path: Path | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
    reranker_model: str | None = None,
) -> dict[str, Any]:
    """Build local providers, invoke the bounded graph, and release HTTP clients."""
    runtime_settings = settings or Settings()
    selected_embedding = embedding_model or runtime_settings.embedding_model
    selected_llm = llm_model or runtime_settings.llm_model
    selected_qdrant_path = qdrant_path or runtime_settings.qdrant_path
    selected_top_k = top_k if top_k is not None else runtime_settings.retrieval_top_k
    selected_candidate_k = (
        candidate_k
        if candidate_k is not None
        else runtime_settings.retrieval_candidate_k
    )
    selected_reranker = (
        reranker_model or runtime_settings.reranker_model if rerank else None
    )
    collection_name = collection_name_for_model(
        runtime_settings.qdrant_collection_prefix,
        selected_embedding,
    )
    store = QdrantVectorStore(
        collection_name=collection_name,
        path=selected_qdrant_path,
    )

    with (
        OllamaEmbeddingProvider(
            base_url=str(runtime_settings.ollama_base_url),
            model_name=selected_embedding,
            timeout_seconds=runtime_settings.ollama_timeout_seconds,
        ) as embedding_provider,
        OllamaStructuredLlm(
            base_url=str(runtime_settings.ollama_base_url),
            model_name=selected_llm,
            timeout_seconds=runtime_settings.ollama_timeout_seconds,
        ) as llm,
    ):
        graph = build_claims_graph(
            AgentDependencies(
                retriever=build_evidence_retriever(
                    embedding_provider,
                    store,
                    reranker_model=selected_reranker,
                    candidate_k=selected_candidate_k,
                    reranker_batch_size=runtime_settings.reranker_batch_size,
                ),
                generator=AnswerGenerator(llm),
                router=DeterministicQueryRouter(),
                top_k=selected_top_k,
                max_retries=runtime_settings.max_agent_retries,
            )
        )
        return graph.invoke({"query": query})
