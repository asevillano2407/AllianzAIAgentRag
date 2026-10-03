"""Provider-independent contracts for two-stage retrieval and reranking."""

from collections.abc import Sequence
from typing import Protocol

from allianz_claims_rag_agent.domain import SourceChunk


class CandidateRetriever(Protocol):
    """Recover a broad candidate set for one or more query variants."""

    def retrieve_many(self, queries: Sequence[str], limit: int) -> list[SourceChunk]: ...


class ChunkReranker(Protocol):
    """Order retrieved chunks by relevance to one natural-language query."""

    @property
    def model_name(self) -> str: ...

    def rerank(
        self,
        query: str,
        chunks: Sequence[SourceChunk],
        limit: int,
    ) -> list[SourceChunk]: ...


class RerankingRetriever:
    """Fuse retrieval and per-query reranking while preserving query coverage."""

    _RRF_OFFSET = 60

    def __init__(
        self,
        candidate_retriever: CandidateRetriever,
        reranker: ChunkReranker,
        candidate_k: int = 12,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve_many(self, queries: Sequence[str], limit: int) -> list[SourceChunk]:
        """Preserve each query's best evidence, then fill by reciprocal rank fusion."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if limit > self._candidate_k:
            raise ValueError("limit cannot exceed candidate_k")
        normalized_queries = list(
            dict.fromkeys(query.strip() for query in queries if query.strip())
        )
        if not normalized_queries:
            raise ValueError("At least one non-empty query is required")

        candidates = self._candidate_retriever.retrieve_many(
            normalized_queries,
            self._candidate_k,
        )
        if not candidates:
            return []
        rankings: list[list[SourceChunk]] = []
        for query in normalized_queries:
            ranking = self._reranker.rerank(query, candidates, len(candidates))
            self._validate_reranked_chunks(candidates, ranking, len(candidates))
            if len(ranking) != len(candidates):
                raise ValueError("reranker must rank every candidate for fusion")
            rankings.append(ranking)

        candidate_by_id = {chunk.chunk_id: chunk for chunk in candidates}
        fused_scores = {
            chunk.chunk_id: 1 / (self._RRF_OFFSET + rank)
            for rank, chunk in enumerate(candidates, start=1)
        }
        for ranking in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                fused_scores[chunk.chunk_id] += 1 / (self._RRF_OFFSET + rank)

        selected_ids: list[str] = []
        for ranking in rankings:
            if ranking and ranking[0].chunk_id not in selected_ids:
                selected_ids.append(ranking[0].chunk_id)
            if len(selected_ids) == limit:
                break

        fused_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )
        selected_ids.extend(
            chunk_id
            for chunk_id in fused_ids
            if chunk_id not in selected_ids
        )
        return [
            candidate_by_id[chunk_id].model_copy(
                update={"score": fused_scores[chunk_id]}
            )
            for chunk_id in selected_ids[:limit]
        ]

    @staticmethod
    def _validate_reranked_chunks(
        candidates: Sequence[SourceChunk],
        reranked: Sequence[SourceChunk],
        limit: int,
    ) -> None:
        if len(reranked) > limit:
            raise ValueError("reranker returned more chunks than requested")
        candidate_ids = {chunk.chunk_id for chunk in candidates}
        reranked_ids = [chunk.chunk_id for chunk in reranked]
        if len(reranked_ids) != len(set(reranked_ids)):
            raise ValueError("reranker returned duplicate chunks")
        if not set(reranked_ids).issubset(candidate_ids):
            raise ValueError("reranker returned a chunk outside the candidate set")
