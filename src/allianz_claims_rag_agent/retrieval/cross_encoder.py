"""Optional local cross-encoder adapter for relevance reranking."""

from collections.abc import Sequence
from math import isfinite
from typing import Protocol

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.errors import ConfigurationError, RerankingError


class CrossEncoderModel(Protocol):
    """Subset of the sentence-transformers CrossEncoder API used by the adapter."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


class LocalCrossEncoderReranker:
    """Score query-chunk pairs with a locally loaded cross-encoder."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        max_length: int = 512,
        device: str | None = None,
        model: CrossEncoderModel | None = None,
    ) -> None:
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("model_name cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self._model_name = normalized_model_name
        self._batch_size = batch_size
        self._model = model or _load_cross_encoder(
            normalized_model_name,
            max_length=max_length,
            device=device,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        chunks: Sequence[SourceChunk],
        limit: int,
    ) -> list[SourceChunk]:
        """Return the highest-scoring candidate chunks in descending order."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not chunks:
            return []

        pairs = [(normalized_query, chunk.text) for chunk in chunks]
        try:
            raw_scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
            scores = [float(score) for score in raw_scores]
        except Exception as exc:
            raise RerankingError("The local reranker could not score candidates") from exc

        if len(scores) != len(chunks):
            raise RerankingError("The local reranker returned an unexpected score count")
        if not all(isfinite(score) for score in scores):
            raise RerankingError("The local reranker returned a non-finite score")

        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            chunk.model_copy(update={"score": score})
            for chunk, score in ranked[:limit]
        ]


def _load_cross_encoder(
    model_name: str,
    *,
    max_length: int,
    device: str | None,
) -> CrossEncoderModel:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ConfigurationError(
            'Reranking requires the optional dependency: pip install -e ".[rerank]"'
        ) from exc

    try:
        return CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
    except Exception as exc:
        raise RerankingError(f"Cannot load local reranker model: {model_name}") from exc
