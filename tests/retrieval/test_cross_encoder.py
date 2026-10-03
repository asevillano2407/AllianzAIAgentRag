"""Tests for the optional local cross-encoder adapter."""

from collections.abc import Sequence

import pytest

from allianz_claims_rag_agent.domain import SourceChunk
from allianz_claims_rag_agent.errors import RerankingError
from allianz_claims_rag_agent.retrieval import LocalCrossEncoderReranker


def _chunk(chunk_id: str) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        text=f"Evidence {chunk_id}",
        source="manual.pdf",
        page=1,
    )


class FakeCrossEncoder:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = scores
        self.received_pairs: Sequence[tuple[str, str]] = []
        self.received_batch_size: int | None = None

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]:
        self.received_pairs = sentences
        self.received_batch_size = batch_size
        assert show_progress_bar is False
        return self.scores


def test_cross_encoder_reranker_orders_chunks_and_replaces_scores() -> None:
    model = FakeCrossEncoder([0.2, 0.9, -0.1])
    reranker = LocalCrossEncoderReranker(
        "fake-model",
        batch_size=2,
        model=model,
    )

    results = reranker.rerank(
        "consulta original",
        [_chunk("a"), _chunk("b"), _chunk("c")],
        limit=2,
    )

    assert [chunk.chunk_id for chunk in results] == ["b", "a"]
    assert [chunk.score for chunk in results] == [0.9, 0.2]
    assert model.received_batch_size == 2
    assert model.received_pairs[0] == ("consulta original", "Evidence a")


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 0.2]])
def test_cross_encoder_reranker_rejects_invalid_scores(scores: list[float]) -> None:
    reranker = LocalCrossEncoderReranker(
        "fake-model",
        model=FakeCrossEncoder(scores),
    )

    with pytest.raises(RerankingError):
        reranker.rerank("consulta", [_chunk("a"), _chunk("b")], limit=2)
