"""Tests for model-specific collection names."""

from allianz_claims_rag_agent.retrieval.naming import collection_name_for_model


def test_collection_name_is_readable_and_deterministic() -> None:
    first = collection_name_for_model("Allianz Manual", "qwen3-embedding:0.6b")
    second = collection_name_for_model("Allianz Manual", "qwen3-embedding:0.6b")

    assert first == second
    assert first.startswith("allianz_manual_qwen3_embedding_0_6b_")


def test_collection_name_changes_with_model_even_after_normalization() -> None:
    colon_name = collection_name_for_model("manual", "model:a")
    slash_name = collection_name_for_model("manual", "model/a")

    assert colon_name != slash_name


def test_collection_name_handles_symbols_only() -> None:
    assert collection_name_for_model("---", ":::").startswith("unnamed_unnamed_")


def test_collection_name_keeps_model_digest_when_readable_part_is_long() -> None:
    first = collection_name_for_model("prefix" * 30, "model-a" * 30)
    second = collection_name_for_model("prefix" * 30, "model-b" * 30)

    assert len(first) <= 120
    assert first != second
