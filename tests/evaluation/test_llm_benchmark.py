"""Tests for frozen-context multi-LLM evaluation."""

from allianz_claims_rag_agent.evaluation.llm_benchmark import (
    DEFAULT_MODELS,
    build_parser,
    summarize,
)


def test_benchmark_parser_uses_three_default_models() -> None:
    args = build_parser().parse_args([])

    assert args.models is None
    assert len(DEFAULT_MODELS) == 3


def test_benchmark_parser_accepts_repeated_models() -> None:
    args = build_parser().parse_args(
        ["--model", "llama3.2:3b", "--model", "qwen3:4b"]
    )

    assert args.models == ["llama3.2:3b", "qwen3:4b"]


def test_benchmark_parser_accepts_repeated_case_ids() -> None:
    args = build_parser().parse_args(["--case-id", "a", "--case-id", "c"])

    assert args.case_ids == ["a", "c"]


def test_benchmark_parser_accepts_optional_reranking() -> None:
    args = build_parser().parse_args(["--rerank", "--candidate-k", "12"])

    assert args.rerank is True
    assert args.candidate_k == 12


def test_summarize_reports_success_rate_and_average_latency() -> None:
    records = [
        {
            "model_name": "model-a",
            "status": "completed",
            "generation": {"total_duration_ms": 100.0},
            "business_evaluation": {"business_correct": True},
        },
        {
            "model_name": "model-a",
            "status": "failed",
            "generation": None,
            "business_evaluation": {"business_correct": False},
        },
        {
            "model_name": "model-b",
            "status": "completed",
            "generation": {"total_duration_ms": 50.0},
            "business_evaluation": {"business_correct": False},
        },
    ]

    summaries = summarize(records)

    assert summaries[0]["technical_success_rate"] == 0.5
    assert summaries[0]["average_generation_duration_ms"] == 100.0
    assert summaries[0]["business_accuracy"] == 0.5
    assert summaries[1]["technical_success_rate"] == 1.0
    assert summaries[1]["business_accuracy"] == 0.0
