"""Tests for retrieval command-line parsing and orchestration."""

import sys

import pytest

from allianz_claims_rag_agent.retrieval import cli
from allianz_claims_rag_agent.retrieval.services import IndexingProgress


def test_search_parser_accepts_a_positive_top_k() -> None:
    args = cli.build_search_parser().parse_args(["consulta", "--top-k", "3"])

    assert args.query == "consulta"
    assert args.top_k == 3


def test_search_parser_rejects_zero_top_k() -> None:
    with pytest.raises(SystemExit):
        cli.build_search_parser().parse_args(["consulta", "--top-k", "0"])


def test_index_main_reports_missing_chunk_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["allianz-index", "--input", "missing.jsonl"])

    assert cli.index_main() == 1
    assert "does not exist" in capsys.readouterr().err


def test_index_progress_is_written_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._report_indexing_progress(IndexingProgress(indexed_chunks=8, total_chunks=160))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Indexed 8/160 chunks\n"
