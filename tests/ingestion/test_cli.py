"""Tests for user-facing ingestion command behaviour."""

from pathlib import Path

from allianz_claims_rag_agent.ingestion import cli


def test_cli_returns_nonzero_for_missing_pdf(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    missing_path = tmp_path / "missing.pdf"
    monkeypatch.setattr("sys.argv", ["allianz-ingest", "--input", str(missing_path)])

    exit_code = cli.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PDF not found" in captured.err
