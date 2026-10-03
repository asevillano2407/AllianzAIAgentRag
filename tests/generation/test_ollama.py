"""Tests for schema-constrained generation through Ollama."""

import json

import httpx
import pytest

from allianz_claims_rag_agent.errors import GenerationError
from allianz_claims_rag_agent.generation import OllamaStructuredLlm


def _provider(transport: httpx.MockTransport) -> OllamaStructuredLlm:
    client = httpx.Client(base_url="http://ollama.test", transport=transport)
    return OllamaStructuredLlm(
        base_url="http://unused.test",
        model_name="test-llm",
        client=client,
    )


def test_generate_json_sends_schema_and_deterministic_options() -> None:
    schema: dict[str, object] = {"type": "object", "required": ["answer"]}

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["format"] == schema
        assert payload["options"] == {"temperature": 0, "seed": 42}
        assert [message["role"] for message in payload["messages"]] == ["system", "user"]
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"answer":"CIDE"}'},
                "prompt_eval_count": 120,
                "eval_count": 20,
                "total_duration": 1_500_000_000,
            },
        )

    result = _provider(httpx.MockTransport(handle)).generate_json("system", "user", schema)

    assert result.content == '{"answer":"CIDE"}'
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 20
    assert result.total_duration_ms == 1_500.0


def test_generate_json_reports_timeout() -> None:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow model", request=request)

    with pytest.raises(GenerationError, match="timed out"):
        _provider(httpx.MockTransport(time_out)).generate_json("system", "user", {})


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": {}},
        {"message": {"content": ""}},
        [],
    ],
)
def test_generate_json_rejects_missing_assistant_content(payload: object) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))

    with pytest.raises(GenerationError, match="response|content"):
        _provider(transport).generate_json("system", "user", {})
