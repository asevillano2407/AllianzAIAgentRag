"""Ollama implementation for schema-constrained local generation."""

from types import TracebackType

import httpx

from allianz_claims_rag_agent.errors import GenerationError
from allianz_claims_rag_agent.generation.base import LlmGeneration


class OllamaStructuredLlm:
    """Call Ollama's local chat API with a JSON schema and deterministic settings."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._model_name = model_name
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        """Return the configured Ollama model name."""
        return self._model_name

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
    ) -> LlmGeneration:
        """Generate one validated-shape JSON string without streaming."""
        try:
            response = self._client.post(
                "/api/chat",
                json={
                    "model": self._model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "think": False,
                    "format": json_schema,
                    "options": {"temperature": 0, "seed": 42},
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.ConnectError as exc:
            raise GenerationError(
                "Cannot connect to Ollama. Confirm that the local service is running."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"Ollama rejected the generation request with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise GenerationError(
                "Ollama generation timed out. Increase ALLIANZ_OLLAMA_TIMEOUT_SECONDS "
                "or select a smaller LLM."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError("Ollama returned an invalid generation response") from exc

        content = _read_content(payload)
        return LlmGeneration(
            content=content,
            prompt_tokens=_optional_int(payload, "prompt_eval_count"),
            completion_tokens=_optional_int(payload, "eval_count"),
            total_duration_ms=_duration_ms(payload),
        )

    def close(self) -> None:
        """Close the internally-created HTTP client."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaStructuredLlm":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _read_content(payload: object) -> str:
    if not isinstance(payload, dict):
        raise GenerationError("Ollama response must be a JSON object")
    message = payload.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("Ollama response does not contain assistant JSON content")
    return content


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _duration_ms(payload: dict[str, object]) -> float | None:
    duration_ns = _optional_int(payload, "total_duration")
    return duration_ns / 1_000_000 if duration_ns is not None else None
