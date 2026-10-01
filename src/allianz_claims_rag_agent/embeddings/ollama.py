"""Local Ollama implementation of the embedding provider contract."""

import math
from collections.abc import Sequence
from types import TracebackType

import httpx

from allianz_claims_rag_agent.errors import EmbeddingError


class OllamaEmbeddingProvider:
    """Generate document and query embeddings through Ollama's local HTTP API."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 120.0,
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

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document chunks without silent truncation."""
        if not texts:
            return []
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed one query in the same vector space as the documents."""
        if not text.strip():
            raise ValueError("text cannot be empty")
        return self._embed([text])[0]

    def close(self) -> None:
        """Close the internally-created HTTP client."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaEmbeddingProvider":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = self._client.post(
                "/api/embed",
                json={
                    "model": self._model_name,
                    "input": inputs,
                    "truncate": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                "Cannot connect to Ollama. Confirm that the local service is running."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama rejected the embedding request with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                "Ollama embedding request timed out. Reduce ALLIANZ_EMBEDDING_BATCH_SIZE "
                "or increase ALLIANZ_OLLAMA_TIMEOUT_SECONDS."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError("Ollama returned an invalid embedding response") from exc

        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise EmbeddingError("Ollama returned an unexpected number of embedding vectors")

        return [self._validate_vector(vector) for vector in embeddings]

    @staticmethod
    def _validate_vector(vector: object) -> list[float]:
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError("Ollama returned an empty or invalid embedding vector")

        validated: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise EmbeddingError("Ollama embedding vectors must contain only numbers")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise EmbeddingError("Ollama embedding vectors must contain only finite numbers")
            validated.append(numeric_value)
        return validated
