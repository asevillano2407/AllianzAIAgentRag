"""Provider-independent contract for structured local LLM generation."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LlmGeneration:
    """Raw structured content and observable inference metadata."""

    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ms: float | None = None


class StructuredLlmProvider(Protocol):
    """Generate JSON constrained by a caller-provided schema."""

    @property
    def model_name(self) -> str:
        """Return the stable identifier of the generation model."""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
    ) -> LlmGeneration:
        """Generate one non-streaming JSON response."""
