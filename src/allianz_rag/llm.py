"""LLM boundary and Vertex AI implementation."""

from collections.abc import Sequence
from typing import Protocol

from allianz_rag.models import ClaimAnalysis, RetrievedChunk
from allianz_rag.prompts import SYSTEM_PROMPT, USER_PROMPT


class AnalysisGenerator(Protocol):
    """Generate a structured analysis from retrieved evidence."""

    def generate(
        self,
        query: str,
        route: str,
        chunks: Sequence[RetrievedChunk],
        *,
        feedback: str = "none",
    ) -> ClaimAnalysis:
        """Return a schema-valid analysis."""


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Serialize evidence with immutable metadata boundaries."""

    blocks = []
    for chunk in chunks:
        blocks.append(
            "\n".join(
                [
                    f"[source={chunk.source}]",
                    f"[page={chunk.page}]",
                    f"[chunk_id={chunk.chunk_id}]",
                    f"[section={chunk.section}]",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


class VertexAIAnalysisGenerator:
    """Gemini structured-output adapter through Vertex AI."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        project: str | None,
        location: str,
    ) -> None:
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Vertex AI dependencies are missing. Install the project dependencies."
            ) from exc

        llm = ChatVertexAI(
            model=model,
            temperature=temperature,
            project=project,
            location=location,
            max_retries=2,
        )
        self._structured_llm = llm.with_structured_output(ClaimAnalysis)

    def generate(
        self,
        query: str,
        route: str,
        chunks: Sequence[RetrievedChunk],
        *,
        feedback: str = "none",
    ) -> ClaimAnalysis:
        """Generate one grounded, schema-constrained response."""

        from langchain_core.messages import HumanMessage, SystemMessage

        message = USER_PROMPT.format(
            context=format_context(chunks),
            query=query,
            route=route,
            feedback=feedback,
        )
        result = self._structured_llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
        )
        if isinstance(result, ClaimAnalysis):
            return result
        return ClaimAnalysis.model_validate(result)
