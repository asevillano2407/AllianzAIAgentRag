"""Application service for grounded and validated answer generation."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from allianz_claims_rag_agent.domain import (
    AnalysisResponse,
    ConfidenceLevel,
    QueryType,
    SourceChunk,
)
from allianz_claims_rag_agent.errors import OutputValidationError
from allianz_claims_rag_agent.generation.base import LlmGeneration, StructuredLlmProvider
from allianz_claims_rag_agent.generation.prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Validated domain response plus local inference metadata."""

    response: AnalysisResponse
    model_name: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_duration_ms: float | None


class AnswerGenerator:
    """Build a grounded prompt, call the LLM, and validate citations."""

    def __init__(self, llm: StructuredLlmProvider) -> None:
        self._llm = llm

    def generate(
        self,
        query: str,
        query_type: QueryType,
        chunks: Sequence[SourceChunk],
    ) -> GeneratedAnswer:
        """Generate one answer using only the supplied retrieved chunks."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if not chunks:
            raise ValueError("At least one retrieved chunk is required")

        schema = AnalysisResponse.model_json_schema()
        generation = self._llm.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(normalized_query, query_type, chunks, schema),
            json_schema=schema,
        )
        response = self._parse_response(generation)
        self._validate_grounding(response, query_type, chunks)
        return GeneratedAnswer(
            response=response,
            model_name=self._llm.model_name,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_duration_ms=generation.total_duration_ms,
        )

    @staticmethod
    def _parse_response(generation: LlmGeneration) -> AnalysisResponse:
        try:
            return AnalysisResponse.model_validate_json(generation.content)
        except ValidationError as exc:
            raise OutputValidationError("The LLM response does not match AnalysisResponse") from exc

    @staticmethod
    def _validate_grounding(
        response: AnalysisResponse,
        expected_query_type: QueryType,
        chunks: Sequence[SourceChunk],
    ) -> None:
        if response.query_type is not expected_query_type:
            raise OutputValidationError("The LLM changed the deterministic query type")
        if response.confidence is not ConfidenceLevel.LOW and not response.citations:
            raise OutputValidationError("A medium or high confidence answer requires citations")

        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for citation in response.citations:
            chunk = chunks_by_id.get(citation.chunk_id)
            if chunk is None:
                raise OutputValidationError("A citation references a chunk not given to the LLM")
            if citation.page != chunk.page:
                raise OutputValidationError("A citation page does not match its source chunk")
            if _normalize_text(citation.quote) not in _normalize_text(chunk.text):
                raise OutputValidationError("A citation quote is not present in its source chunk")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()
