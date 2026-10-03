"""Application service for grounded and validated answer generation."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from unicodedata import combining, normalize

from pydantic import ValidationError

from allianz_claims_rag_agent.domain import (
    AccidentAnalysisResponse,
    AnalysisResponse,
    Citation,
    ConfidenceLevel,
    ConventionApplicability,
    ConventionResponsibility,
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
    adjustments: tuple[str, ...] = ()


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

        response_model = (
            AccidentAnalysisResponse
            if query_type is QueryType.ACCIDENT_DESCRIPTION
            else AnalysisResponse
        )
        schema = response_model.model_json_schema()
        generation = self._llm.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(normalized_query, query_type, chunks, schema),
            json_schema=schema,
        )
        response = self._parse_response(generation, response_model)
        decision_adjustments: tuple[str, ...] = ()
        citation_adjustments: tuple[str, ...] = ()
        if query_type is QueryType.ACCIDENT_DESCRIPTION:
            response, decision_adjustments = _repair_accident_decision_consistency(
                response
            )
            response, citation_adjustments = _repair_accident_citations(response, chunks)
        response, adjustments = _apply_conservative_guardrails(response)
        adjustments = (*decision_adjustments, *citation_adjustments, *adjustments)
        if query_type is QueryType.ACCIDENT_DESCRIPTION:
            response, conclusion_adjustments = _render_accident_conclusion(response)
            adjustments = (*adjustments, *conclusion_adjustments)
        self._validate_grounding(response, query_type, chunks)
        return GeneratedAnswer(
            response=response,
            model_name=self._llm.model_name,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_duration_ms=generation.total_duration_ms,
            adjustments=adjustments,
        )

    @staticmethod
    def _parse_response(
        generation: LlmGeneration,
        response_model: type[AnalysisResponse],
    ) -> AnalysisResponse:
        try:
            return response_model.model_validate_json(generation.content)
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
        if expected_query_type is QueryType.ACCIDENT_DESCRIPTION:
            if response.convention_applicability is None:
                raise OutputValidationError(
                    "An accident analysis requires a convention applicability decision"
                )
            if response.convention_responsibility is None:
                raise OutputValidationError(
                    "An accident analysis requires a convention responsibility decision"
                )
            if (
                response.convention_applicability
                is ConventionApplicability.NOT_APPLICABLE
                and response.convention_responsibility
                is not ConventionResponsibility.NOT_APPLICABLE
            ):
                raise OutputValidationError(
                    "Convention responsibility must be not_applicable when the convention "
                    "is not applicable"
                )
            if (
                response.convention_applicability is ConventionApplicability.APPLICABLE
                and response.convention_responsibility
                is ConventionResponsibility.NOT_APPLICABLE
            ):
                raise OutputValidationError(
                    "An applicable convention cannot have not_applicable responsibility"
                )
            if (
                response.convention_applicability
                is not ConventionApplicability.UNDETERMINED
                and not response.applicability_citations
            ):
                raise OutputValidationError(
                    "A definitive applicability decision requires applicability citations"
                )
            if (
                response.convention_responsibility
                in {
                    ConventionResponsibility.VEHICLE_A,
                    ConventionResponsibility.VEHICLE_B,
                    ConventionResponsibility.SHARED,
                }
                and not response.responsibility_citations
            ):
                raise OutputValidationError(
                    "A definitive responsibility decision requires responsibility citations"
                )
        all_citations = [
            *response.citations,
            *response.applicability_citations,
            *response.responsibility_citations,
        ]
        if response.confidence is not ConfidenceLevel.LOW and not all_citations:
            raise OutputValidationError("A medium or high confidence answer requires citations")

        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for citation in all_citations:
            chunk = chunks_by_id.get(citation.chunk_id)
            if chunk is None:
                raise OutputValidationError("A citation references a chunk not given to the LLM")
            if citation.page != chunk.page:
                raise OutputValidationError("A citation page does not match its source chunk")
            if not _citation_quote_is_present(citation.quote, chunk.text):
                raise OutputValidationError("A citation quote is not present in its source chunk")

        if expected_query_type is QueryType.ACCIDENT_DESCRIPTION:
            _validate_applicability_evidence(response)


def _normalize_text(value: str) -> str:
    normalized = normalize("NFKC", value).casefold().replace("\u00ad", "")
    normalized = re.sub(
        r"(?<=\w)[\-‐‑‒–—]\s*(?:\r?\n)+\s*(?=\w)",
        "",
        normalized,
    )
    decomposed = normalize("NFKD", normalized)
    without_accents = "".join(
        character for character in decomposed if not combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


def _citation_quote_is_present(quote: str, chunk_text: str) -> bool:
    """Match harmless typography differences without allowing truncated negations."""
    normalized_quote = _normalize_text(quote)
    normalized_chunk = _normalize_text(chunk_text)
    if not normalized_quote or normalized_quote not in normalized_chunk:
        return False

    quote_starts_with_negation = normalized_quote.split(maxsplit=1)[0] in {"no", "nunca"}
    if quote_starts_with_negation:
        return True

    for match in re.finditer(re.escape(normalized_quote), normalized_chunk):
        preceding_words = normalized_chunk[: match.start()].rstrip().split()
        if not preceding_words or preceding_words[-1] not in {"no", "nunca"}:
            return True
    return False


def _repair_accident_citations(
    response: AnalysisResponse,
    chunks: Sequence[SourceChunk],
) -> tuple[AnalysisResponse, tuple[str, ...]]:
    """Repair safe metadata errors and degrade only decisions with no valid citation."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    adjustments: list[str] = []

    def repair(citations: Sequence[Citation]) -> list[Citation]:
        repaired: list[Citation] = []
        for citation in citations:
            chunk = chunks_by_id.get(citation.chunk_id)
            if chunk is None or not _citation_quote_is_present(citation.quote, chunk.text):
                if "invalid_citations_removed" not in adjustments:
                    adjustments.append("invalid_citations_removed")
                continue
            if citation.page != chunk.page:
                citation = citation.model_copy(update={"page": chunk.page})
                if "citation_pages_repaired" not in adjustments:
                    adjustments.append("citation_pages_repaired")
            repaired.append(citation)
        return repaired

    general_citations = repair(response.citations)
    applicability_citations = repair(response.applicability_citations)
    responsibility_citations = repair(response.responsibility_citations)
    applicability = response.convention_applicability
    responsibility = response.convention_responsibility
    missing_information = list(response.missing_information)
    confidence = response.confidence

    if (
        applicability is not ConventionApplicability.UNDETERMINED
        and not applicability_citations
    ):
        applicability = ConventionApplicability.UNDETERMINED
        responsibility = ConventionResponsibility.UNDETERMINED
        responsibility_citations = []
        confidence = ConfidenceLevel.LOW
        missing_information.append(
            "No hay una cita válida que permita decidir la aplicabilidad del convenio."
        )
        adjustments.append("applicability_downgraded_missing_valid_citation")
    elif (
        responsibility
        in {
            ConventionResponsibility.VEHICLE_A,
            ConventionResponsibility.VEHICLE_B,
            ConventionResponsibility.SHARED,
        }
        and not responsibility_citations
    ):
        responsibility = ConventionResponsibility.UNDETERMINED
        confidence = ConfidenceLevel.LOW
        missing_information.append(
            "No hay una cita válida que permita atribuir responsabilidad según el convenio."
        )
        adjustments.append("responsibility_downgraded_missing_valid_citation")

    repaired_response = response.model_copy(
        update={
            "convention_applicability": applicability,
            "convention_responsibility": responsibility,
            "applicability_citations": applicability_citations,
            "responsibility_citations": responsibility_citations,
            "citations": general_citations,
            "missing_information": list(dict.fromkeys(missing_information)),
            "confidence": confidence,
        }
    )
    return repaired_response, tuple(dict.fromkeys(adjustments))


def _repair_accident_decision_consistency(
    response: AnalysisResponse,
) -> tuple[AnalysisResponse, tuple[str, ...]]:
    """Repair contradictory accident decisions without inventing responsibility."""
    applicability = response.convention_applicability
    responsibility = response.convention_responsibility

    if (
        applicability is ConventionApplicability.NOT_APPLICABLE
        and responsibility is not ConventionResponsibility.NOT_APPLICABLE
    ):
        return (
            response.model_copy(
                update={
                    "convention_responsibility": ConventionResponsibility.NOT_APPLICABLE,
                    "responsibility_citations": [],
                }
            ),
            ("responsibility_aligned_with_non_applicable_convention",),
        )

    if (
        applicability is ConventionApplicability.APPLICABLE
        and responsibility is ConventionResponsibility.NOT_APPLICABLE
    ):
        missing_information = list(response.missing_information)
        explanation = (
            "La responsabilidad no puede marcarse como no aplicable cuando el convenio "
            "sí resulta aplicable."
        )
        if explanation not in missing_information:
            missing_information.append(explanation)
        return (
            response.model_copy(
                update={
                    "convention_responsibility": ConventionResponsibility.UNDETERMINED,
                    "responsibility_citations": [],
                    "missing_information": missing_information,
                    "confidence": ConfidenceLevel.LOW,
                }
            ),
            ("responsibility_repaired_from_inconsistent_decision",),
        )

    return response, ()


def _apply_conservative_guardrails(
    response: AnalysisResponse,
) -> tuple[AnalysisResponse, tuple[str, ...]]:
    """Downgrade unsupported structured responsibility when text remains neutral."""
    definitive_responsibilities = {
        ConventionResponsibility.VEHICLE_A,
        ConventionResponsibility.VEHICLE_B,
        ConventionResponsibility.SHARED,
    }
    if response.convention_responsibility not in definitive_responsibilities:
        return response, ()

    evidence = " ".join(
        _normalize_text(citation.quote) for citation in response.responsibility_citations
    )
    responsibility_markers = (
        "culpable",
        "responsable",
        "responsabilidad de",
        "responsabilidad compartida",
        "se atribuye",
    )
    if any(marker in evidence for marker in responsibility_markers):
        return response, ()

    missing_information = list(response.missing_information)
    explanation = (
        "La evidencia recuperada no contiene una regla que permita atribuir "
        "responsabilidad según el convenio."
    )
    if explanation not in missing_information:
        missing_information.append(explanation)
    adjusted = response.model_copy(
        update={
            "convention_responsibility": ConventionResponsibility.UNDETERMINED,
            "responsibility_citations": [],
            "missing_information": missing_information,
            "confidence": ConfidenceLevel.LOW,
        }
    )
    return adjusted, ("responsibility_downgraded_missing_normative_evidence",)


def _render_accident_conclusion(
    response: AnalysisResponse,
) -> tuple[AnalysisResponse, tuple[str, ...]]:
    """Render a conclusion that cannot contradict the structured decisions."""
    applicability_messages = {
        ConventionApplicability.APPLICABLE: (
            "El convenio CIDE/ASCIDE es aplicable según la evidencia recuperada."
        ),
        ConventionApplicability.NOT_APPLICABLE: (
            "El convenio CIDE/ASCIDE no es aplicable según la evidencia recuperada."
        ),
        ConventionApplicability.UNDETERMINED: (
            "No puede determinarse la aplicabilidad del convenio CIDE/ASCIDE con la "
            "evidencia disponible."
        ),
        None: (
            "No puede determinarse la aplicabilidad del convenio CIDE/ASCIDE con la "
            "evidencia disponible."
        ),
    }
    responsibility_messages = {
        ConventionResponsibility.VEHICLE_A: (
            "El vehículo A resulta responsable según el convenio."
        ),
        ConventionResponsibility.VEHICLE_B: (
            "El vehículo B resulta responsable según el convenio."
        ),
        ConventionResponsibility.SHARED: (
            "La responsabilidad es compartida según el convenio."
        ),
        ConventionResponsibility.UNDETERMINED: (
            "La responsabilidad no puede determinarse con la evidencia disponible."
        ),
        ConventionResponsibility.NOT_APPLICABLE: (
            "Al no ser aplicable el convenio, no procede atribuir responsabilidad mediante él."
        ),
        None: "La responsabilidad no puede determinarse con la evidencia disponible.",
    }
    conclusion = " ".join(
        [
            applicability_messages[response.convention_applicability],
            responsibility_messages[response.convention_responsibility],
        ]
    )
    if response.conclusion == conclusion:
        return response, ()
    return (
        response.model_copy(update={"conclusion": conclusion}),
        ("conclusion_rendered_from_structured_decisions",),
    )


def _validate_applicability_evidence(response: AnalysisResponse) -> None:
    """Reject only explicit polarity conflicts in applicability evidence."""
    evidence = " ".join(
        _normalize_text(citation.quote) for citation in response.applicability_citations
    )
    if not evidence:
        return

    positive_markers = (
        "son de aplicacion los convenios",
        "es de aplicacion el convenio",
        "no se considera como motivo de exclusion",
        "no se considera la alcoholemia",
    )
    negative_markers = (
        "no son de aplicacion los convenios",
        "no seran de aplicacion los convenios",
        "no es de aplicacion el convenio",
        "invalida la aplicacion de los convenios",
    )
    supports_not_applicable = any(marker in evidence for marker in negative_markers)
    evidence_without_negative_markers = evidence
    for marker in negative_markers:
        evidence_without_negative_markers = evidence_without_negative_markers.replace(
            marker, ""
        )
    supports_applicable = any(
        marker in evidence_without_negative_markers for marker in positive_markers
    )

    if (
        response.convention_applicability is ConventionApplicability.APPLICABLE
        and supports_not_applicable
        and not supports_applicable
    ):
        raise OutputValidationError(
            "The applicability decision contradicts evidence that excludes the convention"
        )
    if (
        response.convention_applicability is ConventionApplicability.NOT_APPLICABLE
        and supports_applicable
        and not supports_not_applicable
    ):
        raise OutputValidationError(
            "The applicability decision contradicts evidence that applies the convention"
        )
