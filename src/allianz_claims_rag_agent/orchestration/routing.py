"""Deterministic routing between manual questions and accident reports."""

import re
from unicodedata import combining, normalize

from allianz_claims_rag_agent.domain import QueryType

_QUESTION_PREFIXES = (
    "como ",
    "cuando ",
    "cual ",
    "cuales ",
    "dime ",
    "donde ",
    "explica ",
    "por que ",
    "puede ",
    "que ",
)

_ACCIDENT_MARKERS = (
    "accidente",
    "alcance",
    "automovil a",
    "automovil b",
    "coche a",
    "coche b",
    "choca",
    "colision",
    "danos",
    "detenido",
    "estacionado",
    "golpea",
    "impacta",
    "implicados",
    "marcha atras",
    "roza",
    "turismo a",
    "turismo b",
    "vehiculo a",
    "vehiculo b",
)

_VEHICLE_LABELS = ("automovil", "coche", "turismo", "vehiculo")


class DeterministicQueryRouter:
    """Classify input without spending an LLM call on a business rule."""

    def route(self, query: str) -> QueryType:
        """Return a stable query type from punctuation and domain markers."""
        normalized = _normalize(query.strip())
        if not normalized:
            raise ValueError("query cannot be empty")
        marker_count = sum(marker in normalized for marker in _ACCIDENT_MARKERS)
        describes_two_vehicles = any(
            f"{vehicle_label} a" in normalized or f"{vehicle_label} b" in normalized
            for vehicle_label in _VEHICLE_LABELS
        )
        if marker_count >= 3 or (marker_count >= 2 and describes_two_vehicles):
            return QueryType.ACCIDENT_DESCRIPTION
        if query.rstrip().endswith("?") or normalized.startswith(_QUESTION_PREFIXES):
            return QueryType.MANUAL_QUESTION
        if marker_count:
            return QueryType.ACCIDENT_DESCRIPTION
        return QueryType.MANUAL_QUESTION


def _normalize(value: str) -> str:
    decomposed = normalize("NFKD", value.casefold())
    without_accents = "".join(
        character for character in decomposed if not combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()
