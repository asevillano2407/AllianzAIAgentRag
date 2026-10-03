"""Deterministic domain vocabulary expansion for accident descriptions."""

import re
from unicodedata import combining, normalize

from allianz_claims_rag_agent.domain import QueryType

_DOMAIN_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "choca por detras",
            "golpea por detras",
            "impacta por detras",
            "colision por detras",
            "alcance trasero",
        ),
        (
            "norma subsidiaria b.9 marcha atrás alcance trasero; vehículo alcanzado con "
            "daños traseros; vehículo que alcanza con daños delanteros; responsabilidad "
            "CIDE ASCIDE"
        ),
    ),
    (
        (
            "cambia de carril",
            "cambiar de carril",
            "cambio de carril",
            "invade el carril",
            "invade un carril",
        ),
        (
            "norma subsidiaria b.10 cambio de carril alcance trasero; invasión de carril; "
            "responsabilidad CIDE ASCIDE"
        ),
    ),
    (
        ("marcha atras", "da marcha atras", "retrocede"),
        "marcha atrás; responsabilidad CIDE ASCIDE",
    ),
    (
        (
            "automovil estacionado",
            "coche aparcado",
            "estaba aparcado",
            "estaba estacionado",
            "vehiculo aparcado",
            "vehiculo estacionado",
        ),
        (
            "norma subsidiaria b.5 vehículos aparcados; vehículo estacionado; culpable el "
            "vehículo que colisiona con el aparcado; responsabilidad CIDE ASCIDE"
        ),
    ),
    (
        ("se incorpora", "incorporacion a la circulacion", "sale del estacionamiento"),
        "incorporación a la circulación; responsabilidad CIDE ASCIDE",
    ),
    (
        (
            "colision en cadena",
            "colision multiple",
            "mas de dos automoviles",
            "mas de dos vehiculos",
            "varios automoviles",
            "varios vehiculos",
        ),
        (
            "sección 26 intervención de más de dos vehículos; colisión en cadena; "
            "aplicación CIDE ASCIDE"
        ),
    ),
    (
        (
            "aleja rapidamente",
            "alejarse rapidamente",
            "contrario desconocido",
            "no hay ninguna nota",
            "se da a la fuga",
            "vehiculo no identificado",
        ),
        (
            "identificación del vehículo contrario; matrícula; marca y modelo; "
            "declaración de accidente CIDE ASCIDE"
        ),
    ),
    (
        (
            "alcoholemia",
            "bajo los efectos del alcohol",
            "bebidas alcoholicas",
            "conductor ebrio",
        ),
        (
            "sección 3 alcoholemia; conducción bajo la influencia de bebidas alcohólicas; "
            "aplicación convenios CIDE ASCIDE"
        ),
    ),
)


def expand_accident_query(query: str) -> str | None:
    """Return manual terminology matched by observable phrases in an accident report."""
    expansions = _matching_expansions(query)
    if not expansions:
        return None
    return "Consulta técnica del manual: " + ". ".join(dict.fromkeys(expansions))


def build_retrieval_queries(query: str, query_type: QueryType) -> list[str]:
    """Retrieve each matched manual concept separately, then preserve the report."""
    if query_type is QueryType.ACCIDENT_DESCRIPTION:
        expansions = _matching_expansions(query)
        if expansions:
            return [
                *(f"Consulta técnica del manual: {expansion}" for expansion in expansions),
                query,
            ]
    return [query]


def _matching_expansions(query: str) -> list[str]:
    normalized_query = _normalize_for_matching(query)
    return list(
        dict.fromkeys(
            expansion
            for patterns, expansion in _DOMAIN_RULES
            if any(pattern in normalized_query for pattern in patterns)
        )
    )


def _normalize_for_matching(value: str) -> str:
    decomposed = normalize("NFKD", value.casefold())
    without_accents = "".join(
        character for character in decomposed if not combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()
