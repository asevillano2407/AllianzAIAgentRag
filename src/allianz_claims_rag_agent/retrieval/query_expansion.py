"""Deterministic domain vocabulary expansion for accident descriptions."""

from unicodedata import combining, normalize

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
            "colisión por alcance trasero; vehículo alcanzado con daños traseros; "
            "vehículo que alcanza con daños delanteros; responsabilidad CIDE ASCIDE"
        ),
    ),
    (
        ("cambia de carril", "cambio de carril", "invade el carril", "invade un carril"),
        "cambio de carril; invasión de carril; responsabilidad CIDE ASCIDE",
    ),
    (
        ("marcha atras", "da marcha atras", "retrocede"),
        "marcha atrás; responsabilidad CIDE ASCIDE",
    ),
    (
        ("vehiculo estacionado", "estaba estacionado", "estaba aparcado", "aparcado"),
        "vehículo estacionado; estacionamiento; responsabilidad CIDE ASCIDE",
    ),
    (
        ("se incorpora", "incorporacion a la circulacion", "sale del estacionamiento"),
        "incorporación a la circulación; responsabilidad CIDE ASCIDE",
    ),
)


def expand_accident_query(query: str) -> str | None:
    """Return manual terminology matched by observable phrases in an accident report."""
    normalized_query = _normalize_for_matching(query)
    expansions = [
        expansion
        for patterns, expansion in _DOMAIN_RULES
        if any(pattern in normalized_query for pattern in patterns)
    ]
    if not expansions:
        return None
    return "Consulta técnica del manual: " + ". ".join(dict.fromkeys(expansions))


def _normalize_for_matching(value: str) -> str:
    decomposed = normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if not combining(character))
