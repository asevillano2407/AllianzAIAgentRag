"""Deterministic Qdrant collection naming."""

import hashlib
import re


def collection_name_for_model(prefix: str, model_name: str) -> str:
    """Build a readable, collision-resistant collection name for one model."""
    normalized_prefix = _slugify(prefix)
    normalized_model = _slugify(model_name)
    model_digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    readable_part = f"{normalized_prefix}_{normalized_model}"
    max_readable_length = 120 - len(model_digest) - 1
    return f"{readable_part[:max_readable_length].rstrip('_')}_{model_digest}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "unnamed"
