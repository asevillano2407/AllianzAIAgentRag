"""Application exceptions with explicit failure categories."""


class ApplicationError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(ApplicationError):
    """Raised when runtime configuration is invalid or incomplete."""


class DocumentProcessingError(ApplicationError):
    """Raised when a source document cannot be processed safely."""


class EmbeddingError(ApplicationError):
    """Raised when text cannot be converted into valid vectors."""


class RetrievalError(ApplicationError):
    """Raised when relevant source fragments cannot be retrieved."""


class RerankingError(ApplicationError):
    """Raised when local candidate reranking cannot be completed."""


class GenerationError(ApplicationError):
    """Raised when the local language model cannot generate a response."""


class OutputValidationError(ApplicationError):
    """Raised when a generated response fails application validation."""
