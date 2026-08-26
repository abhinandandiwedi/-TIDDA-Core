class FusionError(Exception):
    """Base exception for fusion errors."""


class InvalidDetectionError(FusionError, ValueError):
    """Raised when a detection fails validation."""


class EntityNotFoundError(FusionError, KeyError):
    """Raised when an entity ID is not present."""
