"""PDF parser backends: Marker (default) and MinerU (optional)."""

from __future__ import annotations

from .base import BackendOutput, ParserError, ParserUnavailableError, PdfParserBackend

DEFAULT_BACKEND = "marker"
AVAILABLE_BACKENDS = ("marker", "mineru")


def get_backend(name: str | None = None, **options) -> PdfParserBackend:
    """Instantiate a parser backend by name (``marker`` or ``mineru``).

    ``options`` are backend-specific settings (e.g. ``tier`` for MinerU).
    """
    name = (name or DEFAULT_BACKEND).strip().lower()
    if name == "marker":
        from .marker_parser import MarkerBackend
        if options:
            raise ValueError(f"The marker backend takes no options, got: {sorted(options)}")
        return MarkerBackend()
    if name == "mineru":
        from .mineru_parser import MinerUBackend
        return MinerUBackend(**options)
    raise ValueError(
        f"Unknown PDF parser backend '{name}'. Available: {', '.join(AVAILABLE_BACKENDS)}"
    )


__all__ = [
    "AVAILABLE_BACKENDS",
    "BackendOutput",
    "DEFAULT_BACKEND",
    "ParserError",
    "ParserUnavailableError",
    "PdfParserBackend",
    "get_backend",
]
