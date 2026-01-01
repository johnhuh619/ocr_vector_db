"""Use case orchestration for OCR Vector DB.

Implements PKG-API-004: Orchestrate other packages for use cases.
"""

from .ingest import IngestUseCase
from .search import SearchUseCase

__all__ = ["IngestUseCase", "SearchUseCase"]
