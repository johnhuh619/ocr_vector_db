"""Storage layer for OCR Vector DB.

Handles data persistence, schema management, and cascade deletion.

Rules:
- PKG-STO-001~004: Repository interfaces, schema management, CASCADE logic, transactions
- PKG-STO-BAN-001~004: MUST NOT do domain rule enforcement, embedding generation,
  search logic, or file parsing
- DEP-STO-001~004: MUST NOT import ingestion, embedding, retrieval, api
- DEP-STO-ALLOW-001~002: MAY import domain, shared
"""

from .adapters import LangChainAdapter
from .cascade import CascadeDeleter
from .db import DatabaseHelper
from .repositories import (
    BaseRepository,
    ConceptRepository,
    DocumentRepository,
    EmbeddingRepository,
    FragmentRepository,
)
from .schema import DbSchemaManager

__all__ = [
    # Database
    "DatabaseHelper",
    # Schema
    "DbSchemaManager",
    # Cascade
    "CascadeDeleter",
    # Repositories
    "BaseRepository",
    "DocumentRepository",
    "ConceptRepository",
    "FragmentRepository",
    "EmbeddingRepository",
    # Adapters
    "LangChainAdapter",
]
