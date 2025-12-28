"""Document repository implementation.

Provides CRUD operations for Document entities.

Rules:
- PKG-STO-001: Repository interface implementation
- DEP-STO-ALLOW-001~002: MAY import domain, shared
- PKG-STO-BAN-001: MUST NOT enforce domain rules (validation is domain's job)
"""

import json
from typing import List, Optional

import psycopg  # type: ignore

from domain import Document
from shared.config import EmbeddingConfig

from .base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document entities.

    Handles persistence of Document entities to PostgreSQL.
    Uses a simple documents table for metadata storage.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def _pg_conn(self) -> str:
        """Get PostgreSQL connection string."""
        return (self.config.pg_conn or "").replace("postgresql+psycopg", "postgresql")

    def save(self, document: Document) -> Document:
        """Save a Document entity.

        Args:
            document: Document to save

        Returns:
            Saved document
        """
        if not self.config.pg_conn:
            return document

        sql = """
        INSERT INTO documents (id, source, metadata, created_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (id) DO UPDATE SET
          source = EXCLUDED.source,
          metadata = documents.metadata || EXCLUDED.metadata,
          updated_at = now();
        """
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        document.id,
                        document.source,
                        json.dumps(document.metadata or {}),
                    ),
                )
        return document

    def find_by_id(self, entity_id: str) -> Optional[Document]:
        """Find a Document by ID.

        Args:
            entity_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        if not self.config.pg_conn:
            return None

        sql = "SELECT id, source, metadata FROM documents WHERE id = %s"
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))
                row = cur.fetchone()
                if row:
                    return Document(
                        id=row[0],
                        source=row[1],
                        metadata=row[2] or {},
                    )
        return None

    def find_all(self) -> List[Document]:
        """Find all Documents.

        Returns:
            List of all documents
        """
        if not self.config.pg_conn:
            return []

        sql = "SELECT id, source, metadata FROM documents"
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                return [
                    Document(
                        id=row[0],
                        source=row[1],
                        metadata=row[2] or {},
                    )
                    for row in rows
                ]

    def delete(self, entity_id: str) -> None:
        """Delete a Document by ID.

        Note: This does NOT cascade to child Concepts.
        Use CascadeDeleter for cascade deletion.

        Args:
            entity_id: Document ID to delete
        """
        if not self.config.pg_conn:
            return

        sql = "DELETE FROM documents WHERE id = %s"
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))

    def exists(self, entity_id: str) -> bool:
        """Check if a Document exists.

        Args:
            entity_id: Document ID

        Returns:
            True if exists, False otherwise
        """
        if not self.config.pg_conn:
            return False

        sql = "SELECT 1 FROM documents WHERE id = %s"
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))
                return cur.fetchone() is not None

    def ensure_table(self) -> None:
        """Create documents table if it doesn't exist."""
        if not self.config.pg_conn:
            return

        sql = """
        CREATE TABLE IF NOT EXISTS documents (
          id         TEXT PRIMARY KEY,
          source     TEXT NOT NULL,
          metadata   JSONB DEFAULT '{}'::jsonb,
          created_at TIMESTAMPTZ DEFAULT now(),
          updated_at TIMESTAMPTZ DEFAULT now()
        );
        """
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)


__all__ = ["DocumentRepository"]
