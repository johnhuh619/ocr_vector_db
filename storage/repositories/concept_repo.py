"""Concept repository implementation.

Provides CRUD operations for Concept entities.

Rules:
- PKG-STO-001: Repository interface implementation
- DEP-STO-ALLOW-001~002: MAY import domain, shared
- PKG-STO-BAN-001: MUST NOT enforce domain rules (validation is domain's job)
"""

import json
from typing import List, Optional

import psycopg  # type: ignore

from domain import Concept
from shared.config import EmbeddingConfig

from .base import BaseRepository


class ConceptRepository(BaseRepository[Concept]):
    """Repository for Concept entities.

    Handles persistence of Concept entities (semantic parents) to PostgreSQL.
    Concepts group related Fragments together.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def _pg_conn(self) -> str:
        """Get PostgreSQL connection string."""
        return (self.config.pg_conn or "").replace("postgresql+psycopg", "postgresql")

    def save(self, concept: Concept) -> Concept:
        """Save a Concept entity.

        Args:
            concept: Concept to save

        Returns:
            Saved concept
        """
        if not self.config.pg_conn:
            return concept

        sql = """
        INSERT INTO concepts (id, document_id, content, metadata, created_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (id) DO UPDATE SET
          document_id = EXCLUDED.document_id,
          content = EXCLUDED.content,
          metadata = concepts.metadata || EXCLUDED.metadata,
          updated_at = now();
        """
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        concept.id,
                        concept.document_id,
                        concept.content,
                        json.dumps(concept.metadata or {}),
                    ),
                )
        return concept

    def find_by_id(self, entity_id: str) -> Optional[Concept]:
        """Find a Concept by ID.

        Args:
            entity_id: Concept ID

        Returns:
            Concept if found, None otherwise
        """
        if not self.config.pg_conn:
            return None

        sql = "SELECT id, document_id, content, metadata FROM concepts WHERE id = %s"
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))
                row = cur.fetchone()
                if row:
                    return Concept(
                        id=row[0],
                        document_id=row[1],
                        content=row[2],
                        metadata=row[3] or {},
                    )
        return None

    def find_by_document_id(self, document_id: str) -> List[Concept]:
        """Find all Concepts belonging to a Document.

        This is used by CascadeDeleter for CASCADE-001.

        Args:
            document_id: Document ID

        Returns:
            List of Concepts for this Document
        """
        if not self.config.pg_conn:
            return []

        sql = "SELECT id, document_id, content, metadata FROM concepts WHERE document_id = %s"
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                rows = cur.fetchall()
                return [
                    Concept(
                        id=row[0],
                        document_id=row[1],
                        content=row[2],
                        metadata=row[3] or {},
                    )
                    for row in rows
                ]

    def find_all(self) -> List[Concept]:
        """Find all Concepts.

        Returns:
            List of all concepts
        """
        if not self.config.pg_conn:
            return []

        sql = "SELECT id, document_id, content, metadata FROM concepts"
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                return [
                    Concept(
                        id=row[0],
                        document_id=row[1],
                        content=row[2],
                        metadata=row[3] or {},
                    )
                    for row in rows
                ]

    def delete(self, entity_id: str) -> None:
        """Delete a Concept by ID.

        Note: This does NOT cascade to child Fragments.
        Use CascadeDeleter for cascade deletion.

        Args:
            entity_id: Concept ID to delete
        """
        if not self.config.pg_conn:
            return

        sql = "DELETE FROM concepts WHERE id = %s"
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))

    def exists(self, entity_id: str) -> bool:
        """Check if a Concept exists.

        Args:
            entity_id: Concept ID

        Returns:
            True if exists, False otherwise
        """
        if not self.config.pg_conn:
            return False

        sql = "SELECT 1 FROM concepts WHERE id = %s"
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))
                return cur.fetchone() is not None

    def ensure_table(self) -> None:
        """Create concepts table if it doesn't exist."""
        if not self.config.pg_conn:
            return

        sql = """
        CREATE TABLE IF NOT EXISTS concepts (
          id          TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          content     TEXT NOT NULL,
          metadata    JSONB DEFAULT '{}'::jsonb,
          created_at  TIMESTAMPTZ DEFAULT now(),
          updated_at  TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_concepts_document_id ON concepts (document_id);
        """
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)


__all__ = ["ConceptRepository"]
