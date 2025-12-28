"""Fragment repository implementation.

Provides CRUD operations for Fragment entities.

Rules:
- PKG-STO-001: Repository interface implementation
- DEP-STO-ALLOW-001~002: MAY import domain, shared
- PKG-STO-BAN-001: MUST NOT enforce domain rules (validation is domain's job)
"""

import json
from typing import List, Optional

import psycopg  # type: ignore

from domain import Fragment, View
from shared.config import EmbeddingConfig

from .base import BaseRepository


class FragmentRepository(BaseRepository[Fragment]):
    """Repository for Fragment entities.

    Handles persistence of Fragment entities (embeddable content units) to PostgreSQL.
    Fragments are the children of Concepts and the targets for embedding.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def _pg_conn(self) -> str:
        """Get PostgreSQL connection string."""
        return (self.config.pg_conn or "").replace("postgresql+psycopg", "postgresql")

    def save(self, fragment: Fragment) -> Fragment:
        """Save a Fragment entity.

        Note: This does NOT validate FRAG-LEN-001 or other domain rules.
        Domain validation must be done before calling save().

        Args:
            fragment: Fragment to save

        Returns:
            Saved fragment
        """
        if not self.config.pg_conn:
            return fragment

        sql = """
        INSERT INTO fragments (id, concept_id, content, view, language, "order", metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (id) DO UPDATE SET
          concept_id = EXCLUDED.concept_id,
          content = EXCLUDED.content,
          view = EXCLUDED.view,
          language = EXCLUDED.language,
          "order" = EXCLUDED."order",
          metadata = fragments.metadata || EXCLUDED.metadata,
          updated_at = now();
        """
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        fragment.id,
                        fragment.concept_id,
                        fragment.content,
                        fragment.view.value,
                        fragment.language,
                        fragment.order,
                        json.dumps(fragment.metadata or {}),
                    ),
                )
        return fragment

    def find_by_id(self, entity_id: str) -> Optional[Fragment]:
        """Find a Fragment by ID.

        Args:
            entity_id: Fragment ID

        Returns:
            Fragment if found, None otherwise
        """
        if not self.config.pg_conn:
            return None

        sql = """SELECT id, concept_id, content, view, language, "order", metadata
                 FROM fragments WHERE id = %s"""
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))
                row = cur.fetchone()
                if row:
                    return Fragment(
                        id=row[0],
                        concept_id=row[1],
                        content=row[2],
                        view=View(row[3]),
                        language=row[4],
                        order=row[5],
                        metadata=row[6] or {},
                    )
        return None

    def find_by_concept_id(self, concept_id: str) -> List[Fragment]:
        """Find all Fragments belonging to a Concept.

        This is used by CascadeDeleter for CASCADE-002.

        Args:
            concept_id: Concept ID (parent_id)

        Returns:
            List of Fragments for this Concept
        """
        if not self.config.pg_conn:
            return []

        sql = """SELECT id, concept_id, content, view, language, "order", metadata
                 FROM fragments WHERE concept_id = %s ORDER BY "order" """
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (concept_id,))
                rows = cur.fetchall()
                return [
                    Fragment(
                        id=row[0],
                        concept_id=row[1],
                        content=row[2],
                        view=View(row[3]),
                        language=row[4],
                        order=row[5],
                        metadata=row[6] or {},
                    )
                    for row in rows
                ]

    def find_all(self) -> List[Fragment]:
        """Find all Fragments.

        Returns:
            List of all fragments
        """
        if not self.config.pg_conn:
            return []

        sql = """SELECT id, concept_id, content, view, language, "order", metadata
                 FROM fragments ORDER BY concept_id, "order" """
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                return [
                    Fragment(
                        id=row[0],
                        concept_id=row[1],
                        content=row[2],
                        view=View(row[3]),
                        language=row[4],
                        order=row[5],
                        metadata=row[6] or {},
                    )
                    for row in rows
                ]

    def delete(self, entity_id: str) -> None:
        """Delete a Fragment by ID.

        Note: This does NOT cascade to embeddings.
        Use CascadeDeleter for cascade deletion.

        Args:
            entity_id: Fragment ID to delete
        """
        if not self.config.pg_conn:
            return

        sql = "DELETE FROM fragments WHERE id = %s"
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))

    def exists(self, entity_id: str) -> bool:
        """Check if a Fragment exists.

        Args:
            entity_id: Fragment ID

        Returns:
            True if exists, False otherwise
        """
        if not self.config.pg_conn:
            return False

        sql = "SELECT 1 FROM fragments WHERE id = %s"
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_id,))
                return cur.fetchone() is not None

    def ensure_table(self) -> None:
        """Create fragments table if it doesn't exist."""
        if not self.config.pg_conn:
            return

        sql = """
        CREATE TABLE IF NOT EXISTS fragments (
          id          TEXT PRIMARY KEY,
          concept_id  TEXT NOT NULL,
          content     TEXT NOT NULL,
          view        TEXT NOT NULL,
          language    TEXT,
          "order"     INTEGER NOT NULL,
          metadata    JSONB DEFAULT '{}'::jsonb,
          created_at  TIMESTAMPTZ DEFAULT now(),
          updated_at  TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_fragments_concept_id ON fragments (concept_id);
        CREATE INDEX IF NOT EXISTS idx_fragments_view ON fragments (view);
        CREATE INDEX IF NOT EXISTS idx_fragments_language ON fragments (language);
        """
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)


__all__ = ["FragmentRepository"]
