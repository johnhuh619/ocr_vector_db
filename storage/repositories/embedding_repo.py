"""Embedding repository implementation.

Provides operations for managing embeddings in the vector database.

Rules:
- PKG-STO-001: Repository interface implementation
- DEP-STO-ALLOW-002: MAY import shared
- PKG-STO-BAN-002: MUST NOT perform embedding generation (that's embedding layer's job)
"""

import psycopg  # type: ignore

from shared.config import EmbeddingConfig


class EmbeddingRepository:
    """Repository for managing embeddings in PGVector.

    Note: This repository does NOT follow the BaseRepository pattern because
    embeddings are stored in LangChain's PGVector tables and have a different
    structure than domain entities.

    This repository provides operations for:
    - Deleting embeddings by fragment_id (for CASCADE-003)
    - Checking if embeddings exist
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def _pg_conn(self) -> str:
        """Get PostgreSQL connection string."""
        return (self.config.pg_conn or "").replace("postgresql+psycopg", "postgresql")

    def delete_by_fragment_id(self, fragment_id: str) -> None:
        """Delete all embeddings for a Fragment (CASCADE-003).

        Embeddings are stored in langchain_pg_embedding table with
        fragment_id in the cmetadata JSONB column.

        Args:
            fragment_id: Fragment ID whose embeddings should be deleted
        """
        if not self.config.pg_conn:
            return

        # Delete from langchain_pg_embedding where metadata contains fragment_id
        sql = """
        DELETE FROM langchain_pg_embedding
        WHERE cmetadata->>'fragment_id' = %s
        """
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (fragment_id,))

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete embedding by doc_id.

        Args:
            doc_id: Document ID (deterministic hash) to delete
        """
        if not self.config.pg_conn:
            return

        sql = """
        DELETE FROM langchain_pg_embedding
        WHERE cmetadata->>'doc_id' = %s
        """
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (doc_id,))

    def exists_by_doc_id(self, doc_id: str) -> bool:
        """Check if an embedding exists by doc_id.

        Args:
            doc_id: Document ID to check

        Returns:
            True if embedding exists, False otherwise
        """
        if not self.config.pg_conn:
            return False

        sql = """
        SELECT 1 FROM langchain_pg_embedding
        WHERE cmetadata->>'doc_id' = %s
        LIMIT 1
        """
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (doc_id,))
                return cur.fetchone() is not None

    def count_by_fragment_id(self, fragment_id: str) -> int:
        """Count embeddings for a Fragment.

        Args:
            fragment_id: Fragment ID

        Returns:
            Number of embeddings for this fragment
        """
        if not self.config.pg_conn:
            return 0

        sql = """
        SELECT COUNT(*) FROM langchain_pg_embedding
        WHERE cmetadata->>'fragment_id' = %s
        """
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (fragment_id,))
                row = cur.fetchone()
                return row[0] if row else 0


__all__ = ["EmbeddingRepository"]
