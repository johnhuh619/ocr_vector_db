"""Database schema management for OCR Vector DB.

Handles schema creation, index management, and database tuning.

Rules:
- PKG-STO-002: Database schema management responsibility
- DEP-STO-ALLOW-002: MAY import shared (for config)
"""

import re
from typing import Optional

import psycopg  # type: ignore

from shared.config import EmbeddingConfig


class DbSchemaManager:
    """Responsible for ensuring Postgres schema prerequisites exist."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def _pg_conn(self) -> str:
        return (self.config.pg_conn or "").replace("postgresql+psycopg", "postgresql")

    def apply_db_level_tuning(self) -> None:
        """Apply database-level performance tuning parameters."""
        if not self.config.pg_conn:
            return
        params = {
            "ivfflat.probes": self.config.ivfflat_probes,
            "hnsw.ef_search": self.config.hnsw_ef_search,
            "hnsw.ef_construction": self.config.hnsw_ef_construction,
        }
        values = {name: value for name, value in params.items() if value is not None}
        if not values:
            return
        try:
            with psycopg.connect(self._pg_conn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    for setting, value in values.items():
                        cur.execute(f"ALTER DATABASE CURRENT SET {setting} = {int(value)};")
                        print(f"[tuning] set {setting}={int(value)}")
        except Exception as exc:
            print(f"[warn] DB-level tuning not applied: {exc}")

    def ensure_extension_vector(self) -> None:
        """Ensure pgvector extension is installed."""
        if not self.config.pg_conn:
            return
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    def ensure_indexes(self) -> None:
        """Create HNSW vector indexes and JSONB GIN indexes for metadata filtering."""
        if not self.config.pg_conn or not self.config.collection_name:
            return
        table = "langchain_pg_embedding"
        safe_collection = self._sanitize_identifier(self.config.collection_name)
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                # HNSW vector index for cosine similarity
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_hnsw_cosine
                    ON {table} USING hnsw (embedding vector_cosine_ops);
                    """
                )
                # GIN index for JSONB metadata
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_meta_gin
                    ON {table} USING GIN (cmetadata);
                    """
                )
                # BTREE indexes for common filters
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_unit_id_btree
                    ON {table} ((cmetadata->>'unit_id'));
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_unit_role_btree
                    ON {table} ((cmetadata->>'unit_role'));
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_lang_btree
                    ON {table} ((cmetadata->>'lang'));
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_parent_id_btree
                    ON {table} ((cmetadata->>'parent_id'));
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_view_btree
                    ON {table} ((cmetadata->>'view'));
                    """
                )

    def ensure_parent_docstore(self) -> None:
        """Create docstore_parent table for parent documents."""
        if not self.config.pg_conn:
            return
        statements = [
            """
            CREATE TABLE IF NOT EXISTS docstore_parent (
              id         text PRIMARY KEY,
              content    text NOT NULL,
              metadata   jsonb DEFAULT '{}'::jsonb,
              created_at timestamptz DEFAULT now(),
              updated_at timestamptz DEFAULT now()
            );
            """,
            """
            CREATE OR REPLACE FUNCTION set_updated_at()
            RETURNS trigger AS $$
            BEGIN
              NEW.updated_at = now();
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """,
            "DROP TRIGGER IF EXISTS trg_docstore_parent_updated ON docstore_parent;",
            """
            CREATE TRIGGER trg_docstore_parent_updated
            BEFORE UPDATE ON docstore_parent
            FOR EACH ROW EXECUTE PROCEDURE set_updated_at();
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_docstore_parent_meta_gin
            ON docstore_parent USING GIN (metadata);
            """,
        ]
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                for sql in statements:
                    cur.execute(sql)

    def ensure_custom_schema(self, embedding_dim: int) -> None:
        """Create custom child_chunks and parent_docs tables.

        Args:
            embedding_dim: Dimensionality of embedding vectors
        """
        if not self.config.pg_conn:
            return
        statements = [
            f"""
            CREATE TABLE IF NOT EXISTS child_chunks (
              id           BIGSERIAL PRIMARY KEY,
              parent_id    TEXT NOT NULL,
              view         TEXT,
              lang         TEXT,
              content      TEXT NOT NULL,
              content_hash TEXT,
              embedding    vector({embedding_dim}) NOT NULL
            );
            """,
            """
            ALTER TABLE IF EXISTS child_chunks
            ADD COLUMN IF NOT EXISTS content_hash TEXT;
            """,
            "CREATE INDEX IF NOT EXISTS child_chunks_parent_idx ON child_chunks (parent_id);",
            "CREATE INDEX IF NOT EXISTS child_chunks_view_lang_idx ON child_chunks (view, lang);",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS child_chunks_dedupe_idx
              ON child_chunks (parent_id, view, lang, content_hash);
            """,
            """
            CREATE INDEX IF NOT EXISTS child_chunks_vec_idx
              ON child_chunks USING ivfflat (embedding vector_cosine_ops)
              WITH (lists = 100);
            """,
            """
            CREATE TABLE IF NOT EXISTS parent_docs (
              parent_id  TEXT PRIMARY KEY,
              content    TEXT   NOT NULL,
              metadata   JSONB,
              updated_at TIMESTAMPTZ DEFAULT now()
            );
            """,
            "CREATE INDEX IF NOT EXISTS parent_docs_meta_idx ON parent_docs USING GIN (metadata);",
            """
            CREATE OR REPLACE FUNCTION set_updated_at()
            RETURNS trigger AS $$
            BEGIN
              NEW.updated_at = now();
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """,
            "DROP TRIGGER IF EXISTS trg_parent_docs_updated ON parent_docs;",
            """
            CREATE TRIGGER trg_parent_docs_updated
            BEFORE UPDATE ON parent_docs
            FOR EACH ROW EXECUTE PROCEDURE set_updated_at();
            """,
        ]
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                for sql in statements:
                    cur.execute(sql)

    @staticmethod
    def _sanitize_identifier(name: Optional[str]) -> str:
        """Sanitize identifier for use in SQL (prevent injection)."""
        if not name:
            return "default"
        result = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        if not re.match(r"[A-Za-z_]", result):
            result = f"_{result}"
        return result.lower()


__all__ = ["DbSchemaManager"]
