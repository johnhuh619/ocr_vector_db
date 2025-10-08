import json
import re
import time
from typing import Iterable, List, Optional, Sequence

import psycopg  # type: ignore
from langchain_core.documents import Document

from .embeddings_provider import compute_doc_id
from .models import EmbeddingConfig, ParentDocument
from .parsers import iter_by_char_budget
from .utils import HashingService


class DbSchemaManager:
    """Responsible for ensuring Postgres schema prerequisites exist."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def _pg_conn(self) -> str:
        return (self.config.pg_conn or "").replace("postgresql+psycopg", "postgresql")

    def apply_db_level_tuning(self) -> None:
        if not self.config.pg_conn:
            return
        if not any(
            [self.config.ivfflat_probes, self.config.hnsw_ef_search, self.config.hnsw_ef_construction]
        ):
            return
        try:
            with psycopg.connect(self._pg_conn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    if self.config.ivfflat_probes is not None:
                        cur.execute(
                            f"ALTER DATABASE CURRENT SET ivfflat.probes = {int(self.config.ivfflat_probes)};"
                        )
                    if self.config.hnsw_ef_search is not None:
                        cur.execute(
                            f"ALTER DATABASE CURRENT SET hnsw.ef_search = {int(self.config.hnsw_ef_search)};"
                        )
                    if self.config.hnsw_ef_construction is not None:
                        cur.execute(
                            f"ALTER DATABASE CURRENT SET hnsw.ef_construction = {int(self.config.hnsw_ef_construction)};"
                        )
        except Exception as exc:
            print(f"[warn] DB-level tuning not applied: {exc}")

    def ensure_extension_vector(self) -> None:
        if not self.config.pg_conn:
            return
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    def ensure_indexes(self) -> None:
        if not self.config.pg_conn or not self.config.collection_name:
            return
        table = "langchain_pg_embedding"
        safe_collection = self._sanitize_identifier(self.config.collection_name)
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_hnsw_cosine
                    ON {table} USING hnsw (embedding vector_cosine_ops);
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{safe_collection}_meta_gin
                    ON {table} USING GIN (cmetadata);
                    """
                )
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
        if not self.config.pg_conn:
            return
        sql_statements = [
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
            END; $$ LANGUAGE plpgsql;
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
                for sql in sql_statements:
                    cur.execute(sql)

    def ensure_custom_schema(self, embedding_dim: int) -> None:
        if not self.config.pg_conn:
            return
        sql_statements = [
            f"""
            CREATE TABLE IF NOT EXISTS child_chunks (
              id         BIGSERIAL PRIMARY KEY,
              parent_id  TEXT NOT NULL,
              view       TEXT,
              lang       TEXT,
              content    TEXT NOT NULL,
              content_hash TEXT,
              embedding  vector({embedding_dim}) NOT NULL
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
            END; $$ LANGUAGE plpgsql;
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
                for sql in sql_statements:
                    cur.execute(sql)

    @staticmethod
    def _sanitize_identifier(name: Optional[str]) -> str:
        if not name:
            return "default"
        result = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        if not re.match(r"[A-Za-z_]", result):
            result = f"_{result}"
        return result.lower()


class VectorStoreWriter:
    """Handles rate-limited insertions into the PGVector store."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def upsert_batch(self, store, docs: List[Document], batch_size: int = 64) -> None:
        unique: dict[str, Document] = {}
        for doc in docs:
            doc_id = compute_doc_id(doc)
            if doc_id not in unique:
                unique[doc_id] = doc
        docs = list(unique.values())

        char_budget = (
            self.config.max_chars_per_request
            if self.config.max_chars_per_request > 0
            else (4000 if self.config.rate_limit_rpm > 0 else 0)
        )
        groups = list(
            iter_by_char_budget(
                docs,
                char_budget,
                batch_size,
                self.config.max_items_per_request,
            )
        )
        interval = (60.0 / self.config.rate_limit_rpm) if self.config.rate_limit_rpm > 0 else 0.0

        total_groups = len(groups)
        for index, batch in enumerate(groups, 1):
            print(f"[upsert_batch] processing batch {index}/{total_groups} ({len(batch)} docs)")
            ids = [compute_doc_id(doc) for doc in batch]
            attempt = 0
            max_attempts = 6
            backoff = max(20.0, interval) or 20.0
            while True:
                try:
                    try:
                        store.add_documents(batch, ids=ids)
                    except TypeError:
                        store.add_documents(batch)
                    print(f"[upsert_batch] batch {index}/{total_groups} inserted ✅")
                    break
                except Exception as exc:
                    message = str(exc).lower()
                    rate_limited = any(token in message for token in ("ratelimit", "rate limit", "rpm", "tpm"))
                    if not rate_limited or attempt >= max_attempts - 1:
                        print(f"[upsert_batch] batch {index}/{total_groups} failed ❌: {exc}")
                        raise
                    attempt += 1
                    sleep_for = backoff * (1.5 ** attempt)
                    print(
                        f"[rate-limit] retry {attempt}/{max_attempts} in {int(sleep_for)}s (batch {index}/{total_groups})"
                    )
                    time.sleep(sleep_for)
            if interval > 0 and index < total_groups:
                time.sleep(interval)


class ParentChildRepository:
    """Write parent-child records into Postgres tables."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def _pg_conn(self) -> str:
        return (self.config.pg_conn or "").replace("postgresql+psycopg", "postgresql")

    def upsert_parents(self, parents: Sequence[ParentDocument]) -> None:
        if not parents or not self.config.pg_conn:
            return
        sql = """
        INSERT INTO docstore_parent (id, content, metadata)
        VALUES (%s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
          content = EXCLUDED.content,
          metadata = docstore_parent.metadata || EXCLUDED.metadata,
          updated_at = now();
        """
        payload = [
            (parent.parent_id, parent.content, json.dumps(parent.metadata or {}))
            for parent in parents
        ]
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, payload)

    def upsert_parents_custom(self, parents: Sequence[ParentDocument]) -> None:
        if not parents or not self.config.pg_conn:
            return
        sql = """
        INSERT INTO parent_docs (parent_id, content, metadata)
        VALUES (%s, %s, %s)
        ON CONFLICT (parent_id) DO UPDATE SET
          content = EXCLUDED.content,
          metadata = COALESCE(parent_docs.metadata, '{}'::jsonb) || EXCLUDED.metadata,
          updated_at = now();
        """
        payload = [
            (parent.parent_id, parent.content, json.dumps(parent.metadata or {}))
            for parent in parents
        ]
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, payload)

    def upsert_child_chunks_custom(self, embeddings_client, docs: List[Document]) -> None:
        if not docs or not self.config.pg_conn:
            return
        batch_size = 64
        with psycopg.connect(self._pg_conn, autocommit=True) as conn:
            with conn.cursor() as cur:
                for offset in range(0, len(docs), batch_size):
                    batch_docs = docs[offset : offset + batch_size]
                    texts = [doc.page_content for doc in batch_docs]
                    vectors = embeddings_client.embed_documents(texts)
                    rows = []
                    for doc, vector in zip(batch_docs, vectors):
                        parent_id = doc.metadata.get("parent_id") or doc.metadata.get("unit_id")
                        view = doc.metadata.get("view")
                        lang = doc.metadata.get("lang")
                        content = doc.page_content
                        content_hash = doc.metadata.get("content_hash") or HashingService.content_hash(
                            parent_id or "",
                            view or "text",
                            lang,
                            content or "",
                        )
                        rows.append(
                            (
                                parent_id,
                                view,
                                lang,
                                content,
                                content_hash,
                                self._format_vector_literal(vector),
                            )
                        )
                    sql = """
                    INSERT INTO child_chunks (parent_id, view, lang, content, content_hash, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (parent_id, view, lang, content_hash) DO NOTHING
                    """
                    cur.executemany(sql, rows)

    def dual_write_custom_schema(
        self,
        embeddings_client,
        parents: Sequence[ParentDocument],
        docs: List[Document],
    ) -> None:
        if not self.config.pg_conn:
            return
        if not parents and not docs:
            return
        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                if parents:
                    print(f"[dual_write] upserting {len(parents)} parents")
                    sql_parents = """
                        INSERT INTO parent_docs (parent_id, content, metadata)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (parent_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = COALESCE(parent_docs.metadata, '{}'::jsonb) || EXCLUDED.metadata,
                        updated_at = now();
                    """
                    payload = [
                        (parent.parent_id, parent.content, json.dumps(parent.metadata or {}))
                        for parent in parents
                    ]
                    cur.executemany(sql_parents, payload)
                    print(f"[dual_write] parents inserted/updated ✅")
                if docs:
                    batch_size = 64
                    total_batches = (len(docs) + batch_size - 1) // batch_size
                    for index in range(0, len(docs), batch_size):
                        batch_docs = docs[index : index + batch_size]
                        print(
                            f"[dual_write] embedding+inserting batch {index // batch_size + 1}/{total_batches} ({len(batch_docs)} docs)"
                        )
                        texts = [doc.page_content for doc in batch_docs]
                        vectors = embeddings_client.embed_documents(texts)
                        rows = []
                        for doc, vector in zip(batch_docs, vectors):
                            parent_id = doc.metadata.get("parent_id") or doc.metadata.get("unit_id")
                            view = doc.metadata.get("view")
                            lang = doc.metadata.get("lang")
                            content = doc.page_content
                            content_hash = doc.metadata.get("content_hash") or HashingService.content_hash(
                                parent_id or "",
                                view or "text",
                                lang,
                                content or "",
                            )
                            rows.append(
                                (
                                    parent_id,
                                    view,
                                    lang,
                                    content,
                                    content_hash,
                                    self._format_vector_literal(vector),
                                )
                            )
                        sql_children = """
                            INSERT INTO child_chunks (parent_id, view, lang, content, content_hash, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s::vector)
                            ON CONFLICT (parent_id, view, lang, content_hash) DO NOTHING
                        """
                        cur.executemany(sql_children, rows)

    @staticmethod
    def _format_vector_literal(vector: List[float]) -> str:
        return "[" + ",".join(str(float(value)) for value in vector) + "]"


__all__ = ["DbSchemaManager", "ParentChildRepository", "VectorStoreWriter"]
