import os
from typing import Iterable, List, Tuple, Optional

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain.retrievers.multi_vector import MultiVectorRetriever

from app import build_embeddings, get_config
from app.storage import DbSchemaManager


load_dotenv()
CONFIG = get_config()
PG_CONN = CONFIG.pg_conn
COLLECTION = CONFIG.collection_name
EMBEDDING_DIM = CONFIG.embedding_dim


def _plain_conn_str(conn: str) -> str:
    return (conn or "").replace("postgresql+psycopg", "postgresql")


def maybe_apply_db_level_tuning():
    """Apply ANN-related ALTER DATABASE defaults if configured (requires privileges)."""
    manager = DbSchemaManager(CONFIG)
    manager.apply_db_level_tuning()


def inspect_db_level_tuning() -> None:
    """Print current ANN-related database settings for quick diagnostics."""
    import psycopg

    settings = ["ivfflat.probes", "hnsw.ef_search", "hnsw.ef_construction"]
    try:
        with psycopg.connect(_plain_conn_str(PG_CONN)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT name, setting FROM pg_settings WHERE name = ANY(%s) ORDER BY name",
                    (settings,),
                )
                rows = cur.fetchall()
    except Exception as exc:
        print(f"[warn] Unable to read ANN settings: {exc}")
        return
    if not rows:
        print("[diag] ANN settings not found in pg_settings (extension not loaded?)")
        return
    print("[diag] Current ANN defaults:")
    for name, setting in rows:
        print(f"  - {name} = {setting}")

class PostgresByteStore:
    """Minimal key-bytes DocStore adapter backed by Postgres.

    Expects a table created by embedding.ensure_parent_docstore():
      - table: docstore_parent(id text pk, content text, metadata jsonb, ...)
    Stores/reads UTF-8 bytes in the 'content' column.
    """

    def __init__(self, conn_str: str, table: str = "docstore_parent", key_col: str = "id", val_col: str = "content"):
        self.conn_str = _plain_conn_str(conn_str)
        self.table = table
        self.key_col = key_col
        self.val_col = val_col

    def mset(self, items: Iterable[Tuple[str, bytes]]):
        import psycopg

        sql = f"""
        INSERT INTO {self.table} ({self.key_col}, {self.val_col})
        VALUES (%s, %s)
        ON CONFLICT ({self.key_col}) DO UPDATE SET {self.val_col} = EXCLUDED.{self.val_col}, updated_at = now();
        """
        with psycopg.connect(self.conn_str, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, list(items))

    def mget(self, keys: List[str]) -> List[Optional[bytes]]:
        import psycopg

        sql = f"SELECT {self.key_col}, {self.val_col} FROM {self.table} WHERE {self.key_col} = ANY(%s)"
        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (keys,))
                data = {k: (v.encode("utf-8") if isinstance(v, str) else v) for k, v in cur.fetchall()}
        return [data.get(k) for k in keys]


def build_retriever():
    embeddings = build_embeddings()
    maybe_apply_db_level_tuning()
    inspect_db_level_tuning()
    vectorstore = PGVector(
        connection=PG_CONN,
        collection_name=COLLECTION,
        embeddings=embeddings,
        distance_strategy="cosine",
        use_jsonb=True,
        embedding_length=EMBEDDING_DIM,
    )

    docstore = PostgresByteStore(PG_CONN)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="parent_id",
        search_kwargs={"k": 8},
    )
    return retriever


if __name__ == "__main__":
    # Minimal CLI usage example
    import sys
    if len(sys.argv) < 2:
        print("usage: python retriever_multi_view_demo.py <query> [view] [lang]")
        sys.exit(0)

    query = sys.argv[1]
    view = sys.argv[2] if len(sys.argv) > 2 else None
    lang = sys.argv[3] if len(sys.argv) > 3 else None

    r = build_retriever()
    filt = {}
    if view:
        filt["view"] = view
    if lang:
        filt["lang"] = lang
    if filt:
        r.search_kwargs = {"k": 8, "filter": filt}

    docs = r.get_relevant_documents(query)
    for i, d in enumerate(docs, 1):
        print(f"[{i}] id={d.metadata.get('unit_id') or d.metadata.get('parent_id')}\n{d.page_content[:300]}\n---")
