"""Vector similarity search engine.

Handles Fragment embedding search using pgvector.

Rules:
- PKG-RET-001: Search pipeline (MUST)
- SEARCH-SEP-002: Search targets Fragment embeddings (MUST)
- DEP-RET-ALLOW-002: MAY import storage
- DEP-RET-ALLOW-004: MAY import shared
"""

from dataclasses import dataclass
from typing import List, Optional

import psycopg  # type: ignore

from domain import View
from shared.config import EmbeddingConfig
from shared.hashing import HashingService

from .query import QueryPlan


@dataclass
class SearchResult:
    """Result from vector similarity search.

    Attributes:
        fragment_id: ID of matching Fragment
        parent_id: ID of parent Concept (for context expansion)
        view: View type (text, code, image, etc.)
        language: Language (python, javascript, etc.)
        content: Fragment content
        similarity: Cosine similarity score (0-1)
        metadata: Additional metadata
    """

    fragment_id: str
    parent_id: str
    view: View
    language: Optional[str]
    content: str
    similarity: float
    metadata: dict


class VectorSearchEngine:
    """Vector similarity search using pgvector.

    Implements PKG-RET-001 (search pipeline) and SEARCH-SEP-002 (Fragment targeting).

    Example:
        >>> engine = VectorSearchEngine(config)
        >>> plan = QueryPlan(query_text="...", query_embedding=[...], top_k=10)
        >>> results = engine.search(plan)
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @property
    def _pg_conn(self) -> str:
        """Get PostgreSQL connection string."""
        return (self.config.pg_conn or "").replace("postgresql+psycopg", "postgresql")

    def search(self, query_plan: QueryPlan) -> List[SearchResult]:
        """Execute vector similarity search.

        Searches Fragment embeddings in langchain_pg_embedding table.

        Args:
            query_plan: Parsed query with embedding and filters

        Returns:
            List of search results ordered by similarity (highest first)
        """
        if not self.config.pg_conn:
            return []

        # Build WHERE clause for filters
        where_clauses = []
        params: List = [self._format_vector(query_plan.query_embedding)]

        if query_plan.view_filter:
            where_clauses.append("cmetadata->>'view' = %s")
            params.append(query_plan.view_filter.value)

        if query_plan.language_filter:
            where_clauses.append("cmetadata->>'lang' = %s")
            params.append(query_plan.language_filter)

        where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        # Add top_k parameter
        params.append(query_plan.top_k)

        sql = f"""
        SELECT
            cmetadata->>'fragment_id' AS fragment_id,
            cmetadata->>'parent_id' AS parent_id,
            cmetadata->>'view' AS view,
            cmetadata->>'lang' AS lang,
            document AS content,
            1 - (embedding <=> %s::vector) AS similarity,
            cmetadata AS metadata
        FROM langchain_pg_embedding
        WHERE 1=1{where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        # Add vector parameter again for ORDER BY
        params.insert(1, self._format_vector(query_plan.query_embedding))

        with psycopg.connect(self._pg_conn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results = []
        for row in rows:
            fragment_id, parent_id, view_str, lang, content, similarity, metadata = row

            # Parse view
            try:
                view = View(view_str) if view_str else View.TEXT
            except ValueError:
                view = View.TEXT

            results.append(
                SearchResult(
                    fragment_id=fragment_id or "unknown",
                    parent_id=parent_id or "unknown",
                    view=view,
                    language=lang,
                    content=content or "",
                    similarity=float(similarity) if similarity is not None else 0.0,
                    metadata=metadata or {},
                )
            )

        return results

    @staticmethod
    def _format_vector(vector: List[float]) -> str:
        """Format vector for PostgreSQL literal.

        Args:
            vector: List of floats

        Returns:
            PostgreSQL vector literal string
        """
        return "[" + ",".join(str(float(x)) for x in vector) + "]"


__all__ = ["VectorSearchEngine", "SearchResult"]
