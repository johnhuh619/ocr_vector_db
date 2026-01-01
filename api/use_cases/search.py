"""Search use case orchestration.

Implements PKG-API-004: Orchestrate packages for search use case.

Rules:
- DEP-API-ALLOW-003: MAY import embedding
- DEP-API-ALLOW-004: MAY import retrieval
- DEP-API-ALLOW-006: MAY import shared
- PKG-API-BAN-001: MUST NOT implement business logic directly
- PKG-API-BAN-002: MUST NOT access database directly
"""

from typing import List, Optional, Protocol

from retrieval import ExpandedResult, RetrievalPipeline
from shared.config import EmbeddingConfig


class EmbeddingClientProtocol(Protocol):
    """Protocol for embedding client (dependency inversion)."""

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query string."""
        ...


class SearchUseCase:
    """Orchestrates the search pipeline.

    Implements PKG-API-004 (orchestration).

    Pipeline:
    1. Validate query (validators.py)
    2. Interpret query (retrieval layer)
    3. Execute vector search (retrieval layer)
    4. Expand context (retrieval layer)
    5. Format results (formatters.py)

    Example:
        >>> use_case = SearchUseCase(embeddings_client, config)
        >>> results = use_case.execute("python list comprehension", view="code", top_k=5)
    """

    def __init__(
        self,
        embeddings_client: EmbeddingClientProtocol,
        config: EmbeddingConfig,
    ):
        self.pipeline = RetrievalPipeline(embeddings_client, config)

    def execute(
        self,
        query: str,
        view: Optional[str] = None,
        language: Optional[str] = None,
        top_k: int = 10,
        expand_context: bool = True,
    ) -> List[ExpandedResult]:
        """Execute search pipeline.

        Args:
            query: Search query string
            view: Optional view filter (text, code, image, etc.)
            language: Optional language filter (python, javascript, etc.)
            top_k: Number of results to retrieve
            expand_context: Whether to fetch parent context

        Returns:
            List of search results with optional context
        """
        # Delegate to retrieval pipeline (orchestration only)
        return self.pipeline.retrieve(
            query=query,
            view=view,
            language=language,
            top_k=top_k,
            expand_context=expand_context,
        )


__all__ = ["SearchUseCase"]
