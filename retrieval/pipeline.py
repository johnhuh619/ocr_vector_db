"""Retrieval pipeline orchestration.

Coordinates query interpretation, search, context expansion, and grouping.

Rules:
- PKG-RET-001: Search pipeline orchestration (MUST)
- DEP-RET-ALLOW-001~004: MAY import domain, storage, embedding, shared
"""

from typing import List, Optional

from shared.config import EmbeddingConfig

from .context import ContextExpander, ExpandedResult
from .grouping import ResultGrouper
from .query import EmbeddingClientProtocol, QueryInterpreter
from .search import SearchResult, VectorSearchEngine


class RetrievalPipeline:
    """Orchestrates the complete retrieval pipeline.

    Pipeline stages:
    1. Query interpretation (QueryInterpreter)
    2. Vector similarity search (VectorSearchEngine)
    3. Context expansion (ContextExpander)
    4. Result grouping (ResultGrouper)

    Example:
        >>> pipeline = RetrievalPipeline(embeddings_client, config)
        >>> results = pipeline.retrieve("python list comprehension", view="code", top_k=5)
    """

    def __init__(self, embeddings_client: EmbeddingClientProtocol, config: EmbeddingConfig):
        self.config = config
        self.query_interpreter = QueryInterpreter(embeddings_client, config)
        self.search_engine = VectorSearchEngine(config)
        self.context_expander = ContextExpander(config)
        self.grouper = ResultGrouper()

    def retrieve(
        self,
        query: str,
        view: Optional[str] = None,
        language: Optional[str] = None,
        top_k: int = 10,
        expand_context: bool = True,
        deduplicate: bool = True,
    ) -> List[ExpandedResult]:
        """Execute complete retrieval pipeline.

        Args:
            query: User query string
            view: Optional view filter (text, code, image, etc.)
            language: Optional language filter (python, javascript, etc.)
            top_k: Number of results to retrieve
            expand_context: Whether to fetch parent context
            deduplicate: Whether to remove duplicate results

        Returns:
            List of search results with optional parent context
        """
        # Stage 1: Query interpretation
        query_plan = self.query_interpreter.interpret(
            query=query,
            view=view,
            language=language,
            top_k=top_k,
        )

        # Stage 2: Vector similarity search
        search_results = self.search_engine.search(query_plan)

        # Optional: Deduplicate
        if deduplicate:
            search_results = self.grouper.deduplicate_by_content(search_results)

        # Stage 3: Context expansion
        if expand_context:
            expanded_results = self.context_expander.expand(search_results)
        else:
            expanded_results = [ExpandedResult(result=r) for r in search_results]

        return expanded_results

    def retrieve_raw(
        self,
        query: str,
        view: Optional[str] = None,
        language: Optional[str] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Execute search without context expansion.

        Lighter version of retrieve() that returns only Fragment results.

        Args:
            query: User query string
            view: Optional view filter
            language: Optional language filter
            top_k: Number of results to retrieve

        Returns:
            List of Fragment search results
        """
        query_plan = self.query_interpreter.interpret(
            query=query,
            view=view,
            language=language,
            top_k=top_k,
        )
        return self.search_engine.search(query_plan)


__all__ = ["RetrievalPipeline"]
