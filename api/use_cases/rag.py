"""RAG use case orchestration.

Implements PKG-API-004: Orchestrate packages for RAG use case.

Rules:
- DEP-API-ALLOW-003: MAY import embedding
- DEP-API-ALLOW-004: MAY import retrieval
- DEP-API-ALLOW-006: MAY import shared
- DEP-API-ALLOW-007: MAY import generation
- PKG-API-BAN-001: MUST NOT implement business logic directly
- PKG-API-BAN-002: MUST NOT access database directly
"""

from typing import List, Optional, Protocol

from generation import (
    Conversation,
    GeneratedResponse,
    GenerationPipeline,
    GeminiLLMClient,
    QueryOptimizer,
)
from retrieval import RetrievalPipeline
from shared.config import EmbeddingConfig, GenerationConfig


class EmbeddingClientProtocol(Protocol):
    """Protocol for embedding client (dependency inversion)."""

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query string."""
        ...


class RAGUseCase:
    """Orchestrates the full RAG pipeline.

    Implements PKG-API-004 (orchestration).

    Pipeline:
    1. Optimize query (generation layer) - extract keywords, hints
    2. Retrieve relevant context (retrieval layer)
    3. Generate response (generation layer)
    4. Format with source attribution

    Example:
        >>> use_case = RAGUseCase(embeddings_client, embed_config, gen_config)
        >>> response = use_case.execute("How do I use Python decorators?")
        >>> print(response.answer)
        >>> print(response.format_with_sources())
    """

    def __init__(
        self,
        embeddings_client: EmbeddingClientProtocol,
        embed_config: EmbeddingConfig,
        gen_config: GenerationConfig,
    ):
        """Initialize RAGUseCase.

        Args:
            embeddings_client: Client for generating query embeddings
            embed_config: Embedding/retrieval configuration
            gen_config: Generation configuration
        """
        # Retrieval pipeline
        self.retrieval = RetrievalPipeline(embeddings_client, embed_config)

        # LLM client
        self.llm_client = GeminiLLMClient(model=gen_config.llm_model)

        # Query optimizer (optional)
        self.query_optimizer = None
        if gen_config.enable_query_optimization:
            self.query_optimizer = QueryOptimizer(self.llm_client)

        # Generation pipeline
        self.generation = GenerationPipeline(
            self.llm_client,
            temperature=gen_config.temperature,
            max_tokens=gen_config.max_tokens,
        )

        # Conversation state (optional)
        self.conversation = Conversation() if gen_config.enable_conversation else None
        self.gen_config = gen_config

    def execute(
        self,
        query: str,
        *,
        view: Optional[str] = None,
        language: Optional[str] = None,
        top_k: int = 5,
        use_conversation: bool = False,
    ) -> GeneratedResponse:
        """Execute full RAG pipeline.

        Args:
            query: User question
            view: Optional view filter (overrides optimizer hint)
            language: Optional language filter (overrides optimizer hint)
            top_k: Number of results to retrieve
            use_conversation: Whether to use conversation history

        Returns:
            Generated response with sources
        """
        optimized = None

        # Stage 1: Query optimization (optional)
        if self.query_optimizer:
            optimized = self.query_optimizer.optimize(query)

            # Use optimizer hints if not explicitly provided
            if view is None and optimized.view_hint:
                view = optimized.view_hint
            if language is None and optimized.language_hint:
                language = optimized.language_hint

            # Use optimized query for retrieval
            search_query = optimized.rewritten
        else:
            search_query = query

        # Stage 2: Retrieval
        results = self.retrieval.retrieve(
            query=search_query,
            view=view,
            language=language,
            top_k=top_k,
            expand_context=True,
        )

        # Stage 3: Generation
        conversation = None
        if use_conversation and self.conversation:
            conversation = self.conversation

        response = self.generation.generate(
            query=query,  # Use original query for generation
            results=results,
            conversation=conversation,
            optimized_query=optimized,
        )

        # Track conversation
        if use_conversation and self.conversation:
            self.conversation.add_turn(query, response)

        return response

    def clear_conversation(self) -> None:
        """Reset conversation history."""
        if self.conversation:
            self.conversation.clear()


__all__ = ["RAGUseCase"]
