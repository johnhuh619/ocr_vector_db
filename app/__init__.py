"""Reusable embedding pipeline package."""

from functools import lru_cache
from typing import Optional

from .config import load_config
from .embeddings_provider import EmbeddingProviderFactory, validate_embedding_dimension
from .models import EmbeddingConfig
from .pipeline import EmbeddingPipeline
from .utils import format_vector_literal


@lru_cache(maxsize=1)
def get_config() -> EmbeddingConfig:
    """Load and cache embedding configuration from environment."""
    return load_config()


def build_embeddings(config: Optional[EmbeddingConfig] = None, validate: bool = True):
    """Instantiate an embeddings client using the configured provider."""
    cfg = config or get_config()
    client = EmbeddingProviderFactory.create(cfg)
    if validate:
        model_name = cfg.embedding_model if cfg.embedding_provider != "gemini" else cfg.gemini_model
        validate_embedding_dimension(
            client,
            cfg.embedding_dim,
            provider=cfg.embedding_provider,
            model=model_name,
        )
    return client


PG_CONN = get_config().pg_conn

__all__ = [
    "EmbeddingConfig",
    "EmbeddingPipeline",
    "build_embeddings",
    "format_vector_literal",
    "get_config",
    "load_config",
    "PG_CONN",
]
