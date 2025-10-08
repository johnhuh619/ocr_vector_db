from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EmbeddingConfig:
    pg_conn: str
    collection_name: str
    embedding_model: str
    embedding_dim: int
    embedding_provider: str
    gemini_model: str
    custom_schema_write: bool
    rate_limit_rpm: int
    max_chars_per_request: int
    max_items_per_request: int
    max_docs_to_embed: int
    parent_mode: str
    page_regex: str
    section_regex: str
    enable_auto_ocr: bool
    ivfflat_probes: Optional[int] = None
    hnsw_ef_search: Optional[int] = None
    hnsw_ef_construction: Optional[int] = None


@dataclass
class EmbeddingContext:
    config: EmbeddingConfig
    embeddings_client: object
    vector_store: object


@dataclass
class RawSegment:
    kind: str   # text / code / image
    content: str
    language: Optional[str]
    order: int


@dataclass
class UnitizedSegment:
    unit_id: Optional[str]
    role: str
    segment: RawSegment


@dataclass
class ParentDocument:
    parent_id: str
    content: str
    metadata: Dict[str, object]


__all__ = [
    "EmbeddingConfig",
    "EmbeddingContext",
    "RawSegment",
    "UnitizedSegment",
    "ParentDocument",
]
