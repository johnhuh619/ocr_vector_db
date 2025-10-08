import os
from typing import Optional

from dotenv import load_dotenv

from .models import EmbeddingConfig


def _parse_int(value: Optional[str], default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


def load_config() -> EmbeddingConfig:
    load_dotenv()

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "voyage").lower()
    embedding_dim = _parse_int(os.getenv("EMBEDDING_DIM"), 768)
    if os.getenv("EMBEDDING_DIM") in (None, "") and embedding_provider == "gemini":
        embedding_dim = 768

    config = EmbeddingConfig(
        pg_conn=os.getenv("PG_CONN", ""),
        collection_name=os.getenv("COLLECTION_NAME", ""),
        embedding_model=os.getenv("VOYAGE_MODEL", "voyage-3"),
        embedding_dim=embedding_dim,
        embedding_provider=embedding_provider,
        gemini_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
        custom_schema_write=_parse_bool(os.getenv("CUSTOM_SCHEMA_WRITE", "true"), True),
        rate_limit_rpm=_parse_int(os.getenv("RATE_LIMIT_RPM"), 0),
        max_chars_per_request=_parse_int(os.getenv("MAX_CHARS_PER_REQUEST"), 0),
        max_items_per_request=_parse_int(os.getenv("MAX_ITEMS_PER_REQUEST"), 0),
        max_docs_to_embed=_parse_int(os.getenv("MAX_DOCS_TO_EMBED"), 0),
        parent_mode=os.getenv("PARENT_MODE", "unit").lower(),
        page_regex=os.getenv("PAGE_REGEX", r"(?mi)^\s*(?:page|페이지)\s*([0-9]{1,5})\b"),
        section_regex=os.getenv(
            "SECTION_REGEX",
            r"(?m)^(?:#{1,3}\s+.+|Chapter\s+\d+\b|제\s*\d+\s*장\b|\d+\.\d+\s+.+)",
        ),
        enable_auto_ocr=_parse_bool(os.getenv("ENABLE_AUTO_OCR", "false"), False),
        ivfflat_probes=_parse_optional_int(os.getenv("IVFFLAT_PROBES")),
        hnsw_ef_search=_parse_optional_int(os.getenv("HNSW_EF_SEARCH")),
        hnsw_ef_construction=_parse_optional_int(os.getenv("HNSW_EF_CONSTRUCTION")),
    )
    return config


__all__ = ["load_config"]
