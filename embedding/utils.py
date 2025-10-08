import hashlib
import re
import unicodedata
from typing import Optional

from langchain_core.documents import Document


class HashingService:
    """Utility helpers for deterministic hashing of chunk content."""

    @staticmethod
    def content_hash(pid: str, view: str, lang: Optional[str], content: str) -> str:
        key = f"{pid}|{view}|{lang or ''}|{content}".encode("utf-8", errors="ignore")
        return hashlib.md5(key).hexdigest()

    @classmethod
    def compute_doc_id(cls, document: Document) -> str:
        pid = (
            document.metadata.get("parent_id")
            or document.metadata.get("unit_id")
            or "no_parent"
        )
        view = document.metadata.get("view") or "text"
        lang = document.metadata.get("lang")
        ch = cls.content_hash(pid, view, lang, document.page_content or "")
        document.metadata["content_hash"] = ch
        return f"doc:{ch}"


class Slugifier:
    """Normalize strings into slug identifiers suitable for metadata keys."""

    @staticmethod
    def slugify(value: str) -> str:
        if not value:
            return ""
        value = unicodedata.normalize("NFKD", value)
        value = value.encode("ascii", "ignore").decode("ascii")
        value = re.sub(r"[^\w\s-]", "", value).strip().lower()
        value = re.sub(r"[-\s]+", "-", value)
        return value


__all__ = ["HashingService", "Slugifier"]
