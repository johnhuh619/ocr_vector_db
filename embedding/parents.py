import os
import re
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from .models import ParentDocument
from .utils import Slugifier
from .text_utils import TextPreprocessor

HEADER_HINT = re.compile(r"^(?:#{1,3}\s+.+|Chapter\s+\d+\b|제\s*\d+\s*장\b|\d+\.\d+\s+.+)", re.M)
CAPTION_HINT = re.compile(r"(?im)^(?:figure|fig\.|table|그림)\s*\d+[:\.]?\s+.+")
CAPTION_LINE_RE = re.compile(r"(?im)^(?P<prefix>(?:figure|fig\.|table|그림))\s*\d+[:\.]?\s+.+$")


class ParentDocumentBuilder:
    """Synthesize parent documents and manage metadata on child docs."""

    def __init__(self, parent_mode: str, page_regex: str, section_regex: str):
        self.parent_mode = parent_mode
        self.page_pattern = re.compile(page_regex)
        self.section_pattern = re.compile(section_regex)
        self.page_break_pattern = re.compile(r"^\s*-{3,}\s*Page Break\s*-{3,}\s*$", re.I | re.M)

    def build_parent_entries(self, docs: List[Document]) -> List[ParentDocument]:
        by_unit: Dict[str, List[Document]] = {}
        for doc in docs:
            unit_id = doc.metadata.get("unit_id") or doc.metadata.get("parent_id")
            if not unit_id:
                continue
            by_unit.setdefault(unit_id, []).append(doc)

        parents: List[ParentDocument] = []
        for unit_id, childs in by_unit.items():
            content = self.synthesize_parent_content(childs, unit_id)
            meta = dict(childs[0].metadata)
            meta["views"] = list({doc.metadata.get("view") for doc in childs if doc.metadata.get("view")})
            meta["kind"] = "parent"
            parents.append(ParentDocument(parent_id=unit_id, content=content, metadata=meta))
        return parents

    def assign_parent_by_page_section(self, docs: List[Document], source_name: str) -> None:
        if not docs:
            return

        current_page = None
        current_section = None
        page_counter = 0
        sorted_docs = sorted(
            docs,
            key=lambda doc: (doc.metadata.get("order", 0), doc.metadata.get("view", "")),
        )

        for doc in sorted_docs:
            text = doc.page_content or ""
            page_match = self.page_pattern.search(text)
            if page_match:
                current_page = page_match.group(1)
                try:
                    doc.metadata["page"] = int(current_page)
                except Exception:
                    doc.metadata["page"] = current_page
            elif self.page_break_pattern.search(text):
                page_counter += 1
                current_page = str(page_counter)
                doc.metadata["page"] = page_counter

            section_match = self.section_pattern.search(text)
            if section_match:
                current_section = Slugifier.slugify(section_match.group(0))

            base = Slugifier.slugify(os.path.splitext(os.path.basename(source_name))[0])
            parent_id = None
            if self.parent_mode == "page" and current_page:
                parent_id = f"{base}-p{current_page}"
            elif self.parent_mode == "section" and current_section:
                parent_id = f"{base}-s-{current_section}"
            elif self.parent_mode == "page_section":
                if current_page and current_section:
                    parent_id = f"{base}-p{current_page}-s-{current_section}"
                elif current_page:
                    parent_id = f"{base}-p{current_page}"
                elif current_section:
                    parent_id = f"{base}-s-{current_section}"

            if parent_id:
                doc.metadata["parent_id"] = parent_id

    def synthesize_parent_content(self, childs: List[Document], parent_id: str, max_chars: int = 2000) -> str:
        texts = [doc.page_content for doc in childs if doc.metadata.get("view") == "text"]
        if not texts:
            texts = [doc.page_content for doc in childs]
        header = self._extract_first_match(texts, HEADER_HINT)
        if not header:
            header = self._fallback_uppercase_header(texts)
        caption = self._extract_first_match(texts, CAPTION_HINT)
        body = self._build_body_text(childs, texts, max_chars)

        parts = [part for part in (header, caption, body) if part]
        if not parts:
            return f"unit {parent_id}"
        content = "\n\n".join(parts)
        return content[:max_chars]

    def augment_with_captions(self, docs: List[Document]) -> List[Document]:
        extra: List[Document] = []
        for doc in docs:
            if doc.metadata.get("view") != "text":
                continue
            lines = doc.page_content.splitlines()
            for index, line in enumerate(lines):
                match = CAPTION_LINE_RE.match(line.strip())
                if not match:
                    continue
                prefix = match.group("prefix").lower()
                view = "table" if "table" in prefix else "figure"
                tail = ""
                if index + 1 < len(lines):
                    nxt = lines[index + 1].strip()
                    if (
                        0 < len(nxt) <= 160
                        and "```" not in nxt
                        and not TextPreprocessor.CODE_HINT.search(nxt)
                    ):
                        tail = "\n" + nxt
                metadata = dict(doc.metadata)
                metadata["view"] = view
                metadata["kind"] = "caption"
                extra.append(
                    Document(page_content=line.strip() + tail, metadata=metadata)
                )
        return extra

    def _extract_first_match(self, texts: Sequence[str], pattern: re.Pattern) -> Optional[str]:
        for text in texts:
            match = pattern.search(text)
            if match:
                return match.group(0).strip()
        return None

    def _fallback_uppercase_header(self, texts: Sequence[str]) -> Optional[str]:
        for text in texts:
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or len(stripped) > 80:
                    continue
                uppercase_chars = sum(char.isupper() for char in stripped)
                if uppercase_chars >= max(3, len(stripped) // 2):
                    return stripped
        return None

    def _build_body_text(self, childs: List[Document], texts: Sequence[str], max_chars: int) -> str:
        pre_texts = [
            doc.page_content
            for doc in childs
            if doc.metadata.get("unit_role") == "pre_text"
        ]
        if pre_texts:
            return self._first_sentences("\n".join(pre_texts), max_chars=max(600, max_chars - 200))
        if texts:
            return self._first_sentences("\n".join(texts), max_chars=max(600, max_chars - 200))
        return ""

    @staticmethod
    def _first_sentences(text: str, max_chars: int = 1200) -> str:
        stripped = text.strip()
        parts = re.split(r"(?<=[.!?]\s)\s+|\n+", stripped)
        out: List[str] = []
        total = 0
        for part in parts:
            if not part.strip():
                continue
            if total + len(part) + 1 > max_chars:
                break
            out.append(part.strip())
            total += len(part) + 1
            if total >= max_chars:
                break
        return " ".join(out) if out else stripped[:max_chars]


__all__ = ["ParentDocumentBuilder"]
