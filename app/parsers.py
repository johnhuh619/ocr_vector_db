import os
import re
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from .models import RawSegment, UnitizedSegment
from .text_utils import TextPreprocessor


class BaseSegmentParser(ABC):
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor

    @abstractmethod
    def parse(self, path: str) -> List[RawSegment]:
        ...


class OcrParser(BaseSegmentParser):
    def parse_text(self, raw: str) -> List[RawSegment]:
        raw = self.preprocessor.normalize(raw)
        paragraphs = self.preprocessor.split_paragraph(raw)
        segments: List[RawSegment] = []
        for idx, paragraph in enumerate(paragraphs):
            if self.preprocessor.is_code_block(paragraph):
                lang = self.preprocessor.guess_code_lang(paragraph)
                segments.append(RawSegment("code", paragraph, lang, idx))
            else:
                segments.append(RawSegment("text", paragraph, None, idx))
        return segments

    def parse(self, path: str) -> List[RawSegment]:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            raw = handle.read()
        return self.parse_text(raw)


class MarkdownParser(BaseSegmentParser):
    MD_FENCE_RE = re.compile(r"^\s*```\s*([A-Za-z0-9_+-]*)\s*$")
    MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    PAGE_BREAK_LINE_RE = re.compile(r"^\s*-{3,}\s*Page Break\s*-{3,}\s*$", re.I)

    def _norm_lang(self, tag: Optional[str]) -> Optional[str]:
        if not tag:
            return None
        tag = tag.strip().lower()
        if tag in ("py", "python", "python3"):
            return "python"
        if tag in ("js", "javascript", "node", "jsx", "ts", "tsx", "typescript"):
            return "javascript"
        return tag

    def parse_text(self, raw: str) -> List[RawSegment]:
        segments: List[RawSegment] = []
        order = 0
        in_fence = False
        fence_lang: Optional[str] = None
        fence_buf: List[str] = []
        text_buf: List[str] = []

        def flush_text_buf() -> None:
            nonlocal order
            if not text_buf:
                return
            text = "\n".join(text_buf)
            text_buf.clear()
            pos = 0
            for match in self.MD_IMAGE_RE.finditer(text):
                pre = text[pos : match.start()]
                if pre.strip():
                    normalized = self.preprocessor.normalize(pre)
                    if normalized:
                        segments.append(RawSegment("text", normalized, None, order))
                        order += 1
                alt = (match.group(1) or "").strip()
                url = (match.group(2) or "").strip()
                payload = (alt + "\n" + url).strip()
                segments.append(RawSegment("image", payload, "image", order))
                order += 1
                pos = match.end()
            tail = text[pos:]
            if tail.strip():
                normalized_tail = self.preprocessor.normalize(tail)
                if normalized_tail:
                    segments.append(RawSegment("text", normalized_tail, None, order))
                    order += 1

        for line in raw.splitlines():
            fence_match = self.MD_FENCE_RE.match(line)
            if fence_match:
                if not in_fence:
                    flush_text_buf()
                    fence_lang = self._norm_lang((fence_match.group(1) or "").strip())
                    in_fence = True
                    fence_buf = []
                else:
                    code = "\n".join(fence_buf)
                    lang = fence_lang or self._norm_lang(
                        self.preprocessor.guess_code_lang(code) or "unknown"
                    )
                    segments.append(RawSegment("code", code, lang, order))
                    order += 1
                    in_fence = False
                    fence_lang = None
                    fence_buf = []
                continue

            if in_fence:
                fence_buf.append(line)
                continue

            text_buf.append(line)

        if in_fence and fence_buf:
            code = "\n".join(fence_buf)
            lang = fence_lang or self._norm_lang(
                self.preprocessor.guess_code_lang(code) or "unknown"
            )
            segments.append(RawSegment("code", code, lang, order))
            order += 1
        flush_text_buf()
        return segments

    def parse(self, path: str) -> List[RawSegment]:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            raw = handle.read()
        return self.parse_text(raw)


class PdfExtractor:
    """Extract text from PDF while falling back between multiple strategies."""

    def extract(self, pdf_path: str) -> Optional[str]:
        try:
            from pdfminer.high_level import extract_text as _extract

            text = _extract(pdf_path)
            if text and text.strip():
                return text
        except Exception:
            pass

        if shutil.which("pdftotext"):
            try:
                tmp_txt = pdf_path + ".tmp.txt"
                subprocess.run(["pdftotext", "-layout", pdf_path, tmp_txt], check=True)
                with open(tmp_txt, "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
                try:
                    os.remove(tmp_txt)
                except Exception:
                    pass
                if text and text.strip():
                    return text
            except Exception:
                pass
        return None

    @staticmethod
    def is_low_text_density(text: str, min_len: int = 500, min_ratio: float = 0.2) -> bool:
        if not text or len(text.strip()) < min_len:
            return True
        letters = sum(ch.isalnum() for ch in text)
        ratio = letters / max(1, len(text))
        return ratio < min_ratio


class SegmentUnitizer:
    """Group segments into unit ids that preserve Python/JS adjacency."""

    def __init__(self, attach_pre_text: bool = True, attach_post_text: bool = False, bridge_text_max: int = 0, max_pre_text_chars: int = 4000):
        self.attach_pre_text = attach_pre_text
        self.attach_post_text = attach_post_text
        self.bridge_text_max = bridge_text_max
        self.max_pre_text_chars = max_pre_text_chars

    def unitize(self, segments: List[RawSegment]) -> List[UnitizedSegment]:
        output: List[UnitizedSegment] = []
        text_buffer: List[RawSegment] = []
        text_buffer_chars = 0
        i, total = 0, len(segments)

        while i < total:
            segment = segments[i]
            if segment.kind == "text":
                text_buffer.append(segment)
                text_buffer_chars += len(segment.content)
                while text_buffer_chars > self.max_pre_text_chars and text_buffer:
                    old = text_buffer.pop(0)
                    text_buffer_chars -= len(old.content)
                    output.append(UnitizedSegment(None, "other", old))
                i += 1
                continue

            if segment.kind == "code" and segment.language == "python":
                unit_id = str(uuid.uuid4())
                if self.attach_pre_text and text_buffer:
                    for buffered in text_buffer:
                        output.append(UnitizedSegment(unit_id, "pre_text", buffered))
                    text_buffer.clear()
                    text_buffer_chars = 0
                else:
                    while text_buffer:
                        output.append(UnitizedSegment(None, "other", text_buffer.pop(0)))
                    text_buffer_chars = 0

                while i < total and segments[i].kind == "code" and segments[i].language == "python":
                    output.append(UnitizedSegment(unit_id, "python", segments[i]))
                    i += 1

                bridged = 0
                while (
                    bridged < self.bridge_text_max
                    and i < total
                    and segments[i].kind == "text"
                ):
                    output.append(UnitizedSegment(unit_id, "bridge_text", segments[i]))
                    i += 1
                    bridged += 1

                if i < total and segments[i].kind == "code" and segments[i].language == "javascript":
                    while i < total and segments[i].kind == "code" and segments[i].language == "javascript":
                        output.append(UnitizedSegment(unit_id, "javascript", segments[i]))
                        i += 1

                    if self.attach_post_text:
                        while i < total and segments[i].kind == "text":
                            if (
                                i + 1 < total
                                and segments[i + 1].kind == "code"
                                and segments[i + 1].language == "python"
                            ):
                                text_buffer.append(segments[i])
                                text_buffer_chars += len(segments[i].content)
                                i += 1
                                break
                            output.append(UnitizedSegment(unit_id, "post_text", segments[i]))
                            i += 1
                continue

            if segment.kind == "code" and segment.language == "javascript":
                while text_buffer:
                    output.append(UnitizedSegment(None, "other", text_buffer.pop(0)))
                    text_buffer_chars = 0
                output.append(UnitizedSegment(None, "other", segment))
                i += 1
                continue

            while text_buffer:
                output.append(UnitizedSegment(None, "other", text_buffer.pop(0)))
                text_buffer_chars = 0
            output.append(UnitizedSegment(None, "other", segment))
            i += 1

        while text_buffer:
            output.append(UnitizedSegment(None, "other", text_buffer.pop(0)))
            text_buffer_chars = 0
        return output


def iter_by_char_budget(
    docs: List[object],
    max_chars: int,
    default_batch: int,
    max_items: int = 0,
) -> Iterable[List[object]]:
    """Yield batches of documents that respect character and size limits."""
    if max_chars <= 0:
        for i in range(0, len(docs), default_batch):
            yield docs[i : i + default_batch]
        return

    current: List[object] = []
    total_chars = 0
    for doc in docs:
        text = getattr(doc, "page_content", "")
        text_len = len(text or "")
        if current and total_chars + text_len > max_chars:
            yield current
            current = []
            total_chars = 0
        current.append(doc)
        total_chars += text_len
        if max_items and len(current) >= max_items:
            yield current
            current = []
            total_chars = 0
        elif len(current) >= default_batch:
            yield current
            current = []
            total_chars = 0
    if current:
        yield current


__all__ = [
    "BaseSegmentParser",
    "MarkdownParser",
    "OcrParser",
    "PdfExtractor",
    "SegmentUnitizer",
    "iter_by_char_budget",
]
