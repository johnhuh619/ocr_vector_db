"""OCR and plain text parser."""

from typing import List

from shared.text_utils import TextPreprocessor

from ..models import RawSegment
from .base import BaseSegmentParser


class OcrParser(BaseSegmentParser):
    """Parse plain text files and OCR output."""

    def parse_text(self, raw: str) -> List[RawSegment]:
        """
        Parse raw text into segments.

        Args:
            raw: Raw text content

        Returns:
            List of RawSegment objects (text or code)
        """
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
        """
        Parse a plain text file.

        Args:
            path: Path to text file

        Returns:
            List of RawSegment objects
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            raw = handle.read()
        return self.parse_text(raw)


__all__ = ["OcrParser"]
