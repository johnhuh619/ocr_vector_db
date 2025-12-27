"""PDF text extraction with fallback strategies."""

import os
import shutil
import subprocess
from typing import Optional


class PdfExtractor:
    """Extract text from PDF while falling back between multiple strategies."""

    def extract(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF file.

        Tries multiple strategies:
        1. pdfminer.six library
        2. pdftotext command-line tool (if available)

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text or None if extraction fails
        """
        # Strategy 1: pdfminer.six
        try:
            from pdfminer.high_level import extract_text as _extract

            text = _extract(pdf_path)
            if text and text.strip():
                return text
        except Exception:
            pass

        # Strategy 2: pdftotext command
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
        """
        Check if extracted text has low density (likely needs OCR).

        Args:
            text: Extracted text
            min_len: Minimum text length to consider valid
            min_ratio: Minimum ratio of alphanumeric characters

        Returns:
            True if text density is low
        """
        if not text or len(text.strip()) < min_len:
            return True
        letters = sum(ch.isalnum() for ch in text)
        ratio = letters / max(1, len(text))
        return ratio < min_ratio


__all__ = ["PdfExtractor"]
