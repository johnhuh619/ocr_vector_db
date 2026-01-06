"""PyMuPDF-based PDF parser with Gemini Vision OCR support.

This module provides structured block-level PDF parsing using PyMuPDF (fitz),
with optional Gemini Vision API integration for OCR on image blocks.

Block types handled:
- type=0: Text blocks -> kind="text"
- type=1: Image blocks -> kind="image" (OCR via Gemini if enabled)
"""

import base64
import os
from typing import List, Optional, Protocol

from shared.text_utils import TextPreprocessor

from ..models import RawSegment
from .base import BaseSegmentParser
from .ocr import OcrParser


class OcrProvider(Protocol):
    """Protocol for OCR providers."""

    def ocr_image(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Extract text from image bytes."""
        ...


class GeminiVisionOcr:
    """OCR using Google Gemini Vision API.

    Uses the Gemini generative model to extract text from images.
    Requires GOOGLE_API_KEY environment variable.

    Example:
        >>> ocr = GeminiVisionOcr()
        >>> text = ocr.ocr_image(image_bytes)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
    ):
        """Initialize Gemini Vision OCR.

        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
            model: Gemini model to use for vision tasks.
        """
        import google.generativeai as genai

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini Vision OCR")

        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)

    def ocr_image(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Extract text from image using Gemini Vision.

        Args:
            image_bytes: Raw image bytes
            mime_type: MIME type of image (image/png, image/jpeg, etc.)

        Returns:
            Extracted text from image
        """
        try:
            # Log image size for debugging
            image_size_kb = len(image_bytes) / 1024
            
            response = self._model.generate_content(
                [
                    "Extract all text from this image. "
                    "Preserve the original layout and formatting. "
                    "Return only the extracted text, nothing else. "
                    "If there is no text, return an empty string.",
                    {"mime_type": mime_type, "data": base64.standard_b64encode(image_bytes).decode()},
                ]
            )
            
            # Check if response was blocked or has no candidates
            if not response.candidates:
                # Detailed diagnostic logging
                print(f"[ocr] DEBUG: Image size: {image_size_kb:.1f} KB")
                
                # Print all available response attributes
                print(f"[ocr] DEBUG: Response type: {type(response)}")
                print(f"[ocr] DEBUG: Response attributes: {[a for a in dir(response) if not a.startswith('_')]}")
                
                # Try to access text directly (some API versions)
                try:
                    if hasattr(response, 'text'):
                        print(f"[ocr] DEBUG: response.text exists: '{response.text[:100] if response.text else 'None'}...'")
                except Exception as text_err:
                    print(f"[ocr] DEBUG: response.text access error: {text_err}")
                
                # Check prompt_feedback in detail
                if hasattr(response, 'prompt_feedback'):
                    pf = response.prompt_feedback
                    print(f"[ocr] DEBUG: prompt_feedback type: {type(pf)}")
                    print(f"[ocr] DEBUG: prompt_feedback: {pf}")
                    if pf:
                        print(f"[ocr] DEBUG: prompt_feedback attrs: {[a for a in dir(pf) if not a.startswith('_')]}")
                        if hasattr(pf, 'block_reason') and pf.block_reason:
                            print(f"[ocr] Gemini Vision prompt blocked: {pf.block_reason}")
                        if hasattr(pf, 'safety_ratings') and pf.safety_ratings:
                            for rating in pf.safety_ratings:
                                print(f"[ocr] DEBUG: Safety - {rating.category}: {rating.probability}")
                
                print("[ocr] Gemini Vision returned no candidates")
                return ""
            
            # Safely access text from first candidate
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                print("[ocr] DEBUG: Candidate has no content or parts")
                return ""
            
            text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            return text.strip()
        except Exception as e:
            print(f"[ocr] Gemini Vision OCR failed: {e}")
            return ""


class PyMuPdfParser(BaseSegmentParser):
    """PyMuPDF-based structured PDF parser.

    Extracts content from PDF files using PyMuPDF's block-level API,
    providing structured access to text blocks, images, and layout info.

    Features:
    - Block-level text extraction with bbox coordinates
    - Image block detection with optional Gemini Vision OCR
    - Page number tracking
    - Preserves reading order

    Example:
        >>> parser = PyMuPdfParser(preprocessor, ocr=GeminiVisionOcr())
        >>> segments = parser.parse("document.pdf")
    """

    def __init__(
        self,
        preprocessor: TextPreprocessor,
        *,
        ocr: Optional[OcrProvider] = None,
        min_text_length: int = 10,
        enable_auto_ocr: bool = False,
        force_ocr: bool = False,
    ):
        """Initialize PyMuPDF parser.

        Args:
            preprocessor: Text preprocessor for normalization
            ocr: OCR provider for image blocks and page-level OCR (Gemini Vision)
            min_text_length: Minimum text length to include (filters noise)
            enable_auto_ocr: Enable Gemini Vision OCR fallback for sparse text PDFs
            force_ocr: Force OCR mode - render all pages as images and OCR
        """
        super().__init__(preprocessor)
        self.ocr = ocr
        self.min_text_length = min_text_length
        self.enable_auto_ocr = enable_auto_ocr
        self.force_ocr = force_ocr
        self.text_parser = OcrParser(preprocessor)

    def parse(self, path: str) -> List[RawSegment]:
        """Parse PDF file into structured segments.

        Args:
            path: Path to PDF file

        Returns:
            List of RawSegment objects with text, images, and metadata
        """
        import fitz  # PyMuPDF

        try:
            doc = fitz.open(path)
        except Exception as e:
            print(f"[parse] Failed to open PDF: {e}")
            return []

        # Force OCR mode: render all pages as images and OCR via Gemini
        if self.force_ocr:
            if self.ocr:
                segments = self._ocr_all_pages(doc)
                doc.close()
                return segments
            else:
                # Fallback: force_ocr=True but no OCR provider - use text extraction
                print("[parse] WARNING: force_ocr=True but no OCR provider, falling back to text extraction")
                # Continue with normal text extraction below

        segments: List[RawSegment] = []
        order = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_segments, order = self._process_page(page, page_num, order)
            segments.extend(page_segments)

        doc.close()

        # Post-process: detect code blocks within text segments
        segments = self._detect_code_blocks(segments)

        # Check if text is sparse and fallback to Gemini Vision OCR if enabled
        if self._is_sparse(segments) and self.enable_auto_ocr and self.ocr:
            print(f"[parse] Sparse text detected, falling back to Gemini Vision OCR")
            try:
                doc = fitz.open(path)
                ocr_segments = self._ocr_all_pages(doc)
                doc.close()
                if ocr_segments:
                    return ocr_segments
            except Exception as e:
                print(f"[parse] Gemini Vision OCR fallback failed: {e}")

        return segments

    def _process_page(
        self, page, page_num: int, order: int
    ) -> tuple[List[RawSegment], int]:
        """Process a single PDF page.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            order: Current segment order counter

        Returns:
            Tuple of (segments list, updated order counter)
        """
        segments: List[RawSegment] = []
        blocks = page.get_text("dict", flags=11)["blocks"]

        for block in blocks:
            block_type = block.get("type", 0)
            bbox = (
                block.get("bbox", (0, 0, 0, 0))
                if "bbox" in block
                else (0, 0, 0, 0)
            )

            if block_type == 0:  # Text block
                text = self._extract_text_block(block)
                if text and len(text.strip()) >= self.min_text_length:
                    normalized = self.preprocessor.normalize(text)
                    segments.append(
                        RawSegment(
                            kind="text",
                            content=normalized,
                            language=None,
                            order=order,
                            page=page_num,
                            bbox=bbox,
                        )
                    )
                    order += 1

            elif block_type == 1:  # Image block
                segment = self._process_image_block(block, page_num, order, bbox)
                if segment:
                    segments.append(segment)
                    order += 1

        return segments, order

    def _extract_text_block(self, block: dict) -> str:
        """Extract text from a text block.

        Args:
            block: PyMuPDF block dict with 'lines' containing 'spans'

        Returns:
            Extracted text string
        """
        lines = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            if line_text.strip():
                lines.append(line_text)
        return "\n".join(lines)

    def _process_image_block(
        self,
        block: dict,
        page_num: int,
        order: int,
        bbox: tuple,
    ) -> Optional[RawSegment]:
        """Process an image block with optional OCR.

        Args:
            block: PyMuPDF image block dict
            page_num: Page number
            order: Segment order
            bbox: Bounding box coordinates

        Returns:
            RawSegment with OCR text or None if no text extracted
        """
        if not self.ocr:
            return None

        # Extract image bytes
        image_bytes = block.get("image")
        if not image_bytes:
            return None

        # Determine MIME type from extension
        ext = block.get("ext", "png").lower()
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")

        # Run OCR
        print(f"[ocr] DEBUG: Called from _process_image_block for page {page_num}")
        text = self.ocr.ocr_image(image_bytes, mime_type)
        if not text or len(text.strip()) < self.min_text_length:
            return None

        normalized = self.preprocessor.normalize(text)
        return RawSegment(
            kind="image",
            content=normalized,
            language="image",
            order=order,
            page=page_num,
            bbox=bbox,
        )

    def _detect_code_blocks(self, segments: List[RawSegment]) -> List[RawSegment]:
        """Detect and relabel code blocks within text segments.

        Uses heuristics to identify code-like content and updates
        the segment kind and language accordingly.

        Args:
            segments: List of RawSegment objects

        Returns:
            Updated list with code blocks properly labeled
        """
        result = []
        for seg in segments:
            if seg.kind == "text":
                # Use OcrParser's code detection logic
                sub_segments = self.text_parser.parse_text(seg.content)
                for sub in sub_segments:
                    result.append(
                        RawSegment(
                            kind=sub.kind,
                            content=sub.content,
                            language=sub.language,
                            order=seg.order,
                            page=seg.page,
                            bbox=seg.bbox,
                        )
                    )
            else:
                result.append(seg)
        return result

    def _is_sparse(
        self,
        segments: List[RawSegment],
        min_total_chars: int = 500,
        min_alpha_ratio: float = 0.2,
    ) -> bool:
        """Check if extracted text is sparse (likely needs OCR).

        Args:
            segments: Extracted segments
            min_total_chars: Minimum total characters to consider valid
            min_alpha_ratio: Minimum ratio of alphanumeric characters

        Returns:
            True if text is sparse and OCR fallback is recommended
        """
        total_text = "".join(s.content for s in segments if s.kind == "text")

        if len(total_text.strip()) < min_total_chars:
            return True

        alpha_count = sum(1 for c in total_text if c.isalnum())
        ratio = alpha_count / max(1, len(total_text))

        return ratio < min_alpha_ratio

    def _ocr_all_pages(self, doc) -> List[RawSegment]:
        """OCR all pages using Gemini Vision.

        Renders each page as an image and sends to Gemini Vision for OCR.

        Args:
            doc: PyMuPDF document object

        Returns:
            List of RawSegment from OCR text
        """
        if not self.ocr:
            print("[parse] No OCR provider available for page-level OCR")
            return []

        segments: List[RawSegment] = []
        order = 0

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render page as image (~150 DPI for good OCR quality)
            # fitz.Matrix(scale_x, scale_y) - 2x scale = ~144 DPI from 72 DPI base
            import fitz as fitz_module
            scale = 2.0  # 2x scale = ~144 DPI
            pix = page.get_pixmap(matrix=fitz_module.Matrix(scale, scale))
            image_bytes = pix.tobytes("png")

            # OCR via Gemini Vision
            print(f"[ocr] DEBUG: Called from _ocr_all_pages for page {page_num}, image size: {len(image_bytes)/1024:.1f} KB")
            try:
                text = self.ocr.ocr_image(image_bytes, "image/png")
                if text and len(text.strip()) >= self.min_text_length:
                    # Parse the OCR text to detect code blocks
                    page_segments = self.text_parser.parse_text(text)
                    for seg in page_segments:
                        segments.append(
                            RawSegment(
                                kind=seg.kind,
                                content=seg.content,
                                language=seg.language,
                                order=order,
                                page=page_num,
                                bbox=None,
                            )
                        )
                        order += 1
            except Exception as e:
                print(f"[parse] Gemini Vision OCR failed for page {page_num}: {e}")

        return segments


__all__ = ["GeminiVisionOcr", "PyMuPdfParser", "OcrProvider"]
