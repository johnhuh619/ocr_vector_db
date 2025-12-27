"""Shared utilities and configuration for OCR Vector DB."""

from .config import load_config
from .exceptions import SharedError
from .hashing import HashingService, Slugifier, format_vector_literal
from .text_utils import TextPreprocessor

__all__ = [
    "load_config",
    "SharedError",
    "HashingService",
    "Slugifier",
    "format_vector_literal",
    "TextPreprocessor",
]
