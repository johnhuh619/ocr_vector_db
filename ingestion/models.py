"""Data models for ingestion layer."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RawSegment:
    """
    Raw parsed content unit from file parsers.

    This represents a piece of content extracted from a file before
    any semantic grouping or concept identification.
    """

    kind: str  # "text" | "code" | "image"
    content: str
    language: Optional[str]  # e.g., "python", "javascript", "image"
    order: int


@dataclass
class UnitizedSegment:
    """
    Segment grouped into semantic units.

    unit_id groups related segments together (e.g., pre-text + code + post-text).
    """

    unit_id: Optional[str]  # UUID for semantic unit, None for ungrouped
    role: str  # "pre_text", "python", "post_text", "javascript", "bridge_text", "other"
    segment: RawSegment


__all__ = ["RawSegment", "UnitizedSegment"]
