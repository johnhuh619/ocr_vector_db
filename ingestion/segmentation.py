"""Semantic unit grouping for segments."""

import uuid
from typing import List

from .models import RawSegment, UnitizedSegment


class SegmentUnitizer:
    """
    Group segments into semantic units that preserve Python/JS adjacency.

    A semantic unit groups related content together (e.g., explanatory text + code).
    """

    def __init__(
        self,
        attach_pre_text: bool = True,
        attach_post_text: bool = False,
        bridge_text_max: int = 0,
        max_pre_text_chars: int = 4000,
    ):
        """
        Initialize SegmentUnitizer.

        Args:
            attach_pre_text: Attach text before Python code to the same unit
            attach_post_text: Attach text after JavaScript code to the same unit
            bridge_text_max: Maximum text segments to bridge between Python and JS
            max_pre_text_chars: Maximum characters of pre-text to buffer
        """
        self.attach_pre_text = attach_pre_text
        self.attach_post_text = attach_post_text
        self.bridge_text_max = bridge_text_max
        self.max_pre_text_chars = max_pre_text_chars

    def unitize(self, segments: List[RawSegment]) -> List[UnitizedSegment]:
        """
        Group segments into semantic units.

        Args:
            segments: List of RawSegment objects

        Returns:
            List of UnitizedSegment objects with unit_id assignments
        """
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


__all__ = ["SegmentUnitizer"]
