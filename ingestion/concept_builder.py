"""Build domain Concepts from UnitizedSegments."""

import uuid
from typing import Dict, List

from domain import Concept, Document, Fragment, View

from .models import UnitizedSegment


class ConceptBuilder:
    """
    Transform UnitizedSegments into domain Concepts and Fragments.

    This is the bridge between ingestion layer (RawSegment, UnitizedSegment)
    and domain layer (Concept, Fragment).

    Rules enforced:
    - HIER-002: Every Concept belongs to exactly one Document
    - HIER-003: Every Fragment belongs to exactly one Concept
    - FRAG-IMMUT-001: concept_id (parent_id) is set at creation and immutable
    """

    def build(
        self,
        unitized: List[UnitizedSegment],
        document: Document,
        source_basename: str,
    ) -> List[Concept]:
        """
        Build Concepts from UnitizedSegments.

        Args:
            unitized: List of UnitizedSegment objects from SegmentUnitizer
            document: Parent Document entity
            source_basename: Source file basename for generating concept IDs

        Returns:
            List of Concept entities with associated Fragments

        Raises:
            OrphanEntityError: If any Fragment created without valid concept_id
        """
        # Group segments by unit_id
        unit_groups: Dict[str, List[UnitizedSegment]] = {}
        orphan_segments: List[UnitizedSegment] = []

        for unit_seg in unitized:
            if unit_seg.unit_id:
                if unit_seg.unit_id not in unit_groups:
                    unit_groups[unit_seg.unit_id] = []
                unit_groups[unit_seg.unit_id].append(unit_seg)
            else:
                orphan_segments.append(unit_seg)

        concepts: List[Concept] = []
        order = 0

        # Create Concepts from grouped units
        for unit_id, segments in unit_groups.items():
            concept = self._create_concept_from_unit(
                unit_id=unit_id,
                segments=segments,
                document=document,
                order=order,
            )
            concepts.append(concept)
            order += 1

        # Create Concepts for orphan segments (no unit_id)
        if orphan_segments:
            concept = self._create_concept_from_orphans(
                orphan_segments=orphan_segments,
                document=document,
                source_basename=source_basename,
                order=order,
            )
            concepts.append(concept)

        return concepts

    def _create_concept_from_unit(
        self,
        unit_id: str,
        segments: List[UnitizedSegment],
        document: Document,
        order: int,
    ) -> Concept:
        """Create Concept from unitized segments with same unit_id."""
        concept = Concept(
            id=unit_id,  # Use unit_id as concept_id
            document_id=document.id,
            order=order,
            metadata={"unit_type": "semantic_unit"},
        )
        concept.validate()  # Enforce HIER-002

        # Create Fragments for this Concept
        fragments: List[Fragment] = []
        for idx, unit_seg in enumerate(segments):
            fragment = self._create_fragment(
                concept_id=concept.id,
                segment=unit_seg,
                order=idx,
            )
            fragment.validate()  # Enforce HIER-003, FRAG-LEN-001
            fragments.append(fragment)

        # Store fragments in metadata (storage layer will handle persistence)
        concept.metadata["fragments"] = fragments
        return concept

    def _create_concept_from_orphans(
        self,
        orphan_segments: List[UnitizedSegment],
        document: Document,
        source_basename: str,
        order: int,
    ) -> Concept:
        """Create Concept for segments without unit_id."""
        concept_id = f"{source_basename}-orphans-{uuid.uuid4()}"
        concept = Concept(
            id=concept_id,
            document_id=document.id,
            order=order,
            metadata={"unit_type": "orphans"},
        )
        concept.validate()

        fragments: List[Fragment] = []
        for idx, unit_seg in enumerate(orphan_segments):
            fragment = self._create_fragment(
                concept_id=concept.id,
                segment=unit_seg,
                order=idx,
            )
            fragment.validate()
            fragments.append(fragment)

        concept.metadata["fragments"] = fragments
        return concept

    def _create_fragment(
        self,
        concept_id: str,
        segment: UnitizedSegment,
        order: int,
    ) -> Fragment:
        """
        Create Fragment from UnitizedSegment.

        Args:
            concept_id: Parent Concept ID
            segment: UnitizedSegment to convert
            order: Order within Concept

        Returns:
            Fragment entity
        """
        # Map segment kind to View
        view = self._map_kind_to_view(segment.segment.kind)

        fragment = Fragment(
            id=str(uuid.uuid4()),
            concept_id=concept_id,  # Set parent_id at creation (FRAG-IMMUT-001)
            content=segment.segment.content,
            view=view,
            language=segment.segment.language,
            order=order,
            metadata={
                "unit_role": segment.role,
                "original_kind": segment.segment.kind,
            },
        )
        return fragment

    @staticmethod
    def _map_kind_to_view(kind: str) -> View:
        """Map RawSegment kind to domain View."""
        mapping = {
            "text": View.TEXT,
            "code": View.CODE,
            "image": View.IMAGE,
        }
        return mapping.get(kind, View.TEXT)


__all__ = ["ConceptBuilder"]
