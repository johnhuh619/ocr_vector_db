import glob
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings_provider import compute_doc_id, EmbeddingProviderFactory, validate_embedding_dimension
from .models import EmbeddingConfig, RawSegment, UnitizedSegment
from .parents import ParentDocumentBuilder
from .parsers import MarkdownParser, OcrParser, PdfExtractor, SegmentUnitizer
from .storage import DbSchemaManager, ParentChildRepository, VectorStoreWriter
from .text_utils import TextPreprocessor

try:
    from langchain_postgres import PGVector
except Exception:  # pragma: no cover - optional dependency
    PGVector = object  # type: ignore


@dataclass
class InputCollector:
    """Collect input files for processing based on glob patterns."""

    def collect(self, pattern: str) -> List[str]:
        return sorted(glob.glob(pattern))


class DocumentBuilder:
    """Transform unitized segments into LangChain documents with metadata."""

    def __init__(self, text_splitter: Optional[RecursiveCharacterTextSplitter] = None):
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Increased from 1200 for better context
            chunk_overlap=300,  # Increased from 150 for better continuity
            separators=["\n##", "\n###", "\n\n", "\n", " ", ""],
            length_function=len,
            add_start_index=True,
        )

    def build(self, path: str, unitized: Sequence[UnitizedSegment]) -> List[Document]:
        documents: List[Document] = []
        base_metadata = {"source": os.path.basename(path)}

        for unit in unitized:
            metadata = dict(base_metadata)
            metadata["order"] = unit.segment.order
            metadata["kind"] = unit.segment.kind

            if unit.unit_id:
                metadata["unit_id"] = unit.unit_id
                metadata["parent_id"] = unit.unit_id
            else:
                metadata["parent_id"] = f"{os.path.splitext(base_metadata['source'])[0]}-{uuid.uuid4()}"

            if unit.role:
                metadata["unit_role"] = unit.role

            if unit.segment.kind == "text":
                view_metadata = {**metadata, "view": "text"}
                for chunk in self.text_splitter.split_text(unit.segment.content):
                    documents.append(Document(page_content=chunk, metadata=view_metadata))
            elif unit.segment.kind == "code":
                lang = unit.segment.language or "unknown"
                code_metadata = {**metadata, "lang": lang, "view": "code"}
                for chunk in TextPreprocessor.split_code_safely(unit.segment.content):
                    documents.append(Document(page_content=chunk, metadata=code_metadata))
            elif unit.segment.kind == "image":
                alt, url = self._split_image_payload(unit.segment.content)
                image_metadata = {**metadata, "view": "image"}
                if alt:
                    image_metadata["alt"] = alt
                if url:
                    image_metadata["image_url"] = url
                content = (alt or "image") + (f"\n{url}" if url else "")
                documents.append(Document(page_content=content, metadata=image_metadata))
            else:
                fallback_metadata = {**metadata, "view": "text"}
                for chunk in self.text_splitter.split_text(unit.segment.content):
                    documents.append(Document(page_content=chunk, metadata=fallback_metadata))
        return documents

    @staticmethod
    def _split_image_payload(payload: str) -> Tuple[str, str]:
        if not payload:
            return "", ""
        if "\n" in payload:
            alt, url = payload.split("\n", 1)
            return alt.strip(), url.strip()
        return payload.strip(), ""


class EmbeddingPipeline:
    """End-to-end orchestration from raw files to vector store upserts."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.ocr_parser = OcrParser(self.preprocessor)
        self.md_parser = MarkdownParser(self.preprocessor)
        self.segment_unitizer = SegmentUnitizer()
        self.pdf_extractor = PdfExtractor()
        self.schema_manager = DbSchemaManager(config)
        self.repository = ParentChildRepository(config)
        self.vector_writer = VectorStoreWriter(config)
        self.parent_builder = ParentDocumentBuilder(
            parent_mode=config.parent_mode,
            page_regex=config.page_regex,
            section_regex=config.section_regex,
        )
        self.document_builder = DocumentBuilder()
        self.collector = InputCollector()

    def run(self, pattern: str) -> None:
        embeddings = EmbeddingProviderFactory.create(self.config)
        model_name = (
            self.config.embedding_model
            if self.config.embedding_provider != "gemini"
            else self.config.gemini_model
        )
        validate_embedding_dimension(
            embeddings,
            self.config.embedding_dim,
            provider=self.config.embedding_provider,
            model=model_name,
        )
        self.schema_manager.apply_db_level_tuning()

        store = self._create_vector_store(embeddings)

        files = self.collector.collect(pattern)
        if not files:
            print(f"[WARN] No files matched: {pattern}")
            return

        self.schema_manager.ensure_extension_vector()
        self.schema_manager.ensure_parent_docstore()
        if self.config.custom_schema_write:
            self.schema_manager.ensure_custom_schema(self.config.embedding_dim)

        chunk_total = 0
        vector_total = 0
        parent_total = 0
        custom_parent_total = 0
        custom_child_total = 0
        for path in files:
            print(f"[parse] {path}")
            segments = self._parse_file(path)
            if not segments:
                continue

            unitized = self.segment_unitizer.unitize(segments)
            documents = self.document_builder.build(path, unitized)
            try:
                caption_docs = self.parent_builder.augment_with_captions(documents)
                if caption_docs:
                    documents.extend(caption_docs)
                    print(f"[augment] captions added: {len(caption_docs)}")
            except Exception as exc:
                print(f"[warn] caption augmentation skipped: {exc}")
            if not documents:
                continue

            try:
                self.parent_builder.assign_parent_by_page_section(documents, path)
            except Exception as exc:
                print(f"[warn] parent assign skipped for {os.path.basename(path)}: {exc}")

            for doc in documents:
                compute_doc_id(doc)

            parents = self.parent_builder.build_parent_entries(documents)
            parent_total += self.repository.upsert_parents(parents)

            if self.config.custom_schema_write:
                custom_parents, custom_children = self.repository.dual_write_custom_schema(
                    embeddings, parents, documents
                )
                custom_parent_total += custom_parents
                custom_child_total += custom_children

            docs_to_write = documents
            if self.config.max_docs_to_embed > 0:
                docs_to_write = documents[: self.config.max_docs_to_embed]
                print(
                    f"[upsert] {os.path.basename(path)} -> {len(docs_to_write)} / {len(documents)} chunks (limited)"
                )
            else:
                print(f"[upsert] {os.path.basename(path)} -> {len(documents)} chunks")

            vector_total += self.vector_writer.upsert_batch(store, docs_to_write)
            chunk_total += len(documents)

        print(f"[done] total chunk documents: {chunk_total}")
        print(f"[done] total vector writes: {vector_total}")
        print(f"[done] parent rows upserted: {parent_total}")
        if self.config.custom_schema_write:
            print(
                f"[done] custom schema parents: {custom_parent_total}, child chunks: {custom_child_total}"
            )
        print("[index] creating indexes (idempotent)")
        self.schema_manager.ensure_indexes()
        print("[ok] all set")

    def _parse_file(self, path: str) -> Optional[List[RawSegment]]:
        extension = os.path.splitext(path)[1].lower()
        if extension == ".pdf":
            text = self.pdf_extractor.extract(path)
            if not text or self.pdf_extractor.is_low_text_density(text):
                print(
                    f"[warn] PDF text is sparse or empty; consider OCR: "
                    f"ocrmypdf --sidecar {os.path.splitext(path)[0]}.txt --skip-text {path} {os.path.splitext(path)[0]}.ocr.pdf"
                )
                if self.config.enable_auto_ocr and shutil.which("ocrmypdf"):
                    try:
                        sidecar = os.path.splitext(path)[0] + ".txt"
                        subprocess.run(
                            [
                                "ocrmypdf",
                                "--sidecar",
                                sidecar,
                                "--skip-text",
                                path,
                                os.path.splitext(path)[0] + ".ocr.pdf",
                            ],
                            check=True,
                        )
                        with open(sidecar, "r", encoding="utf-8", errors="ignore") as handle:
                            text = handle.read()
                    except Exception as exc:
                        print(f"[warn] auto OCR failed: {exc}")
            if not text:
                print(f"[skip] No extractable text for {path}")
                return None
            return self.ocr_parser.parse_text(text)
        if extension in (".md", ".markdown"):
            return self.md_parser.parse(path)
        return self.ocr_parser.parse(path)

    def _create_vector_store(self, embeddings):
        if PGVector is object:
            raise RuntimeError("langchain_postgres.PGVector is unavailable in this environment")
        return PGVector(
            connection=self.config.pg_conn,
            embeddings=embeddings,
            collection_name=self.config.collection_name,
            distance_strategy="COSINE",
            use_jsonb=True,
            embedding_length=self.config.embedding_dim,
        )


__all__ = ["EmbeddingPipeline"]
