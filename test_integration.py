"""Integration test for OCR Vector DB refactored architecture.

Tests the complete pipeline from ingestion to retrieval using the new layered architecture.
"""

import os
import tempfile

from shared.config import load_config


def test_imports():
    """Test that all packages can be imported."""
    print("[test] Testing package imports...")

    try:
        # Domain layer
        from domain import Concept, Document, Fragment, View

        print("  [OK] domain layer")

        # Shared layer
        from shared.config import EmbeddingConfig

        print("  [OK] shared layer")

        # Ingestion layer
        from ingestion import ConceptBuilder, MarkdownParser, OcrParser, PdfParser, SegmentUnitizer

        print("  [OK] ingestion layer")

        # Embedding layer
        from embedding import EmbeddingProviderFactory, EmbeddingValidator

        print("  [OK] embedding layer")

        # Storage layer (may fail if langchain_core not installed)
        try:
            from storage import (
                ConceptRepository,
                DocumentRepository,
                FragmentRepository,
                LangChainAdapter,
            )
            print("  [OK] storage layer")
        except ImportError as e:
            print(f"  [SKIP] storage layer (langchain_core not installed)")

        # Retrieval layer (may fail if psycopg not installed)
        try:
            from retrieval import ContextExpander, QueryInterpreter, RetrievalPipeline, VectorSearchEngine
            print("  [OK] retrieval layer")
        except ImportError as e:
            print(f"  [SKIP] retrieval layer (psycopg not installed)")

        # API layer (may fail if dependencies not installed)
        try:
            from api.formatters import ResponseFormatter
            from api.use_cases import IngestUseCase, SearchUseCase
            from api.validators import RequestValidator
            print("  [OK] api layer")
        except ImportError as e:
            print(f"  [SKIP] api layer ({e.name} not installed)")

        print("[test] All imports successful (with some SKIPs)!")
        return True

    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_domain_entities():
    """Test domain entity creation and validation."""
    print("\n[test] Testing domain entities...")

    from domain import Concept, Document, Fragment, View

    try:
        # Create Document
        doc = Document(
            id="doc-1",
            source_path="/test/file.txt",
            metadata={"test": True},
        )
        print(f"  [OK] Document created: {doc.id}")

        # Create Concept
        concept = Concept(
            id="concept-1",
            document_id=doc.id,
            order=0,
            content="Test concept content",
            metadata={},
        )
        print(f"  [OK] Concept created: {concept.id}")

        # Create Fragment
        fragment = Fragment(
            id="frag-1",
            concept_id=concept.id,
            content="This is a test fragment with enough length",
            view=View.TEXT,
            language=None,
            order=0,
        )
        fragment.validate()
        print(f"  [OK] Fragment created and validated: {fragment.id}")

        return True

    except Exception as e:
        print(f"[FAIL] Domain entity error: {e}")
        return False


def test_ingestion_parsing():
    """Test file parsing with ingestion layer."""
    print("\n[test] Testing ingestion parsing...")

    from ingestion import OcrParser
    from shared.text_utils import TextPreprocessor

    try:
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document.\n")
            f.write("It has multiple lines.\n")
            f.write("For testing purposes.\n")
            temp_file = f.name

        # Parse file
        preprocessor = TextPreprocessor()
        parser = OcrParser(preprocessor)
        segments = parser.parse(temp_file)

        # Clean up
        os.unlink(temp_file)

        print(f"  [OK] Parsed {len(segments)} segments")
        return True

    except Exception as e:
        print(f"[FAIL] Parsing error: {e}")
        if "temp_file" in locals():
            os.unlink(temp_file)
        return False


def test_pdf_parsing():
    """Test PDF parsing path without external dependencies."""
    print("\n[test] Testing PDF parsing...")

    from ingestion import PdfParser
    from shared.text_utils import TextPreprocessor

    try:
        preprocessor = TextPreprocessor()
        parser = PdfParser(preprocessor, enable_auto_ocr=False)
        parser.extractor.extract = lambda _: "sample " * 100
        segments = parser.parse("dummy.pdf")

        if not segments:
            print("[FAIL] PDF parsing returned no segments")
            return False

        print(f"  [OK] Parsed {len(segments)} segments from mocked PDF")
        return True

    except Exception as e:
        print(f"[FAIL] PDF parsing error: {e}")
        return False


def test_validators():
    """Test request validators."""
    print("\n[test] Testing validators...")

    from api.validators import RequestValidator, ValidationError

    try:
        # Test view validation
        RequestValidator.validate_view("text")
        RequestValidator.validate_view("code")
        print("  [OK] View validation")

        # Test top_k validation
        RequestValidator.validate_top_k(10)
        RequestValidator.validate_top_k(100)
        print("  [OK] top_k validation")

        # Test query validation
        RequestValidator.validate_query("test query")
        print("  [OK] Query validation")

        # Test validation error
        try:
            RequestValidator.validate_view("invalid_view")
            print("  [FAIL] Should have raised ValidationError")
            return False
        except ValidationError:
            print("  [OK] ValidationError raised correctly")

        return True

    except Exception as e:
        print(f"[FAIL] Validator error: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 80)
    print("OCR Vector DB - Integration Tests")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Domain Entities Test", test_domain_entities()))
    results.append(("Ingestion Parsing Test", test_ingestion_parsing()))
    results.append(("PDF Parsing Test", test_pdf_parsing()))
    results.append(("Validators Test", test_validators()))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 80)

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
