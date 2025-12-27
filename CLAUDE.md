# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **OCR Vector Database** system that processes documents (PDFs, Markdown, plain text) and creates a searchable vector database. The system extracts text, segments it into semantic units, generates embeddings via cloud APIs (Voyage AI or Google Gemini), and stores them in PostgreSQL with pgvector for similarity search.

**Core capabilities:**
- Multi-format document parsing (PDF with OCR fallback, Markdown, plain text)
- Intelligent semantic segmentation (text, code, images with multi-view support)
- Parent-child document hierarchy for contextual retrieval
- Rich metadata tagging for precise filtering
- Vector similarity search with pgvector

## Architecture Documentation (MUST READ)

This project has strict architectural rules defined in the following documents. **All development MUST comply with these rules.**

| Document | Purpose | Location |
|----------|---------|----------|
| `ARCHITECTURE.md` | System architecture, design principles, domain model | `docs/ARCHITECTURE.md` |
| `DOMAIN_RULES.md` | Non-negotiable domain invariants (RFC 2119 style) | `docs/DOMAIN_RULES.md` |
| `PACKAGE_RULES.md` | Package structure, dependency direction rules | `docs/PACKAGE_RULES.md` |

### Key Architectural Constraints

**Entity Hierarchy (MUST follow):**
```
Document → Concept → Fragment → Embedding
```

**Core Domain Rules:**
- All Fragments MUST have a valid `parent_id` (no orphan entities)
- Embedding ID MUST be deterministic: `doc_id = hash(parent_id + view + lang + content)`
- Fragments shorter than 10 characters MUST NOT be embedded
- Search target (Fragment embeddings) and context provider (Parent documents) MUST be separated

**Target Package Structure:**
```
domain/      # Pure domain entities (NO infrastructure imports)
ingestion/   # File parsing, segmentation
embedding/   # Vector generation, doc_id computation
retrieval/   # Search pipeline, context assembly
storage/     # Database operations, repositories
api/         # CLI, REST endpoints
shared/      # Common utilities, exceptions
```

**Forbidden Anti-Patterns:**
- Chunk-first design (MUST identify semantic units first)
- Embedding everything (MUST filter by explicit rules)
- View-as-entity confusion (View is an attribute, not an entity)
- domain importing infrastructure (domain MUST NOT import sqlalchemy, langchain, etc.)

### When to Trigger Architecture Review

- Adding new entity types
- Changing embedding target rules
- Modifying ownership chain (parent_id logic)
- Adding package dependencies
- Changing search pipeline structure

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL with pgvector
docker-compose up -d

# Configure environment (edit .env file)
# Required: VOYAGE_API_KEY or GOOGLE_API_KEY, PG_CONN, COLLECTION_NAME
```

### Running the Pipeline

**Simple entry point:**
```bash
python embedding.py "test/*.txt"
python embedding.py "documents/*.md"
```

**Advanced CLI tool (recommended):**
```bash
# Basic ingestion
python tools/ingest.py file1.txt file2.pdf

# With OCR fallback for PDFs
python tools/ingest.py documents/*.pdf --auto-ocr

# Dry run (parse only, no DB writes)
python tools/ingest.py test.txt --dry-run

# With custom schema writes
python tools/ingest.py *.md --custom-schema-write

# Override parent mode
python tools/ingest.py data/*.txt --parent-mode page_section
```

### Search and Retrieval
```bash
# Test basic search
python search_test.py

# Demo multi-vector retrieval
python retriever_multi_view_demo.py "your search query"
```

## Architecture

### Core Pipeline Flow

1. **File Parsing** (`embedding/parsers.py`)
   - PDF: `PdfExtractor` uses pdfminer.six, falls back to OCRMyPDF if text is sparse
   - Markdown: `MarkdownParser` extracts fenced code blocks, images, and text
   - Plain text: `OcrParser` processes raw text

2. **Text Processing** (`embedding/text_utils.py`)
   - `TextPreprocessor.normalize_text()`: Cleans ligatures, quotes, dashes, whitespace
   - Code detection via heuristics (indentation, keywords, syntax patterns)
   - Language detection for code blocks (Python, JavaScript, generic)

3. **Segmentation & Unitization** (`embedding/parsers.py`, `embedding/pipeline.py`)
   - Split into paragraphs and code blocks (`RawSegment` objects)
   - `SegmentUnitizer.unitize()`: Groups related content (pre-text + code + post-text) into semantic units
   - Each unit assigned a `unit_id` for parent-child relationships

4. **Document Building** (`embedding/pipeline.py`)
   - `DocumentBuilder.build()`: Transforms units into LangChain `Document` objects
   - Recursive text splitting (chunk_size=1200, overlap=150)
   - Code splitting preserves structure (avoids breaking mid-function)
   - Attaches metadata: source, view, kind, language, parent_id, unit_id, order

5. **Parent Document Synthesis** (`embedding/parents.py`)
   - `ParentDocumentBuilder`: Groups child documents by unit/page/section
   - Extracts headers and captions for parent content
   - Creates hierarchical relationships for multi-vector retrieval
   - Modes: `unit`, `page`, `section`, `page_section`

6. **Embedding & Storage** (`embedding/embeddings_provider.py`, `embedding/storage.py`)
   - `EmbeddingProviderFactory`: Creates Voyage AI or Gemini embedding clients
   - `VectorStoreWriter`: Upserts to LangChain's PGVector store
   - `ParentChildRepository`: Manages parent documents in custom tables
   - `DbSchemaManager`: Creates HNSW indexes, JSONB GIN indexes, BTREE indexes

### Multi-View Strategy

Documents are segmented into distinct "views" for specialized retrieval:
- **text**: Natural language paragraphs
- **code**: Code blocks with language tags (Python, JavaScript, etc.)
- **image**: Image references with alt text and URLs
- **caption**: Extracted figure/table captions
- **figure**, **table**: Structured content metadata

Each view is processed differently (e.g., code splitting vs. text splitting) and stored with appropriate metadata for filtering during search.

### Parent Modes

Configure via `PARENT_MODE` environment variable:
- **unit** (default): Group by semantic unit (pre+code+post combinations)
- **page**: Group by detected page numbers in content
- **section**: Group by section headers (Markdown headings, "Chapter N")
- **page_section**: Combine both page and section grouping (most granular)

### Database Schema

**Main tables:**
- `langchain_pg_collection`: Collection metadata
- `langchain_pg_embedding`: Vector embeddings with JSONB metadata (LangChain PGVector store)
- `docstore_parent`: Parent documents with metadata
- `child_chunks`: Custom schema for child documents (when `CUSTOM_SCHEMA_WRITE=true`)
- `parent_chunks`: Custom schema for parent documents (when `CUSTOM_SCHEMA_WRITE=true`)

**Indexes:**
- HNSW vector index for similarity search
- JSONB GIN indexes on metadata fields
- BTREE indexes on common filter columns (source, view, lang, page)

## Key Modules

### `embedding/pipeline.py`
Core orchestration class `EmbeddingPipeline`:
- `run(pattern)`: Main entry point - glob files, parse, embed, store
- `DocumentBuilder`: Transforms unitized segments into LangChain documents
- `InputCollector`: Resolves file globs

### `embedding/storage.py`
Database operations:
- `DbSchemaManager`: Schema creation and index management
- `ParentChildRepository`: CRUD for parent documents
- `VectorStoreWriter`: Batch upserts to PGVector with retry logic

### `embedding/parents.py`
Parent document synthesis:
- `ParentDocumentBuilder`: Groups children, extracts headers, builds parent content
- Handles multiple parent modes (unit/page/section/page_section)

### `embedding/parsers.py`
File format parsers:
- `PdfExtractor`: PDF text extraction via pdfminer.six
- `MarkdownParser`: Fence parsing, image extraction
- `OcrParser`: Plain text processing
- `SegmentUnitizer`: Groups segments into semantic units

### `embedding/embeddings_provider.py`
Embedding client abstraction:
- `EmbeddingProviderFactory`: Creates Voyage AI or Gemini clients
- `compute_doc_id()`: Generates stable content-based IDs
- `validate_embedding_dimension()`: Checks embedding dimensions match config

### `embedding/config.py`
Configuration loader:
- `load_config()`: Parses environment variables into `EmbeddingConfig` dataclass
- Defaults: voyage-3 model, 768 dimensions, unit parent mode

### `embedding/text_utils.py`
Text preprocessing utilities:
- `TextPreprocessor.normalize_text()`: Text cleaning
- `TextPreprocessor.split_code_safely()`: Structure-aware code splitting
- Code detection and language guessing

## Configuration

All configuration via `.env` file (see `.env.example` if available):

**Required:**
```
VOYAGE_API_KEY=...               # For Voyage AI embeddings
# OR
GOOGLE_API_KEY=...               # For Gemini embeddings

PG_CONN=postgresql+psycopg://langchain:langchain@localhost:5432/vectordb
COLLECTION_NAME=langchain_book_ocr
```

**Optional:**
```
EMBEDDING_PROVIDER=voyage        # voyage | gemini
VOYAGE_MODEL=voyage-3
GEMINI_EMBED_MODEL=text-embedding-004
EMBEDDING_DIM=768
PARENT_MODE=unit                 # unit | page | section | page_section
CUSTOM_SCHEMA_WRITE=true         # Write to custom child_chunks/parent_chunks tables
ENABLE_AUTO_OCR=false            # Auto-run ocrmypdf for sparse PDFs
RATE_LIMIT_RPM=0                 # Rate limiting (0=disabled)
MAX_DOCS_TO_EMBED=0              # Limit for testing (0=unlimited)
PAGE_REGEX=...                   # Custom page number detection
SECTION_REGEX=...                # Custom section header detection
HNSW_EF_SEARCH=...               # pgvector HNSW search tuning
HNSW_EF_CONSTRUCTION=...         # pgvector HNSW index build tuning
```

## Important Patterns

### Adding New File Format Support
1. Create parser in `embedding/parsers.py` that returns `List[RawSegment]`
2. Add file extension detection in `EmbeddingPipeline.run()` or `tools/ingest.py`
3. Parser should handle: content extraction, kind detection (text/code/image), order assignment

### Modifying Chunking Strategy
- Text chunking: Modify `DocumentBuilder.__init__()` RecursiveCharacterTextSplitter params
- Code chunking: Modify `TextPreprocessor.split_code_safely()` logic
- Default: chunk_size=1200, overlap=150

### Custom Metadata
Add metadata fields in `DocumentBuilder.build()`:
- Metadata propagates to vector store JSONB column
- Add corresponding indexes in `DbSchemaManager.create_indexes()`

### Changing Embedding Provider
1. Set `EMBEDDING_PROVIDER=voyage` or `gemini` in `.env`
2. Update `EMBEDDING_DIM` if dimension differs (Voyage: 1024, Gemini: 768)
3. Factory pattern handles instantiation automatically

## Common Pitfalls

- **PG_CONN typo**: Use `postgresql+psycopg` NOT `postgresql+pycopg` (tools/ingest.py warns about this)
- **Sparse PDF text**: Enable `--auto-ocr` flag or `ENABLE_AUTO_OCR=true` for OCR fallback
- **Embedding dimension mismatch**: Validate with `validate_embedding_dimension()` before upserting
- **Parent mode confusion**: `unit` mode requires semantic units; use `page` or `section` for simpler grouping
- **Custom schema**: When `CUSTOM_SCHEMA_WRITE=true`, data goes to both LangChain tables AND custom tables

## Database Connection

Default setup via docker-compose:
```
Host: localhost
Port: 5432
User: langchain
Password: langchain
Database: vectordb
```

Connection string: `postgresql+psycopg://langchain:langchain@localhost:5432/vectordb`

## Testing

No formal test suite currently configured. Manual testing via:
- `python tools/ingest.py test.txt --dry-run` (parse validation)
- `python search_test.py` (retrieval validation)
- `python retriever_multi_view_demo.py "query"` (end-to-end validation)
