# 리뷰 이슈 개선 계획

## 이슈 요약

| 우선순위 | 이슈 | 영향 |
|---------|------|------|
| 🔴 높음 | Concept ID 비결정적 → 중복 임베딩 | 재수집 시 동일 문서가 다른 ID로 저장 |
| 🔴 높음 | 검색에 collection 스코핑 없음 | 다른 컬렉션 데이터 혼입 |
| 🟡 중간 | 배치/청킹 설정 미적용 | 긴 문단이 모델 제한 초과 가능 |
| 🟡 중간 | PyMuPDF에서 OCR 설정 누락 | 스캔 PDF OCR 미처리 |
| 🟡 중간 | orphans Concept 문제 | Python/JS 외 텍스트 의미 희석 |

---

## 이슈 1: Concept ID 비결정적 생성 (🔴 높음)

### 문제
- `SegmentUnitizer:65` - `uuid.uuid4()` 사용
- `ConceptBuilder:121` - orphans에 UUID 포함
- 재수집 시 concept_id 변경 → doc_id 변경 → 중복 임베딩

### 해결 방안: Content-Based Deterministic ID

```python
# Document ID: 파일 경로 해시
doc_id = hash(source_path)

# Concept ID: 문서 + 유닛 내용 해시
concept_id = hash(document_id + unit_content_hash)

# Fragment ID: 개념 + 순서 + 내용 해시
fragment_id = hash(concept_id + order + content[:100])
```

### 수정 파일
1. `ingestion/segmentation.py:65` - unit_id 결정적 생성
2. `ingestion/concept_builder.py:91,121` - concept_id 결정적 생성
3. `api/use_cases/ingest.py:131-135` - Document ID 결정적 생성
4. (선택) `shared/hashing.py` - 범용 해시 함수 추가

### 구현 예시
```python
# ingestion/segmentation.py
def _generate_unit_id(self, segments: List[RawSegment]) -> str:
    content = "".join(s.content[:100] for s in segments)
    return hashlib.md5(content.encode()).hexdigest()[:16]

# api/use_cases/ingest.py
document = Document(
    id=hashlib.md5(file_path.encode()).hexdigest(),  # 결정적
    source_path=file_path,
)
```

---

## 이슈 2: Collection 스코핑 없음 (🔴 높음)

### 문제
- `retrieval/search.py:95-108` - SQL에 collection_id 필터 없음
- 다른 컬렉션 데이터가 검색 결과에 포함될 수 있음

### 현재 쿼리
```sql
SELECT ... FROM langchain_pg_embedding
WHERE 1=1{where_sql}  -- collection_id 필터 없음!
```

### 해결 방안
```sql
SELECT ... FROM langchain_pg_embedding e
INNER JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = %s  -- 컬렉션 필터 추가
  AND {other_filters}
```

### 수정 파일
1. `retrieval/search.py:95` - SQL에 collection JOIN 추가
2. `retrieval/query.py` - QueryPlan에 collection_name 추가
3. `retrieval/__init__.py` - config 전달 경로 확인

### 구현 예시
```python
# retrieval/search.py
def search(self, query_embedding, collection_name: str, ...):
    sql = """
    SELECT ...
    FROM langchain_pg_embedding e
    INNER JOIN langchain_pg_collection c ON e.collection_id = c.uuid
    WHERE c.name = %s
      AND ...
    """
    params = [collection_name, ...]
```

---

## 이슈 3: 배치/청킹 설정 미적용 (🟡 중간)

### 문제
- `storage/vector_store.py:79` - 단순 고정 배치만 사용
- `max_chars_per_request`, `max_items_per_request` 무시됨

### 현재 구현 (잘못됨)
```python
groups = [deduped[i : i + batch_size] for i in range(0, len(deduped), batch_size)]
```

### 해결 방안: app/storage.py의 iter_by_char_budget 로직 이식

### 수정 파일
1. `storage/vector_store.py:79` - 배치 로직 개선
2. `shared/batching.py` (신규) - iter_by_char_budget 함수

### 구현 예시
```python
# shared/batching.py
def iter_by_char_budget(docs, char_budget, max_batch, max_items):
    batch = []
    chars = 0
    for doc in docs:
        doc_chars = len(doc.page_content)
        if batch and (chars + doc_chars > char_budget or len(batch) >= max_items):
            yield batch
            batch = []
            chars = 0
        batch.append(doc)
        chars += doc_chars
    if batch:
        yield batch

# storage/vector_store.py
from shared.batching import iter_by_char_budget

char_budget = self.config.max_chars_per_request or 4000
groups = list(iter_by_char_budget(deduped, char_budget, batch_size, self.config.max_items_per_request))
```

---

## 이슈 4: PyMuPDF OCR 설정 누락 (🟡 중간)

### 문제
- `api/use_cases/ingest.py:203` - enable_auto_ocr, force_ocr 미전달
- PyMuPdfParser는 이미지 OCR만 지원, 문서 레벨 OCR 없음

### 해결 방안 A: PyMuPdfParser에 텍스트 희박 시 폴백 추가
```python
class PyMuPdfParser:
    def __init__(self, ..., enable_auto_ocr=False, force_ocr=False):
        self.enable_auto_ocr = enable_auto_ocr
        self.force_ocr = force_ocr

    def parse(self, path):
        segments = self._extract_with_pymupdf(path)

        # 텍스트가 희박하면 ocrmypdf 폴백
        if self._is_sparse(segments) and self.enable_auto_ocr:
            return self._fallback_ocr(path)
        return segments
```

### 해결 방안 B: 파서 체인 패턴
```python
# 먼저 PyMuPDF 시도, 실패 시 레거시 PdfParser로 폴백
```

### 수정 파일
1. `ingestion/parsers/pymupdf_parser.py` - OCR 폴백 로직 추가
2. `api/use_cases/ingest.py:193-203` - OCR 설정 전달

---

## 이슈 5: Orphans Concept 문제 (🟡 중간)

### 문제
- `SegmentUnitizer`가 Python/JS 코드 블록 중심
- 나머지 텍스트가 "orphans" Concept로 몰림
- 부모 컨텍스트 2000자 제한으로 의미 경계 희석

### 해결 방안
1. **텍스트 단독 유닛 지원**: 코드 없는 텍스트도 의미 단위로 그룹화
2. **부모 컨텍스트 확장**: 2000자 → 설정 가능하게

### 수정 파일
1. `ingestion/segmentation.py` - 텍스트 유닛 로직 추가
2. `api/use_cases/ingest.py:272` - 컨텍스트 길이 설정화
3. `shared/config.py` - parent_context_limit 설정 추가

---

## 구현 순서 (권장)

### Phase 1: 핵심 수정 (🔴 높음)
1. **Collection 스코핑** - 가장 쉽고 영향 큼
2. **Deterministic ID** - 중복 임베딩 방지

### Phase 2: 안정성 개선 (🟡 중간)
3. **배치 로직 개선** - 모델 제한 초과 방지
4. **PyMuPDF OCR 폴백** - 스캔 PDF 처리

### Phase 3: 품질 향상 (🟡 중간)
5. **Orphans 개선** - 검색 정확도 향상

---

## 파일 변경 요약

| 파일 | 변경 내용 |
|------|----------|
| `retrieval/search.py` | collection JOIN 추가 |
| `ingestion/segmentation.py` | unit_id 결정적 생성 |
| `ingestion/concept_builder.py` | concept_id 결정적 생성 |
| `api/use_cases/ingest.py` | Document ID 결정적 + OCR 설정 전달 |
| `storage/vector_store.py` | 배치 로직 개선 |
| `shared/batching.py` (신규) | iter_by_char_budget 함수 |
| `ingestion/parsers/pymupdf_parser.py` | OCR 폴백 추가 |
| `shared/config.py` | parent_context_limit 추가 |
