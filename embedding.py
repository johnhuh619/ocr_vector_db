import os, re, glob, uuid, math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_voyageai import VoyageAIEmbeddings
from langchain_postgres import PGVector

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

PG_CONN = os.getenv("PG_CONN")
COLLECTION = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")
EMBDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# OCR 전처리
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-"}

# 코드 전처리
CODE_HINT = re.compile(
    r"```|;\s*$|{\s*$|^\s*(def|class|import|from|async|await|try|except|with|for|while|return|lambda|console\.log|function|const|let|var|=>|export\s+default|import\s+.+\s+from)\b",
    re.M,
)

PY_SIGNS = re.compile(r"^\s*(def|class|from|import|try|except|with|async|await|lambda)\b|:\s*$", re.M)
JS_SIGNS = re.compile(r"^\s*(function|const|let|var|class|export|import)\b|=>|;\s*$|{\s*$", re.M)

def normalize(text: str):
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
        
    # 공백/페이지 흔적 정리
    text = re.sub(r"\u00A0", " ", text)        # non-breaking space
    text = re.sub(r"[ \t]+\n", "\n", text)     # 줄 끝 공백 제거
    text = re.sub(r"\n{3,}", "\n\n", text)     # 과도한 개행 축소
    return text.strip()

def split_paragraph(text: str):
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]

# md 일 경우 고려한 메서드
def is_code_block(p: str):
    if "```" in p:
        return True
    hits = len(CODE_HINT.findall(p))
    return hits >= 2 or any(sym in p for sym in ("{", "};", ");")) 

def guess_code_lang(p: str):
    py = len(PY_SIGNS.findall(p))    
    js = len(JS_SIGNS.findall(p))
    if py >= js and py > 0:
        return "python"
    if js >= py and js > 0:
        return "javascript"
    
    if re.search(r"^\s*def\s+", p, re.M) or re.search(r"^\s*class\s+\w+:\s*$", p, re.M):
        return "python"
    if "console.log" in p or "=>" in p or re.search(r";\s*$", p, re.M):
        return "javascript"
    return None

def split_code_safely(code: str, max_chars: int = 1800, overlap_lines: int = 5):
    if len(code) <= max_chars:
        return [code]
    
    lines = code.splitlines()
    chunks = []
    start = 0
    
    # cur   : 청크에 넣을 라인
    # length: 현재까지 누적된 라인
    # i     : 라인 스캔 위한 idx 
    while start < len(lines):
        cur = [] 
        length = 0
        i = start
        while i < len(lines) and (length + len(lines[i])) + 1 <= max_chars:
            cur.append(lines[i])
            length += len(lines[i]) + 1
            i += 1
        if not cur:
            # case: 기준치를 초과하는 긴 라인
            cur = [lines[start][:max_chars]]
            i = start + 1
    
        chunks.append("\n".join(cur))
        
        next_start = i - overlap_lines
        if next_start <= start:
            next_start = i
        start = max(0, next_start)
    
    return chunks
    

@dataclass
class RawSegment:
    kind: str   # txt / code
    content: str
    language: str
    order: int
    
    
def parse_ocr_file(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    
    raw = normalize(raw)
    paras = split_paragraph(raw)
    
    segments: List[RawSegment] = []
    for idx, p in enumerate(paras):
        if is_code_block(p):
            lang = guess_code_lang(p)
            segments.append(RawSegment("code", p,lang, idx))
        else:
            segments.append(RawSegment("txt", p, None, idx))
    
    return segments

# langchain 저서 특화 코드 pairing
def unitize_txt_py_js_streaming(
    segments: List[RawSegment],
    attach_pre_text: bool = True,
    attach_post_text: bool = False,
    bridge_text_max: int = 0,        # py와 js 사이 텍스트 허용 개수(기본 0: 없음)
    max_pre_text_chars: int = 4000   # pre_text로 붙일 텍스트 총 길이 상한(과대포함 방지)
) -> List[Tuple[Optional[str], str, RawSegment]]:
    """
    현실 패턴: [txt*] -> python(code)+ -> [txt{0..bridge_text_max}] -> javascript(code)+ -> [txt* (보통 제외)]
    반환: [(unit_id, unit_role, segment)], unit_role ∈ {"pre_text","python","javascript","post_text","other","bridge_text"}

    설계 포인트
    - 최근 연속 텍스트를 버퍼링했다가 py 시작 시 pre_text로 편입
    - js 이후 텍스트는 기본 제외(attach_post_text=True 때만 포함)
    - py 뒤에 js가 없어도 단일 섹션으로 존중
    - js 단독 등장(앞에 py 없음)은 other 처리
    """
    out: List[Tuple[Optional[str], str, RawSegment]] = []
    text_buf: List[RawSegment] = []   # 아직 섹션 귀속이 정해지지 않은 txt 버퍼
    text_buf_chars = 0

    i, n = 0, len(segments)
    while i < n:
        s = segments[i]

        # 1) 텍스트는 일단 버퍼링 (나중에 py가 오면 pre_text로, 아니면 other로)
        if s.kind == "text":
            text_buf.append(s)
            text_buf_chars += len(s.content)
            # pre_text 과대포함 방지: 너무 길면 오래된 텍스트부터 out(other)로 배출
            while text_buf_chars > max_pre_text_chars and text_buf:
                old = text_buf.pop(0)
                text_buf_chars -= len(old.content)
                out.append((None, "other", old))
            i += 1
            continue

        # 2) 코드 블록인 경우
        if s.kind == "code" and s.language == "python":
            unit_id = str(uuid.uuid4())

            # 2-1) pre_text 부착
            if attach_pre_text and text_buf:
                for t in text_buf:
                    out.append((unit_id, "pre_text", t))
                text_buf.clear()
                text_buf_chars = 0
            else:
                # pre_text를 붙이지 않는다면, 지금까지의 텍스트는 other로 방출
                while text_buf:
                    out.append((None, "other", text_buf.pop(0)))
                text_buf_chars = 0

            # 2-2) python 연속 블록 수집
            while i < n and segments[i].kind == "code" and segments[i].language == "python":
                out.append((unit_id, "python", segments[i]))
                i += 1

            # 2-3) (선택) py와 js 사이 브리지 텍스트 허용
            bridged = 0
            while (bridged < bridge_text_max and i < n and segments[i].kind == "text"):
                out.append((unit_id, "bridge_text", segments[i]))
                i += 1
                bridged += 1

            # 2-4) javascript 연속 블록 수집 (있으면)
            if i < n and segments[i].kind == "code" and segments[i].language == "javascript":
                while i < n and segments[i].kind == "code" and segments[i].language == "javascript":
                    out.append((unit_id, "javascript", segments[i]))
                    i += 1

                # 2-5) JS 이후 텍스트 처리(기본 제외)
                if attach_post_text:
                    while i < n and segments[i].kind == "text":
                        # 다음 py가 나오면 멈추고, 그 텍스트는 다음 섹션의 pre_text 후보가 되게 버퍼로 보냄
                        if (i + 1 < n and segments[i+1].kind == "code" and segments[i+1].language == "python"):
                            text_buf.append(segments[i])
                            text_buf_chars += len(segments[i].content)
                            i += 1
                            break
                        out.append((unit_id, "post_text", segments[i]))
                        i += 1
                # attach_post_text=False(default)이면 JS 뒤 텍스트는 버퍼로 남겨 다음 섹션의 pre_text 후보가 됨
            else:
                # js가 없는 python-only 섹션으로 확정
                # 이후 텍스트는 버퍼에 남겨두고 다음 섹션의 pre_text 후보로 사용
                pass

            continue  # 섹션 처리 끝, 다음 루프

        # 3) JS가 파이썬 없이 단독 등장 → other로 처리
        if s.kind == "code" and s.language == "javascript":
            # 버퍼에 남은 텍스트는 other로 방출(해당 js에 억지로 붙이지 않음)
            while text_buf:
                out.append((None, "other", text_buf.pop(0)))
                text_buf_chars = 0
            out.append((None, "other", s))
            i += 1
            continue

        # 4) 그 외 코드(언어 미상 등)는 other
        while text_buf:
            out.append((None, "other", text_buf.pop(0)))
            text_buf_chars = 0
        out.append((None, "other", s))
        i += 1

    # 루프 종료 후 버퍼 텍스트는 other로 방출
    while text_buf:
        out.append((None, "other", text_buf.pop(0)))
        text_buf_chars = 0

    return out



# 분할 txt/ code (markdown)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap=150,
    separators=["\n##", "\n###", "\n\n", "\n", " ", ""],
    length_function=len,
    add_start_index= True,
)


# 유닛 -> chunk 변환
def build_document_from_unitized(
    path: str,
    unitized: List[Tuple[Optional[str], str, RawSegment]]
) -> List[Document]:
    docs: List[Document] = []
    base_meta = {"source": os.path.basename(path)}
    
    for unit_id, unit_role, seg in unitized:
        meta = dict(base_meta)
        meta['order'] = seg.order
        meta['kind'] = seg.kind
        if unit_id:
            meta["unit_id"] = unit_id
        if unit_role:
            meta["unit_role"] = unit_role
        
        if seg.kind == "text":
            for chunk in TEXT_SPLITTER._split_text(seg.content):
                docs.append(Document(page_content=chunk, metadata=meta))
        else:
            lang = seg.language or "unknown"
            lang_meta = {**meta, "lang": lang}
            for chunk in split_code_safely(seg.content):
                docs.append(Document(page_content=chunk, metadata=lang_meta))
    return docs


def upsert_batch(store: PGVector, docs: List[Document], batch_size: int=64):
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        store.add_documents(batch)


# 인덱스
def ensure_extension_vector():
    import psycopg
    with psycopg.connect(PG_CONN.replace("postgresql+psycopg", "postgresql"), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
def ensure_indexes():
    """pgvector HNSW(cosine) + JSONB GIN + 선택 BTREE 인덱스(메타 equality)"""
    import psycopg
    with psycopg.connect(PG_CONN.replace("postgresql+psycopg", "postgresql"), autocommit=True) as conn:
        with conn.cursor() as cur:
            tbl = "langchain_pg_embedding"

            # HNSW (코사인)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{COLLECTION}_hnsw_cosine
                ON {tbl} USING hnsw (embedding vector_cosine_ops);
            """)

            # 메타데이터 GIN (JSONB)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{COLLECTION}_meta_gin
                ON {tbl} USING GIN (cmetadata);
            """)

            # 선택: unit_id/unit_role/lang equality 빠르게
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{COLLECTION}_unit_id_btree
                ON {tbl} ((cmetadata->>'unit_id'));
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{COLLECTION}_unit_role_btree
                ON {tbl} ((cmetadata->>'unit_role'));
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{COLLECTION}_lang_btree
                ON {tbl} ((cmetadata->>'lang'));
            """)
            
            
            
def main(input_glob:str):
    
    embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
    store = PGVector(
        connection=PG_CONN,
        embeddings=embeddings,
        collection_name=COLLECTION,
        distance_strategy="COSINE",
        use_jsonb=True,
        embedding_length=EMBDDING_DIM
    )
    
    files= sorted(glob.glob(input_glob))
    if not files:
        print(f"[WARN] No files matched: {input_glob}")
        return
    
    ensure_extension_vector()
    
    total_docs= 0
    for path in files:
        print(f"[parse] {path}")
        segs = parse_ocr_file(path)
        
        unitized = unitize_txt_py_js_streaming(
            segs,
            attach_pre_text=True,
            attach_post_text=False,
            bridge_text_max=0           
        )
        
        docs = build_document_from_unitized(path, unitized)
        if not docs:
            continue
        
        print(f"[upsert] {os.path.basename(path)} -> {len(docs)} chunks")
        upsert_batch(store,docs)
        total_docs += len(docs)
        
    
    print(f"[done] total chunks: {total_docs}")
    print("[index] creating indexes (idempotent:멱등성)")
    ensure_indexes()
    print("[ok] all set")
    
    
# 일단 txt라고 가정하고 구현했다.
if __name__ == '__main__':
    import sys
    glob_args = sys.argv[1] if len(sys.argv) > 1 else "test/*.txt"
    main(glob_args)