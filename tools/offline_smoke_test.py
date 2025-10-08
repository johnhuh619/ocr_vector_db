import os
from pathlib import Path

import embedding as emb
import os


def run_offline(path: str):
    p = Path(path)
    if not p.exists():
        print(f"[ERR] not found: {path}")
        return 2
    if p.suffix.lower() == ".pdf":
        txt = emb.extract_text_from_pdf(str(p))
        if not txt or emb.is_low_text_density(txt):
            print("[warn] PDF looks scanned or sparse; run OCR or enable ENABLE_AUTO_OCR=true and install ocrmypdf")
        if not txt:
            return 2
        segs = emb.parse_ocr_text(txt)
    else:
        segs = emb.parse_ocr_file(str(p))
    unitized = emb.unitize_txt_py_js_streaming(segs, attach_pre_text=True, attach_post_text=False, bridge_text_max=0)
    docs = emb.build_document_from_unitized(str(p), unitized)
    try:
        emb.assign_parent_by_page_section(docs, str(p))
    except Exception as e:
        print(f"[warn] parent assign skipped: {e}")

    parents = emb.build_parent_entries(docs)
    print(f"[ok] segments={len(segs)} docs={len(docs)} parents={len(parents)}")

    # show sample
    for i, d in enumerate(docs[:5], 1):
        print(f"[doc {i}] view={d.metadata.get('view')} lang={d.metadata.get('lang')} parent={d.metadata.get('parent_id')} order={d.metadata.get('order')} len={len(d.page_content)}")
    for i, (pid, content, meta) in enumerate(parents[:3], 1):
        print(f"[parent {i}] id={pid} views={meta.get('views')} page={meta.get('page')} order={meta.get('order')} | {content[:120].replace('\n',' ')}")
    return 0


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "test.txt"
    raise SystemExit(run_offline(target))
