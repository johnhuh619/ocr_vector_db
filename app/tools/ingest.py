import os
import sys
import glob as _glob
import argparse
from typing import List, Optional

from dotenv import load_dotenv

import app as emb


def expand_inputs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        hit = _glob.glob(p)
        if not hit and os.path.isfile(p):
            hit = [p]
        files.extend(hit)
    # dedupe and keep order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def warn_pg_conn(conn: Optional[str]):
    if not conn:
        print("[ERR] PG_CONN is not set")
        return
    if "pycopg" in conn and "+pycopg" in conn:
        # common typo: postgresql+pycopg -> postgresql+psycopg
        print("[WARN] PG_CONN uses '+pycopg'; did you mean '+psycopg'? ->", conn)


def run(args: argparse.Namespace) -> int:
    # env + overrides
    load_dotenv()
    if args.auto_ocr:
        os.environ["ENABLE_AUTO_OCR"] = "true"
    if args.custom_schema_write is not None:
        os.environ["CUSTOM_SCHEMA_WRITE"] = "true" if args.custom_schema_write else "false"

    files = expand_inputs(args.inputs)
    if not files:
        print("[WARN] No input files")
        return 2

    # Configure embedding + store unless dry-run
    embeddings = emb.VoyageAIEmbeddings(model=emb.EMBEDDING_MODEL)
    emb.validate_embedding_dimension(embeddings, emb.EMBDDING_DIM)
    if not args.dry_run:
        emb.apply_db_level_tuning()
        store = emb.PGVector(
            connection=emb.PG_CONN,
            embeddings=embeddings,
            collection_name=emb.COLLECTION,
            distance_strategy="COSINE",
            use_jsonb=True,
            embedding_length=emb.EMBDDING_DIM,
        )
        emb.ensure_extension_vector()
        emb.ensure_parent_docstore()
        if args.custom_schema_write or emb.CUSTOM_SCHEMA_WRITE:
            emb.ensure_custom_schema()
    else:
        store = None  # type: ignore

    # Parent mode override
    if args.parent_mode:
        emb.PARENT_MODE = args.parent_mode.lower()
        print(f"[cfg] PARENT_MODE={emb.PARENT_MODE}")

    warn_pg_conn(os.getenv("PG_CONN"))

    total_docs = 0
    total_parents = 0

    for path in files:
        print(f"[parse] {path}")
        ext = os.path.splitext(path)[1].lower()
        # parse to segments
        if ext == ".pdf":
            txt = emb.extract_text_from_pdf(path)
            if not txt or emb.is_low_text_density(txt):
                print(
                    f"[warn] PDF text is sparse/empty; OCR recommended: ocrmypdf --sidecar {os.path.splitext(path)[0]}.txt --skip-text {path} {os.path.splitext(path)[0]}.ocr.pdf"
                )
                if args.auto_ocr and emb.shutil.which("ocrmypdf"):
                    sidecar = os.path.splitext(path)[0] + ".txt"
                    try:
                        emb.subprocess.run(
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
                        with open(sidecar, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read()
                    except Exception as e:
                        print(f"[warn] auto OCR failed: {e}")
            if not txt:
                print(f"[skip] No extractable text for {path}")
                continue
            segs = emb.parse_ocr_text(txt)
        else:
            segs = emb.parse_ocr_file(path)

        unitized = emb.unitize_txt_py_js_streaming(
            segs,
            attach_pre_text=True,
            attach_post_text=False,
            bridge_text_max=0,
        )
        docs = emb.build_document_from_unitized(path, unitized)

        try:
            emb.assign_parent_by_page_section(docs, path)
        except Exception as e:
            print(f"[warn] parent assign skipped: {e}")

        try:
            cap_docs = emb.augment_with_captions(docs)
            if cap_docs:
                docs.extend(cap_docs)
                print(f"[augment] captions added: {len(cap_docs)}")
        except Exception as e:
            print(f"[warn] caption augmentation skipped: {e}")

        parents = emb.build_parent_entries(docs)
        print(f"[ok] {os.path.basename(path)} -> segments={len(segs)} docs={len(docs)} parents={len(parents)}")

        # show a few samples
        for i, d in enumerate(docs[:5], 1):
            print(
                f"[doc {i}] view={d.metadata.get('view')} lang={d.metadata.get('lang')} parent={d.metadata.get('parent_id')} order={d.metadata.get('order')} len={len(d.page_content)}"
            )
        for i, (pid, content, meta) in enumerate(parents[:3], 1):
            preview = (content or "").replace("\n", " ")[:120]
            print(
                f"[parent {i}] id={pid} views={meta.get('views')} page={meta.get('page')} order={meta.get('order')} | {preview}"
            )

        if not args.dry_run:
            try:
                emb.upsert_parents(parents)
                if args.custom_schema_write or emb.CUSTOM_SCHEMA_WRITE:
                    emb.dual_write_custom_schema(embeddings, parents, docs)
                emb.upsert_batch(store, docs)
            except Exception as e:
                print(f"[ERR] DB upsert failed for {path}: {e}")
                return 2

        total_docs += len(docs)
        total_parents += len(parents)

    print(f"[done] total chunks: {total_docs}")
    print(f"[done] total parents: {total_parents}")

    if not args.dry_run and not args.skip_index:
        print("[index] creating indexes (idempotent)")
        emb.ensure_indexes()
        print("[ok] all set")

    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="OCR/PDF ingest to PGVector with multi-view support")
    ap.add_argument("inputs", nargs="+", help="Files or globs (e.g., data/*.txt docs/*.pdf)")
    ap.add_argument("--auto-ocr", action="store_true", help="Enable OCR fallback via ocrmypdf if PDF text is sparse")
    ap.add_argument(
        "--parent-mode",
        choices=["unit", "page", "section", "page_section"],
        help="Override parent grouping mode",
    )
    ap.add_argument("--custom-schema-write", dest="custom_schema_write", action="store_true", help="Also write to custom schema")
    ap.add_argument("--no-custom-schema-write", dest="custom_schema_write", action="store_false", help="Disable custom schema write")
    ap.add_argument("--dry-run", action="store_true", help="Parse and show stats without DB writes")
    ap.add_argument("--skip-index", action="store_true", help="Skip index creation at the end")
    ap.set_defaults(custom_schema_write=None)
    args = ap.parse_args(argv[1:])
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

