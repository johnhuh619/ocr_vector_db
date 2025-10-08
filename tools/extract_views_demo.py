import re
import sys
from pathlib import Path


IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^```(\w+)?\s*$")


def parse_markdown_views(text: str):
    lines = text.splitlines()

    images = []  # {alt, url, caption}
    code_blocks = []  # {lang, content}
    paragraphs = []  # strings (non-code)

    # Extract code blocks and collect non-code lines for paragraph pass
    in_code = False
    code_lang = None
    buf = []
    non_code_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        m = FENCE_RE.match(line)
        if m:
            if not in_code:
                in_code = True
                code_lang = (m.group(1) or "").strip().lower() or None
                buf = []
            else:
                # closing fence
                code_blocks.append({"lang": code_lang or "unknown", "content": "\n".join(buf).strip()})
                in_code = False
                code_lang = None
                buf = []
            i += 1
            continue
        if in_code:
            buf.append(line)
        else:
            non_code_lines.append(line)
        i += 1

    # Paragraphs from non-code segments: split by blank lines
    para_buf = []
    for line in non_code_lines:
        if line.strip() == "":
            if para_buf:
                paragraphs.append("\n".join(para_buf).strip())
                para_buf = []
        else:
            para_buf.append(line)
    if para_buf:
        paragraphs.append("\n".join(para_buf).strip())

    # Images and captions: scan full text by lines
    for idx, line in enumerate(lines):
        for m in IMG_RE.finditer(line):
            alt, url = m.group(1).strip(), m.group(2).strip()
            # caption candidates: next non-empty line or same-line tail
            caption = None
            # next non-empty line (if not a code fence)
            j = idx + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and not FENCE_RE.match(lines[j]):
                cap_line = lines[j].strip()
                if cap_line:
                    caption = cap_line
            # fallback to alt
            if not caption:
                caption = alt
            images.append({"alt": alt, "url": url, "caption": caption})

    return {
        "paragraphs": paragraphs,
        "code_blocks": code_blocks,
        "images": images,
    }


def summarize(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_markdown_views(text)

    out = []
    out.append(f"## File: {md_path}")
    # Paragraphs: show first 1-2
    paras = parsed["paragraphs"]
    out.append(f"- paragraphs: {len(paras)}")
    for p in paras[:2]:
        snippet = p.replace("\n", " ")[:120]
        out.append(f"  - {snippet}…")

    # Code blocks: count and list first two langs
    codes = parsed["code_blocks"]
    out.append(f"- code_blocks: {len(codes)}")
    for cb in codes[:2]:
        first = cb["content"].splitlines()[:2]
        preview = " ".join(first)[:120]
        out.append(f"  - lang={cb['lang']}: {preview}…")

    imgs = parsed["images"]
    out.append(f"- images: {len(imgs)}")
    for im in imgs[:2]:
        out.append(f"  - alt={im['alt']!r} url={im['url']!r} caption={im['caption']!r}")

    return "\n".join(out) + "\n"


def main(argv):
    if len(argv) < 2:
        print("Usage: python tools/extract_views_demo.py <md1> [<md2> ...]")
        return 2
    outputs = []
    for p in argv[1:]:
        md = Path(p)
        if md.exists():
            outputs.append(summarize(md))
        else:
            print(f"[WARN] not found: {p}")
    print("# 샘플 추출 결과\n")
    for o in outputs:
        print(o)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

