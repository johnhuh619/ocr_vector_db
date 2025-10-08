import re
from typing import List, Optional


class TextPreprocessor:
    """High-level text preprocessing helpers for OCR/markdown content."""

    NORMALIZE_MAP = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }

    CODE_HINT = re.compile(
        r"```|;\s*$|{\s*$|^\s*(def|class|import|from|async|await|try|except|with|for|while|return|lambda|console\.log|function|const|let|var|=>|export\s+default|import\s+.+\s+from)\b",
        re.M,
    )
    PY_SIGNS = re.compile(r"^\s*(def|class|from|import|try|except|with|async|await|lambda)\b|:\s*$", re.M)
    JS_SIGNS = re.compile(
        r"^\s*(function|const|let|var|class|export|import)\b|=>|;\s*$|{\s*$",
        re.M,
    )

    def normalize(self, text: str) -> str:
        for src, target in self.NORMALIZE_MAP.items():
            text = text.replace(src, target)
        text = re.sub(r"\u00A0", " ", text)  # non-breaking space
        text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace per line
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive newlines
        return text.strip()

    @staticmethod
    def split_paragraph(text: str) -> List[str]:
        parts = re.split(r"\n{2,}", text)
        return [p.strip() for p in parts if p.strip()]

    def is_code_block(self, paragraph: str) -> bool:
        if "```" in paragraph:
            return True
        if len(self.CODE_HINT.findall(paragraph)) >= 1:
            return True
        if self.PY_SIGNS.search(paragraph) or self.JS_SIGNS.search(paragraph):
            return True
        symbols = sum(paragraph.count(x) for x in (";", "{", "}", "=>", "()", "[]", "):"))
        if symbols >= 2:
            return True
        if len(re.findall(r"(?m)^\s{4,}\S", paragraph)) >= 2:
            return True
        return False

    def guess_code_lang(self, paragraph: str) -> Optional[str]:
        py = len(self.PY_SIGNS.findall(paragraph))
        js = len(self.JS_SIGNS.findall(paragraph))
        if py >= js and py > 0:
            return "python"
        if js >= py and js > 0:
            return "javascript"
        if re.search(r"^\s*def\s+", paragraph, re.M) or re.search(r"^\s*class\s+\w+:\s*$", paragraph, re.M):
            return "python"
        if "console.log" in paragraph or "=>" in paragraph or re.search(r";\s*$", paragraph, re.M):
            return "javascript"
        return None

    @staticmethod
    def split_code_safely(code: str, max_chars: int = 900, overlap_lines: int = 10) -> List[str]:
        blocks = re.split(r"(?m)^(?=def\s+\w+\(|class\s+\w+:)", code)
        chunks: List[str] = []
        for block in blocks:
            if len(block) <= max_chars:
                if block.strip():
                    chunks.append(block)
                continue
            lines = block.splitlines()
            start = 0
            while start < len(lines):
                current, acc, i = [], 0, start
                while i < len(lines) and acc + len(lines[i]) + 1 <= max_chars:
                    current.append(lines[i])
                    acc += len(lines[i]) + 1
                    i += 1
                if not current:
                    current = [lines[start][:max_chars]]
                    i = start + 1
                chunk = "\n".join(current).strip()
                if chunk:
                    chunks.append(chunk)
                next_start = max(0, i - overlap_lines)
                if next_start <= start:
                    next_start = i
                start = next_start
        return chunks


__all__ = ["TextPreprocessor"]
