from __future__ import annotations

import re

LETTERS = "ABCD"
FINAL_RE = re.compile(r"FINAL_ANSWER:\s*\(?([A-D])\)?", re.IGNORECASE)
PAREN_RE = re.compile(r"\(([A-D])\)")
LINE_LETTER_RE = re.compile(r"^\s*\(?([A-D])\)?\s*\.?\s*$", re.IGNORECASE | re.MULTILINE)
# Last standalone A-D token, not inside a longer word (e.g. not "DNA").
STANDALONE_RE = re.compile(r"(?<![A-Za-z0-9])\(?([A-D])\)?(?![A-Za-z0-9])")


def extract_answer(text: str | None) -> str | None:
    """Parse A-D. Failures are missing, never coerced to a wrong letter."""
    if not text:
        return None
    m = FINAL_RE.search(text)
    if m:
        return m.group(1).upper()

    parens = list(PAREN_RE.finditer(text))
    if parens:
        return parens[-1].group(1).upper()

    lines = list(LINE_LETTER_RE.finditer(text))
    if lines:
        return lines[-1].group(1).upper()

    standalone = list(STANDALONE_RE.finditer(text))
    if standalone:
        return standalone[-1].group(1).upper()
    return None


def truncate_cot(text: str, head: int = 1500, tail: int = 500) -> str:
    if text is None:
        return ""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n\n[...truncated...]\n\n{text[-tail:]}"


def parse_verdict(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"VERDICT:\s*(YES|NO)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    stripped = text.strip().splitlines()[-1].strip().upper()
    if stripped in {"YES", "NO"}:
        return stripped
    return None
