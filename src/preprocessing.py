"""Text normalisation and deterministic script detection.

Deliberately conservative: Unicode NFC, control-character removal and whitespace
collapsing only. Romanised Hindi, Indic scripts, punctuation, digits and code-mixing
are preserved because they are the phenomena under study.
"""
from __future__ import annotations

import re
import unicodedata

from .config import MAX_INPUT_CHARS

SCRIPT_RANGES = {
    "Deva": (0x0900, 0x097F),
    "Beng": (0x0980, 0x09FF),
    "Taml": (0x0B80, 0x0BFF),
    "Telu": (0x0C00, 0x0C7F),
    "Knda": (0x0C80, 0x0CFF),
    "Mlym": (0x0D00, 0x0D7F),
}
SCRIPT_TO_LANG = {
    "Deva": "hi-Deva",
    "Beng": "bn-Beng",
    "Taml": "ta-Taml",
    "Telu": "te-Telu",
    "Knda": "kn-Knda",
    "Mlym": "ml-Mlym",
}
_WS = re.compile(r"\s+")
# Keep ZWJ/ZWNJ (U+200C/U+200D): they change rendering of Indic conjuncts.
_BAD_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f﻿​]")


class InvalidInput(ValueError):
    """Raised for empty / non-text input."""


def normalize_text(text: object, max_chars: int = MAX_INPUT_CHARS) -> str:
    if not isinstance(text, str):
        raise InvalidInput("Input must be a string.")
    text = unicodedata.normalize("NFC", text)
    text = _BAD_CTRL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    if not text:
        raise InvalidInput("Enter at least one non-empty message.")
    if not any(ch.isalnum() for ch in text):
        raise InvalidInput("Message contains no letters or digits.")
    return text[:max_chars]


def script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi and ch.isalpha():
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            if ch.isascii() and ch.isalpha() or (0x00C0 <= cp <= 0x024F and ch.isalpha()):
                counts["Latn"] = counts.get("Latn", 0) + 1
    return counts


def detect_script(text: str) -> str:
    """Deterministic Unicode-script detection.

    Returns one of Latn, Deva, Beng, Taml, Telu, Knda, Mlym, ``Deva+Latn`` (both
    present: code-mixed script), ``mixed`` (other multi-script) or ``unknown``.
    The ``Deva+Latn`` rule matches the gold-label rule documented for hinglish_lid
    (script presence in the text).
    """
    c = script_counts(text)
    if not c:
        return "unknown"
    if len(c) == 1:
        return next(iter(c))
    if set(c) == {"Deva", "Latn"}:
        return "Deva+Latn"
    return "mixed"
