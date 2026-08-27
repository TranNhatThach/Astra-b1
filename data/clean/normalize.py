"""
Astra Unicode & Text Normalizer (Phase 1C)
Enforces strict Canonical Decomposition followed by Canonical Composition (NFC),
and tracks replacement character / encoding artifacts.
"""

import unicodedata
from typing import Tuple, Dict, Any


def normalize_text_nfc(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Normalizes text to Unicode NFC and audits modification statistics.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. NFC Normalization
    norm_text = unicodedata.normalize("NFC", text)

    # 2. Audit Statistics
    stats = {
        "original_len": len(text),
        "normalized_len": len(norm_text),
        "changed": text != norm_text,
        "replacement_chars_count": norm_text.count("\ufffd"),
        "null_bytes_count": norm_text.count("\x00"),
    }

    # 3. Clean fatal binary corruption characters
    if stats["null_bytes_count"] > 0:
        norm_text = norm_text.replace("\x00", "")

    return norm_text, stats


def verify_vietnamese_diacritics(text: str) -> bool:
    """
    Verifies that Vietnamese tone marks and special characters (ă, â, đ, ê, ô, ơ, ư)
    are properly normalized into composed NFC characters instead of combining diacritics.
    """
    norm = unicodedata.normalize("NFC", text)
    # Check if combining diacritics (U+0300 to U+036F) remain uncomposed
    combining_marks = [c for c in norm if 0x0300 <= ord(c) <= 0x036F]
    return len(combining_marks) == 0
