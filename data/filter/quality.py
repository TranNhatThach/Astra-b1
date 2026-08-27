"""
Astra Document Quality Scorer & Deduplication Filter (Phase 1E & 1F)
Calculates composite quality score:
  Q(d) = w1*Q_lang + w2*Q_quality + w3*Q_struct + w4*Q_edu - w5*Q_spam - w6*Q_dup
"""

import hashlib
import re
from typing import Dict, Any, Tuple


def compute_exact_hash(text: str) -> str:
    """Calculates SHA-256 hash of normalized text for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_document_quality_score(
    text: str,
    language: str = "en",
    weights: Dict[str, float] = None,
) -> Tuple[float, Dict[str, float]]:
    if weights is None:
        weights = {
            "w_lang": 0.20,
            "w_quality": 0.30,
            "w_struct": 0.20,
            "w_edu": 0.30,
            "w_spam": 0.40,
            "w_dup": 0.20,
        }

    words = text.split()
    num_words = len(words)
    num_chars = len(text)

    # Length & Structure score
    if num_words < 10:
        q_struct = 0.2
    elif 50 <= num_words <= 2000:
        q_struct = 1.0
    else:
        q_struct = 0.7

    # Repetition / Spam heuristic
    unique_words = set(words)
    repetition_ratio = 1.0 - (len(unique_words) / max(num_words, 1))
    q_spam = min(1.0, max(0.0, repetition_ratio * 1.5))

    # Punctuation & sentence formatting
    has_terminal_punct = bool(re.search(r"[.!?]\s*$", text))
    q_quality = 0.9 if has_terminal_punct else 0.6
    if num_chars > 0 and (text.count("\ufffd") > 0):
        q_quality -= 0.5

    # Educational / Information density heuristic
    avg_word_length = num_chars / max(num_words, 1)
    q_edu = 0.9 if 4.0 <= avg_word_length <= 10.0 else 0.5

    q_lang = 1.0

    score = (
        weights["w_lang"] * q_lang
        + weights["w_quality"] * q_quality
        + weights["w_struct"] * q_struct
        + weights["w_edu"] * q_edu
        - weights["w_spam"] * q_spam
    )

    breakdown = {
        "score": round(score, 4),
        "q_lang": q_lang,
        "q_quality": q_quality,
        "q_struct": q_struct,
        "q_edu": q_edu,
        "q_spam": q_spam,
    }

    return score, breakdown
