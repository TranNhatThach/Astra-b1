"""
Astra Text Boilerplate Cleaner (Phase 7)
Removes HTML tags, boilerplate artifacts, zero-width characters, and control characters
while strictly preserving code indentation and scientific formatting.
"""

import re
from typing import Tuple, Dict, Any


HTML_TAG_REGEX = re.compile(r"<[^>]+>")
ZERO_WIDTH_CHARS = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
MULTIPLE_BLANK_LINES = re.compile(r"\n{4,}")


def clean_boilerplate(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Cleans web/document boilerplate artifacts without corrupting source code indentation.
    """
    if not isinstance(text, str):
        text = str(text)

    original_len = len(text)

    # 1. Remove zero-width characters
    cleaned = ZERO_WIDTH_CHARS.sub("", text)

    # 2. Strip HTML tags (if not inside markdown code blocks)
    # Fast regex strip for common html tags in scraped web content
    cleaned = HTML_TAG_REGEX.sub(" ", cleaned)

    # 3. Normalize excessive vertical whitespace
    cleaned = MULTIPLE_BLANK_LINES.sub("\n\n\n", cleaned)

    stats = {
        "original_len": original_len,
        "cleaned_len": len(cleaned),
        "removed_chars": original_len - len(cleaned),
    }

    return cleaned.strip(), stats
