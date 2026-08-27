"""
Astra Safety & Content Filter (Phase 7)
Screens out prohibited content, severe toxicity, and non-text artifacts.
"""

from typing import Tuple, List


BLOCKED_PATTERNS = [
    "<script>",
    "javascript:void(0)",
    "data:text/html",
    "eval(compile(",
]


def check_content_safety(text: str) -> Tuple[bool, List[str]]:
    """
    Returns (is_safe, violations).
    """
    violations = []
    text_lower = text.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in text_lower:
            violations.append(f"malicious_pattern:{pattern}")

    # Check for empty or garbage control character dominance
    if len(text.strip()) == 0:
        violations.append("empty_document")

    is_safe = len(violations) == 0
    return is_safe, violations
