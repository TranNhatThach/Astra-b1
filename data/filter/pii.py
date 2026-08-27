"""
Astra PII Detection & Redaction Filter (Phase 7)
Detects and redacts sensitive Personally Identifiable Information:
  - Emails
  - IPv4 / IPv6 addresses
  - Phone numbers
  - API Keys / Secret Tokens
"""

import re
from typing import Tuple, Dict, Any


EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
IPV4_REGEX = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
API_KEY_REGEX = re.compile(r"\b(?:ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|sk-[a-zA-Z0-9]{32,})\b")


def filter_pii(text: str, redact: bool = True) -> Tuple[str, Dict[str, int]]:
    """
    Redacts detected PII tokens with placeholder markers.
    """
    counts = {
        "email": len(EMAIL_REGEX.findall(text)),
        "ipv4": len(IPV4_REGEX.findall(text)),
        "phone": len(PHONE_REGEX.findall(text)),
        "api_key": len(API_KEY_REGEX.findall(text)),
    }

    if not redact:
        return text, counts

    redacted = EMAIL_REGEX.sub("[EMAIL]", text)
    redacted = IPV4_REGEX.sub("[IP_ADDR]", redacted)
    redacted = PHONE_REGEX.sub("[PHONE]", redacted)
    redacted = API_KEY_REGEX.sub("[SECRET_KEY]", redacted)

    return redacted, counts
