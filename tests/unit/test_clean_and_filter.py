from data.clean.boilerplate import clean_boilerplate
from data.filter.pii import filter_pii
from data.filter.safety import check_content_safety
from data.filter.dedup import Deduplicator


def test_boilerplate_cleaner():
    html_text = "<p>Welcome to <b>Astra</b></p>\u200B\n\n\n\n\nFinal."
    cleaned, stats = clean_boilerplate(html_text)
    assert "<p>" not in cleaned
    assert "\u200B" not in cleaned
    assert stats["removed_chars"] > 0


def test_pii_filter():
    text_with_pii = "Contact admin@astra-research.org or IP 192.168.1.100 with key ghp_123456789012345678901234567890123456."
    redacted, counts = filter_pii(text_with_pii, redact=True)
    assert "[EMAIL]" in redacted
    assert "[IP_ADDR]" in redacted
    assert "[SECRET_KEY]" in redacted
    assert counts["email"] == 1
    assert counts["ipv4"] == 1
    assert counts["api_key"] == 1


def test_safety_filter():
    safe_text = "Standard mathematical formulation of recurrent gradient descent."
    is_safe, flags = check_content_safety(safe_text)
    assert is_safe is True
    assert len(flags) == 0

    malicious_text = "<script>alert('xss')</script>"
    is_safe2, flags2 = check_content_safety(malicious_text)
    assert is_safe2 is False
    assert len(flags2) > 0


def test_deduplicator():
    dedup = Deduplicator(num_perm=32, jaccard_threshold=0.8)
    doc1 = "Deep neural network architectures require careful layer normalization and state management."
    doc2 = "Deep neural network architectures require careful layer normalization and state management."
    doc3 = "Different text discussing completely unrelated mathematical concepts and equations."

    keep1, reason1 = dedup.filter_document(doc1)
    assert keep1 is True
    assert reason1 == "unique"

    keep2, reason2 = dedup.filter_document(doc2)
    assert keep2 is False
    assert reason2 == "exact_duplicate"

    keep3, reason3 = dedup.filter_document(doc3)
    assert keep3 is True
    assert reason3 == "unique"
