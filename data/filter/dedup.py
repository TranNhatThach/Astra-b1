"""
Astra Exact & Near Deduplication Engine (Phase 7)
Implements exact SHA-256 deduplication and MinHash (Jaccard similarity) near-dedup.
"""

import hashlib
from typing import Set, List, Dict, Tuple


class Deduplicator:
    def __init__(self, num_perm: int = 64, jaccard_threshold: float = 0.8):
        self.exact_hashes: Set[str] = set()
        self.num_perm = num_perm
        self.jaccard_threshold = jaccard_threshold
        self.minhash_signatures: List[List[int]] = []

    def is_exact_duplicate(self, text: str) -> bool:
        """Returns True if exact SHA-256 hash has been seen."""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self.exact_hashes:
            return True
        self.exact_hashes.add(h)
        return False

    def _compute_minhash(self, text: str) -> List[int]:
        # 5-gram token shingles
        words = text.split()
        if len(words) < 5:
            shingles = set(words)
        else:
            shingles = {" ".join(words[i:i+5]) for i in range(len(words) - 4)}

        sig = []
        for i in range(self.num_perm):
            min_val = float("inf")
            for shingle in shingles:
                val = int(hashlib.md5(f"{i}_{shingle}".encode("utf-8")).hexdigest()[:8], 16)
                if val < min_val:
                    min_val = val
            sig.append(min_val if min_val != float("inf") else 0)
        return sig

    def is_near_duplicate(self, text: str) -> bool:
        """Estimates Jaccard similarity against previously stored signatures."""
        sig = self._compute_minhash(text)
        if not self.minhash_signatures:
            self.minhash_signatures.append(sig)
            return False

        for existing_sig in self.minhash_signatures[-500:]:  # sliding window check
            matches = sum(1 for a, b in zip(sig, existing_sig) if a == b)
            similarity = matches / self.num_perm
            if similarity >= self.jaccard_threshold:
                return True

        self.minhash_signatures.append(sig)
        return False

    def filter_document(self, text: str) -> Tuple[bool, str]:
        """
        Returns (keep_document, reason_if_dropped).
        """
        if self.is_exact_duplicate(text):
            return False, "exact_duplicate"
        if self.is_near_duplicate(text):
            return False, "near_duplicate"
        return True, "unique"
