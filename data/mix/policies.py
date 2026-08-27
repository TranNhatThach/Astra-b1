"""
Astra Target Data Mixture Policy (Phase 7)
Defines target sampling ratios and domain allocations for Astra pretraining.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MixturePolicy:
    web: float = 0.45
    educational: float = 0.15
    code: float = 0.15
    math: float = 0.10
    vietnamese: float = 0.10
    dialogue: float = 0.05

    def __post_init__(self):
        total = sum([self.web, self.educational, self.code, self.math, self.vietnamese, self.dialogue])
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Mixture weights must sum to 1.0, got {total:.4f}")

    def to_dict(self) -> Dict[str, float]:
        return {
            "web": self.web,
            "educational": self.educational,
            "code": self.code,
            "math": self.math,
            "vietnamese": self.vietnamese,
            "dialogue": self.dialogue,
        }
