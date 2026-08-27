"""
Astra Checkpoint Lineage & Provenance Tracker (Phase 6)
Enforces strict checkpoint continuity and ancestry validation across training runs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Any, List, Optional


class LineageError(Exception):
    """Raised when checkpoint lineage or resumption violates provenance rules."""
    pass


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    experiment_id: str
    step: int
    tokens_seen: int
    checkpoint_path: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_checkpoint_id: Optional[str] = None
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None

    def __post_init__(self):
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id cannot be empty")
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        if self.tokens_seen < 0:
            raise ValueError(f"tokens_seen must be non-negative, got {self.tokens_seen}")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path cannot be empty")


class CheckpointLineage:
    """Manages the directed ancestry of checkpoints within an experiment."""
    def __init__(self):
        self._checkpoints: Dict[str, CheckpointRecord] = {}

    def add_checkpoint(self, record: CheckpointRecord) -> None:
        if record.checkpoint_id in self._checkpoints:
            raise LineageError(f"Checkpoint '{record.checkpoint_id}' already exists in lineage.")

        if record.parent_checkpoint_id is not None:
            if record.parent_checkpoint_id not in self._checkpoints:
                raise LineageError(
                    f"Parent checkpoint '{record.parent_checkpoint_id}' does not exist in lineage."
                )
            parent = self._checkpoints[record.parent_checkpoint_id]
            if record.step < parent.step:
                raise LineageError(
                    f"Invalid lineage: child checkpoint step ({record.step}) cannot be earlier "
                    f"than parent checkpoint step ({parent.step})."
                )

        self._checkpoints[record.checkpoint_id] = record

    def get(self, checkpoint_id: str) -> Optional[CheckpointRecord]:
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> List[CheckpointRecord]:
        return sorted(self._checkpoints.values(), key=lambda c: c.step)

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(c) for c in self.list_checkpoints()]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> CheckpointLineage:
        lineage = cls()
        # Sort by step to ensure parents are added before children
        sorted_data = sorted(data, key=lambda d: d.get("step", 0))
        for item in sorted_data:
            lineage.add_checkpoint(CheckpointRecord(**item))
        return lineage
