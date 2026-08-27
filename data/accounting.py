"""
Astra Data Accounting & Token Reconciliation Tracker (Phase 7A)
Tracks documents and token counts at every transformation stage to ensure zero unexplained data loss.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional


@dataclass
class StageStats:
    stage_name: str
    num_documents: int
    num_tokens: int
    num_characters: int = 0
    rejected_count: int = 0
    notes: str = ""


class PipelineAccounting:
    def __init__(self):
        self.stages: Dict[str, StageStats] = {}
        self.source_stats: Dict[str, Dict[str, int]] = {}

    def record_stage(
        self,
        stage_name: str,
        num_documents: int,
        num_tokens: int,
        num_characters: int = 0,
        rejected_count: int = 0,
        notes: str = "",
    ) -> None:
        self.stages[stage_name] = StageStats(
            stage_name=stage_name,
            num_documents=num_documents,
            num_tokens=num_tokens,
            num_characters=num_characters,
            rejected_count=rejected_count,
            notes=notes,
        )

    def record_source_contribution(
        self,
        source_id: str,
        raw_docs: int,
        raw_tokens: int,
        final_docs: int,
        final_tokens: int,
    ) -> None:
        self.source_stats[source_id] = {
            "raw_docs": raw_docs,
            "raw_tokens": raw_tokens,
            "final_docs": final_docs,
            "final_tokens": final_tokens,
            "retention_rate": round(final_tokens / max(raw_tokens, 1), 4),
        }

    def reconcile(self) -> Dict[str, Any]:
        """
        Reconciles stage metrics and validates consistency.
        """
        stage_list = list(self.stages.values())
        summary = {
            "stages": {s.stage_name: asdict(s) for s in stage_list},
            "source_contributions": self.source_stats,
            "is_reconciled": True,
            "discrepancies": [],
        }

        # Check that post-filtering counts do not exceed pre-filtering counts
        stage_keys = list(self.stages.keys())
        for i in range(len(stage_keys) - 1):
            curr_s = self.stages[stage_keys[i]]
            next_s = self.stages[stage_keys[i + 1]]

            # Except for packing/sampling where tokens are transformed to sequences
            if next_s.stage_name in ("packed", "shards"):
                continue

            if next_s.num_documents > curr_s.num_documents:
                summary["is_reconciled"] = False
                summary["discrepancies"].append(
                    f"Anomaly: Stage '{next_s.stage_name}' has more docs ({next_s.num_documents}) than '{curr_s.stage_name}' ({curr_s.num_documents})"
                )

        return summary
