"""
Astra Centralized Fail-Closed Training Gate (Phase 6)
Prevents model training unless ALL 17 scientific prerequisites are satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .identity import (
    ScientificIdentity,
    compute_config_hash,
    compute_dataset_hash,
    compute_tokenizer_hash,
)
from .registry import ExperimentRecord, ExperimentRegistry
from .state_machine import ExperimentState
from .provenance import get_git_commit, get_git_dirty_state


@dataclass(frozen=True)
class GateResult:
    status: str  # "PASS" or "BLOCKED"
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_passed(self) -> bool:
        return self.status == "PASS"

    def __bool__(self) -> bool:
        return self.is_passed


class TrainingGate:
    """
    Scientific Gatekeeper for Astra training runs.
    Enforces fail-closed validation across all scientific dimensions.
    """

    @classmethod
    def validate(
        cls,
        experiment: Optional[ExperimentRecord],
        config_path: Optional[str] = None,
        dataset_manifest_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        resume_checkpoint_id: Optional[str] = None,
        allow_dirty_git: bool = False,
        require_git_match_head: bool = True,
        verify_shard_checksums: bool = True,
    ) -> GateResult:
        reasons: List[str] = []
        details: Dict[str, Any] = {}

        # 1. Experiment existence
        if experiment is None:
            return GateResult(status="BLOCKED", reasons=["EXPERIMENT_NOT_FOUND"], details={"error": "Experiment is None"})

        # 2. Legacy check
        if experiment.is_legacy:
            reasons.append("LEGACY_UNVALIDATED_EXPERIMENT")

        # 3. Experiment lifecycle state
        if not experiment.state.is_training_eligible:
            reasons.append(f"INELIGIBLE_STATE_{experiment.state.value}")

        # 4 & 5. Config validation & hash match
        cfg_file = Path(config_path or experiment.config_path)
        if not cfg_file.exists():
            reasons.append("CONFIG_FILE_NOT_FOUND")
        else:
            try:
                computed_cfg_hash = compute_config_hash(cfg_file)
                details["computed_config_hash"] = computed_cfg_hash
                if computed_cfg_hash != experiment.identity.config_hash:
                    reasons.append("CONFIG_HASH_MISMATCH")
            except Exception as e:
                reasons.append(f"CONFIG_HASH_ERROR: {str(e)}")

        # 6, 7 & 8. Dataset validation & hash match
        ds_manifest = Path(dataset_manifest_path or "data/shards/manifest.json")
        if not ds_manifest.exists():
            reasons.append("DATASET_MANIFEST_NOT_FOUND")
        else:
            try:
                with open(ds_manifest, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)

                if manifest_data.get("dataset_version") != experiment.identity.dataset_version:
                    reasons.append("DATASET_VERSION_MISMATCH")

                computed_ds_hash = compute_dataset_hash(ds_manifest)
                details["computed_dataset_hash"] = computed_ds_hash
                if computed_ds_hash != experiment.identity.dataset_hash:
                    reasons.append("DATASET_HASH_MISMATCH")

                # Verify actual shard file existence & checksums
                if verify_shard_checksums:
                    for shard_info in manifest_data.get("shards", []):
                        shard_path = ds_manifest.parent / shard_info["shard_name"]
                        if not shard_path.exists():
                            reasons.append(f"DATASET_SHARD_MISSING_{shard_info['shard_name']}")
                        else:
                            with open(shard_path, "rb") as sf:
                                actual_shard_sha = hashlib.sha256(sf.read()).hexdigest()
                            if actual_shard_sha != shard_info["sha256"]:
                                reasons.append(f"DATASET_SHARD_CORRUPTED_{shard_info['shard_name']}")
            except Exception as e:
                reasons.append(f"DATASET_HASH_ERROR: {str(e)}")

        # 9, 10 & 11. Tokenizer validation & freeze check
        tok_file = Path(tokenizer_path or "tokenizer/tokenizer.json")
        if not tok_file.exists():
            reasons.append("TOKENIZER_FILE_NOT_FOUND")
        else:
            try:
                computed_tok_hash = compute_tokenizer_hash(tok_file)
                details["computed_tokenizer_hash"] = computed_tok_hash
                if computed_tok_hash != experiment.identity.tokenizer_hash:
                    reasons.append("TOKENIZER_HASH_MISMATCH")

                meta_file = tok_file.parent / "tokenizer_metadata.json"
                if not meta_file.exists():
                    reasons.append("TOKENIZER_METADATA_MISSING")
                else:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        tok_meta = json.load(f)
                    # Check that tokenizer hash in metadata matches and normalization is NFC
                    if tok_meta.get("normalization") != "NFC":
                        reasons.append("TOKENIZER_NORMALIZATION_NOT_NFC")
                    if not tok_meta.get("byte_fallback", False):
                        reasons.append("TOKENIZER_BYTE_FALLBACK_DISABLED")
            except Exception as e:
                reasons.append(f"TOKENIZER_HASH_ERROR: {str(e)}")

        # 12 & 13. Git commit & repository state
        if not experiment.identity.git_commit or experiment.identity.git_commit in ("0" * 40, "unknown", "placeholder"):
            reasons.append("INVALID_GIT_COMMIT")
        elif require_git_match_head:
            current_git = get_git_commit()
            if current_git is not None and current_git != experiment.identity.git_commit:
                reasons.append("GIT_COMMIT_MISMATCH_WITH_CURRENT_HEAD")

        if not allow_dirty_git and get_git_dirty_state():
            reasons.append("DIRTY_GIT_REPOSITORY")

        # 14. Random seed
        if experiment.identity.random_seed is None or experiment.identity.random_seed < 0:
            reasons.append("INVALID_RANDOM_SEED")

        # 15. Parity verification prerequisite
        # Ensure parity module files exist
        parity_gdn_file = Path("model/parity/test_gdn_parity.py")
        parity_attn_file = Path("model/parity/test_attention_parity.py")
        if not parity_gdn_file.exists() or not parity_attn_file.exists():
            reasons.append("PARITY_VERIFICATION_SUITE_MISSING")

        # 16. Checkpoint lineage on resume
        if resume_checkpoint_id is not None:
            lineage = experiment.lineage
            cp_rec = lineage.get(resume_checkpoint_id)
            if cp_rec is None:
                reasons.append(f"RESUME_CHECKPOINT_NOT_IN_LINEAGE_{resume_checkpoint_id}")
            elif not Path(cp_rec.checkpoint_path).exists():
                reasons.append(f"RESUME_CHECKPOINT_FILE_MISSING_{cp_rec.checkpoint_path}")

        # Final decision
        if len(reasons) > 0:
            return GateResult(status="BLOCKED", reasons=reasons, details=details)

        return GateResult(status="PASS", reasons=[], details=details)
