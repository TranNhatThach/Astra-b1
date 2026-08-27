"""
Astra Scientific Dataset Gate (Phase 7C)
Ensures dataset readiness, license verification, shard integrity, and frozen tokenizer conformance.
"""

from dataclasses import dataclass, field
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

from data.sources.registry import SourceRegistry
from experiments.identity import compute_tokenizer_hash, compute_dataset_hash


@dataclass(frozen=True)
class DatasetGateResult:
    status: str  # "READY_FOR_ASTRA_1B" | "DATASET_NOT_READY"
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.status == "READY_FOR_ASTRA_1B"


class DatasetGate:
    EXPECTED_TOKENIZER_HASH = "514a02f5e8a4eb88b3113c22e022fb1969acddbbf9487f261f615b6e384dc5e8"

    @classmethod
    def validate(
        cls,
        manifest_path: str = "data/shards/manifest.json",
        tokenizer_path: str = "tokenizer/tokenizer.json",
        registry_file: str = "data/sources/registry.json",
    ) -> DatasetGateResult:
        reasons = []
        details = {}

        manifest_file = Path(manifest_path)
        tok_file = Path(tokenizer_path)
        reg_file = Path(registry_file)

        # 1. Manifest existence
        if not manifest_file.exists():
            return DatasetGateResult(
                status="DATASET_NOT_READY",
                reasons=["MISSING_DATASET_MANIFEST"],
                details={"manifest_path": str(manifest_file)},
            )

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # 2. Tokenizer verification
        if not tok_file.exists():
            reasons.append("MISSING_TOKENIZER_FILE")
        else:
            computed_tok_hash = compute_tokenizer_hash(str(tok_file))
            tok_meta_file = tok_file.parent / "tokenizer_metadata.json"
            if tok_meta_file.exists():
                with open(tok_meta_file, "r", encoding="utf-8") as f:
                    tok_meta = json.load(f)
                if tok_meta.get("status") != "FROZEN":
                    reasons.append("TOKENIZER_NOT_FROZEN")
            else:
                reasons.append("MISSING_TOKENIZER_METADATA")

            if computed_tok_hash != cls.EXPECTED_TOKENIZER_HASH:
                reasons.append("TOKENIZER_HASH_MISMATCH")
            details["tokenizer_hash"] = computed_tok_hash

        # 3. Source Registry & Licensing
        if reg_file.exists():
            registry = SourceRegistry(str(reg_file))
            sources = registry.list_sources()
            if len(sources) < 6:
                reasons.append("INSUFFICIENT_APPROVED_SOURCES")
            for s in sources:
                if not s.license or s.license == "UNKNOWN":
                    reasons.append(f"UNRESOLVED_LICENSE_{s.source_id}")
        else:
            reasons.append("MISSING_SOURCE_REGISTRY")

        # 4. Shards Checksums
        shards_dir = manifest_file.parent
        shard_entries = manifest.get("shards", [])
        if not shard_entries:
            reasons.append("ZERO_SHARDS_IN_MANIFEST")

        for s_info in shard_entries:
            s_path = shards_dir / s_info["shard_name"]
            if not s_path.exists():
                reasons.append(f"MISSING_SHARD_{s_info['shard_name']}")
                continue
            with open(s_path, "rb") as sf:
                actual_sha = hashlib.sha256(sf.read()).hexdigest()
            if actual_sha != s_info["sha256"]:
                reasons.append(f"CORRUPTED_SHARD_{s_info['shard_name']}")

        details["total_tokens"] = manifest.get("total_tokens", 0)
        details["num_shards"] = manifest.get("num_shards", 0)

        if reasons:
            return DatasetGateResult(
                status="DATASET_NOT_READY",
                reasons=reasons,
                details=details,
            )

        return DatasetGateResult(
            status="READY_FOR_ASTRA_1B",
            reasons=[],
            details=details,
        )
