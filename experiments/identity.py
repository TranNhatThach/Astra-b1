"""
Astra Canonical Scientific Identity & Deterministic Hashing (Phase 6)
Defines deterministic fingerprinting for configurations, datasets, tokenizers, and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Union
import yaml

from configs.schema import AstraConfig


def _canonicalize_dict(obj: Any) -> Any:
    """Recursively canonicalizes objects for deterministic serialization."""
    if isinstance(obj, dict):
        return {k: _canonicalize_dict(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [_canonicalize_dict(x) for x in obj]
    elif isinstance(obj, float):
        # Format floats consistently to avoid minor precision artifact differences
        return float(f"{obj:.8g}")
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif hasattr(obj, "__dict__"):
        return _canonicalize_dict(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj.__dict__)
    return str(obj)


def compute_config_hash(config_input: Union[AstraConfig, Dict[str, Any], str, Path]) -> str:
    """
    Computes a deterministic, formatting-independent SHA-256 hash for an Astra configuration.
    
    Invariants:
      - Dictionary key insertion order does not affect the hash.
      - Comments and YAML formatting do not affect the hash.
      - Equivalent configs produce identical hashes.
    """
    if isinstance(config_input, (str, Path)):
        p = Path(config_input)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            if p.suffix in (".yaml", ".yml"):
                raw_dict = yaml.safe_load(f)
            else:
                raw_dict = json.load(f)
        cfg_dict = _canonicalize_dict(raw_dict)
    elif isinstance(config_input, AstraConfig):
        cfg_dict = _canonicalize_dict(config_input.to_dict())
    elif isinstance(config_input, dict):
        cfg_dict = _canonicalize_dict(config_input)
    else:
        raise TypeError(f"Unsupported config input type: {type(config_input)}")

    canonical_json = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_tokenizer_hash(tokenizer_path: Union[str, Path]) -> str:
    """
    Computes deterministic SHA-256 hash for the tokenizer asset (tokenizer.json).
    """
    p = Path(tokenizer_path)
    if p.is_dir():
        p = p / "tokenizer.json"
    if not p.exists():
        raise FileNotFoundError(f"Tokenizer asset not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    canonical_json = json.dumps(_canonicalize_dict(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_dataset_hash(manifest_path: Union[str, Path]) -> str:
    """
    Computes canonical dataset SHA-256 hash from its manifest.json, accounting for
    dataset_version, sequence_length, total_tokens, and sorted shard checksums.
    """
    p = Path(manifest_path)
    if p.is_dir():
        p = p / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Extract scientific fields only (strip transient paths or timestamps)
    shards_canonical = []
    for s in manifest.get("shards", []):
        shards_canonical.append({
            "shard_name": s["shard_name"],
            "num_tokens": s["num_tokens"],
            "sha256": s["sha256"],
        })
    shards_canonical.sort(key=lambda x: x["shard_name"])

    dataset_signature = {
        "dataset_version": manifest.get("dataset_version", ""),
        "sequence_length": manifest.get("sequence_length", 0),
        "total_tokens": manifest.get("total_tokens", 0),
        "shards": shards_canonical,
    }

    canonical_json = json.dumps(_canonicalize_dict(dataset_signature), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScientificIdentity:
    """
    Immutable Scientific Identity of an Astra experiment.
    Defines the exact reproducible conditions of a training run.
    """
    git_commit: str
    config_hash: str
    dataset_version: str
    dataset_hash: str
    tokenizer_version: str
    tokenizer_hash: str
    model_architecture: str
    random_seed: int

    def __post_init__(self):
        if not self.git_commit or self.git_commit in ("unknown", "placeholder", "latest"):
            raise ValueError(f"Invalid git_commit: '{self.git_commit}'")
        if not self.config_hash or "placeholder" in self.config_hash:
            raise ValueError(f"Invalid config_hash: '{self.config_hash}'")
        if not self.dataset_version:
            raise ValueError("dataset_version cannot be empty")
        if not self.dataset_hash or "placeholder" in self.dataset_hash:
            raise ValueError(f"Invalid dataset_hash: '{self.dataset_hash}'")
        if not self.tokenizer_version:
            raise ValueError("tokenizer_version cannot be empty")
        if not self.tokenizer_hash or "placeholder" in self.tokenizer_hash:
            raise ValueError(f"Invalid tokenizer_hash: '{self.tokenizer_hash}'")
        if not self.model_architecture:
            raise ValueError("model_architecture cannot be empty")
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError(f"random_seed must be non-negative integer, got {self.random_seed}")

    def compute_identity_hash(self) -> str:
        """Computes a single canonical SHA-256 fingerprint for this scientific identity."""
        data = _canonicalize_dict(asdict(self))
        canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
