"""
Astra Binary Shard Writer & Manifest Generator (Phase 2)
Writes immutable packed binary shards (.bin) and generates provenance manifest.json with SHA256.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Generator
import numpy as np


class BinaryShardWriter:
    """
    Writes packed sequences (input_ids, doc_ids, position_ids) into immutable binary shards.
    Each sample is structured as [3, seq_len] uint32.
    """
    def __init__(
        self,
        output_dir: str = "data/shards",
        dataset_version: str = "astra-data-v0.1",
        tokenizer_hash: str = "unknown",
        seq_len: int = 4096,
        max_samples_per_shard: int = 12500,  # ~50M tokens per shard at T=4096
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_version = dataset_version
        self.tokenizer_hash = tokenizer_hash
        self.seq_len = seq_len
        self.max_samples_per_shard = max_samples_per_shard

        self.current_shard_idx = 0
        self.current_samples: List[np.ndarray] = []
        self.shards_metadata: List[Dict[str, Any]] = []
        self.total_tokens_written = 0

    def add_sample(self, input_ids: np.ndarray, doc_ids: np.ndarray, position_ids: np.ndarray) -> None:
        # Stack into [3, seq_len]
        sample = np.stack([input_ids, doc_ids, position_ids], axis=0).astype(np.uint32)
        self.current_samples.append(sample)
        self.total_tokens_written += self.seq_len

        if len(self.current_samples) >= self.max_samples_per_shard:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self.current_samples:
            return

        shard_name = f"shard-{self.current_shard_idx:06d}.bin"
        shard_path = self.output_dir / shard_name

        data_array = np.stack(self.current_samples, axis=0)  # [num_samples, 3, seq_len]
        data_bytes = data_array.tobytes()

        with open(shard_path, "wb") as f:
            f.write(data_bytes)

        sha256_hash = hashlib.sha256(data_bytes).hexdigest()
        self.shards_metadata.append(
            {
                "shard_name": shard_name,
                "num_samples": len(self.current_samples),
                "num_tokens": len(self.current_samples) * self.seq_len,
                "sha256": sha256_hash,
            }
        )

        self.current_shard_idx += 1
        self.current_samples = []

    def close(self) -> Dict[str, Any]:
        self._flush_shard()

        manifest = {
            "dataset_version": self.dataset_version,
            "tokenizer_hash": self.tokenizer_hash,
            "sequence_length": self.seq_len,
            "num_shards": len(self.shards_metadata),
            "total_tokens": self.total_tokens_written,
            "dtype": "uint32",
            "sample_shape": [3, self.seq_len],
            "shards": self.shards_metadata,
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[OK] Manifest written to {manifest_path} ({len(self.shards_metadata)} shards, {self.total_tokens_written:,} tokens)")
        return manifest
