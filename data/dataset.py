"""
Astra Offline Memory-Mapped Dataset (Phase 2)
Provides zero-copy, low-RAM PyTorch DataLoader dataset for pre-sharded binary training data.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from torch.utils.data import Dataset


class AstraOfflineDataset(Dataset):
    """
    Reads packed binary shards using numpy memory mapping.
    Never loads entire datasets into RAM; streams directly from local storage/NVMe.
    """
    def __init__(self, shards_dir: str = "data/shards"):
        self.shards_dir = Path(shards_dir)
        manifest_path = self.shards_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.seq_len = self.manifest["sequence_length"]
        self.total_tokens = self.manifest["total_tokens"]
        self.shards_info = self.manifest["shards"]

        # Build index mapping: global_sample_idx -> (shard_idx, local_sample_idx)
        self.sample_map: List[tuple[int, int]] = []
        self.mmaps: List[np.memmap] = []

        for shard_idx, s_info in enumerate(self.shards_info):
            shard_path = self.shards_dir / s_info["shard_name"]
            num_samples = s_info["num_samples"]

            mmap = np.memmap(
                shard_path,
                dtype=np.uint32,
                mode="r",
                shape=(num_samples, 3, self.seq_len),
            )
            self.mmaps.append(mmap)

            for local_idx in range(num_samples):
                self.sample_map.append((shard_idx, local_idx))

    def __len__(self) -> int:
        return len(self.sample_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self.sample_map[idx]
        sample_data = self.mmaps[shard_idx][local_idx]  # [3, seq_len]

        # Extract rows
        input_ids = torch.from_numpy(sample_data[0].astype(np.int64))
        doc_ids = torch.from_numpy(sample_data[1].astype(np.int64))
        position_ids = torch.from_numpy(sample_data[2].astype(np.int64))

        return {
            "input_ids": input_ids,
            "doc_ids": doc_ids,
            "position_ids": position_ids,
        }
