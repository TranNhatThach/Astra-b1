"""
Astra Phase 7B (100M Real-Data Pilot & Training Smoke Test) Test Suite
"""

import json
import hashlib
from pathlib import Path
import torch

from data.dataset import AstraOfflineDataset
from experiments.identity import (
    ScientificIdentity,
    compute_config_hash,
    compute_dataset_hash,
    compute_tokenizer_hash,
)
from experiments.provenance import get_git_commit
from experiments.state_machine import ExperimentState
from experiments.registry import ExperimentRecord
from experiments.gate import TrainingGate


def test_100m_manifest_and_shards_integrity():
    manifest_file = Path("data/shards/manifest.json")
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dataset_version"] == "astra-pilot-100m-v0.1"
    assert manifest["total_tokens"] >= 99_000_000
    assert manifest["num_shards"] >= 2

    # Verify each shard exists and matches SHA-256
    for s_info in manifest["shards"]:
        s_path = manifest_file.parent / s_info["shard_name"]
        assert s_path.exists()
        with open(s_path, "rb") as sf:
            assert hashlib.sha256(sf.read()).hexdigest() == s_info["sha256"]


def test_100m_dataloader_and_memmap():
    ds = AstraOfflineDataset(shards_dir="data/shards")
    assert len(ds) >= 20000

    sample = ds[0]
    assert "input_ids" in sample
    assert "doc_ids" in sample
    assert "position_ids" in sample
    assert len(sample["input_ids"]) == 4096


def test_100m_checkpoint_integrity():
    ckpt_file = Path("checkpoints/smoke_step_5.pt")
    assert ckpt_file.exists()

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 5
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert isinstance(ckpt["loss"], float)


def test_100m_pilot_training_gate_pass():
    cfg_file = "configs/astra_100m.yaml"
    ds_manifest = "data/shards/manifest.json"
    tok_file = "tokenizer/tokenizer.json"

    cfg_hash = compute_config_hash(cfg_file)
    ds_hash = compute_dataset_hash(ds_manifest)
    tok_hash = compute_tokenizer_hash(tok_file)
    git_commit = get_git_commit() or ("0" * 40)

    identity = ScientificIdentity(
        git_commit=git_commit,
        config_hash=cfg_hash,
        dataset_version="astra-pilot-100m-v0.1",
        dataset_hash=ds_hash,
        tokenizer_version="astra-tok-v0.1",
        tokenizer_hash=tok_hash,
        model_architecture="astra_100m",
        random_seed=42,
    )

    record = ExperimentRecord(
        experiment_id="exp_astra_100m_pilot_real_pass",
        description="Astra-100M pilot real data training gate validation",
        identity=identity,
        config_path=cfg_file,
        state=ExperimentState.VALIDATED,
    )

    res = TrainingGate.validate(
        experiment=record,
        config_path=cfg_file,
        dataset_manifest_path=ds_manifest,
        tokenizer_path=tok_file,
        allow_dirty_git=True,
    )

    assert res.is_passed, f"Gate failed with reasons: {res.reasons}"
    assert res.status == "PASS"
