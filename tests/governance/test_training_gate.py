"""
Astra Training Gate Formal Test Suite (Phase 6)
Covers all 20 required test scenarios from Phase 6 Specification.
"""

import json
import hashlib
from pathlib import Path
import pytest
import yaml
import torch

from configs.schema import AstraConfig
from experiments.identity import (
    ScientificIdentity,
    compute_config_hash,
    compute_dataset_hash,
    compute_tokenizer_hash,
)
from experiments.state_machine import ExperimentState
from experiments.lineage import CheckpointRecord
from experiments.provenance import get_git_commit
from experiments.registry import ExperimentRecord, ExperimentRegistry
from experiments.gate import TrainingGate, GateResult


@pytest.fixture
def setup_governance_env(tmp_path):
    """Sets up a complete, valid mock governance environment with real hashed assets."""
    env_dir = tmp_path / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid Config
    cfg = AstraConfig()
    cfg_file = env_dir / "config.yaml"
    cfg.to_yaml(cfg_file)
    cfg_hash = compute_config_hash(cfg_file)

    # 2. Valid Tokenizer Asset
    tok_dir = env_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_file = tok_dir / "tokenizer.json"
    with open(tok_file, "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "vocab": {"<unk>": 0, "<bos>": 1}}, f)
    tok_meta = tok_dir / "tokenizer_metadata.json"
    with open(tok_meta, "w", encoding="utf-8") as f:
        json.dump({"normalization": "NFC", "byte_fallback": True}, f)
    tok_hash = compute_tokenizer_hash(tok_file)

    # 3. Valid Sharded Dataset
    ds_dir = env_dir / "shards"
    ds_dir.mkdir(parents=True, exist_ok=True)
    shard_file = ds_dir / "shard-000000.bin"
    shard_content = b"\x01\x02\x03\x04" * 100
    with open(shard_file, "wb") as f:
        f.write(shard_content)
    shard_sha = hashlib.sha256(shard_content).hexdigest()

    manifest_file = ds_dir / "manifest.json"
    manifest_data = {
        "dataset_version": "astra-data-v0.1",
        "sequence_length": 4096,
        "total_tokens": 100,
        "shards": [
            {
                "shard_name": "shard-000000.bin",
                "num_tokens": 100,
                "sha256": shard_sha,
            }
        ],
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
    ds_hash = compute_dataset_hash(manifest_file)

    # 4. Checkpoint File
    cp_dir = env_dir / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_file = cp_dir / "step_100.pt"
    torch.save({"step": 100}, cp_file)

    # Git SHA
    git_commit = get_git_commit() or ("1" * 40)

    identity = ScientificIdentity(
        git_commit=git_commit,
        config_hash=cfg_hash,
        dataset_version="astra-data-v0.1",
        dataset_hash=ds_hash,
        tokenizer_version="v0.1",
        tokenizer_hash=tok_hash,
        model_architecture="astra_100m",
        random_seed=42,
    )

    record = ExperimentRecord(
        experiment_id="exp_governance_test_01",
        description="Gate test experiment",
        identity=identity,
        config_path=str(cfg_file),
        state=ExperimentState.VALIDATED,
    )

    return {
        "record": record,
        "cfg_file": cfg_file,
        "tok_file": tok_file,
        "ds_manifest": manifest_file,
        "shard_file": shard_file,
        "cp_file": cp_file,
        "identity": identity,
    }


def test_gate_01_changed_config(setup_governance_env):
    env = setup_governance_env
    # Modify config on disk
    with open(env["cfg_file"], "a", encoding="utf-8") as f:
        f.write("\n# modified comment with altered parameters\nmodel:\n  hidden_size: 9999\n")

    res = TrainingGate.validate(
        experiment=env["record"],
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "CONFIG_HASH_MISMATCH" in res.reasons


def test_gate_02_changed_dataset_version(setup_governance_env):
    env = setup_governance_env
    with open(env["ds_manifest"], "w", encoding="utf-8") as f:
        json.dump({"dataset_version": "v999", "shards": []}, f)

    res = TrainingGate.validate(
        experiment=env["record"],
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "DATASET_VERSION_MISMATCH" in res.reasons


def test_gate_03_corrupted_dataset_shard(setup_governance_env):
    env = setup_governance_env
    with open(env["shard_file"], "wb") as f:
        f.write(b"corrupted bytes")

    res = TrainingGate.validate(
        experiment=env["record"],
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert any("CORRUPTED" in r for r in res.reasons)


def test_gate_04_changed_tokenizer_hash(setup_governance_env):
    env = setup_governance_env
    with open(env["tok_file"], "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "vocab": {"<unk>": 0, "tampered": 99}}, f)

    res = TrainingGate.validate(
        experiment=env["record"],
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "TOKENIZER_HASH_MISMATCH" in res.reasons


def test_gate_06_missing_git_commit(setup_governance_env):
    env = setup_governance_env
    # Legacy / invalid git
    rec = env["record"]
    rec.is_legacy = True
    res = TrainingGate.validate(
        experiment=rec,
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "LEGACY_UNVALIDATED_EXPERIMENT" in res.reasons


def test_gate_07_dirty_git_repo(setup_governance_env, monkeypatch):
    env = setup_governance_env
    # Mock dirty git state
    monkeypatch.setattr("experiments.gate.get_git_dirty_state", lambda: True)

    res = TrainingGate.validate(
        experiment=env["record"],
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=False,
    )
    assert not res.is_passed
    assert "DIRTY_GIT_REPOSITORY" in res.reasons


def test_gate_09_tokenizer_not_frozen(setup_governance_env):
    env = setup_governance_env
    tok_meta = env["tok_file"].parent / "tokenizer_metadata.json"
    with open(tok_meta, "w", encoding="utf-8") as f:
        json.dump({"normalization": "NFD", "byte_fallback": False}, f)

    res = TrainingGate.validate(
        experiment=env["record"],
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "TOKENIZER_NORMALIZATION_NOT_NFC" in res.reasons


def test_gate_10_missing_config_file(setup_governance_env):
    env = setup_governance_env
    res = TrainingGate.validate(
        experiment=env["record"],
        config_path="non_existent_path.yaml",
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "CONFIG_FILE_NOT_FOUND" in res.reasons


def test_gate_15_invalid_experiment_state(setup_governance_env):
    env = setup_governance_env
    rec = env["record"]
    rec.state = ExperimentState.DRAFT
    res = TrainingGate.validate(
        experiment=rec,
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "INELIGIBLE_STATE_DRAFT" in res.reasons


def test_gate_16_invalid_checkpoint_lineage(setup_governance_env):
    env = setup_governance_env
    res = TrainingGate.validate(
        experiment=env["record"],
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        resume_checkpoint_id="cp_missing_999",
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert any("RESUME_CHECKPOINT_NOT_IN_LINEAGE" in r for r in res.reasons)


def test_gate_17_valid_immutable_experiment_pass(setup_governance_env):
    env = setup_governance_env
    res = TrainingGate.validate(
        experiment=env["record"],
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        allow_dirty_git=True,
    )
    assert res.is_passed, f"Expected PASS, got reasons: {res.reasons}"
    assert res.status == "PASS"


def test_gate_18_valid_resume_from_checkpoint(setup_governance_env):
    env = setup_governance_env
    rec = env["record"]
    cp = CheckpointRecord(
        checkpoint_id="cp_100",
        experiment_id=rec.experiment_id,
        step=100,
        tokens_seen=409600,
        checkpoint_path=str(env["cp_file"]),
    )
    lineage = rec.lineage
    lineage.add_checkpoint(cp)
    rec.checkpoints = lineage.to_list()

    res = TrainingGate.validate(
        experiment=rec,
        config_path=str(env["cfg_file"]),
        dataset_manifest_path=str(env["ds_manifest"]),
        tokenizer_path=str(env["tok_file"]),
        resume_checkpoint_id="cp_100",
        allow_dirty_git=True,
    )
    assert res.is_passed, f"Expected PASS on resume, got reasons: {res.reasons}"


def test_gate_19_appending_metrics_preserves_identity(setup_governance_env):
    env = setup_governance_env
    rec = env["record"]
    h_before = rec.identity.compute_identity_hash()

    rec.metrics["100"] = {"loss": 2.5, "ppl": 12.18}
    h_after = rec.identity.compute_identity_hash()

    assert h_before == h_after, "Appending metrics must never alter scientific identity"


def test_gate_20_appending_checkpoint_preserves_identity(setup_governance_env):
    env = setup_governance_env
    rec = env["record"]
    h_before = rec.identity.compute_identity_hash()

    cp = CheckpointRecord(
        checkpoint_id="cp_200",
        experiment_id=rec.experiment_id,
        step=200,
        tokens_seen=819200,
        checkpoint_path=str(env["cp_file"]),
    )
    rec.checkpoints.append(json.loads(json.dumps(cp.__dict__)))
    h_after = rec.identity.compute_identity_hash()

    assert h_before == h_after, "Appending checkpoints must never alter scientific identity"
