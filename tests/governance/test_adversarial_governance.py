"""
Astra Adversarial Governance & Security Test Suite (Phase 6)
Attempts to bypass scientific governance through direct tampering, out-of-band edits,
and unverified state mutations.
"""

import json
from pathlib import Path
import pytest

from experiments.identity import (
    ScientificIdentity,
    compute_config_hash,
    compute_dataset_hash,
    compute_tokenizer_hash,
)
from experiments.state_machine import ExperimentState, StateTransitionError, validate_transition
from experiments.registry import ExperimentRecord, ExperimentRegistry, ExperimentIdentityConflictError
from experiments.gate import TrainingGate


def test_adversarial_direct_registry_json_mutation(tmp_path):
    """
    Simulates an adversary directly modifying registry.json to alter a config hash.
    The TrainingGate must recalculate hashes from source files and block training.
    """
    reg_file = tmp_path / "registry.json"
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write("model:\n  hidden_size: 512\n")

    cfg_hash = compute_config_hash(cfg_file)

    ident = ScientificIdentity(
        git_commit="2" * 40,
        config_hash=cfg_hash,
        dataset_version="v1",
        dataset_hash="3" * 64,
        tokenizer_version="v1",
        tokenizer_hash="4" * 64,
        model_architecture="astra_100m",
        random_seed=42,
    )

    record = ExperimentRecord(
        experiment_id="exp_adv_01",
        description="Adversarial test",
        identity=ident,
        config_path=str(cfg_file),
        state=ExperimentState.VALIDATED,
    )

    registry = ExperimentRegistry(registry_file=str(reg_file))
    registry.register(record)

    # Adversary edits registry.json directly on disk to change config_hash to a fake value
    with open(reg_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["exp_adv_01"]["identity"]["config_hash"] = "9" * 64
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Reload registry
    tampered_registry = ExperimentRegistry(registry_file=str(reg_file))
    tampered_rec = tampered_registry.get("exp_adv_01")

    # Gate must detect mismatch with actual file on disk
    res = TrainingGate.validate(
        experiment=tampered_rec,
        config_path=str(cfg_file),
        dataset_manifest_path="non_existent",
        tokenizer_path="non_existent",
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "CONFIG_HASH_MISMATCH" in res.reasons


def test_adversarial_out_of_band_tokenizer_mutation(tmp_path):
    """
    An adversary edits the tokenizer vocabulary after experiment registration.
    """
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_file = tok_dir / "tokenizer.json"
    with open(tok_file, "w", encoding="utf-8") as f:
        json.dump({"vocab": {"<unk>": 0}}, f)

    tok_hash = compute_tokenizer_hash(tok_file)

    ident = ScientificIdentity(
        git_commit="2" * 40,
        config_hash="5" * 64,
        dataset_version="v1",
        dataset_hash="3" * 64,
        tokenizer_version="v1",
        tokenizer_hash=tok_hash,
        model_architecture="astra_100m",
        random_seed=42,
    )

    rec = ExperimentRecord(
        experiment_id="exp_adv_tok",
        description="Tokenizer mutation test",
        identity=ident,
        config_path="some_cfg.yaml",
        state=ExperimentState.VALIDATED,
    )

    # Modify tokenizer on disk
    with open(tok_file, "w", encoding="utf-8") as f:
        json.dump({"vocab": {"<unk>": 0, "backdoor_token": 1}}, f)

    res = TrainingGate.validate(
        experiment=rec,
        tokenizer_path=str(tok_file),
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert "TOKENIZER_HASH_MISMATCH" in res.reasons


def test_adversarial_illegal_state_skip():
    """
    An adversary attempts to bypass validation and force state from DRAFT directly to RUNNING.
    """
    with pytest.raises(StateTransitionError):
        validate_transition(ExperimentState.DRAFT, ExperimentState.RUNNING)
