import json
from pathlib import Path
from experiments.identity import (
    ScientificIdentity,
    compute_config_hash,
    compute_dataset_hash,
    compute_tokenizer_hash,
)
from experiments.provenance import get_git_commit
from experiments.state_machine import ExperimentState
from experiments.registry import ExperimentRecord, ExperimentRegistry
from experiments.gate import TrainingGate


def test_training_gate_pass_with_real_pilot_assets():
    cfg_file = "configs/astra_100m.yaml"
    ds_manifest = "data/shards/manifest.json"
    tok_file = "tokenizer/tokenizer.json"

    assert Path(cfg_file).exists()
    assert Path(ds_manifest).exists()
    assert Path(tok_file).exists()

    cfg_hash = compute_config_hash(cfg_file)
    ds_hash = compute_dataset_hash(ds_manifest)
    tok_hash = compute_tokenizer_hash(tok_file)
    git_commit = get_git_commit() or ("0" * 40)

    with open(ds_manifest, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    ds_version = m_data.get("dataset_version", "astra-pilot-100m-v0.1")

    identity = ScientificIdentity(
        git_commit=git_commit,
        config_hash=cfg_hash,
        dataset_version=ds_version,
        dataset_hash=ds_hash,
        tokenizer_version="astra-tok-v0.1",
        tokenizer_hash=tok_hash,
        model_architecture="astra_100m",
        random_seed=42,
    )

    record = ExperimentRecord(
        experiment_id="exp_astra_100m_pilot_real_pass",
        description="Astra-100M pilot training gate verification",
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
