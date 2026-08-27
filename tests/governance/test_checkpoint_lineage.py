import pytest
from experiments.lineage import CheckpointRecord, CheckpointLineage, LineageError


def test_checkpoint_lineage_monotonic():
    lineage = CheckpointLineage()

    cp1 = CheckpointRecord(
        checkpoint_id="cp_100",
        experiment_id="exp_01",
        step=100,
        tokens_seen=409600,
        checkpoint_path="checkpoints/cp_100.pt",
    )
    cp2 = CheckpointRecord(
        checkpoint_id="cp_200",
        experiment_id="exp_01",
        step=200,
        tokens_seen=819200,
        checkpoint_path="checkpoints/cp_200.pt",
        parent_checkpoint_id="cp_100",
    )

    lineage.add_checkpoint(cp1)
    lineage.add_checkpoint(cp2)

    assert len(lineage.list_checkpoints()) == 2
    assert lineage.get("cp_200").parent_checkpoint_id == "cp_100"


def test_checkpoint_lineage_invalid_step():
    lineage = CheckpointLineage()

    cp1 = CheckpointRecord(
        checkpoint_id="cp_200",
        experiment_id="exp_01",
        step=200,
        tokens_seen=819200,
        checkpoint_path="checkpoints/cp_200.pt",
    )
    cp_invalid = CheckpointRecord(
        checkpoint_id="cp_150",
        experiment_id="exp_01",
        step=150,  # Invalid: earlier than parent 200
        tokens_seen=600000,
        checkpoint_path="checkpoints/cp_150.pt",
        parent_checkpoint_id="cp_200",
    )

    lineage.add_checkpoint(cp1)
    with pytest.raises(LineageError):
        lineage.add_checkpoint(cp_invalid)


def test_checkpoint_lineage_missing_parent():
    lineage = CheckpointLineage()
    cp = CheckpointRecord(
        checkpoint_id="cp_300",
        experiment_id="exp_01",
        step=300,
        tokens_seen=1200000,
        checkpoint_path="checkpoints/cp_300.pt",
        parent_checkpoint_id="cp_non_existent",
    )
    with pytest.raises(LineageError):
        lineage.add_checkpoint(cp)
