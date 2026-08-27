import pytest
from experiments.identity import ScientificIdentity
from experiments.state_machine import ExperimentState, StateTransitionError, validate_transition
from experiments.registry import (
    ExperimentRecord,
    ExperimentRegistry,
    ImmutableExperimentError,
    ExperimentIdentityConflictError,
)


def make_valid_identity(seed: int = 42) -> ScientificIdentity:
    return ScientificIdentity(
        git_commit="a" * 40,
        config_hash="b" * 64,
        dataset_version="astra-data-v0.1",
        dataset_hash="c" * 64,
        tokenizer_version="v0.1",
        tokenizer_hash="d" * 64,
        model_architecture="astra_100m",
        random_seed=seed,
    )


def test_scientific_identity_validation():
    # Valid
    ident = make_valid_identity()
    assert len(ident.compute_identity_hash()) == 64

    # Invalid git commit
    with pytest.raises(ValueError):
        ScientificIdentity(
            git_commit="unknown",
            config_hash="b" * 64,
            dataset_version="v1",
            dataset_hash="c" * 64,
            tokenizer_version="v1",
            tokenizer_hash="d" * 64,
            model_architecture="100m",
            random_seed=42,
        )

    # Invalid placeholder hash
    with pytest.raises(ValueError):
        ScientificIdentity(
            git_commit="a" * 40,
            config_hash="sha256_placeholder",
            dataset_version="v1",
            dataset_hash="c" * 64,
            tokenizer_version="v1",
            tokenizer_hash="d" * 64,
            model_architecture="100m",
            random_seed=42,
        )


def test_state_machine_transitions():
    # Allowed: DRAFT -> VALIDATED -> RUNNING -> COMPLETED
    validate_transition(ExperimentState.DRAFT, ExperimentState.VALIDATED)
    validate_transition(ExperimentState.VALIDATED, ExperimentState.RUNNING)
    validate_transition(ExperimentState.RUNNING, ExperimentState.COMPLETED)

    # Illegal: DRAFT -> RUNNING
    with pytest.raises(StateTransitionError):
        validate_transition(ExperimentState.DRAFT, ExperimentState.RUNNING)

    # Illegal: COMPLETED -> RUNNING
    with pytest.raises(StateTransitionError):
        validate_transition(ExperimentState.COMPLETED, ExperimentState.RUNNING)


def test_registry_immutability_lock(tmp_path):
    reg_file = tmp_path / "reg_test.json"
    registry = ExperimentRegistry(registry_file=str(reg_file))

    ident1 = make_valid_identity(seed=42)
    record = ExperimentRecord(
        experiment_id="exp_test_01",
        description="Testing immutability lock",
        identity=ident1,
        config_path="configs/astra_100m.yaml",
        state=ExperimentState.VALIDATED,
    )
    registry.register(record)

    # Attempt to re-register with a different seed -> Conflict Error
    ident2 = make_valid_identity(seed=999)
    record2 = ExperimentRecord(
        experiment_id="exp_test_01",
        description="Tampered seed",
        identity=ident2,
        config_path="configs/astra_100m.yaml",
        state=ExperimentState.VALIDATED,
    )
    with pytest.raises(ExperimentIdentityConflictError):
        registry.register(record2)
