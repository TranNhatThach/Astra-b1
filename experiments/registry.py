"""
Astra Hardened Experiment Governance Registry (Phase 6)
Enforces immutable scientific identities, strict lifecycle state transitions,
and auditable checkpoint provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from .identity import ScientificIdentity
from .state_machine import ExperimentState, validate_transition
from .lineage import CheckpointRecord, CheckpointLineage, LineageError


class ImmutableExperimentError(Exception):
    """Raised when an attempt is made to mutate an immutable scientific identity."""
    pass


class ExperimentIdentityConflictError(Exception):
    """Raised when an experiment_id is re-registered with conflicting scientific parameters."""
    pass


@dataclass
class ExperimentRecord:
    experiment_id: str
    description: str
    identity: ScientificIdentity
    config_path: str
    state: ExperimentState = ExperimentState.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    hardware_provenance: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_legacy: bool = False

    @property
    def is_locked(self) -> bool:
        return self.state.is_locked

    @property
    def lineage(self) -> CheckpointLineage:
        return CheckpointLineage.from_list(self.checkpoints)


class ExperimentRegistry:
    """
    Centralized governance registry enforcing strict immutability and state machine transitions.
    """
    def __init__(self, registry_file: str = "experiments/registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_file.exists():
            return

        with open(self.registry_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for exp_id, exp_data in data.items():
            # Support both new schema and legacy schema
            if "identity" in exp_data:
                identity = ScientificIdentity(**exp_data["identity"])
                state = ExperimentState(exp_data.get("state", "DRAFT"))
                record = ExperimentRecord(
                    experiment_id=exp_data["experiment_id"],
                    description=exp_data.get("description", ""),
                    identity=identity,
                    config_path=exp_data.get("config_path", ""),
                    state=state,
                    created_at=exp_data.get("created_at", ""),
                    updated_at=exp_data.get("updated_at", ""),
                    hardware_provenance=exp_data.get("hardware_provenance", {}),
                    checkpoints=exp_data.get("checkpoints", []),
                    metrics=exp_data.get("metrics", {}),
                    is_legacy=exp_data.get("is_legacy", False),
                )
            else:
                # Legacy format backward compatibility
                # Parse as legacy draft record without inventing scientific hashes
                record = ExperimentRecord(
                    experiment_id=exp_data.get("experiment_id", exp_id),
                    description=exp_data.get("description", "Legacy unmigrated experiment"),
                    identity=ScientificIdentity(
                        git_commit=exp_data.get("git_commit", "0" * 40),
                        config_hash=exp_data.get("config_hash", "0" * 64),
                        dataset_version=exp_data.get("dataset_version", "legacy"),
                        dataset_hash="0" * 64,
                        tokenizer_version="legacy_v0",
                        tokenizer_hash=exp_data.get("tokenizer_hash", "0" * 64),
                        model_architecture=exp_data.get("target_scale", "100m"),
                        random_seed=exp_data.get("random_seed", 42),
                    ),
                    config_path=exp_data.get("config_name", ""),
                    state=ExperimentState.DRAFT,
                    created_at=exp_data.get("created_at", ""),
                    updated_at=exp_data.get("updated_at", ""),
                    hardware_provenance=exp_data.get("hardware", {}),
                    checkpoints=[],
                    metrics=exp_data.get("metrics", {}),
                    is_legacy=True,
                )
            self.experiments[exp_id] = record

    def save(self) -> None:
        out_dict = {}
        for exp_id, rec in self.experiments.items():
            rec_dict = asdict(rec)
            rec_dict["state"] = rec.state.value
            out_dict[exp_id] = rec_dict

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)

    def register(self, record: ExperimentRecord) -> None:
        """
        Registers a new experiment record.
        Fails if an existing locked experiment is re-registered with conflicting parameters.
        """
        if record.experiment_id in self.experiments:
            existing = self.experiments[record.experiment_id]
            if existing.identity.compute_identity_hash() != record.identity.compute_identity_hash():
                raise ExperimentIdentityConflictError(
                    f"Conflict: Experiment '{record.experiment_id}' already exists with a different scientific identity. "
                    f"Existing identity hash: {existing.identity.compute_identity_hash()}, "
                    f"Attempted new identity hash: {record.identity.compute_identity_hash()}. "
                    f"A new experiment_id must be assigned for new scientific conditions."
                )
            if existing.is_locked and existing.state != record.state:
                raise ImmutableExperimentError(
                    f"Cannot overwrite locked experiment '{record.experiment_id}' in state {existing.state.value}."
                )

        self.experiments[record.experiment_id] = record
        self.save()

    def transition_state(self, experiment_id: str, new_state: ExperimentState) -> None:
        if experiment_id not in self.experiments:
            raise KeyError(f"Experiment '{experiment_id}' not found in registry.")

        record = self.experiments[experiment_id]
        validate_transition(record.state, new_state)

        record.state = new_state
        record.updated_at = datetime.now().isoformat()
        self.save()

    def add_checkpoint(self, experiment_id: str, checkpoint: CheckpointRecord) -> None:
        if experiment_id not in self.experiments:
            raise KeyError(f"Experiment '{experiment_id}' not found in registry.")

        record = self.experiments[experiment_id]
        if record.state not in (ExperimentState.RUNNING, ExperimentState.VALIDATED):
            raise ImmutableExperimentError(
                f"Cannot add checkpoint to experiment '{experiment_id}' in terminal state {record.state.value}."
            )

        lineage = record.lineage
        lineage.add_checkpoint(checkpoint)
        record.checkpoints = lineage.to_list()
        record.updated_at = datetime.now().isoformat()
        self.save()

    def append_metrics(self, experiment_id: str, step: int, metrics_dict: Dict[str, Any]) -> None:
        if experiment_id not in self.experiments:
            raise KeyError(f"Experiment '{experiment_id}' not found in registry.")

        record = self.experiments[experiment_id]
        if record.state == ExperimentState.COMPLETED:
            raise ImmutableExperimentError(
                f"Cannot append metrics to completed experiment '{experiment_id}'."
            )

        step_key = str(step)
        record.metrics[step_key] = {
            "timestamp": datetime.now().isoformat(),
            "values": metrics_dict,
        }
        record.updated_at = datetime.now().isoformat()
        self.save()

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        return self.experiments.get(experiment_id)
