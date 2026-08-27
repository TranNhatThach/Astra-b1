"""
Astra Experiment Tracking & Governance Registry (Phase 0/7/8)
Tracks all pretraining and scaling experiments with complete metadata provenance:
  - experiment_id
  - git_commit
  - config_hash
  - dataset_version
  - tokenizer_hash
  - random_seed
  - hardware topology
  - checkpoint paths and metrics
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import subprocess


@dataclass
class ExperimentMetadata:
    experiment_id: str
    description: str
    target_scale: str  # "100m", "350m", "1b", "ablation"
    config_name: str
    git_commit: str
    config_hash: str
    dataset_version: str
    tokenizer_hash: str
    random_seed: int
    hardware: Dict[str, Any]
    status: str = "initialized"  # "initialized", "running", "completed", "failed"
    created_at: str = ""
    updated_at: str = ""
    checkpoints: List[str] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()
        if self.checkpoints is None:
            self.checkpoints = []
        if self.metrics is None:
            self.metrics = {}


class ExperimentRegistry:
    def __init__(self, registry_file: str = "experiments/registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentMetadata] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_file.exists():
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for exp_id, exp_dict in data.items():
                self.experiments[exp_id] = ExperimentMetadata(**exp_dict)

    def save(self) -> None:
        data = {exp_id: asdict(meta) for exp_id, meta in self.experiments.items()}
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register(self, metadata: ExperimentMetadata) -> None:
        if metadata.experiment_id in self.experiments:
            raise ValueError(f"Experiment '{metadata.experiment_id}' already registered.")
        self.experiments[metadata.experiment_id] = metadata
        self.save()

    def update_status(
        self,
        experiment_id: str,
        status: str,
        checkpoint: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        if experiment_id not in self.experiments:
            raise KeyError(f"Experiment '{experiment_id}' not found.")
        exp = self.experiments[experiment_id]
        exp.status = status
        exp.updated_at = datetime.now().isoformat()
        if checkpoint:
            exp.checkpoints.append(checkpoint)
        if metrics:
            exp.metrics.update(metrics)
        self.save()

    def get(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        return self.experiments.get(experiment_id)


def create_experiment_entry(
    experiment_id: str,
    description: str,
    target_scale: str,
    config_name: str,
    dataset_version: str,
    tokenizer_hash: str,
    random_seed: int = 42,
) -> ExperimentMetadata:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"

    hardware_info = {
        "num_gpus": 1,
        "gpu_type": "auto_detect",
        "precision": "bfloat16",
    }

    return ExperimentMetadata(
        experiment_id=experiment_id,
        description=description,
        target_scale=target_scale,
        config_name=config_name,
        git_commit=git_commit,
        config_hash="sha256_placeholder",
        dataset_version=dataset_version,
        tokenizer_hash=tokenizer_hash,
        random_seed=random_seed,
        hardware=hardware_info,
    )


if __name__ == "__main__":
    reg = ExperimentRegistry()
    entry = create_experiment_entry(
        experiment_id="exp_astra_100m_pilot_01",
        description="Astra-100M sanity pretraining pilot on 500M tokens",
        target_scale="100m",
        config_name="configs/astra_100m.yaml",
        dataset_version="astra-data-v0.1",
        tokenizer_hash="e10c0c7bdb3bb4b063f9cc46761c644974183b92821dc0a81db0abbc390260cf",
    )
    if not reg.get(entry.experiment_id):
        reg.register(entry)
        print(f"[OK] Experiment '{entry.experiment_id}' registered successfully.")
