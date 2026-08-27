from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Any, Dict
import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "astra-1b"
    vocab_size: int = 65536
    hidden_size: int = 2048
    intermediate_size: int = 3456
    num_layers: int = 24
    layer_pattern: Tuple[str, ...] = ("gdn", "gdn", "gdn", "attention")
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 4096

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        for p in self.layer_pattern:
            if p not in ("gdn", "attention"):
                raise ValueError(f"Invalid layer type in pattern: {p}")


@dataclass(frozen=True)
class GDNConfig:
    num_heads: int = 16
    head_dim: int = 96
    conv_kernel: int = 4
    key_norm_eps: float = 1e-6
    retention_bias_init: float = 3.0
    update_bias_init: float = -0.5
    output_bias_init: float = 2.0
    gated: bool = True

    @property
    def projection_dim(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True)
class AttentionConfig:
    num_q_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 1000000.0
    gated: bool = True
    gate_bias_init: float = 2.0

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim


@dataclass(frozen=True)
class MTPConfig:
    enabled: bool = True
    horizon: int = 2
    loss_weight: float = 0.2

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError(f"MTP horizon must be >= 1, got {self.horizon}")
        if self.loss_weight < 0.0:
            raise ValueError(f"MTP loss_weight must be >= 0.0, got {self.loss_weight}")


@dataclass(frozen=True)
class TrainingConfig:
    dtype: str = "bfloat16"
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    context_stages: Tuple[int, ...] = (4096, 8192, 16384, 32768)


@dataclass(frozen=True)
class AstraConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    gdn: GDNConfig = field(default_factory=GDNConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AstraConfig:
        model_data = dict(data.get("model", {}))
        if "layer_pattern" in model_data and isinstance(model_data["layer_pattern"], list):
            model_data["layer_pattern"] = tuple(model_data["layer_pattern"])
        
        training_data = dict(data.get("training", {}))
        if "context_stages" in training_data and isinstance(training_data["context_stages"], list):
            training_data["context_stages"] = tuple(training_data["context_stages"])

        return cls(
            model=ModelConfig(**model_data),
            gdn=GDNConfig(**data.get("gdn", {})),
            attention=AttentionConfig(**data.get("attention", {})),
            mtp=MTPConfig(**data.get("mtp", {})),
            training=TrainingConfig(**training_data),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AstraConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        def _to_list(obj):
            if isinstance(obj, dict):
                return {k: _to_list(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_list(x) for x in obj]
            return obj
        clean_dict = _to_list(asdict(self))
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(clean_dict, f, default_flow_style=False, sort_keys=False)

