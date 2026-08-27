from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest
from configs.schema import (
    AstraConfig,
    ModelConfig,
    GDNConfig,
    AttentionConfig,
    MTPConfig,
    TrainingConfig,
)


def test_config_yaml_loading():
    configs_dir = Path(__file__).parents[2] / "configs"
    
    for name in ["astra_100m.yaml", "astra_350m.yaml", "astra_1b.yaml"]:
        yaml_path = configs_dir / name
        assert yaml_path.exists(), f"{name} should exist"
        cfg = AstraConfig.from_yaml(yaml_path)
        assert isinstance(cfg, AstraConfig)
        assert cfg.model.hidden_size > 0
        assert cfg.model.num_layers > 0
        assert cfg.gdn.num_heads > 0
        assert cfg.attention.num_q_heads > 0
        assert cfg.mtp.horizon >= 1


def test_config_immutability():
    cfg = AstraConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.model.hidden_size = 4096


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=-10)

    with pytest.raises(ValueError):
        ModelConfig(layer_pattern=("invalid_layer",))

    with pytest.raises(ValueError):
        MTPConfig(horizon=0)

    with pytest.raises(ValueError):
        MTPConfig(loss_weight=-0.5)


def test_config_projection_dims():
    gdn_cfg = GDNConfig(num_heads=16, head_dim=96)
    assert gdn_cfg.projection_dim == 1536

    attn_cfg = AttentionConfig(num_q_heads=16, num_kv_heads=4, head_dim=128)
    assert attn_cfg.q_dim == 2048
    assert attn_cfg.kv_dim == 512
