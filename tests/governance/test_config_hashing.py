import json
from configs.schema import AstraConfig, ModelConfig, GDNConfig, AttentionConfig
from experiments.identity import compute_config_hash


def test_config_hashing_identity_and_determinism():
    cfg1 = AstraConfig()
    cfg2 = AstraConfig()

    h1 = compute_config_hash(cfg1)
    h2 = compute_config_hash(cfg2)
    assert h1 == h2, "Equivalent configs must produce identical hashes"


def test_config_hashing_reordered_keys():
    dict_a = {
        "model": {"name": "astra-100m", "hidden_size": 768, "num_layers": 12},
        "gdn": {"num_heads": 12, "head_dim": 64},
    }
    dict_b = {
        "gdn": {"head_dim": 64, "num_heads": 12},
        "model": {"num_layers": 12, "hidden_size": 768, "name": "astra-100m"},
    }

    ha = compute_config_hash(dict_a)
    hb = compute_config_hash(dict_b)
    assert ha == hb, "Key insertion order must not affect canonical hash"


def test_config_hashing_distinct_configs():
    cfg1 = AstraConfig(model=ModelConfig(hidden_size=768))
    cfg2 = AstraConfig(model=ModelConfig(hidden_size=1024))

    h1 = compute_config_hash(cfg1)
    h2 = compute_config_hash(cfg2)
    assert h1 != h2, "Scientifically distinct configs must produce different hashes"
