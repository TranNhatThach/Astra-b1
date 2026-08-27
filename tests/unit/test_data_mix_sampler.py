import pytest
from data.mix.policies import MixturePolicy
from data.mix.sampler import DeterministicMixtureSampler


def test_mixture_policy_validation():
    # Valid default
    policy = MixturePolicy()
    assert sum(policy.to_dict().values()) == 1.0

    # Invalid weights sum
    with pytest.raises(ValueError):
        MixturePolicy(web=0.9, educational=0.9)


def test_deterministic_mixture_sampler():
    corpora = {
        "web": ["web doc 1", "web doc 2", "web doc 3"],
        "educational": ["edu doc 1", "edu doc 2"],
        "code": ["code doc 1", "code doc 2"],
        "math": ["math doc 1"],
        "vietnamese": ["vi doc 1", "vi doc 2"],
        "dialogue": ["dial doc 1"],
    }

    sampler1 = DeterministicMixtureSampler(domain_corpora=corpora, seed=42)
    stream1 = list(sampler1.sample_stream(total_samples=20))

    sampler2 = DeterministicMixtureSampler(domain_corpora=corpora, seed=42)
    stream2 = list(sampler2.sample_stream(total_samples=20))

    assert len(stream1) == 20
    assert stream1 == stream2, "Same random seed must produce identical deterministic sampling stream"
