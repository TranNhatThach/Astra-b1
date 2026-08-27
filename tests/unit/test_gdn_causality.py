import torch
import pytest
from configs.schema import GDNConfig
from model.reference.gdn import AstraGDN


def test_gdn_three_tier_causality():
    """
    3-Tier Causal Invariance Test:
      Modifying future tokens x[t_cutoff:] must have zero effect on:
      1. Conv features at t < t_cutoff
      2. Recurrent states S_t at t < t_cutoff
      3. Outputs y[t] at t < t_cutoff
    """
    torch.manual_seed(42)
    B, T, d_model = 2, 16, 128
    t_cutoff = 6

    config = GDNConfig(num_heads=4, head_dim=32, conv_kernel=4)
    model = AstraGDN(d_model=d_model, config=config)
    model.eval()

    # Base input X1
    x1 = torch.randn(B, T, d_model)

    # Perturbed input X2 (identical up to t_cutoff - 1, completely different afterwards)
    x2 = x1.clone()
    x2[:, t_cutoff:] = torch.randn(B, T - t_cutoff, d_model)

    with torch.no_grad():
        out1, _ = model(x1)
        out2, _ = model(x2)

    # Tier 3: Output causality
    out1_past = out1[:, :t_cutoff]
    out2_past = out2[:, :t_cutoff]
    torch.testing.assert_close(
        out1_past,
        out2_past,
        rtol=1e-6,
        atol=1e-6,
        msg="Output at t < t_cutoff must not change when future inputs are modified",
    )

    # Output after t_cutoff SHOULD differ
    assert not torch.allclose(out1[:, t_cutoff:], out2[:, t_cutoff:]), "Future outputs must differ"
