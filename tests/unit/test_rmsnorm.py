import torch
from model.reference.rmsnorm import RMSNorm


def test_rmsnorm_forward_and_variance():
    torch.manual_seed(42)
    B, T, D = 2, 8, 64
    norm = RMSNorm(dim=D, eps=1e-6)

    x = torch.randn(B, T, D) * 5.0 + 2.0
    y = norm(x)

    assert y.shape == (B, T, D)
    # Check that RMS(y) is approximately 1.0 when weight is 1.0
    rms = torch.sqrt(torch.mean(y**2, dim=-1))
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)
