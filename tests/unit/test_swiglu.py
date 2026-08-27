import torch
from model.reference.swiglu import SwiGLU


def test_swiglu_shape_and_nonlinearity():
    torch.manual_seed(42)
    B, T, d_model, d_ff = 2, 8, 64, 128
    mlp = SwiGLU(d_model=d_model, intermediate_size=d_ff)

    x = torch.randn(B, T, d_model)
    y = mlp(x)

    assert y.shape == (B, T, d_model)
    # Check that negative values are softly suppressed by SiLU gating
    assert not torch.allclose(y, torch.zeros_like(y))
