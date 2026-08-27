import torch
import pytest
from configs.schema import GDNConfig
from model.reference.gdn import AstraGDN


def test_gdn_state_initialization():
    B, T, d_model = 2, 8, 128
    config = GDNConfig(num_heads=4, head_dim=32, conv_kernel=4)
    model = AstraGDN(d_model=d_model, config=config)

    x = torch.randn(B, T, d_model)
    out_none, state_none = model(x, state=None)

    explicit_zero_state = torch.zeros(
        B, config.num_heads, config.head_dim, config.head_dim, dtype=x.dtype, device=x.device
    )
    out_zero, state_zero = model(x, state=explicit_zero_state)

    torch.testing.assert_close(out_none, out_zero)
    torch.testing.assert_close(state_none, state_zero)


def test_gdn_state_detach_contract():
    B, T, d_model = 2, 8, 128
    config = GDNConfig(num_heads=4, head_dim=32, conv_kernel=4)
    model = AstraGDN(d_model=d_model, config=config)

    x = torch.randn(B, T, d_model, requires_grad=True)

    # With detach_state=True
    _, detached_state = model(x, detach_state=True)
    assert not detached_state.requires_grad
    assert detached_state.grad_fn is None

    # With detach_state=False
    _, attached_state = model(x, detach_state=False)
    assert attached_state.requires_grad
    assert attached_state.grad_fn is not None
