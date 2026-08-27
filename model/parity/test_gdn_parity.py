import pytest
import torch
from configs.schema import GDNConfig
from model.reference.gdn import AstraGDN
from model.optimized.gdn_fla import AstraGDNFLA


def rel_err(ref: torch.Tensor, opt: torch.Tensor, eps: float = 1e-8) -> float:
    diff = torch.linalg.norm(ref.float() - opt.float()).item()
    norm = torch.linalg.norm(ref.float()).item()
    return diff / max(norm, eps)


@pytest.mark.parametrize("seq_len", [1, 7, 31, 64, 127, 257, 1024, 4096, 4097, 4225])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gdn_forward_and_state_parity(seq_len, dtype):
    torch.manual_seed(42)
    B, d_model = 2, 64
    config = GDNConfig(num_heads=2, head_dim=32, conv_kernel=4)

    ref_model = AstraGDN(d_model=d_model, config=config).to(dtype=dtype)
    opt_model = AstraGDNFLA(d_model=d_model, config=config, chunk_size=64).to(dtype=dtype)

    # Copy weights identically to guarantee zero parameter divergence
    opt_model.load_state_dict(ref_model.state_dict())

    ref_model.eval()
    opt_model.eval()

    x = torch.randn(B, seq_len, d_model, dtype=dtype)

    with torch.no_grad():
        y_ref, s_ref = ref_model(x)
        y_opt, s_opt = opt_model(x)

    tol = 1e-4 if dtype == torch.float32 else 1e-2

    err_y = rel_err(y_ref, y_opt)
    err_s = rel_err(s_ref, s_opt)

    assert err_y < tol, f"T={seq_len} {dtype} forward rel_err {err_y} >= {tol}"
    assert err_s < tol, f"T={seq_len} {dtype} state rel_err {err_s} >= {tol}"


def test_gdn_gradient_parity():
    """
    Verifies 4-tier Gradient Parity between Reference and Optimized backends:
      1. Input gradient nabla_x L
      2. Parameter gradients nabla_W L across all projections
    """
    torch.manual_seed(42)
    B, T, d_model = 2, 128, 64
    config = GDNConfig(num_heads=2, head_dim=32, conv_kernel=4)

    ref_model = AstraGDN(d_model=d_model, config=config)
    opt_model = AstraGDNFLA(d_model=d_model, config=config, chunk_size=64)

    opt_model.load_state_dict(ref_model.state_dict())

    x_ref = torch.randn(B, T, d_model, requires_grad=True)
    x_opt = x_ref.clone().detach().requires_grad_(True)

    y_ref, _ = ref_model(x_ref)
    loss_ref = y_ref.sum()
    loss_ref.backward()

    y_opt, _ = opt_model(x_opt)
    loss_opt = y_opt.sum()
    loss_opt.backward()

    # 1. Input gradient parity
    grad_x_err = rel_err(x_ref.grad, x_opt.grad)
    assert grad_x_err < 1e-4, f"Input grad rel_err {grad_x_err} >= 1e-4"

    # 2. Parameter gradient parity
    for name, ref_param in ref_model.named_parameters():
        opt_param = dict(opt_model.named_parameters())[name]
        assert ref_param.grad is not None
        assert opt_param.grad is not None
        param_grad_err = rel_err(ref_param.grad, opt_param.grad)
        assert param_grad_err < 1e-4, f"Param {name} grad rel_err {param_grad_err} >= 1e-4"
