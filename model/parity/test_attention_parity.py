import pytest
import torch
from configs.schema import AttentionConfig
from model.reference.attention import ReferenceGQA
from model.optimized.attention_flash import FlashAttentionGQA


def rel_err(ref: torch.Tensor, opt: torch.Tensor, eps: float = 1e-8) -> float:
    diff = torch.linalg.norm(ref.float() - opt.float()).item()
    norm = torch.linalg.norm(ref.float()).item()
    return diff / max(norm, eps)


@pytest.mark.parametrize("seq_len", [1, 7, 31, 64, 127, 257, 1024])
def test_attention_forward_parity(seq_len):
    torch.manual_seed(42)
    B, d_model = 2, 64
    config = AttentionConfig(num_q_heads=4, num_kv_heads=2, head_dim=32)

    ref_attn = ReferenceGQA(d_model=d_model, config=config)
    opt_attn = FlashAttentionGQA(d_model=d_model, config=config)

    opt_attn.load_state_dict(ref_attn.state_dict())
    ref_attn.eval()
    opt_attn.eval()

    x = torch.randn(B, seq_len, d_model)

    with torch.no_grad():
        out_ref = ref_attn(x)
        out_opt = opt_attn(x)

    err = rel_err(out_ref, out_opt)
    assert err < 1e-4, f"Attention T={seq_len} forward rel_err {err} >= 1e-4"


def test_attention_gradient_parity():
    torch.manual_seed(42)
    B, T, d_model = 2, 64, 64
    config = AttentionConfig(num_q_heads=4, num_kv_heads=2, head_dim=32)

    ref_attn = ReferenceGQA(d_model=d_model, config=config)
    opt_attn = FlashAttentionGQA(d_model=d_model, config=config)

    opt_attn.load_state_dict(ref_attn.state_dict())

    x_ref = torch.randn(B, T, d_model, requires_grad=True)
    x_opt = x_ref.clone().detach().requires_grad_(True)

    out_ref = ref_attn(x_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    out_opt = opt_attn(x_opt)
    loss_opt = out_opt.sum()
    loss_opt.backward()

    grad_x_err = rel_err(x_ref.grad, x_opt.grad)
    assert grad_x_err < 1e-4, f"Attention input grad rel_err {grad_x_err} >= 1e-4"

    for name, ref_p in ref_attn.named_parameters():
        opt_p = dict(opt_attn.named_parameters())[name]
        param_err = rel_err(ref_p.grad, opt_p.grad)
        assert param_err < 1e-4, f"Attention param {name} grad rel_err {param_err} >= 1e-4"
