import torch
from configs.schema import AttentionConfig
from model.reference.attention import ReferenceGQA


def test_gqa_causality_invariance():
    torch.manual_seed(42)
    B, T, d_model = 2, 16, 128
    t_cutoff = 6

    config = AttentionConfig(num_q_heads=4, num_kv_heads=2, head_dim=32)
    attn = ReferenceGQA(d_model=d_model, config=config)
    attn.eval()

    x1 = torch.randn(B, T, d_model)
    x2 = x1.clone()
    x2[:, t_cutoff:] = torch.randn(B, T - t_cutoff, d_model)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Outputs at t < t_cutoff must be identical
    torch.testing.assert_close(
        out1[:, :t_cutoff],
        out2[:, :t_cutoff],
        rtol=1e-6,
        atol=1e-6,
        msg="GQA output at t < t_cutoff must not be influenced by future inputs",
    )
    assert not torch.allclose(out1[:, t_cutoff:], out2[:, t_cutoff:])
