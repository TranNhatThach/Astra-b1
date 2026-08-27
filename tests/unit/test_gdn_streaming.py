import torch
import pytest
from configs.schema import GDNConfig
from model.reference.gdn import AstraGDN


def compute_relative_l2_error(ref: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    diff_norm = torch.linalg.norm(ref - target).item()
    ref_norm = torch.linalg.norm(ref).item()
    return diff_norm / max(ref_norm, eps)


def compute_max_abs_error(ref: torch.Tensor, target: torch.Tensor) -> float:
    return torch.max(torch.abs(ref - target)).item()


def test_gdn_recurrent_streaming_equivalence():
    """
    Test that full-sequence processing and token-by-token recurrent streaming
    produce mathematically equivalent outputs and final states (conv_kernel=1).
    """
    torch.manual_seed(42)
    B, T, d_model = 2, 32, 128
    config = GDNConfig(num_heads=4, head_dim=32, conv_kernel=1)
    model = AstraGDN(d_model=d_model, config=config)
    model.eval()

    x = torch.randn(B, T, d_model)

    # Path A: Full Sequence in one pass
    with torch.no_grad():
        y_full, s_full = model(x)

    # Path B: Token-by-token streaming
    stream_outputs = []
    current_state = None
    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t : t + 1]  # [B, 1, d_model]
            y_t, current_state = model(x_t, state=current_state)
            stream_outputs.append(y_t)

    y_stream = torch.cat(stream_outputs, dim=1)  # [B, T, d_model]
    s_stream = current_state

    # Verify metrics
    rel_l2_y = compute_relative_l2_error(y_full, y_stream)
    max_abs_y = compute_max_abs_error(y_full, y_stream)
    rel_l2_s = compute_relative_l2_error(s_full, s_stream)

    assert rel_l2_y < 1e-6, f"Output relative error {rel_l2_y} exceeded 1e-6"
    assert max_abs_y < 1e-5, f"Output max error {max_abs_y} exceeded 1e-5"
    assert rel_l2_s < 1e-6, f"State relative error {rel_l2_s} exceeded 1e-6"
