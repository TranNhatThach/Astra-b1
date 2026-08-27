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


@pytest.mark.parametrize("chunk_size", [1, 8, 32, 128, 512])
def test_gdn_chunk_equivalence(chunk_size):
    """
    Chunk-Equivalence Invariant Test:
    Processing sequence T in chunks of arbitrary size (1, 8, 32, 128, 512)
    with state carryover must yield mathematically equivalent outputs and final states
    as processing the full sequence at once.
    """
    torch.manual_seed(42)
    B, T, d_model = 2, 512, 128
    config = GDNConfig(num_heads=4, head_dim=32, conv_kernel=1)
    model = AstraGDN(d_model=d_model, config=config)
    model.eval()

    x = torch.randn(B, T, d_model)

    # 1. Full sequence pass
    with torch.no_grad():
        y_full, s_full = model(x)

    # 2. Chunkwise pass
    chunk_outputs = []
    current_state = None
    num_chunks = (T + chunk_size - 1) // chunk_size

    with torch.no_grad():
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, T)
            x_chunk = x[:, start_idx:end_idx]

            y_chunk, current_state = model(x_chunk, state=current_state)
            chunk_outputs.append(y_chunk)

    y_chunked = torch.cat(chunk_outputs, dim=1)
    s_chunked = current_state

    rel_err_y = compute_relative_l2_error(y_full, y_chunked)
    max_err_y = compute_max_abs_error(y_full, y_chunked)
    rel_err_s = compute_relative_l2_error(s_full, s_chunked)

    assert rel_err_y < 1e-6, f"Chunk size {chunk_size} output relative error {rel_err_y} >= 1e-6"
    assert max_err_y < 1e-5, f"Chunk size {chunk_size} output max error {max_err_y} >= 1e-5"
    assert rel_err_s < 1e-6, f"Chunk size {chunk_size} state relative error {rel_err_s} >= 1e-6"
