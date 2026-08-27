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


@pytest.mark.parametrize(
    "total_len,chunk_size,expected_tail",
    [
        (4097, 128, 1),    # 32 chunks of 128 + 1 token tail
        (4225, 512, 129),  # 8 chunks of 512 + 129 token tail
        (257, 64, 1),      # 4 chunks of 64 + 1 token tail
        (1025, 256, 1),    # 4 chunks of 256 + 1 token tail
    ],
)
def test_gdn_chunk_tail_torture(total_len, chunk_size, expected_tail):
    """
    Chunk-Boundary Torture Test:
    Tests non-aligned sequence lengths where the final chunk is a small remainder
    (e.g., exactly 1 token or 129 tokens) to expose tail accumulation bugs.
    """
    torch.manual_seed(42)
    B, d_model = 1, 64
    config = GDNConfig(num_heads=2, head_dim=32, conv_kernel=1)
    model = AstraGDN(d_model=d_model, config=config)
    model.eval()

    x = torch.randn(B, total_len, d_model)

    # Path A: Full Sequence pass
    with torch.no_grad():
        y_full, s_full = model(x)

    # Path B: Chunkwise pass with awkward tail
    chunk_outputs = []
    current_state = None
    num_chunks = (total_len + chunk_size - 1) // chunk_size

    with torch.no_grad():
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_len)
            x_chunk = x[:, start_idx:end_idx]

            # Verify the tail size on the last chunk
            if i == num_chunks - 1:
                assert x_chunk.shape[1] == expected_tail, f"Expected tail {expected_tail}, got {x_chunk.shape[1]}"

            y_chunk, current_state = model(x_chunk, state=current_state)
            chunk_outputs.append(y_chunk)

    y_chunked = torch.cat(chunk_outputs, dim=1)
    s_chunked = current_state

    rel_err_y = compute_relative_l2_error(y_full, y_chunked)
    max_err_y = compute_max_abs_error(y_full, y_chunked)
    rel_err_s = compute_relative_l2_error(s_full, s_chunked)

    assert rel_err_y < 1e-6, f"Tail torture (T={total_len}, chunk={chunk_size}) output rel_err {rel_err_y} >= 1e-6"
    assert max_err_y < 1e-5, f"Tail torture (T={total_len}, chunk={chunk_size}) output max_err {max_err_y} >= 1e-5"
    assert rel_err_s < 1e-6, f"Tail torture (T={total_len}, chunk={chunk_size}) state rel_err {rel_err_s} >= 1e-6"
