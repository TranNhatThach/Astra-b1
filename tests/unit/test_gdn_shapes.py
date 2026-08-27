import torch
import pytest
from configs.schema import GDNConfig
from model.reference.gdn import AstraGDN


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [1, 7, 64, 257])
def test_gdn_output_and_state_shapes(batch_size, seq_len):
    d_model = 256
    config = GDNConfig(num_heads=4, head_dim=32, conv_kernel=4)
    model = AstraGDN(d_model=d_model, config=config)

    x = torch.randn(batch_size, seq_len, d_model)
    out, state = model(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert state.shape == (batch_size, config.num_heads, config.head_dim, config.head_dim)
    assert out.dtype == x.dtype
    assert state.dtype == x.dtype
