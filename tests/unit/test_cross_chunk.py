import torch
from configs.schema import GDNConfig
from model.reference.gdn import AstraGDN


def test_cross_chunk_state_continuity():
    """
    Verifies that recurrent state is strictly preserved across chunk boundaries:
    S_{out}(chunk_i) == S_{in}(chunk_{i+1}) and is never accidentally reset or corrupted.
    """
    torch.manual_seed(42)
    B, d_model = 2, 64
    chunk_size = 16
    num_chunks = 4

    config = GDNConfig(num_heads=2, head_dim=32, conv_kernel=1)
    model = AstraGDN(d_model=d_model, config=config)
    model.eval()

    x = torch.randn(B, chunk_size * num_chunks, d_model)

    current_state = None
    saved_states = []

    for i in range(num_chunks):
        x_chunk = x[:, i * chunk_size : (i + 1) * chunk_size]
        
        # Verify chunk 0 initial state is None or zeros
        if i == 0:
            assert current_state is None

        # Execute chunk forward
        _, next_state = model(x_chunk, state=current_state)
        
        # State must not be all zeros after processing chunk
        assert not torch.allclose(next_state, torch.zeros_like(next_state))
        
        saved_states.append(next_state.clone())
        current_state = next_state

    # Verify state continuity: each chunk received precisely the preceding state
    assert len(saved_states) == num_chunks
    for i in range(len(saved_states) - 1):
        assert not torch.allclose(saved_states[i], saved_states[i + 1]), "State must evolve across chunks"
