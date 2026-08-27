import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Rotates query and key tensors according to position indices.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calculate inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor of shape [B, T, num_q_heads, head_dim]
            k: Key tensor of shape [B, T, num_kv_heads, head_dim]
            position_ids: Position indices of shape [B, T]
        """
        B, T, _, D = q.shape
        device = q.device
        dtype = q.dtype

        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # Compute freqs: [B, T, D/2]
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        pos = position_ids.to(device=device, dtype=torch.float32)
        freqs = torch.einsum("bt,d->btd", pos, inv_freq)  # [B, T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)           # [B, T, D]

        cos = emb.cos().unsqueeze(2).to(dtype=dtype)      # [B, T, 1, D]
        sin = emb.sin().unsqueeze(2).to(dtype=dtype)      # [B, T, 1, D]

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot
