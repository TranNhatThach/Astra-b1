from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.schema import AttentionConfig
from .rope import RotaryEmbedding


class ReferenceGQA(nn.Module):
    """
    Reference Gated Grouped Query Attention (GQA).
    
    Features:
      - Multi-query / Grouped-query attention (e.g. 16 Q heads, 4 KV heads)
      - Rotary Position Embedding (RoPE)
      - Causal self-attention masking
      - Learned output gate: g_a = sigmoid(W_g x + b_g)
    """
    def __init__(
        self,
        d_model: int = 2048,
        config: Optional[AttentionConfig] = None,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()
        self.config = config or AttentionConfig()
        self.d_model = d_model
        self.num_q_heads = self.config.num_q_heads
        self.num_kv_heads = self.config.num_kv_heads
        self.head_dim = self.config.head_dim
        self.num_groups = self.num_q_heads // self.num_kv_heads

        self.q_dim = self.num_q_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, self.q_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(self.q_dim, d_model, bias=False)

        if self.config.gated:
            self.gate_proj = nn.Linear(d_model, self.q_dim, bias=True)
            nn.init.constant_(self.gate_proj.bias, self.config.gate_bias_init)
        else:
            self.gate_proj = None

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=self.config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, d_model]
            position_ids: Position indices of shape [B, T]
            attention_mask: Optional mask of shape [B, 1, T, T] or [B, T]
        """
        B, T, _ = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # 2. Apply RoPE
        q, k = self.rotary_emb(q, k, position_ids=position_ids)

        # 3. Transpose to [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)  # [B, num_q_heads, T, head_dim]
        k = k.transpose(1, 2)  # [B, num_kv_heads, T, head_dim]
        v = v.transpose(1, 2)  # [B, num_kv_heads, T, head_dim]

        # 4. Repeat KV heads for grouped-query attention
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)  # [B, num_q_heads, T, head_dim]
            v = v.repeat_interleave(self.num_groups, dim=1)  # [B, num_q_heads, T, head_dim]

        # 5. Scaled Dot-Product Attention with Causal Mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_q_heads, T, T]

        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, T] -> [B, 1, 1, T]
                mask = (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -10000.0
                scores = scores + mask
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
        context = torch.matmul(attn_weights, v)  # [B, num_q_heads, T, head_dim]

        # Reshape to [B, T, q_dim]
        context = context.transpose(1, 2).contiguous().view(B, T, self.q_dim)

        # 6. Gating
        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(x))
            context = context * gate

        # 7. Output projection
        return self.out_proj(context)
