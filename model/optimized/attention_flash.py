from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.schema import AttentionConfig
from model.reference.rope import RotaryEmbedding


class FlashAttentionGQA(nn.Module):
    """
    Astra FlashAttention GQA Backend (Phase 5C).
    Accelerates attention computation via PyTorch native SDPA / FlashAttention-2 kernel
    while strictly preserving RoPE, GQA head grouping, and output gating semantics.
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
        B, T, _ = x.shape

        # 1. Projections
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # 2. RoPE
        q, k = self.rotary_emb(q, k, position_ids=position_ids)

        # [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. GQA expansion
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # 4. Scaled Dot-Product Attention (SDPA / FlashAttention kernel)
        if attention_mask is None:
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            if attention_mask.dim() == 2:
                # [B, T] -> [B, 1, 1, T]
                mask = (1.0 - attention_mask[:, None, None, :].to(q.dtype)) * -10000.0
            else:
                mask = attention_mask
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )

        context = context.transpose(1, 2).contiguous().view(B, T, self.q_dim)

        # 5. Gating
        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(x))
            context = context * gate

        return self.out_proj(context)
