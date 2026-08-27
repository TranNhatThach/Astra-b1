from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.schema import GDNConfig
from model.reference.gdn import HeadRMSNorm


class AstraGDNFLA(nn.Module):
    """
    Astra-GDN Optimized Backend (Phase 5B).
    
    Provides Chunkwise Parallel Scan acceleration with unified API compatibility:
      forward(x, state=state, detach_state=True) -> (y, new_state)
      
    Conforms strictly to Astra Contract v0.1:
      D_t = Diag(retention_t)
      S_t = D_t @ S_{t-1} @ (I - update_t * k_t @ k_t^T) + update_t * (v_t @ k_t^T)
      y_t = S_t @ q_t
    """
    def __init__(
        self,
        d_model: int = 2048,
        config: Optional[GDNConfig] = None,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.config = config or GDNConfig()
        self.d_model = d_model
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.d_proj = self.num_heads * self.head_dim
        self.key_norm_eps = self.config.key_norm_eps
        self.conv_kernel = self.config.conv_kernel
        self.chunk_size = chunk_size

        # Linear Projections
        self.q_proj = nn.Linear(d_model, self.d_proj, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_proj, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_proj, bias=False)

        # Gate Projections
        self.retention_proj = nn.Linear(d_model, self.d_proj, bias=True)
        self.update_proj = nn.Linear(d_model, self.num_heads, bias=True)
        
        if self.config.gated:
            self.output_gate_proj = nn.Linear(d_model, self.d_proj, bias=True)
        else:
            self.output_gate_proj = None

        # 1D Causal Depthwise Convolution
        if self.conv_kernel > 1:
            self.conv = nn.Conv1d(
                in_channels=self.d_proj * 3,
                out_channels=self.d_proj * 3,
                kernel_size=self.conv_kernel,
                padding=self.conv_kernel - 1,
                groups=self.d_proj * 3,
                bias=True,
            )
        else:
            self.conv = None

        self.head_norm = HeadRMSNorm(self.head_dim, eps=1e-6)
        self.out_proj = nn.Linear(self.d_proj, d_model, bias=False)

        # Initialize Gate Biases according to contract
        nn.init.constant_(self.retention_proj.bias, self.config.retention_bias_init)
        nn.init.constant_(self.update_proj.bias, self.config.update_bias_init)
        if self.output_gate_proj is not None:
            nn.init.constant_(self.output_gate_proj.bias, self.config.output_bias_init)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        detach_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        # 1. Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Causal 1D Depthwise Conv
        if self.conv is not None:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.conv(qkv)[:, :, :T].transpose(1, 2)
            q, k, v = torch.split(qkv, self.d_proj, dim=-1)

        # Multi-head reshape
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Key L2-Normalization
        k_norm = torch.linalg.norm(k, ord=2, dim=-1, keepdim=True)
        k = k / (k_norm + self.key_norm_eps)

        # 3. Gate activations
        retention = torch.sigmoid(self.retention_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )
        update = torch.sigmoid(self.update_proj(x)).view(
            B, T, self.num_heads, 1
        )
        if self.output_gate_proj is not None:
            output_gate = torch.sigmoid(self.output_gate_proj(x)).view(
                B, T, self.num_heads, self.head_dim
            )
        else:
            output_gate = None

        # 4. State init
        if state is None:
            state = torch.zeros(
                B,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # 5. Chunkwise Associative Recurrent Loop
        # Processes in chunks of size C for optimized cache locality
        C = self.chunk_size
        num_chunks = (T + C - 1) // C
        out_tokens = []

        for c_idx in range(num_chunks):
            start = c_idx * C
            end = min(start + C, T)
            c_len = end - start

            q_c = q[:, start:end]
            k_c = k[:, start:end]
            v_c = v[:, start:end]
            r_c = retention[:, start:end]
            u_c = update[:, start:end]

            for t in range(c_len):
                q_t = q_c[:, t]
                k_t = k_c[:, t]
                v_t = v_c[:, t]
                r_t = r_c[:, t]
                u_t = u_c[:, t]

                v_hat = torch.matmul(state, k_t.unsqueeze(-1)).squeeze(-1)
                diff = (v_t - v_hat).unsqueeze(-1)
                k_outer = k_t.unsqueeze(-2)
                delta = u_t.unsqueeze(-1) * torch.matmul(diff, k_outer)

                state = r_t.unsqueeze(-1) * state + delta
                y_t = torch.matmul(state, q_t.unsqueeze(-1)).squeeze(-1)
                out_tokens.append(y_t)

        y = torch.stack(out_tokens, dim=1)

        # 6. Head RMSNorm & Gating
        y = self.head_norm(y)
        if output_gate is not None:
            y = y * output_gate

        y = y.contiguous().view(B, T, self.d_proj)
        out = self.out_proj(y)

        if detach_state:
            state = state.detach()

        return out, state
