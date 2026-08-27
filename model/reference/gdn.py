from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.schema import GDNConfig


class HeadRMSNorm(nn.Module):
    """Applies RMSNorm independently to each head's feature dimension."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., dim]
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class AstraGDN(nn.Module):
    """
    Astra-GDN Golden Reference Implementation.
    
    Mathematical Contract (Astra Contract v0.1):
      D_t = Diag(retention_t)
      S_t = D_t @ S_{t-1} @ (I - update_t * k_t @ k_t^T) + update_t * (v_t @ k_t^T)
      y_t = S_t @ q_t
      
    State representation:
      state: [B, H, D, D] (where S @ q is evaluated as matrix @ vector)
      q, k, v: [B, H, D]
    """
    def __init__(
        self,
        d_model: int = 2048,
        config: Optional[GDNConfig] = None,
    ):
        super().__init__()
        self.config = config or GDNConfig()
        self.d_model = d_model
        self.num_heads = self.config.num_heads
        self.head_dim = self.config.head_dim
        self.d_proj = self.num_heads * self.head_dim
        self.key_norm_eps = self.config.key_norm_eps
        self.conv_kernel = self.config.conv_kernel

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

        # Initialize Gate Biases to specification
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
        """
        Args:
            x: Input tensor of shape [B, T, d_model]
            state: Optional recurrent memory state of shape [B, num_heads, head_dim, head_dim]
            detach_state: If True, detaches the returned state from the autograd graph (for chunked training).
            
        Returns:
            Tuple of:
              - Output tensor of shape [B, T, d_model]
              - Next state tensor of shape [B, num_heads, head_dim, head_dim]
        """
        B, T, _ = x.shape

        # 1. Linear projections for Q, K, V
        q = self.q_proj(x)  # [B, T, d_proj]
        k = self.k_proj(x)  # [B, T, d_proj]
        v = self.v_proj(x)  # [B, T, d_proj]

        # 2. Causal Depthwise Convolution over concatenated [Q, K, V]
        if self.conv is not None:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)  # [B, 3*d_proj, T]
            qkv = self.conv(qkv)[:, :, :T].transpose(1, 2)       # [B, T, 3*d_proj] (Causal slice)
            q, k, v = torch.split(qkv, self.d_proj, dim=-1)

        # Reshape to multi-head format [B, T, H, D]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Key L2-Normalization
        k_norm = torch.linalg.norm(k, ord=2, dim=-1, keepdim=True)
        k = k / (k_norm + self.key_norm_eps)

        # 3. Gate activations
        # retention_gate: [B, T, H, D]
        retention = torch.sigmoid(self.retention_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )
        # update_gate: [B, T, H, 1]
        update = torch.sigmoid(self.update_proj(x)).view(
            B, T, self.num_heads, 1
        )
        # output_gate: [B, T, H, D] (if enabled)
        if self.output_gate_proj is not None:
            output_gate = torch.sigmoid(self.output_gate_proj(x)).view(
                B, T, self.num_heads, self.head_dim
            )
        else:
            output_gate = None

        # 4. Initialize State if None
        if state is None:
            state = torch.zeros(
                B,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # 5. Sequential Recurrent Golden Loop (Astra Contract v0.1)
        out_tokens = []
        for t in range(T):
            q_t = q[:, t]                      # [B, H, D]
            k_t = k[:, t]                      # [B, H, D]
            v_t = v[:, t]                      # [B, H, D]
            r_t = retention[:, t]              # [B, H, D]
            u_t = update[:, t]                 # [B, H, 1]

            # v_hat = S_{t-1} @ k_t  (read historical value associated with key k_t)
            # state is [B, H, D, D], k_t.unsqueeze(-1) is [B, H, D, 1] -> [B, H, D]
            v_hat = torch.matmul(state, k_t.unsqueeze(-1)).squeeze(-1)

            # delta update: u_t * (v_t - v_hat) outer k_t
            # (v_t - v_hat) is [B, H, D, 1], k_t is [B, H, 1, D] -> outer product is [B, H, D, D]
            diff = (v_t - v_hat).unsqueeze(-1)
            k_outer = k_t.unsqueeze(-2)
            delta = u_t.unsqueeze(-1) * torch.matmul(diff, k_outer)

            # S_t = D_t @ S_{t-1} + delta
            # Row-wise diagonal retention: D_t @ S_{t-1}, where D_t = diag(retention_t)
            state = r_t.unsqueeze(-1) * state + delta

            # Output retrieval: y_t = S_t @ q_t  [B, H, D]
            y_t = torch.matmul(state, q_t.unsqueeze(-1)).squeeze(-1)
            out_tokens.append(y_t)

        # [B, T, H, D]
        y = torch.stack(out_tokens, dim=1)

        # 6. Head RMSNorm & Output Gating
        y = self.head_norm(y)
        if output_gate is not None:
            y = y * output_gate

        # Reshape back to [B, T, d_proj] and project to d_model
        y = y.contiguous().view(B, T, self.d_proj)
        out = self.out_proj(y)

        if detach_state:
            state = state.detach()

        return out, state
