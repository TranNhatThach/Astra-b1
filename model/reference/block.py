from __future__ import annotations

from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn

from configs.schema import AstraConfig
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .gdn import AstraGDN
from .attention import ReferenceGQA


class HybridBlock(nn.Module):
    """
    Unified Hybrid Layer Block for Astra-1B.
    
    Structure:
      x
      │
      ├── RMSNorm
      ▼
      Mixer (AstraGDN or ReferenceGQA)
      │
      ▼
      u = x + residual_gate * Mixer(RMSNorm(x))
      │
      ├── RMSNorm
      ▼
      SwiGLU
      │
      ▼
      y = u + SwiGLU(RMSNorm(u))
    """
    def __init__(
        self,
        layer_type: Literal["gdn", "attention"],
        config: AstraConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        d_model = config.model.hidden_size
        d_ff = config.model.intermediate_size
        norm_eps = config.model.norm_eps

        # Pre-mix RMSNorm
        self.input_norm = RMSNorm(d_model, eps=norm_eps)

        # Sequence Mixer
        if layer_type == "gdn":
            self.mixer = AstraGDN(d_model=d_model, config=config.gdn)
        elif layer_type == "attention":
            self.mixer = ReferenceGQA(
                d_model=d_model,
                config=config.attention,
                max_position_embeddings=config.model.max_position_embeddings,
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # Learnable residual gate scalar
        self.residual_gate = nn.Parameter(torch.ones(1))

        # Pre-FFN RMSNorm & SwiGLU
        self.post_attention_norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLU(d_model=d_model, intermediate_size=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        detach_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            (output_hidden_states, next_recurrent_state_if_gdn)
        """
        # 1. Mixer branch (Pre-Norm)
        normed_x = self.input_norm(x)
        
        if self.layer_type == "gdn":
            mixer_out, new_state = self.mixer(
                normed_x, state=state, detach_state=detach_state
            )
        else:
            mixer_out = self.mixer(
                normed_x, position_ids=position_ids, attention_mask=attention_mask
            )
            new_state = None

        # Residual connection with learnable gate
        u = x + self.residual_gate * mixer_out

        # 2. Feed-Forward branch (Pre-Norm)
        normed_u = self.post_attention_norm(u)
        mlp_out = self.mlp(normed_u)
        y = u + mlp_out

        return y, new_state
