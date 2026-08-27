from __future__ import annotations

from typing import Optional, Tuple, Dict, List, Any
import torch
import torch.nn as nn

from configs.schema import AstraConfig
from .rmsnorm import RMSNorm
from .block import HybridBlock
from .mtp import MTPModule, compute_boundary_aware_loss


class AstraModel(nn.Module):
    """
    Core Transformer/Stateful Hybrid Backbone for Astra-1B.
    """
    def __init__(self, config: AstraConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.model.vocab_size
        self.hidden_size = config.model.hidden_size
        self.num_layers = config.model.num_layers

        # Input Embedding
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Build Layers according to repeating pattern
        pattern = list(config.model.layer_pattern)
        repeats = self.num_layers // len(pattern)
        self.layer_types = (pattern * repeats)[: self.num_layers]

        self.layers = nn.ModuleList(
            [
                HybridBlock(layer_type=l_type, config=config, layer_idx=idx)
                for idx, l_type in enumerate(self.layer_types)
            ]
        )

        # Final RMSNorm
        self.norm = RMSNorm(self.hidden_size, eps=config.model.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[List[Optional[torch.Tensor]]] = None,
        detach_state: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Args:
            input_ids: [B, T]
            position_ids: [B, T]
            attention_mask: [B, T]
            states: List of state tensors per layer (None for attention layers)
            detach_state: Whether to detach recurrent states from autograd graph
            
        Returns:
            (hidden_states [B, T, hidden_size], next_states list)
        """
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)

        if states is None:
            states = [None] * self.num_layers

        next_states = []
        for idx, layer in enumerate(self.layers):
            layer_state = states[idx]
            x, new_state = layer(
                x,
                position_ids=position_ids,
                attention_mask=attention_mask,
                state=layer_state,
                detach_state=detach_state,
            )
            next_states.append(new_state)

        hidden_states = self.norm(x)
        return hidden_states, next_states


class AstraForCausalLM(nn.Module):
    """
    Astra-1B Full Language Model with Tied Output Head and MTP-2 Auxiliary Objective.
    """
    def __init__(self, config: Optional[AstraConfig] = None):
        super().__init__()
        self.config = config or AstraConfig()
        self.model = AstraModel(self.config)

        # Language Model Head
        self.lm_head = nn.Linear(
            self.config.model.hidden_size, self.config.model.vocab_size, bias=False
        )

        # Tied Word Embeddings
        if self.config.model.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Multi-Token Prediction Head
        if self.config.mtp.enabled:
            self.mtp_module = MTPModule(
                d_model=self.config.model.hidden_size,
                norm_eps=self.config.model.norm_eps,
                config=self.config.mtp,
            )
        else:
            self.mtp_module = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[List[Optional[torch.Tensor]]] = None,
        detach_state: bool = False,
        compute_loss: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional boundary-aware loss calculation.
        """
        hidden_states, next_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            states=states,
            detach_state=detach_state,
        )

        # Main LM Logits (t -> t+1)
        logits_ar = self.lm_head(hidden_states)

        # MTP Logits (t -> t+2)
        if self.mtp_module is not None:
            mtp_hidden = self.mtp_module(hidden_states)
            logits_mtp = self.lm_head(mtp_hidden)
        else:
            logits_mtp = None

        result = {
            "logits": logits_ar,
            "logits_mtp": logits_mtp,
            "hidden_states": hidden_states,
            "states": next_states,
        }

        if compute_loss:
            loss_dict = compute_boundary_aware_loss(
                logits_ar=logits_ar,
                logits_mtp=logits_mtp,
                input_ids=input_ids,
                doc_ids=doc_ids,
                mtp_loss_weight=self.config.mtp.loss_weight,
            )
            result.update(loss_dict)

        return result
