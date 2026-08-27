from __future__ import annotations

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.schema import MTPConfig
from .rmsnorm import RMSNorm


class MTPModule(nn.Module):
    """
    Multi-Token Prediction (MTP-2) Auxiliary Module.
    Transforms current hidden states h_t into future representations h_{t+2}
    before projecting through the shared LM head.
    """
    def __init__(self, d_model: int = 2048, norm_eps: float = 1e-6, config: Optional[MTPConfig] = None):
        super().__init__()
        self.config = config or MTPConfig()
        self.proj1 = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.proj2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # h_t -> proj1 -> RMSNorm -> proj2
        h = self.proj1(hidden_states)
        h = self.norm(h)
        return self.proj2(h)


def compute_boundary_aware_loss(
    logits_ar: torch.Tensor,
    logits_mtp: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    doc_ids: Optional[torch.Tensor] = None,
    mtp_loss_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Computes Autoregressive and MTP-2 cross-entropy losses with document-boundary masking.
    
    Args:
        logits_ar: Logits for next token prediction [B, T, V]
        logits_mtp: Logits for 2-step future prediction [B, T, V]
        input_ids: Ground truth token sequence [B, T]
        doc_ids: Document index per token [B, T] (determines document boundaries)
        mtp_loss_weight: lambda_MTP scalar weighting
        
    Returns:
        Dictionary with 'loss', 'loss_ar', 'loss_mtp', 'valid_ar_tokens', 'valid_mtp_tokens'
    """
    B, T, V = logits_ar.shape
    device = input_ids.device

    # 1. Autoregressive (t -> t+1)
    shift_logits_ar = logits_ar[:, :-1, :].contiguous().view(-1, V)
    shift_labels_ar = input_ids[:, 1:].contiguous().view(-1)

    if doc_ids is not None:
        # doc(t) == doc(t+1)
        valid_ar = (doc_ids[:, :-1] == doc_ids[:, 1:]).contiguous().view(-1).float()
    else:
        valid_ar = torch.ones_like(shift_labels_ar, dtype=torch.float32)

    loss_ar_per_token = F.cross_entropy(
        shift_logits_ar, shift_labels_ar, reduction="none"
    )
    num_valid_ar = valid_ar.sum().clamp(min=1.0)
    loss_ar = (loss_ar_per_token * valid_ar).sum() / num_valid_ar

    # 2. MTP-2 (t -> t+2)
    loss_mtp = torch.tensor(0.0, device=device, dtype=loss_ar.dtype)
    num_valid_mtp = torch.tensor(0.0, device=device)

    if logits_mtp is not None and T > 2:
        shift_logits_mtp = logits_mtp[:, :-2, :].contiguous().view(-1, V)
        shift_labels_mtp = input_ids[:, 2:].contiguous().view(-1)

        if doc_ids is not None:
            # doc(t) == doc(t+2)
            valid_mtp = (doc_ids[:, :-2] == doc_ids[:, 2:]).contiguous().view(-1).float()
        else:
            valid_mtp = torch.ones_like(shift_labels_mtp, dtype=torch.float32)

        loss_mtp_per_token = F.cross_entropy(
            shift_logits_mtp, shift_labels_mtp, reduction="none"
        )
        num_valid_mtp = valid_mtp.sum().clamp(min=1.0)
        loss_mtp = (loss_mtp_per_token * valid_mtp).sum() / num_valid_mtp

    total_loss = loss_ar + mtp_loss_weight * loss_mtp

    return {
        "loss": total_loss,
        "loss_ar": loss_ar,
        "loss_mtp": loss_mtp,
        "num_valid_ar": num_valid_ar,
        "num_valid_mtp": num_valid_mtp,
    }
