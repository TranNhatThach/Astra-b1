import torch
import torch.nn.functional as F
from model.reference.mtp import compute_boundary_aware_loss


def test_mtp_alignment_and_zero_loss_on_perfect_logits():
    B, T, V = 1, 6, 32
    input_ids = torch.tensor([[5, 8, 12, 19, 23, 30]])
    doc_ids = torch.zeros_like(input_ids)

    # Construct one-hot logits with huge values at the exact correct targets
    logits_ar = torch.zeros(B, T, V)
    logits_mtp = torch.zeros(B, T, V)

    # AR targets: input_ids[t+1]
    for t in range(T - 1):
        target_token = input_ids[0, t + 1].item()
        logits_ar[0, t, target_token] = 1000.0

    # MTP targets: input_ids[t+2]
    for t in range(T - 2):
        target_token = input_ids[0, t + 2].item()
        logits_mtp[0, t, target_token] = 1000.0

    loss_dict = compute_boundary_aware_loss(
        logits_ar=logits_ar,
        logits_mtp=logits_mtp,
        input_ids=input_ids,
        doc_ids=doc_ids,
        mtp_loss_weight=0.2,
    )

    assert loss_dict["loss_ar"].item() < 1e-4
    assert loss_dict["loss_mtp"].item() < 1e-4
    assert loss_dict["loss"].item() < 1e-4
