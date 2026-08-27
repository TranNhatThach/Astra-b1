import torch
from model.reference.mtp import compute_boundary_aware_loss


def test_mtp_and_ar_document_boundary_masking():
    """
    Test deterministic document boundary masking contract:
      doc A = [10, 11, 12, 13] (doc_id = 1)
      doc B = [20, 21, 22, 23] (doc_id = 2)
      
    Expected:
      AR valid:   [1, 1, 1, 0, 1, 1, 1]  (total 6 valid transitions)
      MTP2 valid: [1, 1, 0, 0, 1, 1]     (total 4 valid transitions)
    """
    input_ids = torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]])  # [1, 8]
    doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]])            # [1, 8]
    vocab_size = 50

    # Create dummy logits
    logits_ar = torch.randn(1, 8, vocab_size)
    logits_mtp = torch.randn(1, 8, vocab_size)

    loss_dict = compute_boundary_aware_loss(
        logits_ar=logits_ar,
        logits_mtp=logits_mtp,
        input_ids=input_ids,
        doc_ids=doc_ids,
        mtp_loss_weight=0.2,
    )

    assert loss_dict["num_valid_ar"].item() == 6.0
    assert loss_dict["num_valid_mtp"].item() == 4.0
    assert loss_dict["loss"].item() > 0.0
