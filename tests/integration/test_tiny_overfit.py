import torch
import torch.optim as optim
from configs.schema import AstraConfig, ModelConfig, GDNConfig, AttentionConfig, MTPConfig
from model.reference.astra import AstraForCausalLM


def test_tiny_dataset_overfit():
    """
    Integration Test: Verify that Astra can completely overfit a tiny dataset
    and that loss decreases monotonically towards zero (Sanity Check before large compute).
    """
    torch.manual_seed(42)
    config = AstraConfig(
        model=ModelConfig(
            name="astra-tiny",
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_layers=4,
            layer_pattern=("gdn", "gdn", "gdn", "attention"),
            tie_word_embeddings=True,
        ),
        gdn=GDNConfig(num_heads=2, head_dim=32, conv_kernel=4),
        attention=AttentionConfig(num_q_heads=2, num_kv_heads=1, head_dim=32),
        mtp=MTPConfig(enabled=True, horizon=2, loss_weight=0.2),
    )

    model = AstraForCausalLM(config)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    B, T = 2, 16
    input_ids = torch.randint(0, 64, (B, T))
    doc_ids = torch.zeros(B, T, dtype=torch.long)

    # Initial loss
    with torch.no_grad():
        init_loss = model(input_ids=input_ids, doc_ids=doc_ids, compute_loss=True)["loss"].item()

    # Optimization loop (50 steps)
    losses = []
    for step in range(50):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, doc_ids=doc_ids, compute_loss=True)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]

    # Verify dramatic loss decrease (memorization)
    assert final_loss < 0.1, f"Model failed to overfit: initial={init_loss:.4f}, final={final_loss:.4f}"
    assert final_loss < init_loss * 0.05, f"Loss did not drop by at least 95%: init={init_loss}, final={final_loss}"
