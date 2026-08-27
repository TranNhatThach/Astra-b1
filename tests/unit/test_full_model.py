import torch
from configs.schema import AstraConfig, ModelConfig, GDNConfig, AttentionConfig, MTPConfig
from model.reference.astra import AstraForCausalLM


def test_astra_full_model_forward_and_backward():
    torch.manual_seed(42)
    # Small test config: 4 layers (3 GDN + 1 Attention)
    config = AstraConfig(
        model=ModelConfig(
            name="astra-test",
            vocab_size=128,
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

    # Check tied embeddings
    assert model.lm_head.weight is model.model.embed_tokens.weight

    B, T = 2, 8
    input_ids = torch.randint(0, 128, (B, T))
    doc_ids = torch.zeros(B, T, dtype=torch.long)

    out = model(input_ids=input_ids, doc_ids=doc_ids, compute_loss=True)

    assert "loss" in out
    assert "logits" in out
    assert "logits_mtp" in out
    assert out["logits"].shape == (B, T, 128)
    assert out["logits_mtp"].shape == (B, T, 128)

    # Test backward
    loss = out["loss"]
    loss.backward()

    # Verify gradients flowed to all major parameter tensors
    assert model.model.embed_tokens.weight.grad is not None
    assert model.model.layers[0].mixer.q_proj.weight.grad is not None  # GDN layer
    assert model.model.layers[3].mixer.q_proj.weight.grad is not None  # Attention layer
    assert model.model.layers[0].mlp.w_gate.weight.grad is not None     # SwiGLU
    assert model.mtp_module.proj1.weight.grad is not None              # MTP head
