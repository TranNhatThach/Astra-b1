import io
import torch
from configs.schema import AstraConfig, ModelConfig, GDNConfig, AttentionConfig
from model.reference.astra import AstraForCausalLM


def test_model_checkpoint_roundtrip_and_determinism():
    torch.manual_seed(42)
    config = AstraConfig(
        model=ModelConfig(
            name="astra-ckpt-test",
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_layers=4,
            layer_pattern=("gdn", "gdn", "gdn", "attention"),
            tie_word_embeddings=True,
        ),
        gdn=GDNConfig(num_heads=2, head_dim=32),
        attention=AttentionConfig(num_q_heads=2, num_kv_heads=1, head_dim=32),
    )

    model1 = AstraForCausalLM(config)
    model1.eval()

    # Save to buffer
    buffer = io.BytesIO()
    torch.save(model1.state_dict(), buffer)
    buffer.seek(0)

    # Load into fresh model
    model2 = AstraForCausalLM(config)
    model2.load_state_dict(torch.load(buffer, weights_only=True))
    model2.eval()

    input_ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        out1 = model1(input_ids)
        out2 = model2(input_ids)

    torch.testing.assert_close(out1["logits"], out2["logits"], rtol=0.0, atol=0.0)
