import unicodedata
from tokenizers import Tokenizer
from tokenizer.train_tokenizer import train_astra_tokenizer
from tokenizer.evaluate_tokenizer import evaluate_tokenizer


def test_tokenizer_nfc_and_byte_fallback(tmp_path):
    corpus = [
        "Tiếng Việt: Hà Nội, Đà Nẵng, TP. Hồ Chí Minh.",
        "Toán: E = mc^2, \\int_0^1 x dx = 0.5.",
        "Code: def forward(x):\n    return x\n",
    ] * 50

    tok_dir = tmp_path / "tok_test"
    tokenizer = train_astra_tokenizer(corpus, vocab_size=500, output_dir=str(tok_dir))
    tok_file = str(tok_dir / "tokenizer.json")

    eval_res = evaluate_tokenizer(tok_file)
    assert eval_res["unicode_nfc_test"]["passed"] is True
    assert 0 < eval_res["vocab_size"] <= 500

    # Test byte fallback on unseen emojis
    emoji_text = "Thử nghiệm emoji: 🚀🧪🔥"
    encoded = tokenizer.encode(emoji_text)
    decoded = tokenizer.decode(encoded.ids)
    assert decoded.strip() == emoji_text.strip()
