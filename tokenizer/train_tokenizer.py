"""
Astra-1B BPE Tokenizer Trainer (Phase 1)
Trains a 65,536-vocab Byte-Pair Encoding (BPE) Tokenizer with:
  - Strict Unicode NFC Normalization
  - Byte-Level Pre-tokenization with Byte Fallback (zero <unk> on valid UTF-8)
  - Dedicated Vietnamese, Code, English, and Math token efficiency
"""

import json
import hashlib
from pathlib import Path
from typing import List, Iterable, Optional
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, trainers, models


def build_astra_tokenizer(
    vocab_size: int = 65536,
    special_tokens_path: Optional[str] = None,
) -> Tokenizer:
    # 1. Initialize BPE model with byte fallback
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))

    # 2. Normalizer: Canonical NFC
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
    ])

    # 3. Pre-tokenizer: ByteLevel (preserves whitespace and prevents character destruction)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])

    # 4. Decoder: ByteLevel Decoder
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def train_astra_tokenizer(
    training_data: Iterable[str] | List[str],
    vocab_size: int = 65536,
    special_tokens_file: Optional[str] = None,
    output_dir: str = "tokenizer",
) -> Tokenizer:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if special_tokens_file is None:
        special_tokens_file = str(Path(__file__).parent / "special_tokens.json")

    with open(special_tokens_file, "r", encoding="utf-8") as f:
        spec_data = json.load(f)

    special_tokens = [
        spec_data["unk_token"],
        spec_data["bos_token"],
        spec_data["eos_token"],
        spec_data["pad_token"],
    ] + spec_data.get("additional_special_tokens", [])

    tokenizer = build_astra_tokenizer(vocab_size=vocab_size)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    if isinstance(training_data, list) and len(training_data) > 0 and Path(training_data[0]).is_file():
        tokenizer.train(training_data, trainer=trainer)
    else:
        tokenizer.train_from_iterator(training_data, trainer=trainer)

    # Save tokenizer.json
    save_path = out_dir / "tokenizer.json"
    tokenizer.save(str(save_path))

    # Compute tokenizer hash
    with open(save_path, "rb") as f:
        tok_hash = hashlib.sha256(f.read()).hexdigest()

    meta_path = out_dir / "tokenizer_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": tokenizer.get_vocab_size(),
                "tokenizer_hash": tok_hash,
                "normalization": "NFC",
                "byte_fallback": True,
            },
            f,
            indent=2,
        )

    print(f"[OK] Tokenizer saved to {save_path} (vocab={tokenizer.get_vocab_size()}, SHA256={tok_hash})")
    return tokenizer


if __name__ == "__main__":
    # Sample corpus covering Vietnamese, English, Code, Math, Unicode for testing
    sample_corpus = [
        "Astra-1B là một mô hình ngôn ngữ lớn lai ghép (Hybrid Large Language Model).",
        "Tiếng Việt có dấu: ă, â, đ, ê, ô, ơ, ư, và các thanh điệu huyền, sắc, hỏi, ngã, nặng.",
        "Toán học: \\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}, E = mc^2, f(x) = \\sigma(Wx + b).",
        "Python Code:\ndef forward(x, state=None):\n    y = S @ q\n    return y, state\n",
        "English: The primary objective is building a reproducible, research-grade hybrid model.",
    ] * 200

    train_astra_tokenizer(sample_corpus, vocab_size=1024)
