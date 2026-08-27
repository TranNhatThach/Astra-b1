"""
Astra Tokenizer Evaluation Suite (Phase 1)
Evaluates token efficiency (tokens/word, bytes/token) and Unicode lossless roundtrip
across Vietnamese, English, Source Code, and Mathematics.
"""

import json
import unicodedata
from pathlib import Path
from typing import Dict, Any, List
from tokenizers import Tokenizer


BENCHMARK_SAMPLES = {
    "vietnamese": (
        "Mô hình ngôn ngữ Astra được thiết kế theo kiến trúc lai ghép giữa Gated DeltaNet "
        "và Attention. Tiếng Việt cần được chuẩn hóa NFC đầy đủ: các nguyên âm như ă, â, đ, ê, "
        "ô, ơ, ư cùng dấu thanh huyền, sắc, hỏi, ngã, nặng phải được bảo tồn chính xác."
    ),
    "english": (
        "Astra-1B is a research-grade, ~1.0B parameter autoregressive Large Language Model "
        "combining Gated DeltaNet with Gated Grouped Query Attention, SwiGLU FFN, and MTP."
    ),
    "code": (
        "def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):\n"
        "    q = self.q_proj(x)\n"
        "    k = F.normalize(self.k_proj(x), p=2, dim=-1)\n"
        "    v = self.v_proj(x)\n"
        "    state = retention.unsqueeze(-1) * state + delta\n"
        "    return self.out_proj(y), state\n"
    ),
    "math": (
        "\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}, \\quad "
        "S_t = D_t S_{t-1}(I - u_t k_t k_t^T) + u_t v_t k_t^T, \\quad "
        "\\mathcal{L}_{total} = \\mathcal{L}_{AR} + 0.2 \\mathcal{L}_{MTP2}"
    ),
}


def evaluate_tokenizer(tokenizer_path: str = "tokenizer/tokenizer.json") -> Dict[str, Any]:
    tokenizer = Tokenizer.from_file(tokenizer_path)

    results = {
        "vocab_size": tokenizer.get_vocab_size(),
        "domains": {},
        "unicode_nfc_test": {"passed": True, "errors": []},
    }

    for domain, text in BENCHMARK_SAMPLES.items():
        # Ensure text is normalized NFC
        norm_text = unicodedata.normalize("NFC", text)
        encoded = tokenizer.encode(norm_text)
        decoded = tokenizer.decode(encoded.ids)

        num_words = len(norm_text.split())
        num_chars = len(norm_text)
        num_bytes = len(norm_text.encode("utf-8"))
        num_tokens = len(encoded.ids)

        tokens_per_word = num_tokens / max(num_words, 1)
        bytes_per_token = num_bytes / max(num_tokens, 1)
        chars_per_token = num_chars / max(num_tokens, 1)

        # Lossless Roundtrip Check
        roundtrip_match = unicodedata.normalize("NFC", decoded.strip()) == norm_text.strip()
        if not roundtrip_match:
            results["unicode_nfc_test"]["passed"] = False
            results["unicode_nfc_test"]["errors"].append(
                f"Mismatch in domain {domain}: decoded differs from original NFC text"
            )

        # Check for unk tokens
        unk_id = tokenizer.token_to_id("<unk>")
        unk_count = encoded.ids.count(unk_id) if unk_id is not None else 0

        results["domains"][domain] = {
            "num_words": num_words,
            "num_chars": num_chars,
            "num_bytes": num_bytes,
            "num_tokens": num_tokens,
            "tokens_per_word": round(tokens_per_word, 3),
            "bytes_per_token": round(bytes_per_token, 3),
            "chars_per_token": round(chars_per_token, 3),
            "unk_count": unk_count,
            "roundtrip_match": roundtrip_match,
        }

    return results


def generate_tokenizer_report(tokenizer_path: str = "tokenizer/tokenizer.json", output_dir: str = "tokenizer"):
    out_dir = Path(output_dir)
    res = evaluate_tokenizer(tokenizer_path)

    # 1. Export JSON
    json_path = out_dir / "tokenizer_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    # 2. Export Markdown
    md_path = out_dir / "tokenizer_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Astra Tokenizer Evaluation Report\n\n")
        f.write(f"- **Vocabulary Size:** {res['vocab_size']:,}\n")
        f.write(f"- **Unicode NFC Lossless:** {'[PASS]' if res['unicode_nfc_test']['passed'] else '[FAIL]'}\n\n")
        f.write("## Tokenization Efficiency Matrix\n\n")
        f.write("| Domain | Words | Tokens | Tokens/Word | Chars/Token | Bytes/Token | <unk> Count | Roundtrip |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for dom, stats in res["domains"].items():
            f.write(
                f"| **{dom.capitalize()}** | {stats['num_words']} | {stats['num_tokens']} | "
                f"{stats['tokens_per_word']} | {stats['chars_per_token']} | {stats['bytes_per_token']} | "
                f"{stats['unk_count']} | {'[OK]' if stats['roundtrip_match'] else '[MISMATCH]'} |\n"
            )

    print(f"[OK] Tokenizer evaluation reports generated at {json_path} and {md_path}")


if __name__ == "__main__":
    generate_tokenizer_report()
