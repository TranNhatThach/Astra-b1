"""
Astra Tokenizer Freeze Utility (Phase 7)
Validates, fingerprints, and freezes Tokenizer v0.1 for production pretraining.
Once FROZEN, the tokenizer cannot be modified in place.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any

from .evaluate_tokenizer import evaluate_tokenizer, generate_tokenizer_report
from experiments.identity import compute_tokenizer_hash


def freeze_tokenizer(tokenizer_dir: str = "tokenizer") -> Dict[str, Any]:
    tok_path = Path(tokenizer_dir)
    tok_json = tok_path / "tokenizer.json"

    if not tok_json.exists():
        raise FileNotFoundError(f"Tokenizer asset not found at {tok_json}")

    # 1. Run full evaluation
    eval_res = evaluate_tokenizer(str(tok_json))
    if not eval_res["unicode_nfc_test"]["passed"]:
        raise ValueError(
            f"Tokenizer freeze aborted: Unicode NFC lossless test failed! Errors: {eval_res['unicode_nfc_test']['errors']}"
        )

    # 2. Compute canonical hash
    tok_hash = compute_tokenizer_hash(tok_json)

    # 3. Create frozen metadata
    meta = {
        "status": "FROZEN",
        "version": "astra-tok-v0.1",
        "vocab_size": eval_res["vocab_size"],
        "tokenizer_hash": tok_hash,
        "normalization": "NFC",
        "byte_fallback": True,
        "frozen_at": datetime.now().isoformat(),
        "evaluation_summary": {
            dom: {"tokens_per_word": stats["tokens_per_word"], "bytes_per_token": stats["bytes_per_token"]}
            for dom, stats in eval_res["domains"].items()
        },
    }

    meta_file = tok_path / "tokenizer_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 4. Generate report
    generate_tokenizer_report(str(tok_json), output_dir=tokenizer_dir)

    print(f"[OK] Tokenizer successfully FROZEN at {tok_json} (SHA256={tok_hash})")
    return meta


if __name__ == "__main__":
    freeze_tokenizer()
