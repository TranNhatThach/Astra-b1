"""
Astra End-to-End Data Pipeline & Pilot Shard Generator (Phase 7)
Orchestrates:
  Source Ingestion -> Clean/Normalize -> PII & Safety Filter -> Quality Filter ->
  MinHash Dedup -> Deterministic Mixture -> Tokenization -> Packing -> Binary Shards -> Audit
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from tokenizers import Tokenizer

from .clean.boilerplate import clean_boilerplate
from .clean.normalize import normalize_text_nfc, verify_vietnamese_diacritics
from .filter.pii import filter_pii
from .filter.safety import check_content_safety
from .filter.quality import compute_document_quality_score
from .filter.dedup import Deduplicator
from .mix.policies import MixturePolicy
from .mix.sampler import DeterministicMixtureSampler
from .pack.pack import pack_documents
from .shard.shard_writer import BinaryShardWriter
from experiments.identity import compute_tokenizer_hash, compute_dataset_hash


# Representative multi-domain corpora for pilot dataset generation
SEED_CORPORA = {
    "web": [
        "The development of scalable large language models requires deep hardware awareness and numerical stability.",
        "Modern deep learning architectures increasingly rely on hybrid state-space and linear attention layers.",
        "Distributed gradient all-reduce algorithms allow training billion-parameter neural networks across clusters.",
        "Operating systems manage memory paging, device interrupts, and scheduling threads efficiently.",
        "Artificial intelligence systems require robust data governance, licensing audits, and reproducible benchmarks.",
    ] * 40,
    "educational": [
        "In linear algebra, an associative scan computes cumulative operations in logarithmic depth across sequence elements.",
        "Backpropagation through time in recurrent neural networks calculates partial derivatives via the chain rule.",
        "Gradient checkpointing trades computation for GPU memory by recalculating activations during the backward pass.",
        "Grouped-Query Attention (GQA) reduces key-value cache memory bandwidth while preserving multi-head representational power.",
    ] * 40,
    "code": [
        "def chunkwise_scan(q, k, v, retention, update, state):\n    diff = v - state @ k\n    state = retention * state + update * (diff @ k.T)\n    return state @ q, state\n",
        "class RMSNorm(nn.Module):\n    def __init__(self, d_model, eps=1e-6):\n        super().__init__()\n        self.weight = nn.Parameter(torch.ones(d_model))\n",
        "int main() {\n    std::cout << \"Astra-1B High Performance Linear Attention\" << std::endl;\n    return 0;\n}\n",
        "fn forward_pass(inputs: &[f32], weights: &[f32]) -> Vec<f32> {\n    inputs.iter().zip(weights.iter()).map(|(x, w)| x * w).collect()\n}\n",
    ] * 40,
    "math": [
        "Theorem: The stationary state matrix S_t converges when the diagonal decay operator satisfies ||D_t|| < 1.",
        "Equation: S_t = D_t S_{t-1} (I - u_t k_t k_t^T) + u_t v_t k_t^T, \\quad y_t = S_t q_t.",
        "Integral: \\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}, \\quad \\nabla_W \\mathcal{L} = \\sum_{t=1}^T \\frac{\\partial \\mathcal{L}}{\\partial y_t} \\frac{\\partial y_t}{\\partial W}.",
    ] * 40,
    "vietnamese": [
        "Astra-1B là mô hình ngôn ngữ lớn lai ghép đầu tiên kết hợp Gated DeltaNet và GQA Attention cho tiếng Việt.",
        "Hệ thống ngữ pháp tiếng Việt phong phú với các thanh điệu huyền, sắc, hỏi, ngã, nặng cùng nguyên âm đặc trưng.",
        "Hà Nội và Thành phố Hồ Chí Minh là hai trung tâm kinh tế, văn hóa, khoa học và công nghệ lớn của Việt Nam.",
        "Nghiên cứu trí tuệ nhân tạo cần đảm bảo tính minh bạch, khả năng tái lập và kiểm toán dữ liệu nghiêm ngặt.",
    ] * 40,
    "dialogue": [
        "User: Giải thích kiến trúc lai ghép Gated DeltaNet?\nAssistant: Gated DeltaNet sử dụng trạng thái hồi quy O(1) kết hợp cơ chế gated update và decay.",
        "User: How do you prevent loss spikes in pretraining?\nAssistant: By using RMSNorm pre-normalization, output gates, and gradient clipping at 1.0.",
    ] * 40,
}


def run_data_pipeline(
    tokenizer_path: str = "tokenizer/tokenizer.json",
    shards_output_dir: str = "data/shards",
    seq_len: int = 4096,
    max_samples_per_shard: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    shards_dir = Path(shards_output_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean & Filter
    deduplicator = Deduplicator()
    cleaned_corpora: Dict[str, List[str]] = {}
    audit_stats = {
        "raw_documents": 0,
        "kept_documents": 0,
        "rejected_pii": 0,
        "rejected_safety": 0,
        "rejected_quality": 0,
        "rejected_duplicate": 0,
        "domain_counts": {},
    }

    for domain, raw_docs in SEED_CORPORA.items():
        cleaned_corpora[domain] = []
        for text in raw_docs:
            audit_stats["raw_documents"] += 1

            # A. Clean boilerplate
            text, _ = clean_boilerplate(text)

            # B. NFC Normalization & Diacritics check
            text, _ = normalize_text_nfc(text)

            # C. Safety filter
            is_safe, _ = check_content_safety(text)
            if not is_safe:
                audit_stats["rejected_safety"] += 1
                continue

            # D. PII Filter
            text, pii_counts = filter_pii(text, redact=True)

            # E. Quality Score
            q_score, _ = compute_document_quality_score(text)
            if q_score < 0.4:
                audit_stats["rejected_quality"] += 1
                continue

            # F. Deduplication
            keep, _ = deduplicator.filter_document(text)
            if not keep:
                audit_stats["rejected_duplicate"] += 1
                continue

            cleaned_corpora[domain].append(text)
            audit_stats["kept_documents"] += 1

        audit_stats["domain_counts"][domain] = len(cleaned_corpora[domain])

    # 2. Deterministic Mixture Sampling
    policy = MixturePolicy()
    sampler = DeterministicMixtureSampler(domain_corpora=cleaned_corpora, policy=policy, seed=seed)
    sampled_stream = list(sampler.sample_stream(total_samples=100))

    # 3. Tokenization & Packing
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tok_hash = compute_tokenizer_hash(tokenizer_path)

    tokenized_docs = []
    for item in sampled_stream:
        enc = tokenizer.encode(item["text"])
        tokenized_docs.append(enc.ids)

    packed_samples = list(pack_documents(tokenized_docs, seq_len=seq_len, eos_token_id=2, pad_token_id=3))

    # 4. Binary Shard Writing
    writer = BinaryShardWriter(
        output_dir=str(shards_dir),
        dataset_version="astra-data-v0.1",
        tokenizer_hash=tok_hash,
        seq_len=seq_len,
        max_samples_per_shard=max_samples_per_shard,
    )
    for sample in packed_samples:
        writer.add_sample(sample["input_ids"], sample["doc_ids"], sample["position_ids"])

    manifest = writer.close()
    dataset_hash = compute_dataset_hash(shards_dir / "manifest.json")

    # 5. Export Audit Reports
    audit_report = {
        "dataset_version": "astra-data-v0.1",
        "dataset_hash": dataset_hash,
        "tokenizer_hash": tok_hash,
        "num_shards": manifest["num_shards"],
        "total_tokens": manifest["total_tokens"],
        "sequence_length": seq_len,
        "filtering_statistics": audit_stats,
        "mixture_policy": policy.to_dict(),
    }

    report_json = Path("data/audit_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    report_md = Path("data/audit_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Astra-1B Pilot Dataset Audit Report (Phase 7)\n\n")
        f.write(f"- **Dataset Version:** `{manifest['dataset_version']}`\n")
        f.write(f"- **Dataset Hash:** `{dataset_hash}`\n")
        f.write(f"- **Tokenizer Hash:** `{tok_hash}`\n")
        f.write(f"- **Total Shards:** `{manifest['num_shards']}` | **Total Tokens:** `{manifest['total_tokens']:,}`\n")
        f.write(f"- **Sequence Length:** `{seq_len}`\n\n")
        f.write("## 1. Document Curation & Filtering Statistics\n\n")
        f.write(f"- Raw Documents Ingested: {audit_stats['raw_documents']:,}\n")
        f.write(f"- Curated Documents Kept: {audit_stats['kept_documents']:,}\n")
        f.write(f"- Duplicates Filtered: {audit_stats['rejected_duplicate']:,}\n")
        f.write(f"- Quality Rejected: {audit_stats['rejected_quality']:,}\n\n")
        f.write("## 2. Domain Distribution\n\n")
        f.write("| Domain | Unique Docs | Target Mixture Ratio |\n")
        f.write("| :--- | :--- | :--- |\n")
        for dom, count in audit_stats["domain_counts"].items():
            f.write(f"| {dom.capitalize()} | {count} | {policy.to_dict().get(dom, 0.0)*100:.1f}% |\n")

    print(f"[OK] Pilot dataset successfully generated and audited: {manifest['total_tokens']:,} tokens across {manifest['num_shards']} shards")
    return audit_report


if __name__ == "__main__":
    run_data_pipeline()
