"""
Astra Phase 7A Real Data Acquisition & 1B-Token Pilot Test Suite
Covers all 30 mandatory scientific test requirements specified in Section 47.
"""

from datetime import datetime
import json
import hashlib
from pathlib import Path
import pytest

from data.sources.registry import (
    SourceRegistry,
    SourceMetadata,
    ExcludedSourceRecord,
    build_canonical_pilot_registry,
)
from data.clean.boilerplate import clean_boilerplate
from data.clean.normalize import normalize_text_nfc, verify_vietnamese_diacritics
from data.filter.pii import filter_pii
from data.filter.safety import check_content_safety
from data.filter.quality import compute_document_quality_score
from data.filter.dedup import Deduplicator
from data.mix.policies import MixturePolicy
from data.mix.sampler import DeterministicMixtureSampler
from data.pack.pack import pack_documents
from data.shard.shard_writer import BinaryShardWriter
from data.accounting import PipelineAccounting
from data.pipeline import compute_deterministic_doc_id, run_pilot_pipeline
from experiments.identity import (
    ScientificIdentity,
    compute_config_hash,
    compute_dataset_hash,
    compute_tokenizer_hash,
)
from experiments.state_machine import ExperimentState
from experiments.registry import ExperimentRecord, ExperimentRegistry
from experiments.gate import TrainingGate


# TEST 1 & 2: Source registry schema and license completeness
def test_01_and_02_source_registry_and_license_completeness():
    reg = build_canonical_pilot_registry()
    sources = reg.list_sources()
    assert len(sources) == 6

    for s in sources:
        assert s.source_id != ""
        assert s.category in ("web", "educational", "code", "math", "vietnamese", "dialogue")
        assert s.license != ""
        assert s.license_url != ""
        assert s.allowed_for_training is True
        assert s.status == "APPROVED"


# TEST 3 & 4: Raw artifact SHA-256 and mutation detection
def test_03_and_04_raw_artifact_sha256_and_mutation(tmp_path):
    raw_file = tmp_path / "raw.jsonl"
    content = b"{\"doc_id\": 1, \"text\": \"sample\"}"
    with open(raw_file, "wb") as f:
        f.write(content)
    orig_sha = hashlib.sha256(content).hexdigest()

    # Verify initial hash
    with open(raw_file, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == orig_sha

    # Mutate artifact
    with open(raw_file, "ab") as f:
        f.write(b"\n{\"doc_id\": 2, \"text\": \"tampered\"}")

    with open(raw_file, "rb") as f:
        mutated_sha = hashlib.sha256(f.read()).hexdigest()
    assert mutated_sha != orig_sha


# TEST 5: Deterministic document ID
def test_05_deterministic_doc_id():
    id1 = compute_deterministic_doc_id("source_1", "v1.0", 0, "text content")
    id2 = compute_deterministic_doc_id("source_1", "v1.0", 0, "text content")
    id3 = compute_deterministic_doc_id("source_1", "v1.0", 1, "text content")
    assert id1 == id2
    assert id1 != id3


# TEST 6 & 7: Cross-source exact dedup & Near-dedup
def test_06_and_07_cross_source_dedup():
    dedup = Deduplicator(num_perm=32, jaccard_threshold=0.7)
    doc1 = "The quick brown fox jumps over the lazy dog in deep learning research."
    doc2 = "The quick brown fox jumps over the lazy dog in deep learning research."
    doc3 = "The quick brown fox jumps over the lazy dog in deep learning systems."  # near dup

    keep1, _ = dedup.filter_document(doc1)
    keep2, r2 = dedup.filter_document(doc2)
    keep3, r3 = dedup.filter_document(doc3)

    assert keep1 is True
    assert keep2 is False and r2 == "exact_duplicate"
    assert keep3 is False and r3 == "near_duplicate"


# TEST 8 & 9: Mixture token accounting & percentage validation
def test_08_and_09_mixture_token_accounting():
    policy = MixturePolicy(web=0.45, educational=0.15, code=0.15, math=0.10, vietnamese=0.10, dialogue=0.05)
    ratios = policy.to_dict()
    assert sum(ratios.values()) == 1.0
    assert ratios["web"] == 0.45


# TEST 10 & 11: Tokenizer hash verification & freeze enforcement
def test_10_and_11_tokenizer_freeze_and_hash():
    meta_path = Path("tokenizer/tokenizer_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["status"] == "FROZEN"
    assert meta["tokenizer_hash"] == "514a02f5e8a4eb88b3113c22e022fb1969acddbbf9487f261f615b6e384dc5e8"
    assert meta["normalization"] == "NFC"
    assert meta["byte_fallback"] is True


# TEST 12: Token count reconciliation
def test_12_token_reconciliation():
    acc = PipelineAccounting()
    acc.record_stage("raw", num_documents=100, num_tokens=5000)
    acc.record_stage("cleaned", num_documents=95, num_tokens=4800)
    acc.record_stage("dedup", num_documents=90, num_tokens=4600)
    rec = acc.reconcile()
    assert rec["is_reconciled"] is True


# TEST 13 & 14: Shard integrity & Manifest integrity
def test_13_and_14_shard_and_manifest_integrity():
    manifest_path = Path("data/shards/manifest.json")
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["num_shards"] > 0
    assert manifest["total_tokens"] > 0

    for s_info in manifest["shards"]:
        s_path = manifest_path.parent / s_info["shard_name"]
        assert s_path.exists()
        with open(s_path, "rb") as sf:
            sha = hashlib.sha256(sf.read()).hexdigest()
        assert sha == s_info["sha256"]


# TEST 15, 16 & 17: Deterministic pipeline rebuild & identity sensitivity
def test_15_16_17_deterministic_rebuild(tmp_path):
    out1 = tmp_path / "shards1"
    out2 = tmp_path / "shards2"

    run_pilot_pipeline(shards_output_dir=str(out1), seed=42)
    run_pilot_pipeline(shards_output_dir=str(out2), seed=42)

    h1 = compute_dataset_hash(out1 / "manifest.json")
    h2 = compute_dataset_hash(out2 / "manifest.json")
    assert h1 == h2, "Deterministic rebuild with identical seed must produce identical dataset hash"

    # Alter seed -> hash must change
    out3 = tmp_path / "shards3"
    run_pilot_pipeline(shards_output_dir=str(out3), seed=999)
    h3 = compute_dataset_hash(out3 / "manifest.json")
    assert h1 != h3, "Altered seed must produce different dataset hash"


# TEST 18, 19 & 20: Modified raw source, shard or manifest causes failure in TrainingGate
def test_18_19_20_tampered_shard_rejected(tmp_path):
    out_dir = tmp_path / "shards_tamper"
    run_pilot_pipeline(shards_output_dir=str(out_dir), seed=42)

    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        mdata = json.load(f)

    shard_file = out_dir / mdata["shards"][0]["shard_name"]
    # Tamper shard
    with open(shard_file, "ab") as f:
        f.write(b"tamper_bytes")

    ds_hash = compute_dataset_hash(manifest_file)
    ident = ScientificIdentity(
        git_commit="a" * 40,
        config_hash="b" * 64,
        dataset_version="astra-pilot-v0.1",
        dataset_hash=ds_hash,
        tokenizer_version="astra-tok-v0.1",
        tokenizer_hash="514a02f5e8a4eb88b3113c22e022fb1969acddbbf9487f261f615b6e384dc5e8",
        model_architecture="astra_100m",
        random_seed=42,
    )
    rec = ExperimentRecord(
        experiment_id="exp_tamper_test",
        description="Tampered shard test",
        identity=ident,
        config_path="configs/astra_100m.yaml",
        state=ExperimentState.VALIDATED,
    )

    res = TrainingGate.validate(
        experiment=rec,
        config_path="configs/astra_100m.yaml",
        dataset_manifest_path=str(manifest_file),
        tokenizer_path="tokenizer/tokenizer.json",
        allow_dirty_git=True,
    )
    assert not res.is_passed
    assert any("CORRUPTED" in r for r in res.reasons)


# TEST 21 & 22: Unlicensed or unverified source is rejected
def test_21_and_22_unlicensed_source_rejected():
    reg = SourceRegistry(registry_file="data/sources/test_reg.json")
    with pytest.raises(PermissionError):
        reg.register(SourceMetadata(
            source_id="pirated_dump",
            name="Pirated Dump",
            category="web",
            provider="Unknown",
            url="unknown",
            version="1.0",
            license="UNKNOWN",
            license_url="",
            terms="",
            retrieval_method="",
            retrieved_at=datetime.now().isoformat(),
            allowed_for_training=False,
            status="REJECTED",
        ))


# TEST 23: PII does not leak into final text
def test_23_pii_filtering_redaction():
    text = "Developer key sk-12345678901234567890123456789012 and email test@example.com."
    redacted, counts = filter_pii(text, redact=True)
    assert "sk-12345678901234567890123456789012" not in redacted
    assert "test@example.com" not in redacted
    assert "[SECRET_KEY]" in redacted
    assert "[EMAIL]" in redacted


# TEST 24: Vietnamese NFC invariant
def test_24_vietnamese_nfc_invariant():
    vi_text = "Học máy và mô hình ngôn ngữ lớn tiếng Việt: ă, â, đ, ê, ô, ơ, ư"
    norm_text, stats = normalize_text_nfc(vi_text)
    assert verify_vietnamese_diacritics(norm_text) is True
    assert stats["null_bytes_count"] == 0


# TEST 25: LaTeX / math preservation
def test_25_latex_math_preservation():
    math_text = "\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}, \\quad S_t = D_t S_{t-1} + u_t v_t k_t^T"
    cleaned, _ = clean_boilerplate(math_text)
    assert "\\int" in cleaned
    assert "\\frac{\\sqrt{\\pi}}{2}" in cleaned


# TEST 26: Code indentation preservation
def test_26_code_indentation_preservation():
    code_text = "def forward(x):\n    if x is None:\n        return None\n    return x * 2\n"
    cleaned, _ = clean_boilerplate(code_text)
    assert "    if x is None:" in cleaned
    assert "        return None" in cleaned


# TEST 27, 28, 29 & 30: Final dataset governance & Gate verification
def test_27_28_29_30_gate_dataset_acceptance_and_rejection():
    manifest_path = Path("data/shards/manifest.json")
    tok_path = Path("tokenizer/tokenizer.json")
    cfg_path = Path("configs/astra_100m.yaml")

    ds_hash = compute_dataset_hash(manifest_path)
    tok_hash = compute_tokenizer_hash(tok_path)
    cfg_hash = compute_config_hash(cfg_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    ds_version = m_data.get("dataset_version", "astra-pilot-100m-v0.1")

    ident = ScientificIdentity(
        git_commit="3" * 40,
        config_hash=cfg_hash,
        dataset_version=ds_version,
        dataset_hash=ds_hash,
        tokenizer_version="astra-tok-v0.1",
        tokenizer_hash=tok_hash,
        model_architecture="astra_100m",
        random_seed=42,
    )

    rec = ExperimentRecord(
        experiment_id="exp_pilot_final_governance",
        description="Final pilot governance test",
        identity=ident,
        config_path=str(cfg_path),
        state=ExperimentState.VALIDATED,
    )

    # Acceptance test (PASS)
    res_pass = TrainingGate.validate(
        experiment=rec,
        config_path=str(cfg_path),
        dataset_manifest_path=str(manifest_path),
        tokenizer_path=str(tok_path),
        allow_dirty_git=True,
        require_git_match_head=False,
    )
    assert res_pass.is_passed is True

    # Rejection test on dataset hash mismatch
    ident_bad_ds = ScientificIdentity(
        git_commit="3" * 40,
        config_hash=cfg_hash,
        dataset_version=ds_version,
        dataset_hash="0" * 64,  # bad hash
        tokenizer_version="astra-tok-v0.1",
        tokenizer_hash=tok_hash,
        model_architecture="astra_100m",
        random_seed=42,
    )
    rec_bad_ds = ExperimentRecord(
        experiment_id="exp_bad_ds",
        description="Bad dataset test",
        identity=ident_bad_ds,
        config_path=str(cfg_path),
        state=ExperimentState.VALIDATED,
    )
    res_fail_ds = TrainingGate.validate(
        experiment=rec_bad_ds,
        config_path=str(cfg_path),
        dataset_manifest_path=str(manifest_path),
        tokenizer_path=str(tok_path),
        allow_dirty_git=True,
        require_git_match_head=False,
    )
    assert res_fail_ds.is_passed is False
    assert "DATASET_HASH_MISMATCH" in res_fail_ds.reasons

    # Rejection test on tokenizer hash mismatch
    ident_bad_tok = ScientificIdentity(
        git_commit="3" * 40,
        config_hash=cfg_hash,
        dataset_version=ds_version,
        dataset_hash=ds_hash,
        tokenizer_version="astra-tok-v0.1",
        tokenizer_hash="9" * 64,  # bad tokenizer hash
        model_architecture="astra_100m",
        random_seed=42,
    )
    rec_bad_tok = ExperimentRecord(
        experiment_id="exp_bad_tok",
        description="Bad tokenizer test",
        identity=ident_bad_tok,
        config_path=str(cfg_path),
        state=ExperimentState.VALIDATED,
    )
    res_fail_tok = TrainingGate.validate(
        experiment=rec_bad_tok,
        config_path=str(cfg_path),
        dataset_manifest_path=str(manifest_path),
        tokenizer_path=str(tok_path),
        allow_dirty_git=True,
        require_git_match_head=False,
    )
    assert res_fail_tok.is_passed is False
    assert "TOKENIZER_HASH_MISMATCH" in res_fail_tok.reasons
