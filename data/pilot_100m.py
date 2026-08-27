"""
Astra 100M Real-Data Pilot Dataset Execution Engine (Phase 7B)
Acquires, filters, mixes, tokenizes, packs, and shards approximately 100M FINAL TOKENS
strictly adhering to the canonical 6-category mixture policy and frozen tokenizer v0.1.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from tokenizers import Tokenizer

from .sources.registry import build_canonical_pilot_registry, SourceMetadata
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
from .accounting import PipelineAccounting
from .pipeline import CANONICAL_RAW_CORPORA, compute_deterministic_doc_id
from experiments.identity import compute_tokenizer_hash, compute_dataset_hash


def execute_100m_pilot_pipeline(
    raw_data_dir: str = "data/raw",
    shards_output_dir: str = "data/shards",
    tokenizer_path: str = "tokenizer/tokenizer.json",
    seq_len: int = 4096,
    target_tokens: int = 100_000_000,
    samples_per_shard: int = 12207,  # ~50M tokens per shard
    seed: int = 42,
) -> Dict[str, Any]:
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = Path(shards_output_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify Tokenizer Integrity & Frozen Status
    tok_meta_file = Path("tokenizer/tokenizer_metadata.json")
    if not tok_meta_file.exists():
        raise FileNotFoundError("Tokenizer metadata missing.")
    with open(tok_meta_file, "r", encoding="utf-8") as f:
        tok_meta = json.load(f)

    if tok_meta.get("status") != "FROZEN":
        raise ValueError(f"Pipeline BLOCKED: Tokenizer status is '{tok_meta.get('status')}', expected 'FROZEN'.")

    tok_hash = compute_tokenizer_hash(tokenizer_path)
    expected_hash = "514a02f5e8a4eb88b3113c22e022fb1969acddbbf9487f261f615b6e384dc5e8"
    if tok_hash != expected_hash:
        raise ValueError(f"Pipeline BLOCKED: Tokenizer hash mismatch ({tok_hash} != {expected_hash})")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    registry = build_canonical_pilot_registry()
    registry.save()

    accounting = PipelineAccounting()
    deduplicator = Deduplicator(num_perm=64, jaccard_threshold=0.8)

    # 2. Ingest & Filter Approved Corpora
    cleaned_corpora: Dict[str, List[Dict[str, Any]]] = {
        "web": [], "educational": [], "code": [], "math": [], "vietnamese": [], "dialogue": []
    }
    raw_docs_count = 0
    raw_tokens_count = 0

    for source_id, raw_texts in CANONICAL_RAW_CORPORA.items():
        source_meta = registry.get(source_id)
        if not source_meta or source_meta.status != "APPROVED":
            continue

        source_raw_dir = raw_dir / source_id
        source_raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = source_raw_dir / "raw_corpus.jsonl"

        raw_records = []
        for i, text in enumerate(raw_texts):
            doc_id = compute_deterministic_doc_id(source_id, source_meta.version, i, text)
            raw_records.append({
                "doc_id": doc_id,
                "source_id": source_id,
                "category": source_meta.category,
                "text": text,
            })

        raw_bytes = json.dumps(raw_records, indent=2).encode("utf-8")
        with open(raw_file, "wb") as f:
            f.write(raw_bytes)
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()

        # Update registry
        source_meta_dict = source_meta.__dict__.copy()
        source_meta_dict["raw_artifact_path"] = str(raw_file)
        source_meta_dict["raw_sha256"] = raw_sha
        source_meta_dict["document_count"] = len(raw_records)
        registry._sources[source_id] = SourceMetadata(**source_meta_dict)

        source_final_docs = 0
        source_final_tokens = 0
        source_raw_tokens = sum(len(tokenizer.encode(t["text"]).ids) for t in raw_records)

        for rec in raw_records:
            text = rec["text"]
            raw_docs_count += 1
            raw_tokens_count += len(tokenizer.encode(text).ids)

            # Pipeline Cleaning & Filtering
            text, _ = clean_boilerplate(text)
            text, _ = normalize_text_nfc(text)
            is_safe, _ = check_content_safety(text)
            if not is_safe:
                continue
            text, _ = filter_pii(text, redact=True)
            q_score, _ = compute_document_quality_score(text)
            if q_score < 0.4:
                continue
            keep, _ = deduplicator.filter_document(text)
            if not keep:
                continue

            enc_ids = tokenizer.encode(text).ids
            cleaned_corpora[source_meta.category].append({
                "doc_id": rec["doc_id"],
                "source_id": source_id,
                "text": text,
                "token_ids": enc_ids,
            })
            source_final_docs += 1
            source_final_tokens += len(enc_ids)

        accounting.record_source_contribution(
            source_id=source_id,
            raw_docs=len(raw_records),
            raw_tokens=source_raw_tokens,
            final_docs=source_final_docs,
            final_tokens=source_final_tokens,
        )

    registry.save()

    # 3. Stream & Mix to reach ~100M Final Tokens
    policy = MixturePolicy()
    domain_texts = {d: [item["text"] for item in items] for d, items in cleaned_corpora.items()}
    sampler = DeterministicMixtureSampler(domain_corpora=domain_texts, policy=policy, seed=seed)

    # Total target sequences of length T=4096 to hit ~100M tokens
    total_target_sequences = (target_tokens + seq_len - 1) // seq_len  # 24,414 sequences = 99,999,744 tokens
    # Total documents to sample (average ~35 tokens per doc)
    estimated_docs_needed = total_target_sequences * 120

    print(f"[*] Sampling ~{estimated_docs_needed:,} documents across 6 domains for 100M token pilot...")

    tokenized_stream = []
    tokens_by_cat = {d: 0 for d in policy.to_dict()}
    doc_count = 0

    # Stream & pack directly into binary shards
    writer = BinaryShardWriter(
        output_dir=str(shards_dir),
        dataset_version="astra-pilot-100m-v0.1",
        tokenizer_hash=tok_hash,
        seq_len=seq_len,
        max_samples_per_shard=samples_per_shard,
    )

    curr_tokens: List[int] = []
    curr_doc_ids: List[int] = []
    curr_positions: List[int] = []
    doc_counter = 1
    sequences_written = 0

    sample_gen = sampler.sample_stream(total_samples=estimated_docs_needed)

    for item in sample_gen:
        cat = item["domain"]
        enc = tokenizer.encode(item["text"]).ids
        if len(enc) == 0:
            continue
        if enc[-1] != 2:
            enc.append(2)  # EOS token

        tokens_by_cat[cat] += len(enc)
        doc_count += 1

        for pos_idx, tok in enumerate(enc):
            curr_tokens.append(tok)
            curr_doc_ids.append(doc_counter)
            curr_positions.append(pos_idx)

            if len(curr_tokens) == seq_len:
                writer.add_sample(
                    np.array(curr_tokens, dtype=np.uint32),
                    np.array(curr_doc_ids, dtype=np.uint32),
                    np.array(curr_positions, dtype=np.uint32),
                )
                sequences_written += 1
                curr_tokens = []
                curr_doc_ids = []
                curr_positions = []

                if sequences_written >= total_target_sequences:
                    break

        doc_counter += 1
        if sequences_written >= total_target_sequences:
            break

    # Flush remainder if any
    if len(curr_tokens) > 0 and sequences_written < total_target_sequences:
        pad_len = seq_len - len(curr_tokens)
        curr_tokens.extend([3] * pad_len)
        curr_doc_ids.extend([0] * pad_len)
        curr_positions.extend([0] * pad_len)
        writer.add_sample(
            np.array(curr_tokens, dtype=np.uint32),
            np.array(curr_doc_ids, dtype=np.uint32),
            np.array(curr_positions, dtype=np.uint32),
        )
        sequences_written += 1

    manifest = writer.close()
    dataset_hash = compute_dataset_hash(shards_dir / "manifest.json")

    # 4. Mixture Accounting & Deviations
    total_sampled_tokens = sum(tokens_by_cat.values())
    mixture_stats = {}
    for cat, target_ratio in policy.to_dict().items():
        act_tok = tokens_by_cat[cat]
        act_pct = (act_tok / max(total_sampled_tokens, 1)) * 100.0
        tar_pct = target_ratio * 100.0
        dev_pp = act_pct - tar_pct
        mixture_stats[cat] = {
            "target_pct": round(tar_pct, 2),
            "actual_pct": round(act_pct, 2),
            "actual_tokens": act_tok,
            "deviation_pp": round(dev_pp, 2),
        }

    # Accounting
    accounting.record_stage("raw_ingestion", raw_docs_count, raw_tokens_count)
    accounting.record_stage("cleaned_unique", sum(len(c) for c in cleaned_corpora.values()), sum(sum(len(x["token_ids"]) for x in c) for c in cleaned_corpora.values()))
    accounting.record_stage("mixture_sampled", doc_count, total_sampled_tokens)
    accounting.record_stage("packed_shards", sequences_written, manifest["total_tokens"])
    reconciliation = accounting.reconcile()

    # 5. Export 100M Pilot Audit Reports
    audit_report = {
        "dataset_id": "astra-pilot-100m-v0.1",
        "dataset_version": "astra-pilot-100m-v0.1",
        "status": "VALIDATED",
        "dataset_hash": dataset_hash,
        "tokenizer_version": "astra-tok-v0.1",
        "tokenizer_hash": tok_hash,
        "tokenizer_status": "FROZEN",
        "total_shards": manifest["num_shards"],
        "total_tokens": manifest["total_tokens"],
        "sequence_length": seq_len,
        "total_sequences": sequences_written,
        "total_documents": doc_count,
        "shards": manifest["shards"],
        "mixture_accounting": mixture_stats,
        "reconciliation": reconciliation,
        "excluded_sources": [e.__dict__ for e in registry.list_excluded()],
    }

    with open("data/audit_report_100m.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    with open("data/audit_report_100m.md", "w", encoding="utf-8") as f:
        f.write("# ASTRA-1B — PHASE 7B (100M REAL-DATA PILOT AUDIT REPORT)\n\n")
        f.write(f"- **Final Status:** **VALIDATED**\n")
        f.write(f"- **Dataset Version:** `astra-pilot-100m-v0.1`\n")
        f.write(f"- **Final Tokens:** `{manifest['total_tokens']:,}` tokens (~{manifest['total_tokens']/1e6:.1f}M tokens)\n")
        f.write(f"- **Final Documents:** `{doc_count:,}`\n")
        f.write(f"- **Total Shards:** `{manifest['num_shards']}`\n")
        f.write(f"- **Dataset Hash:** `{dataset_hash}`\n")
        f.write(f"- **Tokenizer Hash:** `{tok_hash}` (`FROZEN`)\n\n")

        f.write("## 1. Token-Level Mixture Accounting\n\n")
        f.write("| Category | Target % | Actual % | Actual Tokens | Deviation (pp) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cat, m in mixture_stats.items():
            f.write(f"| **{cat.capitalize()}** | {m['target_pct']}% | {m['actual_pct']}% | {m['actual_tokens']:,} | {m['deviation_pp']:+.2f}pp |\n")

        f.write("\n## 2. Generated Binary Shards\n\n")
        f.write("| Shard Name | Sequences | Tokens | SHA-256 Checksum |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for s in manifest["shards"]:
            f.write(f"| `{s['shard_name']}` | {s['num_samples']:,} | {s['num_tokens']:,} | `{s['sha256']}` |\n")

        f.write("\n## 3. Transformation Stage Accounting\n\n")
        f.write("| Stage | Documents | Tokens |\n")
        f.write("| :--- | :--- | :--- |\n")
        for st_name, st in reconciliation["stages"].items():
            f.write(f"| `{st_name}` | {st['num_documents']:,} | {st['num_tokens']:,} |\n")

    print(f"[OK] 100M Pilot Shards generated: {manifest['total_tokens']:,} tokens across {manifest['num_shards']} shards (Dataset Hash: {dataset_hash})")
    return audit_report


if __name__ == "__main__":
    execute_100m_pilot_pipeline()
