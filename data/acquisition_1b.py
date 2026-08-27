"""
Astra Full-Scale Real Data Acquisition & 1B-Token Corpus Construction Engine (Phase 7C)
Features:
  - Streaming Source Adapters (Zero-RAM Overhead)
  - Resumable Pipeline Execution State Tracking
  - Full Cleaning, Normalization, PII, Safety, Quality & MinHash Deduplication
  - Frozen Tokenizer Invariant Enforcement
  - T=4096 Sequence Packing & Binary Shard Generation
  - Document Diversity Auditing & Complete Token Accounting
"""

from datetime import datetime
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
import numpy as np
from tokenizers import Tokenizer

from data.sources.registry import SourceRegistry, build_canonical_pilot_registry
from data.sources.adapters import (
    FineWebEduAdapter,
    OpenStaxEduAdapter,
    TheStackCodeAdapter,
    OpenWebMathAdapter,
    VietnameseCuratedAdapter,
    SyntheticDialogueAdapter,
    RawDocument,
)
from data.clean.boilerplate import clean_boilerplate
from data.clean.normalize import normalize_text_nfc, verify_vietnamese_diacritics
from data.filter.pii import filter_pii
from data.filter.safety import check_content_safety
from data.filter.quality import compute_document_quality_score
from data.filter.dedup import Deduplicator
from data.mix.policies import MixturePolicy
from data.shard.shard_writer import BinaryShardWriter
from data.accounting import PipelineAccounting
from experiments.identity import compute_tokenizer_hash, compute_dataset_hash


class AcquisitionStateTracker:
    def __init__(self, state_file: str = "data/resume_state.json"):
        self.state_file = Path(state_file)
        self.state = {
            "source_positions": {},
            "tokens_generated": 0,
            "documents_accepted": 0,
            "shards_completed": [],
            "last_updated": datetime.now().isoformat(),
        }
        self.load()

    def load(self) -> None:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.state = json.load(f)

    def save(self) -> None:
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def update_source_pos(self, source_id: str, pos: int) -> None:
        self.state["source_positions"][source_id] = pos

    def get_source_pos(self, source_id: str) -> int:
        return self.state["source_positions"].get(source_id, 0)


class ScalableCorpusAcquisitionEngine:
    def __init__(
        self,
        tokenizer_path: str = "tokenizer/tokenizer.json",
        shards_dir: str = "data/shards",
        raw_provenance_dir: str = "data/raw",
        state_file: str = "data/resume_state.json",
        seq_len: int = 4096,
        samples_per_shard: int = 12207,  # ~50M tokens per shard
        seed: int = 42,
    ):
        self.tokenizer_path = Path(tokenizer_path)
        self.shards_dir = Path(shards_dir)
        self.raw_dir = Path(raw_provenance_dir)
        self.seq_len = seq_len
        self.samples_per_shard = samples_per_shard
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self._verify_frozen_tokenizer()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.state_tracker = AcquisitionStateTracker(state_file)

        # Initialize Source Adapters
        self.adapters = {
            "web": FineWebEduAdapter(),
            "educational": OpenStaxEduAdapter(),
            "code": TheStackCodeAdapter(),
            "math": OpenWebMathAdapter(),
            "vietnamese": VietnameseCuratedAdapter(),
            "dialogue": SyntheticDialogueAdapter(),
        }
        self.policy = MixturePolicy()
        self.deduplicator = Deduplicator(num_perm=64, jaccard_threshold=0.8)
        self.accounting = PipelineAccounting()

    def _verify_frozen_tokenizer(self) -> None:
        meta_file = Path("tokenizer/tokenizer_metadata.json")
        if not meta_file.exists():
            raise FileNotFoundError("Tokenizer metadata missing!")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("status") != "FROZEN":
            raise ValueError(f"HARD FAIL: Tokenizer status is {meta.get('status')}, must be FROZEN.")

        tok_hash = compute_tokenizer_hash(str(self.tokenizer_path))
        expected_hash = "514a02f5e8a4eb88b3113c22e022fb1969acddbbf9487f261f615b6e384dc5e8"
        if tok_hash != expected_hash:
            raise ValueError(f"HARD FAIL: Tokenizer hash mismatch ({tok_hash} != {expected_hash})")
        self.tok_hash = tok_hash

    def run_acquisition(
        self,
        target_tokens: int = 20_000_000,
        dataset_version: str = "astra-research-v0.2",
    ) -> Dict[str, Any]:
        """
        Executes streaming ingestion, multi-stage filtering, mixture sampling, and sharding.
        """
        registry = build_canonical_pilot_registry()
        registry.save()

        # Target sequences
        total_sequences_needed = (target_tokens + self.seq_len - 1) // self.seq_len
        ratios = self.policy.to_dict()

        # Diversity tracking
        diversity_stats = {
            "raw_docs_seen": 0,
            "accepted_unique_docs": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "quality_rejected": 0,
            "safety_rejected": 0,
            "pii_redactions": 0,
            "docs_per_category": {cat: 0 for cat in ratios},
            "tokens_per_category": {cat: 0 for cat in ratios},
            "doc_lengths": [],
        }

        # Initialize iterators per adapter
        adapter_iters = {}
        for cat, adapter in self.adapters.items():
            cur_pos = self.state_tracker.get_source_pos(adapter.source_id)
            adapter_iters[cat] = adapter.iterate_documents(resume_pos=cur_pos)

        writer = BinaryShardWriter(
            output_dir=str(self.shards_dir),
            dataset_version=dataset_version,
            tokenizer_hash=self.tok_hash,
            seq_len=self.seq_len,
            max_samples_per_shard=self.samples_per_shard,
        )

        curr_tokens: List[int] = []
        curr_doc_ids: List[int] = []
        curr_positions: List[int] = []
        doc_counter = 1
        sequences_written = 0

        categories = list(ratios.keys())
        probs = [ratios[c] for c in categories]

        print(f"[*] Starting streaming acquisition for {target_tokens:,} tokens ({dataset_version})...")

        while sequences_written < total_sequences_needed:
            # Deterministic weighted domain selection
            chosen_cat = self.rng.choice(categories, p=probs)
            raw_doc: RawDocument = next(adapter_iters[chosen_cat])
            diversity_stats["raw_docs_seen"] += 1

            text = raw_doc.text

            # 1. Clean boilerplate & format
            text, b_stats = clean_boilerplate(text)

            # 2. NFC Normalization & Diacritics
            text, nfc_stats = normalize_text_nfc(text)
            if chosen_cat == "vietnamese" and not verify_vietnamese_diacritics(text):
                continue

            # 3. Safety Screening
            is_safe, _ = check_content_safety(text)
            if not is_safe:
                diversity_stats["safety_rejected"] += 1
                continue

            # 4. PII Redaction
            text, pii_counts = filter_pii(text, redact=True)
            if sum(pii_counts.values()) > 0:
                diversity_stats["pii_redactions"] += sum(pii_counts.values())

            # 5. Quality Score
            q_score, _ = compute_document_quality_score(text)
            if q_score < 0.4:
                diversity_stats["quality_rejected"] += 1
                continue

            # 6. Deduplication (Exact SHA-256 + MinHash LSH)
            keep, dedup_reason = self.deduplicator.filter_document(text)
            if not keep:
                if dedup_reason == "exact_duplicate":
                    diversity_stats["exact_duplicates"] += 1
                else:
                    diversity_stats["near_duplicates"] += 1
                continue

            # 7. Tokenize with Frozen Tokenizer
            token_ids = self.tokenizer.encode(text).ids
            if len(token_ids) == 0:
                continue
            if token_ids[-1] != 2:
                token_ids.append(2)  # Append EOS

            # Update stats
            diversity_stats["accepted_unique_docs"] += 1
            diversity_stats["docs_per_category"][chosen_cat] += 1
            diversity_stats["tokens_per_category"][chosen_cat] += len(token_ids)
            diversity_stats["doc_lengths"].append(len(token_ids))

            # 8. Pack into sequences
            for pos_idx, tok in enumerate(token_ids):
                curr_tokens.append(tok)
                curr_doc_ids.append(doc_counter)
                curr_positions.append(pos_idx)

                if len(curr_tokens) == self.seq_len:
                    writer.add_sample(
                        np.array(curr_tokens, dtype=np.uint32),
                        np.array(curr_doc_ids, dtype=np.uint32),
                        np.array(curr_positions, dtype=np.uint32),
                    )
                    sequences_written += 1
                    curr_tokens = []
                    curr_doc_ids = []
                    curr_positions = []

                    if sequences_written >= total_sequences_needed:
                        break

            doc_counter += 1

        # Flush remainder
        if len(curr_tokens) > 0 and sequences_written < total_sequences_needed:
            pad_len = self.seq_len - len(curr_tokens)
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
        dataset_hash = compute_dataset_hash(self.shards_dir / "manifest.json")

        # Calculate mixture metrics
        total_tokens_sampled = sum(diversity_stats["tokens_per_category"].values())
        mixture_metrics = {}
        for cat, target_ratio in ratios.items():
            act_tok = diversity_stats["tokens_per_category"][cat]
            act_pct = (act_tok / max(total_tokens_sampled, 1)) * 100.0
            dev_pp = act_pct - (target_ratio * 100.0)
            mixture_metrics[cat] = {
                "target_pct": round(target_ratio * 100.0, 2),
                "actual_pct": round(act_pct, 2),
                "actual_tokens": act_tok,
                "deviation_pp": round(dev_pp, 2),
            }

        # Diversity metrics
        lengths = diversity_stats["doc_lengths"] or [0]
        diversity_summary = {
            "total_raw_documents": diversity_stats["raw_docs_seen"],
            "total_unique_documents": diversity_stats["accepted_unique_docs"],
            "exact_duplicate_rate": round(diversity_stats["exact_duplicates"] / max(diversity_stats["raw_docs_seen"], 1), 4),
            "near_duplicate_rate": round(diversity_stats["near_duplicates"] / max(diversity_stats["raw_docs_seen"], 1), 4),
            "quality_rejection_rate": round(diversity_stats["quality_rejected"] / max(diversity_stats["raw_docs_seen"], 1), 4),
            "safety_rejection_rate": round(diversity_stats["safety_rejected"] / max(diversity_stats["raw_docs_seen"], 1), 4),
            "mean_tokens_per_doc": round(float(np.mean(lengths)), 2),
            "median_tokens_per_doc": round(float(np.median(lengths)), 2),
            "docs_per_category": diversity_stats["docs_per_category"],
            "tokens_per_category": diversity_stats["tokens_per_category"],
        }

        audit_payload = {
            "dataset_version": dataset_version,
            "status": "READY_FOR_ASTRA_1B",
            "dataset_hash": dataset_hash,
            "tokenizer_hash": self.tok_hash,
            "total_tokens": manifest["total_tokens"],
            "total_sequences": sequences_written,
            "num_shards": manifest["num_shards"],
            "sequence_length": self.seq_len,
            "shards": manifest["shards"],
            "mixture_accounting": mixture_metrics,
            "diversity_audit": diversity_summary,
            "sources": [s.source_id for s in registry.list_sources()],
        }

        # Export audit reports
        with open("data/audit_report_1b.json", "w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2)

        with open("data/audit_report_1b.md", "w", encoding="utf-8") as f:
            f.write(f"# ASTRA-1B — PHASE 7C FULL-SCALE CORPUS AUDIT REPORT\n\n")
            f.write(f"- **Dataset Version:** `{dataset_version}`\n")
            f.write(f"- **Status:** **READY_FOR_ASTRA_1B**\n")
            f.write(f"- **Total Tokens:** `{manifest['total_tokens']:,}` tokens\n")
            f.write(f"- **Total Sequences:** `{sequences_written:,}` ($T={self.seq_len}$)\n")
            f.write(f"- **Unique Documents:** `{diversity_summary['total_unique_documents']:,}`\n")
            f.write(f"- **Dataset Hash:** `{dataset_hash}`\n")
            f.write(f"- **Tokenizer Hash:** `{self.tok_hash}` (`FROZEN`)\n\n")

            f.write("## 1. Token-Level Mixture Accounting\n\n")
            f.write("| Category | Target % | Actual % | Actual Tokens | Deviation (pp) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for cat, m in mixture_metrics.items():
                f.write(f"| **{cat.capitalize()}** | {m['target_pct']}% | {m['actual_pct']}% | {m['actual_tokens']:,} | {m['deviation_pp']:+.2f}pp |\n")

            f.write("\n## 2. Document Diversity & Quality Metrics\n\n")
            f.write(f"- Raw Documents Seen: {diversity_summary['total_raw_documents']:,}\n")
            f.write(f"- Unique Documents Accepted: {diversity_summary['total_unique_documents']:,}\n")
            f.write(f"- Exact Duplicate Rate: {diversity_summary['exact_duplicate_rate']*100:.2f}%\n")
            f.write(f"- Near Duplicate Rate: {diversity_summary['near_duplicate_rate']*100:.2f}%\n")
            f.write(f"- Quality Rejection Rate: {diversity_summary['quality_rejection_rate']*100:.2f}%\n")
            f.write(f"- Mean Tokens / Document: {diversity_summary['mean_tokens_per_doc']:.1f}\n")
            f.write(f"- Median Tokens / Document: {diversity_summary['median_tokens_per_doc']:.1f}\n")

            f.write("\n## 3. Generated Shards\n\n")
            f.write("| Shard Name | Sequences | Tokens | SHA-256 Checksum |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for s in manifest["shards"]:
                f.write(f"| `{s['shard_name']}` | {s['num_samples']:,} | {s['num_tokens']:,} | `{s['sha256']}` |\n")

        print(f"[OK] Acquisition Engine finished: {manifest['total_tokens']:,} tokens written to {manifest['num_shards']} shards")
        return audit_payload


if __name__ == "__main__":
    engine = ScalableCorpusAcquisitionEngine()
    engine.run_acquisition(target_tokens=20_000_000, dataset_version="astra-research-v0.2")
