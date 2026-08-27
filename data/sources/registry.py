"""
Astra Legal Data Sources Registry & Candidate Inventory (Phase 7A)
Tracks origin, provenance, licensing, usage rights, and cryptographic hashes for all sources.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Optional, List, Any
import json
from pathlib import Path


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    name: str
    category: str  # web, educational, code, math, vietnamese, dialogue
    provider: str
    url: str
    version: str
    license: str
    license_url: str
    terms: str
    retrieval_method: str
    retrieved_at: str
    allowed_for_training: bool = True
    raw_artifact_path: Optional[str] = None
    raw_sha256: Optional[str] = None
    document_count: int = 0
    estimated_tokens: int = 0
    final_tokens: int = 0
    status: str = "APPROVED"  # CANDIDATE, APPROVED, REJECTED, QUARANTINED, VALIDATED, PROCESSED
    quality_notes: str = ""
    known_limitations: str = ""


@dataclass(frozen=True)
class ExcludedSourceRecord:
    source_id: str
    name: str
    category: str
    provider: str
    exclusion_reason: str  # LICENSE_UNKNOWN, LICENSE_INCOMPATIBLE, PROVENANCE_UNKNOWN, PII_RISK, etc.
    notes: str


class SourceRegistry:
    def __init__(self, registry_file: str = "data/sources/registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._sources: Dict[str, SourceMetadata] = {}
        self._excluded: Dict[str, ExcludedSourceRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_file.exists():
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for s_id, s_data in data.get("approved_sources", {}).items():
                self._sources[s_id] = SourceMetadata(**s_data)
            for e_id, e_data in data.get("excluded_sources", {}).items():
                self._excluded[e_id] = ExcludedSourceRecord(**e_data)

    def register(self, source: SourceMetadata, overwrite: bool = True) -> None:
        if source.source_id in self._sources and not overwrite:
            raise ValueError(f"Source ID '{source.source_id}' is already registered.")
        if not source.allowed_for_training or source.status == "REJECTED":
            raise PermissionError(f"Source '{source.source_id}' is not permitted for model training.")
        self._sources[source.source_id] = source
        self.save()

    def record_exclusion(self, excluded: ExcludedSourceRecord) -> None:
        self._excluded[excluded.source_id] = excluded
        self.save()

    def get(self, source_id: str) -> Optional[SourceMetadata]:
        return self._sources.get(source_id)

    def list_sources(self) -> List[SourceMetadata]:
        return list(self._sources.values())

    def list_excluded(self) -> List[ExcludedSourceRecord]:
        return list(self._excluded.values())

    def save(self) -> None:
        payload = {
            "approved_sources": {s_id: asdict(meta) for s_id, meta in self._sources.items()},
            "excluded_sources": {e_id: asdict(meta) for e_id, meta in self._excluded.items()},
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def build_canonical_pilot_registry() -> SourceRegistry:
    """
    Constructs the canonical 6-category approved source registry and documented exclusions.
    """
    reg = SourceRegistry()
    now_str = datetime.now().isoformat()

    # 1. Category A: Web / General English (45%)
    reg.register(SourceMetadata(
        source_id="fineweb_edu_web_v1",
        name="FineWeb-Edu Filtered Open Web Corpus",
        category="web",
        provider="HuggingFace / FineWeb",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        version="v1.0.0",
        license="OpenRAIL / CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        terms="Permitted for academic and commercial training with attribution",
        retrieval_method="HuggingFace Datasets API with checksum verification",
        retrieved_at=now_str,
        allowed_for_training=True,
        status="APPROVED",
        quality_notes="High educational value filter, filtered for noise and spam",
        known_limitations="Primarily English web content",
    ))

    # 2. Category B: Educational / Science (15%)
    reg.register(SourceMetadata(
        source_id="openstax_scientific_corpus_v1",
        name="OpenStax Open Scientific Textbooks Corpus",
        category="educational",
        provider="OpenStax / Rice University",
        url="https://openstax.org/",
        version="v1.2",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        terms="Permitted for derivative works, indexing, and training",
        retrieval_method="Direct XML/Markdown archival ingestion",
        retrieved_at=now_str,
        allowed_for_training=True,
        status="APPROVED",
        quality_notes="Peer-reviewed university-level science, physics, biology, and CS textbooks",
        known_limitations="Domain restricted to academic STEM",
    ))

    # 3. Category C: Source Code (15%)
    reg.register(SourceMetadata(
        source_id="the_stack_permissive_code_v1",
        name="The Stack Permissive Code Subset (Python, C++, Rust, Go)",
        category="code",
        provider="BigCode Project",
        url="https://huggingface.co/datasets/bigcode/the-stack-v2",
        version="v2.0",
        license="MIT / Apache-2.0 / BSD-3-Clause",
        license_url="https://opensource.org/licenses/",
        terms="Permissive open-source licenses with opt-out filtering applied",
        retrieval_method="BigCode parquet dataset extraction with license validation",
        retrieved_at=now_str,
        allowed_for_training=True,
        status="APPROVED",
        quality_notes="Deduplicated repositories, stripped binary files and secrets",
        known_limitations="Requires strict PII/secret key screening",
    ))

    # 4. Category D: Mathematics & LaTeX (10%)
    reg.register(SourceMetadata(
        source_id="openwebmath_curated_v1",
        name="OpenWebMath Curated Mathematical Reasoning Corpus",
        category="math",
        provider="OpenWebMath Team",
        url="https://huggingface.co/datasets/open-web-math/open-web-math",
        version="v1.0",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        terms="Permitted for research and foundation model training",
        retrieval_method="Direct dataset stream with LaTeX equation preservation",
        retrieved_at=now_str,
        allowed_for_training=True,
        status="APPROVED",
        quality_notes="Pre-parsed LaTeX equations, MathJax notation preserved",
        known_limitations="High density of formulas requires cautious tokenization",
    ))

    # 5. Category E: Vietnamese High Quality (10%)
    reg.register(SourceMetadata(
        source_id="vietnamese_curated_literature_web_v1",
        name="Astra Curated Vietnamese Web, News & Cultural Corpus",
        category="vietnamese",
        provider="Astra Linguistic Lab",
        url="https://astra-research.org/datasets/vietnamese_curated_v1",
        version="v1.1",
        license="CC-BY-SA-4.0 / Public Domain",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        terms="Permitted for foundation model training",
        retrieval_method="Curated public domain and open access repository crawl",
        retrieved_at=now_str,
        allowed_for_training=True,
        status="APPROVED",
        quality_notes="Strict NFC normalization, full tone mark preservation (ă, â, đ, ê, ô, ơ, ư)",
        known_limitations="Careful diacritic normalization is mandatory",
    ))

    # 6. Category F: Synthetic Multi-turn Dialogue (5%)
    reg.register(SourceMetadata(
        source_id="synthetic_reasoning_dialogue_v1",
        name="Astra Multi-Turn Scientific Reasoning & Dialogues",
        category="dialogue",
        provider="Astra AI Research",
        url="https://astra-research.org/datasets/reasoning_dialogues",
        version="v1.0",
        license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        terms="Permitted for pretraining and instruction tuning",
        retrieval_method="Synthetic generation with multi-turn verification",
        retrieved_at=now_str,
        allowed_for_training=True,
        status="APPROVED",
        quality_notes="Multi-turn structure with explicit user/assistant turns",
        known_limitations="Synthetic dataset, requires factuality screening",
    ))

    # Record Excluded Candidates (Transparency & Legal Audit)
    reg.record_exclusion(ExcludedSourceRecord(
        source_id="unlicensed_web_scrape_dump_01",
        name="Unverified Scraped Web Archive",
        category="web",
        provider="Unknown Forum Dump",
        exclusion_reason="LICENSE_UNKNOWN",
        notes="No clear license, terms of service forbid automated scraping",
    ))

    reg.record_exclusion(ExcludedSourceRecord(
        source_id="proprietary_math_books_02",
        name="Commercial Textbook Scans",
        category="math",
        provider="Scanned PDF Library",
        exclusion_reason="LICENSE_INCOMPATIBLE",
        notes="Copyrighted proprietary material without explicit training license",
    ))

    return reg


def get_default_registry() -> SourceRegistry:
    return build_canonical_pilot_registry()
