"""
Astra Data Sources Registry (Phase 1A)
Tracks origin, provenance, licensing, and legal auditability for all raw data sources.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional, List
import json
from pathlib import Path


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    name: str
    language: str
    domain: str
    license: str
    terms: str
    retrieved_at: str
    allowed_for_training: bool = True
    sha256: Optional[str] = None
    document_count: Optional[int] = None
    token_count: Optional[int] = None


class SourceRegistry:
    def __init__(self):
        self._sources: Dict[str, SourceMetadata] = {}

    def register(self, source: SourceMetadata) -> None:
        if source.source_id in self._sources:
            raise ValueError(f"Source ID '{source.source_id}' is already registered.")
        if not source.allowed_for_training:
            raise PermissionError(f"Source '{source.source_id}' is not permitted for model training.")
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> Optional[SourceMetadata]:
        return self._sources.get(source_id)

    def list_sources(self) -> List[SourceMetadata]:
        return list(self._sources.values())

    def save_manifest(self, path: str = "data/sources/manifest.json") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {s_id: asdict(meta) for s_id, meta in self._sources.items()}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_manifest(cls, path: str = "data/sources/manifest.json") -> "SourceRegistry":
        reg = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s_id, s_data in data.items():
            reg.register(SourceMetadata(**s_data))
        return reg


def get_default_registry() -> SourceRegistry:
    registry = SourceRegistry()
    
    # 1. High-quality Web
    registry.register(SourceMetadata(
        source_id="fineweb_edu_sample",
        name="FineWeb-Edu Filtered Sample",
        language="en",
        domain="web_educational",
        license="OpenRAIL / CC-BY",
        terms="Permitted for pretraining",
        retrieved_at=datetime.now().isoformat(),
    ))
    
    # 2. Vietnamese High-Quality Corpus
    registry.register(SourceMetadata(
        source_id="vietnamese_curated_v1",
        name="Astra Curated Vietnamese Web & Literature",
        language="vi",
        domain="vietnamese_general",
        license="Open Access / Public Domain",
        terms="Permitted for pretraining",
        retrieved_at=datetime.now().isoformat(),
    ))

    # 3. Open Source Code
    registry.register(SourceMetadata(
        source_id="starcoder_permissive_code",
        name="Permissive Open-Source Code (Python, C++, Rust)",
        language="code",
        domain="code",
        license="MIT / Apache-2.0 / BSD",
        terms="Permitted for code model training",
        retrieved_at=datetime.now().isoformat(),
    ))

    return registry
