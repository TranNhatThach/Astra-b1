"""
Astra Source Adapter Interface (Phase 7C)
Defines the unified contract for streaming ingestion, provenance capture, and document normalization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from typing import Dict, Any, Generator, Optional
from data.sources.registry import SourceMetadata


@dataclass(frozen=True)
class RawDocument:
    source_id: str
    source_version: str
    source_record_id: str
    category: str
    language: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    doc_id: str = ""

    def __post_init__(self):
        c_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        object.__setattr__(self, "content_hash", c_hash)
        seed_str = f"{self.source_id}:{self.source_version}:{self.source_record_id}:{c_hash[:16]}"
        d_id = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        object.__setattr__(self, "doc_id", d_id)


class SourceAdapter(ABC):
    def __init__(self, source_id: str, version: str, category: str, language: str = "en"):
        self.source_id = source_id
        self.version = version
        self.category = category
        self.language = language

    @abstractmethod
    def iterate_documents(
        self,
        max_docs: Optional[int] = None,
        resume_pos: int = 0,
    ) -> Generator[RawDocument, None, None]:
        """
        Stream raw documents incrementally with deterministic pagination/resumability.
        """
        pass

    @abstractmethod
    def get_license_info(self) -> Dict[str, Any]:
        """
        Returns documented license and usage rights metadata.
        """
        pass

    def get_metadata(self) -> SourceMetadata:
        lic = self.get_license_info()
        return SourceMetadata(
            source_id=self.source_id,
            name=lic.get("name", self.source_id),
            category=self.category,
            provider=lic.get("provider", "Unknown"),
            url=lic.get("url", ""),
            version=self.version,
            license=lic.get("license", "UNKNOWN"),
            license_url=lic.get("license_url", ""),
            terms=lic.get("terms", ""),
            retrieval_method=lic.get("retrieval_method", "Streaming API"),
            retrieved_at=datetime.now().isoformat(),
            allowed_for_training=lic.get("allowed_for_training", True),
            status="APPROVED" if lic.get("allowed_for_training", True) else "REJECTED",
        )
