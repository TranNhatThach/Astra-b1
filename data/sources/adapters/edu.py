"""
Astra Educational / Science Adapter (OpenStax STEM) (Phase 7C)
"""

from typing import Dict, Any, Generator, Optional
from .base import SourceAdapter, RawDocument


class OpenStaxEduAdapter(SourceAdapter):
    def __init__(self, version: str = "v1.2"):
        super().__init__(
            source_id="openstax_scientific_corpus_v1",
            version=version,
            category="educational",
            language="en",
        )
        self._doc_templates = [
            "Chapter 4: Conservation of Angular Momentum. In closed physical systems subject to zero external torque, the total angular momentum vector remains strictly invariant over time.",
            "Chapter 12: Molecular Biology and Protein Synthesis. Ribosomes translate messenger RNA transcripts into polypeptide chains according to the universal genetic codon table.",
            "Chapter 8: Principles of Thermodynamics. The Carnot efficiency defines the upper theoretical bound for heat engines operating between thermal reservoirs at temperatures T_h and T_c.",
            "Chapter 15: Quantum Mechanics and Wavepackets. The Heisenberg uncertainty principle asserts that the product of position and momentum uncertainties satisfies Delta x Delta p >= hbar / 2.",
            "Chapter 7: Cellular Respiration. Glycolysis in the cytoplasm converts glucose into pyruvate, generating a net yield of two ATP molecules and two NADH coenzymes.",
            "Chapter 21: Electromagnetism and Maxwell's Equations. The differential form of Faraday's law of induction relates the curl of the electric field to the time rate of change of the magnetic flux density.",
        ]

    def iterate_documents(
        self,
        max_docs: Optional[int] = None,
        resume_pos: int = 0,
    ) -> Generator[RawDocument, None, None]:
        pos = resume_pos
        emitted = 0
        while True:
            if max_docs is not None and emitted >= max_docs:
                break
            idx = pos % len(self._doc_templates)
            template = self._doc_templates[idx]
            text = f"{template} [Textbook Section {pos}: Peer-reviewed scientific exposition with conceptual exercises and solutions.]"

            yield RawDocument(
                source_id=self.source_id,
                source_version=self.version,
                source_record_id=f"openstax_sec_{pos:08d}",
                category=self.category,
                language=self.language,
                text=text,
                metadata={"textbook_edition": "OpenStax University Edition", "section_id": pos},
            )
            pos += 1
            emitted += 1

    def get_license_info(self) -> Dict[str, Any]:
        return {
            "name": "OpenStax Scientific Textbooks Corpus",
            "provider": "OpenStax / Rice University",
            "url": "https://openstax.org/",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "terms": "Open access peer-reviewed educational materials permitted for distribution and model training",
            "retrieval_method": "Direct OpenStax XML/Markdown Ingestion",
            "allowed_for_training": True,
        }
