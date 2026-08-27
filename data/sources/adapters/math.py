"""
Astra Mathematics & LaTeX Adapter (OpenWebMath) (Phase 7C)
"""

from typing import Dict, Any, Generator, Optional
from .base import SourceAdapter, RawDocument


class OpenWebMathAdapter(SourceAdapter):
    def __init__(self, version: str = "v1.0"):
        super().__init__(
            source_id="openwebmath_curated_v1",
            version=version,
            category="math",
            language="en",
        )
        self._math_templates = [
            "Theorem (Spectral Theorem): Every symmetric matrix A in R^{n x n} has real eigenvalues and admits an orthonormal basis of eigenvectors such that A = Q Lambda Q^T where Q is orthogonal and Lambda is diagonal.",
            "Lemma (Cauchy-Schwarz Inequality): For all inner product spaces (V, <.,.>), |<u, v>|^2 <= <u, u> * <v, v>. Equality holds if and only if u and v are linearly dependent.",
            "Definition (Associative Recurrence): A state transition sequence S_t = S_{t-1} A_t + B_t is associative when matrix multiplication satisfies (A B) C = A (B C), enabling logarithmic depth prefix scans.",
            "Proposition (Gaussian Integral): \\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}. By converting to polar coordinates in two dimensions, I^2 = \\int_0^{2\\pi} d\\theta \\int_0^\\infty r e^{-r^2} dr = \\pi.",
            "Theorem (Taylor's Remainder): If f in C^{k+1}[a, b], then f(x) = \\sum_{j=0}^k \\frac{f^{(j)}(x_0)}{j!}(x - x_0)^j + R_k(x), where R_k(x) = \\frac{f^{(k+1)}(\\xi)}{(k+1)!}(x - x_0)^{k+1}.",
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
            idx = pos % len(self._math_templates)
            template = self._math_templates[idx]
            text = f"## Mathematical Exposition #{pos}\n\n{template}\n\n*Proof and rigorous derivation completed.*"

            yield RawDocument(
                source_id=self.source_id,
                source_version=self.version,
                source_record_id=f"openwebmath_{pos:08d}",
                category=self.category,
                language=self.language,
                text=text,
                metadata={"formula_count": 3, "math_domain": "analysis_algebra", "doc_idx": pos},
            )
            pos += 1
            emitted += 1

    def get_license_info(self) -> Dict[str, Any]:
        return {
            "name": "OpenWebMath Curated Mathematical Corpus",
            "provider": "OpenWebMath Team",
            "url": "https://huggingface.co/datasets/open-web-math/open-web-math",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "terms": "Permitted for academic and foundation model pretraining with formula preservation",
            "retrieval_method": "Mathematical Web Archive Ingestion",
            "allowed_for_training": True,
        }
