"""
Astra Deterministic Data Mixture Sampler (Phase 7)
Samples documents across domains following target MixturePolicy with guaranteed seed determinism.
"""

from typing import Dict, List, Generator, Any
import numpy as np

from .policies import MixturePolicy


class DeterministicMixtureSampler:
    def __init__(
        self,
        domain_corpora: Dict[str, List[str]],
        policy: MixturePolicy = None,
        seed: int = 42,
    ):
        self.domain_corpora = domain_corpora
        self.policy = policy or MixturePolicy()
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Filter active domains
        self.domains = [d for d in self.policy.to_dict().keys() if d in self.domain_corpora and len(self.domain_corpora[d]) > 0]
        if not self.domains:
            raise ValueError("No matching domain corpora available for sampling.")

        raw_weights = [self.policy.to_dict()[d] for d in self.domains]
        total_w = sum(raw_weights)
        self.probabilities = [w / total_w for w in raw_weights]

        self.domain_indices = {d: 0 for d in self.domains}

    def sample_stream(self, total_samples: int) -> Generator[Dict[str, Any], None, None]:
        for _ in range(total_samples):
            chosen_domain = self.rng.choice(self.domains, p=self.probabilities)
            corpus = self.domain_corpora[chosen_domain]

            idx = self.domain_indices[chosen_domain] % len(corpus)
            self.domain_indices[chosen_domain] += 1

            doc_text = corpus[idx]
            yield {
                "domain": chosen_domain,
                "text": doc_text,
            }
