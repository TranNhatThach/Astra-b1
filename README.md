# Astra-1B: Hybrid Stateful Large Language Model

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-36%20Passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-orange.svg)]()

**Astra-1B** is a research-grade, ~1.0B parameter autoregressive Large Language Model built upon a hybrid sequence-processing backbone: **Gated DeltaNet (GDN)** combined with **Gated Grouped-Query Attention (GQA)**, **SwiGLU FFN**, **RMSNorm**, **RoPE**, and **Multi-Token Prediction (MTP-2)**.

---

## 🏛️ Architectural Philosophy: Separation of Concerns

Astra-1B deliberately assigns sequence-processing primitives to dedicated functional roles:

$$\underbrace{\text{Astra-GDN}}_{\text{Recurrent Memory } (O(1)\text{ inference})} + \underbrace{\text{GQA Attention}}_{\text{Exact Retrieval}} + \underbrace{\text{SwiGLU}}_{\text{Nonlinear Representation}} + \underbrace{\text{Residual Gate}}_{\text{Feature Routing}} + \underbrace{\text{MTP-2}}_{\text{Auxiliary Predictive Signal}}$$

* **Layer Layout:** 24 Layers structured in a repeating pattern: $(3\times\text{GDN} + 1\times\text{Attention})\times 6$ (18 GDN layers, 6 Attention layers).
* **KV-Cache Reduction:** Reduces runtime KV-cache memory consumption by 75% compared to dense Transformers.
* **Tied Embeddings:** $W_{LM} = E^\top$, saving $\approx 134\text{M}$ parameters with a 65,536 vocabulary.
* **Total Parameter Budget:** $\approx 998\text{M}$ parameters.

---

## 📐 Mathematical Contract (Astra Contract v0.1)

### Astra-GDN Recurrent Affine Update
$$S_t = D_t S_{t-1} (I - u_t k_t k_t^\top) + u_t v_t k_t^\top$$
$$y_t = S_t q_t$$

Where:
* $D_t = \operatorname{Diag}(r_t)$ is the diagonal retention decay with $r_t = \sigma(b_r + \Delta r_t) \in \mathbb{R}^{d_h}$.
* $u_t = \sigma(b_u + \Delta u_t) \in \mathbb{R}^1$ is the scalar update gate.
* $k_t \in \mathbb{R}^{d_h}$ is $L_2$-normalized: $k_t = \frac{k_t}{\|k_t\|_2 + \epsilon}$.
* $S \in \mathbb{R}^{D \times D}$ ($96 \times 96$ per head).

### Document-Boundary Loss Masking
$$M_t^{AR} = \mathbf{1}[\text{doc}_t = \text{doc}_{t+1}], \qquad M_t^{MTP2} = \mathbf{1}[\text{doc}_t = \text{doc}_{t+2}]$$
$$\mathcal{L}_{total} = \mathcal{L}_{AR} + 0.2 \mathcal{L}_{MTP2}$$

---

## 📂 Repository Structure

```text
Astra-1B/
├── configs/
│   ├── schema.py               # Immutable Dataclass config schema with YAML parser
│   ├── astra_100m.yaml         # Sanity & correctness verification model
│   ├── astra_350m.yaml         # Scaling law validation model
│   └── astra_1b.yaml           # Full ~998M parameter model
│
├── model/
│   ├── __init__.py
│   ├── reference/              # [ORACLE] Golden PyTorch Implementation
│   │   ├── gdn.py              # AstraGDN
│   │   ├── attention.py        # ReferenceGQA (16Q / 4KV)
│   │   ├── swiglu.py           # SwiGLU FFN
│   │   ├── rmsnorm.py          # RMSNorm
│   │   ├── rope.py             # Rotary Position Embedding
│   │   ├── block.py            # HybridBlock
│   │   ├── mtp.py              # MTP-2 & Boundary-Aware Loss
│   │   └── astra.py            # AstraModel & AstraForCausalLM
│   │
│   ├── optimized/              # Fast Backends (FLA & Triton - Phase 5/6)
│   └── parity/                 # Automated Reference vs Optimized Parity Suite
│
├── tests/
│   ├── unit/                   # Fast equation, causality, state & shape tests
│   └── integration/            # Tiny dataset overfit & checkpoint tests
│
└── benchmarks/                 # Performance harness & memory profiling
```

---

## 🧪 Verification & Testing Suite

Run all mathematical verification, causality, and integration tests:

```bash
pytest -v
```

### Verified Test Matrix (100% Passed)
1. **Manual Equation Match:** Verified against analytical step-by-step NumPy calculations ($rel\_err = 0.0$).
2. **Mathematical Invariants:** Tested Zero-Update Invariant ($S_t = S_{t-1}$) and Exact Retrieval Invariant ($S_t k_t = v_t$).
3. **3-Tier Causality Invariance:** Rigorously proven that modifying future tokens $x_{>t}$ has zero effect on Conv features, recurrent state, and outputs at $\le t$.
4. **Streaming Equivalence:** Full-sequence forward vs token-by-token streaming matches with relative error $< 10^{-6}$.
5. **Chunk Equivalence:** Full-sequence vs arbitrary chunk sizes ($1, 8, 32, 128, 512$) matches with relative error $< 10^{-6}$.
6. **Document Boundary Masking:** Exact loss masking across document boundaries for packed sequence training.
7. **Tiny Overfit Test:** Loss decreases monotonically by $> 95\%$ in 50 steps.
8. **Checkpoint Determinism:** State dictionary save/load preserves exact numerical outputs ($rel\_err = 0.0$).

---

## 🚀 Quickstart

```python
import torch
from configs.schema import AstraConfig
from model import AstraForCausalLM

# Load 1B configuration
config = AstraConfig.from_yaml("configs/astra_1b.yaml")
model = AstraForCausalLM(config)

# Forward pass with stateful streaming
input_ids = torch.randint(0, config.model.vocab_size, (2, 64))
out = model(input_ids=input_ids, compute_loss=False)

print(f"Logits shape: {out['logits'].shape}")
print(f"Recurrent states: {len(out['states'])} layers")
```
