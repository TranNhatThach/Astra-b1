# Astra-1B

## Engineering Specification v1.0

**Project type:** Decoder-style autoregressive Large Language Model
**Target scale:** ~1.0B trainable parameters
**Architecture:** Hybrid Gated DeltaNet + Gated GQA Attention
**Training:** From scratch
**Primary objective:** Build a reproducible, research-grade 1B LLM whose architecture can later scale to 3B, 7B and beyond.

> **Important:** Astra-1B is a proposed architecture. It is inspired by ideas from modern LLM research, including Gated DeltaNet, GQA, RoPE, SwiGLU and MTP, but it is not an implementation of Qwen3.8 or any other proprietary model.

---

# 1. Design Objectives

Astra-1B is designed around five principles:

$$
\boxed{
Memory + Retrieval + Nonlinearity + Efficiency + Scalability
}
$$

The model deliberately gives different sequence-processing mechanisms different jobs:

$$
\boxed{
GDN \rightarrow persistent state / cheap sequence processing
}
$$

$$
\boxed{
Attention \rightarrow exact content retrieval
}
$$

$$
\boxed{
SwiGLU \rightarrow nonlinear feature transformation
}
$$

$$
\boxed{
Residual stream \rightarrow information highway
}
$$

$$
\boxed{
MTP \rightarrow additional predictive training signal
}
$$

The primary research question is:

> Can a carefully balanced GDN-Attention hybrid outperform a same-budget dense Transformer on quality, training efficiency and long-context efficiency?

---

# 2. System Architecture

High-level pipeline:

```text
                         RAW DATA
                            │
                            ▼
                ┌─────────────────────┐
                │ Quality / Language  │
                │ Filtering           │
                └──────────┬──────────┘
                           │
                           ▼
                    Deduplication
                           │
                           ▼
                     Tokenization
                           │
                           ▼
                    Token Packing
                           │
                           ▼
                   Training Shards
                           │
                           ▼
                       Astra-1B
                           │
              ┌────────────┴────────────┐
              │                         │
          Base LM                   MTP-2
              │                         │
              └────────────┬────────────┘
                           ▼
                         logits
                           │
                           ▼
                       sampling
                           │
                           ▼
                         text
```

Model:

```text
Token IDs
   │
   ▼
Embedding
   │
   ▼
24 Hybrid Blocks
   │
   ├── GDN
   ├── GDN
   ├── GDN
   └── Attention
       │
       └── pattern repeats × 6
   │
   ▼
Final RMSNorm
   │
   ▼
LM Head
   │
   ├── next-token prediction
   └── MTP-2 auxiliary prediction
```

---

# 3. Model Configuration

## 3.1 Core configuration

| Parameter                |                                            Value |
| ------------------------ | -----------------------------------------------: |
| Model name               |                                         Astra-1B |
| Parameters               | ~0.998B before optional metadata/bias variations |
| Layers                   |                                               24 |
| Hidden size              |                                             2048 |
| FFN intermediate size    |                                             3456 |
| Vocabulary               |                                           65,536 |
| Max base context         |                                             4096 |
| Long-context training    |                                   8K → 16K → 32K |
| Normalization            |                                          RMSNorm |
| Activation               |                                           SwiGLU |
| Position encoding        |                                             RoPE |
| Embedding                |                                             tied |
| Attention type           |                                              GQA |
| Q heads                  |                                               16 |
| KV heads                 |                                                4 |
| Attention head dimension |                                              128 |
| GDN heads                |                                               16 |
| GDN head dimension       |                                               96 |
| GDN projected dimension  |                                             1536 |
| Attention layers         |                                                6 |
| GDN layers               |                                               18 |
| MTP depth                |                               2 prediction steps |
| Precision                |                                             BF16 |
| Optimizer                |                                            AdamW |
| Gradient clipping        |                                              1.0 |
| Initial peak LR          |                               \(3\times10^{-4}\) |
| Weight decay             |                                              0.1 |
| LR schedule              |                                  warmup + cosine |
| Dropout                  |                                      0 initially |

---

# 4. Layer Layout

The canonical repeating pattern is:

$$
\boxed{
(GDN,GDN,GDN,ATTN)\times6
}
$$

Exact layer sequence:

```text
Layer 01  GDN
Layer 02  GDN
Layer 03  GDN
Layer 04  Attention

Layer 05  GDN
Layer 06  GDN
Layer 07  GDN
Layer 08  Attention

Layer 09  GDN
Layer 10  GDN
Layer 11  GDN
Layer 12  Attention

Layer 13  GDN
Layer 14  GDN
Layer 15  GDN
Layer 16  Attention

Layer 17  GDN
Layer 18  GDN
Layer 19  GDN
Layer 20  Attention

Layer 21  GDN
Layer 22  GDN
Layer 23  GDN
Layer 24  Attention
```

This produces:

$$
18\ GDN + 6\ Attention
$$

or:

$$
75\%\ GDN,\qquad25\%\ Attention
$$

---

# 5. Input Embedding

Vocabulary:

$$
V=65,536
$$

Embedding matrix:

$$
E\in\mathbb{R}^{65536\times2048}
$$

Parameter count:

$$
65536\times2048
=
134,217,728
$$

The language-model output head is tied:

$$
W_{LM}=E^T
$$

Therefore the model does not allocate another 134M parameters for the vocabulary projection.

---

# 6. RMSNorm

For hidden vector \(x\):

$$
RMSNorm(x)
=
\gamma
\frac{x}
{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}
$$

with:

$$
d=2048
$$

Recommended:

$$
\epsilon=10^{-6}
$$

Use Pre-Norm.

---

# 7. GDN Block

## 7.1 Purpose

The GDN block is responsible primarily for:

* persistent state
* long-range sequential information
* efficient streaming
* avoiding quadratic attention at most layers

The conceptual update is:

$$
S_t = F(S_{t-1},x_t)
$$

followed by:

$$
y_t=G(S_t,x_t)
$$

The important design principle is that the hidden history is represented by a compact learned state rather than a complete token-token attention matrix.

---

# 8. Astra-GDN Tensor Specification

For input:

$$
X\in\mathbb{R}^{B\times T\times2048}
$$

set:

$$
H=16
$$

$$
d_h=96
$$

so:

$$
D_{gdn}=16\times96=1536
$$

Projection dimensions:

```text
Q : 2048 → 1536
K : 2048 → 1536
V : 2048 → 1536
G : 2048 → 1536
O : 1536 → 2048
```

Shapes:

```text
Q = [B,T,1536]
K = [B,T,1536]
V = [B,T,1536]
G = [B,T,1536]
O = [B,T,2048]
```

Parameters per GDN projection:

$$
2048\times1536
=
3,145,728
$$

Five projections:

$$
5\times3,145,728
=
15,728,640
$$

This keeps the 1B budget under control.

---

# 9. GDN State

For each head:

$$
S_t^{(h)}
\in
\mathbb{R}^{96\times96}
$$

The core delta-style update should follow the form:

$$
\tilde S_t
=
S_{t-1}
+
\beta_t
\left(
v_t-\hat v_t
\right)
k_t^T
$$

followed by retention:

$$
S_t
=
\gamma_t\odot S_{t-1}
+
\tilde S_t
$$

where:

$$
\beta_t\in[0,1]
$$

is an update gate and:

$$
\gamma_t\in[0,1]
$$

controls state retention.

The exact optimized implementation should use fused recurrent/scan kernels rather than constructing the full \(T\times T\) attention matrix.

---

# 10. GDN Gating

Use a sigmoid gate:

$$
g_t=\sigma(Gx_t)
$$

and:

$$
y_t=g_t\odot y_t^{state}
$$

This prevents every state update from having equal influence.

The gate should also support a learnable bias initialized toward conservative updating.

Recommended initialization:

$$
b_g\approx2
$$

to avoid aggressively overwriting state during the first training steps.

---

# 11. Local Convolution

Add a lightweight causal depthwise convolution before the recurrent state update.

Recommended:

$$
k=4
$$

This supplies short-range local mixing to complement the recurrent memory.

Pipeline:

```text
input
  │
  ▼
projection
  │
  ▼
depthwise causal conv
  │
  ▼
Q/K/V
  │
  ▼
delta-state update
  │
  ▼
gate
  │
  ▼
output projection
```

---

# 12. GDN Residual Block

Recommended implementation:

```text
x
│
├── RMSNorm
│
▼
GDN
│
▼
residual gate
│
▼
x + gated(GDN(x))
│
├── RMSNorm
│
▼
SwiGLU
│
▼
Residual
```

Formally:

$$
u=x+g_r\odot GDN(RMSNorm(x))
$$

then:

$$
y=u+SwiGLU(RMSNorm(u))
$$

where \(g_r\) is a learnable residual gate.

For v1, initialize:

$$
g_r=1
$$

or use a sigmoid parameter initialized so the effective gate is close to 1.

---

# 13. Attention Block

Six layers use Gated Grouped Query Attention.

Configuration:

$$
16\ Q\ heads
$$

$$
4\ KV\ heads
$$

$$
d_h=128
$$

Therefore:

$$
D_Q=16\times128=2048
$$

$$
D_{KV}=4\times128=512
$$

---

# 14. Attention Projections

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

with:

```text
W_Q : 2048 × 2048
W_K : 2048 × 512
W_V : 2048 × 512
W_O : 2048 × 2048
```

Total attention parameters:

$$
4,194,304
+
1,048,576
+
1,048,576
+
4,194,304
$$

$$
=
10,485,760
$$

per attention layer.

---

# 15. GQA Routing

Every group of four query heads shares one K/V head.

```text
Q1 Q2 Q3 Q4 → KV1
Q5 Q6 Q7 Q8 → KV2
Q9 Q10 Q11 Q12 → KV3
Q13 Q14 Q15 Q16 → KV4
```

This reduces KV-cache memory compared with full multi-head attention.

---

# 16. RoPE

Apply Rotary Position Embedding to Q and K:

$$
Q' = RoPE(Q,p)
$$

$$
K' = RoPE(K,p)
$$

Use a long-context-compatible RoPE base.

Initial training:

$$
T=4096
$$

Later context stages progressively increase sequence length.

---

# 17. Attention Gating

The Attention block uses a learned gate:

$$
g_a=\sigma(W_gx)
$$

then:

$$
y=x+g_a\odot Attention(RMSNorm(x))
$$

This lets the model learn how strongly to inject exact-attention information into the residual stream.

---

# 18. Feed-Forward Network

Every layer uses SwiGLU.

For:

$$
x\in\mathbb{R}^{2048}
$$

with:

$$
d_{ff}=3456
$$

define:

$$
u=W_u x
$$

$$
g=SiLU(W_gx)
$$

$$
FFN(x)=W_d(g\odot u)
$$

Parameter count:

$$
3\times2048\times3456
=
22,020,096
$$

per layer.

Across 24 layers:

$$
528,482,304
$$

parameters.

---

# 19. Total Parameter Budget

Approximate major components:

| Component              | Parameters |
| ---------------------- | ---------: |
| Shared token embedding |    134.22M |
| 24 × SwiGLU            |    528.48M |
| 18 × GDN               |    283.12M |
| 6 × GQA Attention      |     62.91M |
| Norms                  |     ~0.10M |
| MTP projections        |     ~8.39M |
| **Total**              |  **~998M** |

Target:

$$
\boxed{\approx1.0B}
$$

Small differences can occur depending on bias, gate and implementation choices.

---

# 20. MTP-2

The main training target is standard next-token prediction:

$$
L_{AR}
=
-\sum_t
\log P(x_{t+1}|x_{\le t})
$$

Add a second predictive horizon:

$$
L_{MTP2}
=
-\sum_t
\log P(x_{t+2}|x_{\le t})
$$

Total:

$$
\boxed{
L=L_{AR}+0.2L_{MTP2}
}
$$

For implementation, reuse the tied vocabulary projection.

Only the intermediate transformation is separate.

This keeps parameter overhead small.

---

# 21. MTP Architecture

```text
hidden_t
   │
   ├───────────────► LM Head ───► x[t+1]
   │
   ▼
MTP Projection 1
   │
   ▼
MTP Projection 2
   │
   ▼
LM Head ───────────► x[t+2]
```

MTP must remain an auxiliary objective.

The main language-model loss always dominates.

---

# 22. Weight Tying

Use:

$$
W_{embedding}=W_{LM}^T
$$

Do not duplicate the vocabulary matrix.

This saves approximately:

$$
134M
$$

parameters.

---

# 23. Tokenizer Specification

Tokenizer:

$$
\boxed{BPE}
$$

Vocabulary:

$$
65,536
$$

Required coverage:

```text
English
Vietnamese
Unicode
source code
mathematics
numbers
scientific notation
URLs
structured text
```

Tokenizer training corpus must represent the final data mixture.

Do not train the tokenizer exclusively on English web data.

---

# 24. Vietnamese Tokenization Requirement

Because Vietnamese is an explicit target language, evaluate:

```text
character/token ratio
word/token ratio
syllable/token ratio
```

for:

```text
Vietnamese
English
code
math
```

Particular attention should be paid to:

```text
dấu tiếng Việt
Unicode normalization
đ
ă
â
ê
ô
ơ
ư
```

The tokenizer must preserve Unicode correctly.

---

# 25. Data Pipeline

The production pipeline:

```text
SOURCE
  │
  ▼
RAW STORAGE
  │
  ▼
Language Identification
  │
  ▼
Boilerplate Removal
  │
  ▼
Quality Filtering
  │
  ▼
PII / Safety Filtering
  │
  ▼
Exact Dedup
  │
  ▼
Near Dedup
  │
  ▼
Quality Scoring
  │
  ▼
Mixture Sampling
  │
  ▼
Tokenizer
  │
  ▼
Document → Token stream
  │
  ▼
Sequence packing
  │
  ▼
Sharded dataset
  │
  ▼
Training
```

---

# 26. Initial Data Mixture

Initial target:

| Dataset family                  | Share |
| ------------------------------- | ----: |
| High-quality web                |   45% |
| Books / educational / reference |   15% |
| Code                            |   15% |
| Mathematics                     |   10% |
| Encyclopedia                    |    5% |
| QA / dialogue                   |    5% |
| Vietnamese-focused data         |    5% |

This is a starting hypothesis, not a permanent recipe.

---

# 27. Document Quality Score

For document \(d\):

$$
Q(d)
=
w_1Q_{lang}
+w_2Q_{quality}
+w_3Q_{structure}
+w_4Q_{educational}
-w_5Q_{spam}
-w_6Q_{dup}
$$

The score is used for sampling, not simply hard filtering.

Recommended strategy:

```text
very low quality      → remove
low quality           → mostly remove
medium quality        → downsample
high quality          → retain
exceptional quality   → upsample
```

---

# 28. Deduplication

Use multiple levels.

## Level 1: exact

Hash normalized document:

$$
h=SHA256(normalize(d))
$$

## Level 2: near duplicate

Use MinHash / LSH or equivalent.

## Level 3: semantic duplicate

Only apply to expensive high-value subsets.

For code:

```text
file-level dedup
function-level dedup
repository-level dedup
```

---

# 29. Data Storage Format

Training-ready data should be immutable shards.

Suggested:

```text
data/
├── train/
│   ├── shard-000000.bin
│   ├── shard-000001.bin
│   └── ...
├── validation/
│   └── ...
└── metadata/
    ├── manifest.json
    └── statistics.json
```

Each shard should contain:

```text
token IDs
document boundaries
sample boundaries
metadata
checksum
```

The training process must never depend on downloading raw web documents while GPUs are training.

---

# 30. Sequence Packing

Do not train one padded document per GPU sample.

Instead:

```text
doc A ─┐
doc B ─┼──► packed token stream
doc C ─┤
doc D ─┘
```

Then slice into:

$$
T=4096
$$

token sequences.

This minimizes padding waste.

---

# 31. Context Curriculum

Training stages:

```text
Stage 1
4K context

Stage 2
8K

Stage 3
16K

Stage 4
32K
```

Do not train the entire 1B model at 32K from step 1.

Initial stages maximize:

$$
tokens/sec
$$

Later stages specialize the model for long context.

---

# 32. Optimizer

Baseline:

$$
\boxed{AdamW}
$$

Parameters:

```text
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
```

Exclude from weight decay:

```text
RMSNorm parameters
bias parameters
gating bias parameters
```

---

# 33. Learning Rate

Initial peak:

$$
3\times10^{-4}
$$

Warmup:

$$
1\%-2\%
$$

of total optimizer steps.

Then cosine decay:

$$
\eta(t)
=
\eta_{min}
+
\frac12(\eta_{max}-\eta_{min})
\left(
1+\cos\frac{\pi t}{T}
\right)
$$

Suggested final LR:

$$
3\times10^{-5}
$$

with a final cooldown.

---

# 34. Precision

Primary training precision:

$$
\boxed{BF16}
$$

Use:

```text
BF16 parameters
BF16 activations
FP32 optimizer states where needed
FP32 reductions for sensitive statistics
```

Avoid FP16 unless hardware constraints force it.

---

# 35. Gradient Clipping

Global norm clipping:

$$
||g||_2\le1.0
$$

Algorithm:

```text
calculate global grad norm
       │
       ▼
if norm > 1
       │
       ▼
scale gradients
```

Record gradient norms every logging interval.

---

# 36. Training Infrastructure

Recommended software:

```text
Python
PyTorch
CUDA
FlashAttention
FSDP
torch.compile
NCCL
Weights & Biases or MLflow
```

For the first distributed implementation:

$$
\boxed{FSDP}
$$

should be the default.

Use tensor parallelism only when scaling eventually requires it.

---

# 37. GPU Strategy

Since GPU rental is available, prioritize:

```text
H100
H200
A100 80GB
```

The first production target is:

```text
8 GPUs
```

rather than immediately building a huge cluster.

The model is small enough that engineering efficiency matters more than raw GPU count.

---

# 38. Estimated Compute

A useful first-order estimate for training is:

$$
FLOPs
\approx
6NT
$$

where:

$$
N\approx10^9
$$

and \(T\) is the number of training tokens.

For 50B tokens:

$$
F
\approx
6\times10^9\times50\times10^9
=
3\times10^{20}
$$

FLOPs.

For 100B:

$$
F
\approx
6\times10^{20}
$$

FLOPs.

Actual wall-clock time depends heavily on implementation efficiency, GDN kernels, communication, data throughput and achieved MFU.

---

# 39. Compute Plan

Do not purchase the entire 100B-token run up front.

Use checkpoints:

```text
10B tokens
      ↓
20B
      ↓
30B
      ↓
50B
      ↓
100B if justified
```

At every checkpoint:

```text
validation loss
PPL
downstream benchmarks
tokens/sec
GPU utilization
```

must be compared.

Stop scaling when quality improvement no longer justifies compute.

---

# 40. Pretraining Phases

## Phase 0: Architecture validation

Model:

$$
100M
$$

Train:

$$
0.5B-1B
$$

tokens.

Required:

```text
loss decreases
no NaN
generation works
checkpoint restore works
GDN state works
attention mask works
MTP loss works
distributed training works
```

---

## Phase 1: Scaling validation

Model:

$$
350M
$$

Train:

$$
3B-10B
$$

tokens.

Goal:

$$
\text{Does architecture scale normally?}
$$

This stage is mandatory.

---

## Phase 2: Astra-1B

Train:

$$
20B
$$

tokens first.

Evaluate.

Then:

$$
50B
$$

Evaluate again.

Then optionally:

$$
100B
$$

---

# 41. Required Checkpoints

Save:

```text
latest
best_val_loss
every 1000-5000 steps
every major token milestone
```

Each checkpoint:

```text
model weights
optimizer state
scheduler state
step
tokens_seen
RNG state
data-loader state
configuration hash
git commit
tokenizer version
```

A checkpoint without these metadata cannot be considered reproducible.

---

# 42. Checkpoint Format

Repository should eventually produce:

```text
astra-1b/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── generation_config.json
├── chat_template.jinja
├── model.safetensors
└── README.md
```

During training, temporary distributed checkpoints can use FSDP-native checkpointing.

The final public artifact should use `safetensors`.

---

# 43. Repository Structure

```text
astra-llm/
│
├── README.md
├── LICENSE
├── pyproject.toml
│
├── configs/
│   ├── 100m.yaml
│   ├── 350m.yaml
│   └── 1b.yaml
│
├── model/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── rmsnorm.py
│   ├── swiglu.py
│   ├── gdn.py
│   ├── attention.py
│   ├── block.py
│   ├── mtp.py
│   └── model.py
│
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── evaluate_tokenizer.py
│   └── special_tokens.json
│
├── data/
│   ├── ingest/
│   ├── clean/
│   ├── dedup/
│   ├── score/
│   ├── tokenize/
│   ├── pack/
│   └── manifest/
│
├── training/
│   ├── train.py
│   ├── distributed.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── checkpoint.py
│   └── logger.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── benchmark.py
│   ├── vietnamese.py
│   ├── math.py
│   ├── code.py
│   └── long_context.py
│
├── inference/
│   ├── generate.py
│   └── server.py
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_gdn.py
│   ├── test_attention.py
│   ├── test_mtp.py
│   ├── test_forward.py
│   └── test_checkpoint.py
│
└── scripts/
    ├── pretrain.sh
    ├── eval.sh
    └── export_hf.sh
```

---

# 44. Configuration File

Canonical `1b.yaml`:

```yaml
model:
  name: astra-1b

  vocab_size: 65536
  hidden_size: 2048
  num_layers: 24
  intermediate_size: 3456

  norm_eps: 1.0e-6
  rope_theta: 1000000

  tie_word_embeddings: true

  layer_pattern:
    - gdn
    - gdn
    - gdn
    - attention

  attention:
    num_q_heads: 16
    num_kv_heads: 4
    head_dim: 128
    gated: true

  gdn:
    num_heads: 16
    head_dim: 96
    projection_dim: 1536
    conv_kernel: 4
    gated: true

  mtp:
    enabled: true
    depth: 2
    loss_weight: 0.2

training:
  dtype: bfloat16

  optimizer:
    name: adamw
    lr: 3.0e-4
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 0.1

  warmup_ratio: 0.02
  min_lr: 3.0e-5
  max_grad_norm: 1.0

  sequence_length:
    stage1: 4096
    stage2: 8192
    stage3: 16384
    stage4: 32768

distributed:
  backend: nccl
  strategy: fsdp

logging:
  interval_steps: 10
  eval_interval_steps: 1000
  checkpoint_interval_steps: 2000
```

---

# 45. Training Loop

Pseudo-flow:

```text
initialize distributed runtime
        │
        ▼
load configuration
        │
        ▼
initialize tokenizer
        │
        ▼
initialize model
        │
        ▼
initialize optimizer
        │
        ▼
load dataset shard
        │
        ▼
for each batch
        │
        ▼
forward
        │
        ├── main LM logits
        └── MTP logits
        │
        ▼
compute loss
        │
        ▼
backward
        │
        ▼
gradient clipping
        │
        ▼
optimizer step
        │
        ▼
scheduler step
        │
        ▼
logging
        │
        ▼
validation
        │
        ▼
checkpoint
```

---

# 46. Correctness Tests Before Real Training

The 100M model must pass:

### Shape tests

Every module must produce expected tensor dimensions.

### Causal test

Token \(t\) must not access:

$$
x_{>t}
$$

### GDN state test

Streaming inference should match full-sequence inference within numerical tolerance.

### Attention test

Compare optimized kernel to naive reference implementation on tiny sequences.

### MTP test

Verify:

$$
x_t\rightarrow x_{t+1}
$$

and:

$$
x_t\rightarrow x_{t+2}
$$

are aligned correctly.

### Gradient test

Run finite-difference or PyTorch gradcheck on reduced-size modules.

---

# 47. Golden Reference Implementation

Before writing optimized CUDA/fused code, implement every module in a slow reference form.

For GDN:

```text
reference_gdn.py
optimized_gdn.py
```

For attention:

```text
reference_attention.py
optimized_attention.py
```

Then test:

$$
||Y_{reference}-Y_{optimized}||
$$

against tolerance.

Only optimize after correctness is established.

---

# 48. Performance Engineering

Monitor:

$$
tokens/sec
$$

$$
TFLOPS/GPU
$$

$$
MFU
$$

$$
GPU\ memory
$$

$$
communication\ overhead
$$

$$
data-loader\ throughput
$$

The training system is considered healthy only when GPUs are consistently fed.

A cheap GPU waiting for Python's dataloader is still an expensive GPU.

---

# 49. Evaluation Suite

Every checkpoint should run:

## Language modeling

$$
Perplexity
$$

on:

```text
English validation
Vietnamese validation
Code validation
Math validation
```

## Knowledge

General knowledge benchmark.

## Reasoning

Arithmetic, symbolic and language reasoning.

## Code

Code completion and code reasoning.

## Vietnamese

Vietnamese reading comprehension, generation and QA.

## Long context

Needle-in-haystack and synthetic retrieval tasks at:

```text
4K
8K
16K
32K
```

---

# 50. Critical Metrics

Do not use only validation loss.

The dashboard should contain:

```text
train_loss
val_loss
perplexity
MTP_loss
learning_rate
grad_norm
tokens/sec
GPU_utilization
MFU
memory_usage
checkpoint_size
```

And benchmark:

```text
knowledge
reasoning
math
code
Vietnamese
long-context
```

---

# 51. Research Ablation Plan

Once baseline Astra-1B is trained, branch into controlled experiments.

## A0

Dense Transformer baseline.

```text
24 × Attention
```

## A1

GDN only.

```text
24 × GDN
```

## A2

Hybrid.

```text
18 × GDN
6 × Attention
```

## A3

Hybrid + residual gating.

## A4

Hybrid + MTP.

## A5

Hybrid + N-gram embedding.

This generates an actual research matrix.

---

# 52. Primary Comparison

For fair comparison:

$$
\boxed{
same\ parameters
}
$$

$$
\boxed{
same\ data
}
$$

$$
\boxed{
same\ training\ tokens
}
$$

$$
\boxed{
same\ optimizer
}
$$

$$
\boxed{
same\ hardware\ budget
}
$$

Otherwise an architecture comparison becomes contaminated by compute differences.

---

# 53. Long Context Experiment

At each context size:

$$
4096,\ 8192,\ 16384,\ 32768
$$

measure:

$$
quality
$$

$$
tokens/sec
$$

$$
memory/token
$$

$$
latency/token
$$

$$
retrieval\ accuracy
$$

The central question:

> Does the GDN majority reduce the cost of long context without destroying exact retrieval capability?

---

# 54. Streaming Inference Test

Astra must support:

```text
token 1
token 2
token 3
...
token N
```

without recomputing all previous GDN history.

The desired property is:

$$
O(1)
$$

state growth per GDN stream, apart from fixed state dimensions.

Attention still requires KV cache.

Therefore the hybrid architecture should have:

```text
GDN:
fixed recurrent state

Attention:
KV cache
```

---

# 55. Inference Cache

Attention cache:

```text
K cache
V cache
```

GDN cache:

```text
state matrix
conv state
gate state if required
```

The inference engine should maintain both independently.

---

# 56. Memory Model

At generation time:

```text
weights
+
attention KV cache
+
GDN recurrent states
+
temporary activations
```

The key advantage of Astra's architecture is that GDN layers do not require a full KV cache.

---

# 57. Post-Training

After base pretraining:

```text
Astra-1B Base
      │
      ▼
instruction SFT
      │
      ▼
preference optimization
      │
      ▼
tool-calling SFT
      │
      ▼
Astra-1B-Instruct
```

Do not perform instruction tuning before base-model quality is established.

---

# 58. SFT Dataset Format

Canonical format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Astra."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

The chat template must become part of the released model artifact.

---

# 59. Tool Calling Extension

For later agent capability:

```text
user
  │
  ▼
model
  │
  ▼
tool call
  │
  ▼
environment
  │
  ▼
tool result
  │
  ▼
model
  │
  ▼
final answer
```

Training examples should include:

```text
tool selection
tool arguments
tool result interpretation
error recovery
multi-step execution
```

This is deliberately separated from base pretraining.

---

# 60. Release Structure

Final public repository:

```text
Astra-1B/
│
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── chat_template.jinja
│
├── model-00001-of-000XX.safetensors
├── ...
├── model.safetensors.index.json
│
├── README.md
└── LICENSE
```

This gives Astra the same general portability model as modern Hugging Face repositories.

---

# 61. Experiment Tracking

Every run must have:

```text
experiment_id
git_commit
config_hash
dataset_version
tokenizer_version
seed
GPU topology
CUDA version
PyTorch version
compiler version
```

No anonymous experiment is allowed into the final research table.

---

# 62. Reproducibility

At minimum save:

$$
seed
$$

$$
config
$$

$$
dataset\ manifest
$$

$$
tokenizer
$$

$$
git\ commit
$$

$$
checkpoint
$$

$$
training\ logs
$$

The model should be restartable from any major checkpoint.

---

# 63. Failure Detection

The trainer must automatically detect:

```text
NaN loss
Inf loss
exploding gradient
dead gradients
data corruption
unexpected token distribution
GPU OOM
NCCL failure
checkpoint corruption
```

On failure:

```text
freeze logs
record exception
save diagnostic metadata
attempt safe restart
```

Do not silently continue after NaN.

---

# 64. Dataset Health Monitoring

During training periodically sample batches and report:

```text
language distribution
average sequence length
special-token frequency
document boundary rate
code percentage
Vietnamese percentage
duplicate rate
token entropy
```

This prevents a catastrophic data-pipeline error from burning thousands of GPU-hours unnoticed.

---

# 65. Model Health Monitoring

Periodically inspect:

$$
||W||_2
$$

$$
||g||_2
$$

activation variance

gate saturation

GDN state norm

attention entropy

MTP accuracy.

In particular, watch:

$$
g_{GDN}\approx0
$$

or:

$$
g_{GDN}\approx1
$$

for long periods.

A permanently saturated gate may indicate a failed routing design.

---

# 66. Minimum Acceptance Criteria

Astra-1B v1 is accepted only if:

```text
[✓] 100M reference model trains
[✓] 350M model scales without instability
[✓] 1B model trains continuously
[✓] checkpoint resume is exact
[✓] GDN streaming works
[✓] attention masking is correct
[✓] MTP loss decreases
[✓] validation loss decreases
[✓] no persistent NaN/Inf
[✓] long-context evaluation works
[✓] Hugging Face export works
[✓] inference works
```

---

# 67. Research Acceptance Criteria

The architecture is scientifically interesting only if at least one of these is demonstrated:

$$
\boxed{
quality\ improvement
}
$$

or:

$$
\boxed{
same\ quality,\ lower\ compute
}
$$

or:

$$
\boxed{
same\ quality,\ lower\ inference\ memory
}
$$

or:

$$
\boxed{
better\ long-context\ scaling
}
$$

Otherwise Astra remains an engineering exercise rather than a research contribution.

---

# 68. Version Roadmap

## Astra-100M

Purpose:

$$
\text{correctness}
$$

## Astra-350M

Purpose:

$$
\text{scaling validation}
$$

## Astra-1B v1

Architecture:

$$
GDN + Attention
$$

## Astra-1B v2

Add:

$$
\text{Gated Residual}
$$

## Astra-1B v3

Add:

$$
\text{N-gram Embedding}
$$

## Astra-1B v4

Experiment with:

$$
\text{Sparse Attention}
$$

## Astra-3B

Scale the best configuration.

## Astra-7B

Introduce more aggressive hardware-aware optimization.

## Astra-MoE

Only after dense scaling behavior is understood.

---

# 69. Future MoE Branch

Do not add MoE to v1.

When the architecture is stable, define:

$$
y=\sum_{i\in TopK(x)}g_iE_i(x)
$$

with:

```text
router
8-32 experts
top-2 routing
capacity management
auxiliary load-balance loss
```

The first MoE experiment should compare:

```text
Astra Dense 1B
vs
Astra MoE same active compute
```

rather than simply making the MoE enormous.

---

# 70. Future Sparse Attention Branch

The eventual sparse-attention layer could be:

```text
Query
  │
  ▼
cheap relevance indexer
  │
  ▼
top relevant blocks
  │
  ▼
exact attention
```

This creates the next architectural step:

$$
\boxed{
GDN = memory
}
$$

$$
\boxed{
Sparse Attention = retrieval
}
$$

$$
\boxed{
MoE = capacity
}
$$

That is the point at which Astra begins to approach the architectural philosophy of current high-end hybrid LLMs.

---

# 71. Final Engineering Blueprint

The complete v1 system is therefore:

```text
                         ASTRA-1B
                            │
                     Tokenizer 65K
                            │
                       Embedding
                            │
                            ▼
        ┌─────────────────────────────────────┐
        │             24 Layers               │
        │                                     │
        │ GDN → SwiGLU                        │
        │ GDN → SwiGLU                        │
        │ GDN → SwiGLU                        │
        │ ATT → SwiGLU                        │
        │                                     │
        │ repeat × 6                          │
        └─────────────────────────────────────┘
                            │
                       RMSNorm
                            │
                  ┌─────────┴─────────┐
                  │                   │
              LM Head              MTP-2
                  │                   │
                  └────────┬──────────┘
                           ▼
                        logits
                           │
                      sampling
                           │
                         text
```

Core numbers:

$$
\boxed{
24\ layers
}
$$

$$
\boxed{
18\ GDN + 6\ Attention
}
$$

$$
\boxed{
d_{model}=2048
}
$$

$$
\boxed{
d_{ff}=3456
}
$$

$$
\boxed{
V=65,536
}
$$

$$
\boxed{
16Q/4KV
}
$$

$$
\boxed{
16\ GDN\ heads\times96
}
$$

$$
\boxed{
1B\ parameters
}
$$

$$
\boxed{
4K\rightarrow8K\rightarrow16K\rightarrow32K
}
$$

$$
\boxed{
20B\rightarrow50B\rightarrow100B\ tokens
}
$$

---

# 72. Execution Order

The actual project should be executed in exactly this order:

```text
01. Implement tokenizer
        ↓
02. Build dataset ingestion
        ↓
03. Build filtering + dedup
        ↓
04. Create packed token shards
        ↓
05. Implement reference GDN
        ↓
06. Implement reference Attention
        ↓
07. Implement reference Transformer block
        ↓
08. Implement MTP
        ↓
09. Build 100M model
        ↓
10. Unit tests
        ↓
11. Tiny overfit test
        ↓
12. 100M training
        ↓
13. Optimize kernels
        ↓
14. Distributed training
        ↓
15. 350M scaling test
        ↓
16. 1B initialization
        ↓
17. 20B-token run
        ↓
18. Evaluate
        ↓
19. 50B-token run
        ↓
20. Ablation experiments
        ↓
21. Long-context stages
        ↓
22. Select best checkpoint
        ↓
23. SFT
        ↓
24. Preference optimization
        ↓
25. Tool / agent tuning
        ↓
26. Final HF export
```

# 73. Architect's Decision

The most important architectural decision is not actually the equation inside GDN.

It is the **separation of responsibilities**:

$$
\boxed{
\underbrace{GDN}_{memory}
+
\underbrace{Attention}_{retrieval}
+
\underbrace{SwiGLU}_{transformation}
+
\underbrace{Residual}_{communication}
+
\underbrace{MTP}_{training\ signal}
}
$$

Then the project deliberately evolves through controlled experiments rather than throwing every fashionable component into version 1.

That gives us a much more valuable asset than a random 1B checkpoint:

$$
\boxed{
\text{Astra becomes an experimental LLM platform.}
}
$$

Every later architectural idea can be inserted, measured and compared against the same clean 1B reference point. That is the foundation needed to move from **“I trained an LLM”** to **“I can design and experimentally validate LLM architectures.”**
