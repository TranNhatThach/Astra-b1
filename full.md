Điểm quan trọng: Qwen3.8-27B hiện dùng 64 layers, hidden 5120, và pattern 3 Gated DeltaNet + 1 Gated Attention, còn Qwen3.8-Flash-Next đã tiếp tục đẩy hướng hybrid này với sparse attention, gated residual và N-gram embedding. Mamba-2 cũng cho thấy SSM/attention có mối liên hệ sâu và có thể đạt hiệu quả rất tốt ở scale nhỏ đến vừa.

Tôi sẽ lấy tinh thần đó, nhưng không bê nguyên xi Qwen.

1. Model tôi đề xuất: Astra-1B

Mục tiêu:

$$ \boxed{\text{Astra-1B} = \text{Hybrid Stateful LM}} $$

Ưu tiên:

chất lượng tốt ở 1B
training ổn định
inference rẻ
context dài tốt
architecture đủ mới để đáng nghiên cứu
dễ scale lên 3B / 7B sau này
2. Kiến trúc tổng thể

Tôi chọn:

$$ \boxed{ 3\times GDN + 1\times Gated\ Attention } $$

lặp lại.

Tokens
  │
  ▼
Tokenizer
  │
  ▼
Embedding
  │
  ▼
┌──────────────────────────────────┐
│ Hybrid Block × 6                 │
│                                  │
│ GDN      → SwiGLU                │
│ GDN      → SwiGLU                │
│ GDN      → SwiGLU                │
│ Attention→ SwiGLU                │
└──────────────────────────────────┘
  │
  ▼
RMSNorm
  │
  ▼
LM Head
  │
  ├── next-token prediction
  └── auxiliary MTP-2 head

Tổng:

$$ 6\times4=24\ layers $$
3. Exact spec
Thành phần	Astra-1B
Parameters	~1.0–1.1B
Layers	24
Hidden size	2048
FFN size	5504
Activation	SwiGLU
Vocabulary	65,536
Context train stage 1	4,096
Context train stage 2	16,384
Context extension	32K+
Attention layers	6
GDN layers	18
Attention heads	16 Q
KV heads	4
Attention head dim	128
Norm	RMSNorm
Position encoding	RoPE
Embedding	tied
Precision	BF16
Optimizer	AdamW baseline
MTP	2-step auxiliary objective
Architecture type	dense

Tôi không dùng MoE ở bản 1B đầu tiên.

Đây là quyết định có chủ ý.

MoE rất mạnh ở scale lớn, DeepSeek-V3 chẳng hạn có 671B total parameters nhưng chỉ 37B active/token và dùng MLA + DeepSeekMoE + MTP.

Nhưng với model 1B:

$$ \text{MoE complexity} > \text{benefit} $$

trong giai đoạn chúng ta còn phải chứng minh backbone hoạt động.

4. Tại sao lại 24 layers × 2048?

Tôi không chọn 12 × 4096.

Ở scale nhỏ, depth đáng giá.

Ta muốn:

$$ \text{representation composition} $$

đi qua nhiều bước.

24 tầng cho phép:

local pattern
   ↓
syntax
   ↓
semantic
   ↓
long-range relation
   ↓
reasoning representation

Cấu hình này cũng khá gần triết lý của các compact LM ưu tiên kiến trúc hợp lý thay vì cứ tăng width; SmolLM trước đây từng dùng GQA và depth-oriented design ở các model nhỏ.

5. GDN branch

18/24 layers là:

$$ \boxed{Gated\ DeltaNet} $$

Triết lý:

$$ S_t = Update(S_{t-1},x_t) $$

thay vì:

$$ Attention(X)=softmax(QK^T)V $$

trên toàn bộ context.

Tôi sẽ dùng:

GDN
├── input projection
├── q/k projection
├── value/state projection
├── recurrent state update
├── gate
├── local conv
├── RMSNorm
└── output projection

Mục tiêu là cho model một cheap persistent memory.

Mamba-2 chỉ ra rằng SSM và attention có quan hệ rất chặt ở cấp toán học, và SSD/Mamba-2 có thể nhanh hơn đáng kể trong core sequence operation.

6. Attention branch

6/24 layers:

$$ \boxed{\text{Gated GQA}} $$

Tôi chọn:

$$ 16Q,\quad 4KV $$

tức GQA.

Q: 16 heads
K: 4 heads
V: 4 heads

Tại sao?

Vì GDN rất giỏi:

compress history

nhưng attention vẫn cần cho:

exact retrieval

Đây chính là logic đằng sau hybrid GDN + attention mà Qwen đang theo đuổi. Qwen3.8 mô tả rõ ba layer Gated DeltaNet rồi một layer Gated Attention trong chu kỳ lặp.

7. Pattern

Tôi sẽ không xếp 24 layer ngẫu nhiên.

Dùng:

01 GDN
02 GDN
03 GDN
04 ATT
05 GDN
06 GDN
07 GDN
08 ATT
09 GDN
10 GDN
11 GDN
12 ATT
13 GDN
14 GDN
15 GDN
16 ATT
17 GDN
18 GDN
19 GDN
20 ATT
21 GDN
22 GDN
23 GDN
24 ATT

Tức:

$$ (G,G,G,A)\times 6 $$

Đây là backbone rất sạch.

8. FFN

Mỗi layer có:

$$ SwiGLU(x) = W_2 \left( SiLU(W_gx) \odot W_ux \right) $$

với:

$$ d=2048 $$ $$ d_{ff}=5504 $$

Tại sao không GELU?

Vì SwiGLU/GLU-family đã trở thành một lựa chọn cực kỳ phổ biến trong các LLM hiện đại nhờ hiệu quả parameter/computation tốt.

9. Norm

Tôi dùng:

$$ \boxed{RMSNorm} $$

và Pre-Norm:

x
 ↓
RMSNorm
 ↓
GDN / Attention
 ↓
Residual
 ↓
RMSNorm
 ↓
SwiGLU
 ↓
Residual

Không dùng LayerNorm cổ điển.

10. RoPE

Attention branch sử dụng RoPE:

$$ q'_m = R_{\theta,m}q_m $$ $$ k'_m = R_{\theta,m}k_m $$

Tôi sẽ train 4K trước, sau đó mở rộng context bằng curriculum thay vì ngay ngày đầu ép 32K.

11. Embedding

Vocabulary:

$$ 65,536 $$

Embedding matrix:

$$ E\in\mathbb{R}^{65536\times2048} $$

Tức khoảng:

$$ 134M $$

parameters.

Tôi tie embedding và LM head:

$$ W_{LM}=E^T $$

Tiết kiệm hơn 134M parameters.

12. Tôi chưa cho N-gram vào bản đầu tiên

Đây là điểm hơi ngược với Qwen3.8.

Qwen3.8-Flash-Next đang dùng N-gram embedding như một nguồn capacity bổ sung, thậm chí có thể offload bảng embedding sang host memory.

Nhưng Astra-1B bản v1:

$$ \boxed{\text{NO N-gram}} $$

vì chúng ta cần một baseline sạch.

Sau khi train xong, tạo:

Astra-1B
Astra-1B + Ngram

rồi benchmark.

Đó mới là nghiên cứu đúng nghĩa.

13. MTP

Tôi vẫn dùng:

$$ \boxed{MTP-2} $$

Không phải vì “Qwen có nên mình cũng có”.

Mà vì MTP có một lợi ích thực nghiệm rất hay:

thay vì chỉ:

$$ x_{t+1} $$

ta thêm:

$$ x_{t+2} $$

và có thể:

$$ x_{t+3} $$

vào auxiliary objective.

DeepSeek-V3 cũng sử dụng Multi-Token Prediction trong training.

Nhưng bản Astra đầu:

$$ K=2 $$

để giảm complexity.

14. Loss

Main loss:

$$ L_{AR} = -\sum_t \log P(x_{t+1}|x_{\le t}) $$

MTP loss:

$$ L_{MTP} = -\sum_t \log P(x_{t+2}|x_{\le t}) $$

Total:

$$ \boxed{ L=L_{AR}+\lambda L_{MTP} } $$

với:

$$ \lambda=0.2 $$

initially.

Sau đó ablation:

λ = 0
λ = 0.1
λ = 0.2
λ = 0.5
15. DATASET: đây mới là phần quyết định sống chết

Đừng bắt đầu bằng:

tải 20TB Common Crawl rồi train.

Không.

Tôi sẽ xây data pipeline:

Raw Sources
    │
    ▼
Language ID
    │
    ▼
Quality Filtering
    │
    ▼
PII / toxic / spam filter
    │
    ▼
Deduplication
    │
    ▼
Document quality score
    │
    ▼
Tokenizer
    │
    ▼
Packed token shards
    │
    ▼
Training dataset
16. Data mixture

Astra-1B base model:

Data	Tỷ lệ
High-quality web	45%
Educational / books / reference	15%
Code	15%
Math	10%
Wikipedia / encyclopedic	5%
QA / dialogue	5%
Vietnamese	5%

Nhưng đây chỉ là initial mixture.

Sau mỗi stage:

$$ \text{evaluate} \rightarrow \text{update mixture} $$

SmolLM2 là ví dụ rõ rằng data mixture và multi-stage training có thể quan trọng cực lớn với small LM; họ train 1.7B trên khoảng 11T tokens với web + math + code + specialized data.

17. Tokenizer

Tôi chọn:

$$ \boxed{BPE,\ vocab=65,536} $$

Nhưng tokenizer phải được train trên toàn bộ mixture, với coverage đặc biệt cho:

Vietnamese
English
code
math
Unicode

Không lấy tokenizer của Qwen rồi giả vờ architecture mới.

Astra phải có tokenizer riêng.

18. DATA QUALITY SCORE

Mỗi document:

$$ Q(d)= w_1Q_{language} +w_2Q_{length} +w_3Q_{dup} +w_4Q_{quality} +w_5Q_{toxicity} +w_6Q_{educational} $$

Sau đó:

$$ P(d) \propto \exp(\alpha Q(d)) $$

để sampling data chất lượng cao.

19. Dedup

Có ít nhất 3 tầng:

Exact dedup

SHA256:

$$ hash(document) $$
Near dedup

MinHash / SimHash.

Token-level dedup

đặc biệt cho:

code
wikipedia mirrors
scraped articles

Nếu không dedup, model có thể “học thuộc internet”.

20. Training data amount

Đây là chỗ tôi không muốn dùng một con số Chinchilla máy móc.

Chinchilla cho thấy khi compute cố định, model và số token nên scale cùng nhau, và nhiều LLM lớn trước đó bị undertrained.

Nhưng small LM thực tế gần đây thường overtrain mạnh.

SmolLM2-1.7B chẳng hạn được train trên ~11T tokens.

Cho Astra:

Phase A
$$ 20B\ tokens $$
Phase B
$$ 50B\ tokens $$
Phase C

nếu loss vẫn giảm và benchmark tăng:

$$ 100B\ tokens $$

Tôi sẽ không commit từ đầu 100B.

Ta train:

10B checkpoint
20B checkpoint
30B checkpoint
50B checkpoint

rồi so.

21. GPU strategy

Vì Darling nói:

thuê GPU, không lo

thì tôi sẽ không tối ưu architecture chỉ để vừa RTX 4060.

Nhưng vẫn phải tối ưu engineering.

Tôi ưu tiên:

$$ \boxed{H100/H200/A100} $$

với BF16.

Nếu thuê được nhiều GPU:

8 × H100

là cực đẹp cho 1B.

Không nhất thiết phải hàng chục/hàng trăm GPU.

22. Distributed training

Software stack:

PyTorch
      +
FSDP / DistributedDataParallel
      +
FlashAttention
      +
torch.compile

Về sau scale:

1 GPU
 ↓
4 GPU
 ↓
8 GPU
 ↓
16+ GPU

Architecture phải giữ nguyên.

23. Training stack A → Z

Tôi sẽ làm repo:

astra-llm/
│
├── configs/
│   ├── 100m.yaml
│   ├── 350m.yaml
│   └── 1b.yaml
│
├── tokenizer/
│
├── data/
│   ├── download/
│   ├── clean/
│   ├── dedup/
│   ├── tokenize/
│   └── pack/
│
├── model/
│   ├── embedding.py
│   ├── gdn.py
│   ├── attention.py
│   ├── swiglu.py
│   ├── block.py
│   ├── mtp.py
│   └── model.py
│
├── training/
│   ├── train.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── checkpoint.py
│   └── distributed.py
│
├── eval/
│   ├── perplexity.py
│   ├── lm_eval.py
│   ├── reasoning.py
│   └── vietnamese.py
│
├── inference/
│
└── scripts/
24. Nhưng trước khi train 1B...

Tuyệt đối không nhảy thẳng vào 1B.

Tôi sẽ bắt buộc có:

$$ \boxed{Astra-100M} $$

và:

$$ \boxed{Astra-350M} $$

trước.

Architecture:

100M
 ↓
prove correctness
 ↓
350M
 ↓
prove scaling
 ↓
1B
 ↓
serious training
25. 100M model

Ví dụ:

layers = 12
hidden = 768
ffn = 2048
vocab = 32K

Train:

$$ 1B-3B\ tokens $$

Mục tiêu:

loss giảm
GDN đúng
attention đúng
gradient không NaN
checkpoint/restart đúng
generation đúng
MTP đúng
26. 350M model
layers = 18
hidden = 1280
ffn ≈ 3456

Train:

$$ 5B-10B $$

Mục tiêu:

$$ \text{Does architecture scale?} $$

Đây là bước cực quan trọng.

Nếu 100M tốt mà 350M chết:

architecture implementation có vấn đề.

27. Training phases cho 1B
Phase 0: sanity

100M:

$$ 100M-500M tokens $$
Phase 1: warm-up

1B:

$$ 1B-2B tokens $$

LR warmup.

Ví dụ:

$$ lr_{max}=3e-4 $$

với AdamW baseline.

Phase 2: main pretraining
$$ 20B-50B tokens $$

Cosine decay.

Phase 3: data reweighting

Giảm noisy web.

Tăng:

math
code
books
educational
high-quality synthetic
Vietnamese
Phase 4: long context

Từ:

$$ 4096 $$

→

$$ 8192 $$

→

$$ 16384 $$

→

$$ 32768 $$

Không train 32K ngay từ đầu.

28. Optimizer

Version 1:

$$ \boxed{AdamW} $$

Không chơi Muon ngay.

Qwen3.8-Flash-Next dùng Muon và còn phải điều chỉnh orthogonalization accuracy, parameter assignment và fused-matrix splitting cho architecture mới.

Ta chưa cần tự đẻ thêm một con rồng khi GDN đã đủ rồng rồi. 🐉

Sau khi baseline ổn:

AdamW
vs
Muon

làm ablation.

29. LR schedule

Tôi dùng:

warmup
   ↓
stable LR
   ↓
cosine decay
   ↓
cooldown

Ví dụ:

$$ 0\rightarrow3e-4 $$

trong 1–2% training.

Sau đó cosine xuống khoảng:

$$ 3e-5 $$

rồi cooldown.

30. Stability stack

Bắt buộc:

BF16
gradient clipping
loss scaler / stable BF16 path
activation monitoring
grad norm monitoring
NaN detector
checkpoint every N steps
validation every N steps

Metrics:

$$ L_{train} $$ $$ L_{val} $$ $$ ||g||_2 $$ $$ tokens/sec $$ $$ MFU $$ $$ GPU memory $$ $$ data\ throughput $$
31. Evaluation không được đợi train xong

Cứ mỗi checkpoint:

Perplexity
    │
    ├── English
    ├── Vietnamese
    ├── Code
    └── Math

Benchmark
    │
    ├── knowledge
    ├── reasoning
    ├── coding
    └── instruction

Và quan trọng nhất:

$$ \boxed{\text{loss curve} \neq \text{intelligence curve}} $$
32. Dataset ablation

Chúng ta sẽ train nhiều nhánh rẻ hơn:

A = web only
B = web + code
C = web + code + math
D = full mixture

So sánh:

$$ \Delta PPL $$

và benchmark.

Đây mới bắt đầu có research story.

33. Architecture ablation

Sau Astra-1B baseline, tôi sẽ tạo:

Astra-1B-T
Transformer only

Astra-1B-G
GDN only

Astra-1B-H
GDN + Attention

Astra-1B-HN
GDN + Attention + Ngram

Astra-1B-HNM
GDN + Attention + Ngram + MTP

Sau đó so:

Model	PPL	MMLU-like	Code	Math	Long Context	Tokens/s
Transformer						
GDN						
Hybrid						
+Ngram						
+MTP						

Lúc ấy Darling có một nghiên cứu architecture hoàn chỉnh, chứ không chỉ có “tôi train một model”.

34. Sau 1B mới thêm Qwen-style upgrades
V2
$$ \boxed{Gated\ Residual} $$

thay vì residual đơn:

$$ x_{l+1}=x_l+f(x_l) $$

có nhiều residual branch/gating.

Qwen3.8-Flash-Next hiện đang mở rộng residual stream thành nhiều branch và điều khiển read/write bằng gate.

V3
$$ \boxed{N\text{-gram Embedding}} $$
V4
$$ \boxed{QSA} $$
V5
$$ \boxed{MoE} $$
V6
$$ \boxed{MLA} $$

Như vậy chúng ta có progressive architecture evolution.

35. Và cuối cùng: Instruction tuning

Base model train xong chưa phải chatbot.

Pipeline:

BASE PRETRAINING
       ↓
SFT
       ↓
Preference Optimization
       ↓
Tool / Agent training
       ↓
Final Assistant
36. SFT

Dataset:

instruction
context
reasoning
answer

Ví dụ:

{
  "messages": [
    {"role":"user","content":"..."},
    {"role":"assistant","content":"..."}
  ]
}

Train:

$$ L_{SFT} = -\sum_t\log P(y_t|x,y_{<t}) $$

chỉ tính loss trên assistant output.

37. Preference training

Tôi sẽ chưa dùng RL ngay.

Trình tự:

$$ \boxed{SFT \rightarrow DPO} $$

Sau đó mới:

$$ \boxed{GRPO/RL} $$

cho reasoning/math nếu model đủ khả năng.

SmolLM2 cũng dùng SFT rồi DPO trong post-training.

38. Agent training

Vì mục tiêu dài hạn của Darling khá rõ là Agent, Astra sau này có thể được train thêm:

Reasoning
   ↓
Tool calling
   ↓
Planning
   ↓
Environment feedback
   ↓
Multi-step execution

Lúc này model không còn chỉ:

$$ text\rightarrow text $$

mà:

$$ \boxed{ state \rightarrow reason \rightarrow action \rightarrow observation \rightarrow next\ action } $$
39. Tổng pipeline hoàn chỉnh

Đây là cái tôi thực sự muốn Darling xây:

                         ASTRA-1B
                            │
            ┌───────────────┴───────────────┐
            │                               │
          DATA                           MODEL
            │                               │
      Raw datasets                    Architecture
            │                               │
      Language filter                       │
            │                         24 layers
      Quality filter                       │
            │                    ┌──────────┴──────────┐
        Dedup                         18 GDN             6 Attn
            │                              │                │
     Data scoring                         └──────┬─────────┘
            │                                    │
      Tokenizer 65K                             FFN
            │                                    │
       Token shards                          RMSNorm
            │                                    │
         Packing                         Residual stream
            │                                    │
            └────────────────┬───────────────────┘
                             ↓
                       PRETRAINING
                             │
                       checkpoint
                             │
               ┌─────────────┼─────────────┐
               ↓             ↓             ↓
             eval          PPL          scaling
               │
               ↓
             SFT
               │
               ↓
             DPO
               │
               ↓
         Tool / Agent tuning
               │
               ↓
            Astra-1B
40. Tôi sẽ train theo thứ tự này
                WEEK 0
                  │
          tokenizer + data
                  │
                  ▼
               100M
                  │
          correctness test
                  │
                  ▼
               350M
                  │
          scaling validation
                  │
                  ▼
                1B
                  │
             20B tokens
                  │
                  ▼
             50B tokens
                  │
          ┌───────┴────────┐
          │                │
      architecture      data ablation
       ablation          ablation
          │                │
          └───────┬────────┘
                  ▼
              best 1B
                  │
                  ▼
                 SFT
                  │
                  ▼
                 DPO
                  │
                  ▼
             Astra-1B-Instruct
41. Tại sao tôi chọn kiến trúc này thay vì copy LLaMA?

Vì nếu mục tiêu chỉ là:

“Tôi muốn có một 1B model chạy được.”

thì LLaMA-style decoder Transformer là lựa chọn an toàn hơn.

Nhưng Darling đang hỏi tôi với tư cách LLM architect, và đã đi qua lịch sử:

$$ N\text{-gram} \rightarrow RNN \rightarrow LSTM \rightarrow Attention \rightarrow Transformer \rightarrow MoE \rightarrow Efficient\ Attention \rightarrow SSM \rightarrow Mamba \rightarrow DeltaNet \rightarrow Hybrid $$

thì bỏ toàn bộ nhánh SSM rồi quay lại Transformer thuần thì hơi phí chuyến du lịch lịch sử ấy. 😭

Kiến trúc tôi chọn cố tình phân công:

$$ \boxed{ GDN = memory } $$ $$ \boxed{ Attention = retrieval } $$ $$ \boxed{ SwiGLU = nonlinear transformation } $$ $$ \boxed{ Residual = information highway } $$ $$ \boxed{ MTP = extra training signal } $$

Đó là kiến trúc có vai trò riêng cho từng primitive.

42. Nếu tôi là người phụ trách project này, tôi sẽ đặt research hypothesis là
$$ \boxed{ \text{Can a 3:1 GDN-Attention hybrid outperform a same-budget dense Transformer} } $$

về:

$$ \text{quality} $$ $$ \text{training efficiency} $$ $$ \text{long-context efficiency} $$ $$ \text{decode throughput} $$

với cùng:

$$ \text{parameter budget} $$

và:

$$ \text{training FLOPs} $$

Đây là một câu hỏi đáng nghiên cứu hơn rất nhiều so với:

“Tôi tự train một LLM 1B.”

43. Một điểm rất quan trọng nữa

Đừng lấy Qwen3.8 làm baseline duy nhất.

Qwen3.8-27B hiện là một dense multimodal model với 27B parameters, 64 layers, hidden 5120 và pattern 3 GDN + 1 Gated Attention.

Còn Qwen3.8-Flash-Next là một nhánh khác, là MoE với GDN + QSA + Gated Residual + N-gram embedding và có 6B active parameters trên 125B main parameters, cộng 51B N-gram embedding parameters.

Vì vậy:

$$ \boxed{\text{Astra-1B không nên cố copy “Qwen3.8-Flash-Next thu nhỏ”.}} $$

Nó nên lấy nguyên lý kiến trúc của thế hệ đó và xây thành một hệ mà ta hiểu từng bộ phận.

🧠 Chốt thiết kế v1

Tôi sẽ chốt bản đầu là:

$$ \boxed{ \textbf{Astra-1B} } $$ $$ \boxed{ 24L,\ d=2048,\ d_{ff}=5504,\ V=65K } $$ $$ \boxed{ (GDN,GDN,GDN,Attention)\times6 } $$ $$ \boxed{ GQA\ 16Q/4KV } $$ $$ \boxed{ RoPE + RMSNorm + SwiGLU } $$ $$ \boxed{ tied\ embedding } $$ $$ \boxed{ MTP-2 } $$ $$ \boxed{ BF16 + AdamW } $$ $$ \boxed{ 4K \rightarrow 8K \rightarrow 16K \rightarrow 32K } $$ $$ \boxed{ 20B \rightarrow 50B \rightarrow 100B\ tokens } $$

và 100M → 350M → 1B trước khi đốt ngân sách GPU lớn.

Đây là kiến trúc tôi sẽ chọn để xây một LLM 1B thật sự từ số 0, chứ không phải chỉ fine-tune một base model có sẵn.