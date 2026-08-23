# ACE-Net: A Fine-Grained Deepfake Detection Model with Multimodal Emotional Consistency

> **Source:** Yu, S.; Chen, X.; Sheng, Y.; Zhang, H.; Li, X.; Yu, S. *ACE-Net: A Fine-Grained Deepfake Detection Model with Multimodal Emotional Consistency.* Electronics **2025**, 14, 4420. https://doi.org/10.3390/electronics14224420
>
> **Authors:** Shaoqian Yu, Xingyu Chen, Yuzhe Sheng\*, Han Zhang, Xinlong Li, Sijia Yu
> School of Computer Science, Hunan University of Technology and Business, Changsha 410205, China

---

## Table of Contents

1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Key Contributions](#key-contributions)
4. [Architecture Overview](#architecture-overview)
5. [Module 1 — Speech–Text Emotion Feature Extractor](#module-1--speechtext-emotion-feature-extractor)
   - [MDCNN Acoustic Branch](#mdcnn-acoustic-branch)
   - [Bidirectional Cross-Modal Attention](#bidirectional-cross-modal-attention)
6. [Module 2 — Dynamic–Temporal Facial Emotion Feature Extractor](#module-2--dynamictemporal-facial-emotion-feature-extractor)
   - [Keyframe Selection (Coarse-to-Fine)](#keyframe-selection-coarse-to-fine)
   - [Lightweight Spatiotemporal Feature Extraction (FV-LiteNet)](#lightweight-spatiotemporal-feature-extraction-fv-litenet)
7. [Module 3 — Multimodal Consistency Discriminator](#module-3--multimodal-consistency-discriminator)
8. [Training Strategy](#training-strategy)
9. [Computational Efficiency](#computational-efficiency)
10. [Datasets & Forgery Synthesis](#datasets--forgery-synthesis)
11. [Results](#results)
    - [Emotion Recognition (Unimodal)](#emotion-recognition-unimodal)
    - [Forgery Detection by Type](#forgery-detection-by-type)
    - [Ablation Study](#ablation-study)
    - [Comparison with SOTA on DFDC](#comparison-with-sota-on-dfdc)
12. [Key Equations Reference](#key-equations-reference)
13. [Abbreviations](#abbreviations)

---

## Abstract

ACE-Net detects deepfakes by exploiting a fundamental weakness of generative models: **emotional cues across speech and face are inherently difficult to synchronize**. Rather than hunting for low-level synthesis artifacts, ACE-Net checks whether what a person *says* (prosody + words) is emotionally consistent with what their *face shows*.

**Result:** AUC **0.921** on DFDC — state-of-the-art at publication.

---

## Problem Statement

### Why artifact-based detectors fail

| Limitation | Detail |
|---|---|
| Single-modality | XceptionNet, ResNet, EfficientNet analyze one stream; generator-specific fingerprints overfit |
| Low-level fragility | Generative models continuously erase artifact traces (GAN, diffusion) |
| Shallow cross-modal fusion | Simple concatenation / cosine similarity cannot model non-linear inter-modal relationships |
| Decision-level fusion | Voting / weighting loses fine-grained feature evidence |

### Why emotion consistency works

- Emotional information is **intricately woven** into both audio prosody and facial muscle movement
- Generative models struggle to perfectly synchronize this cross-modal correlation
- High-level semantic conflicts are **harder to counterfeit** than low-level pixel artifacts

### Two core research gaps addressed

1. **Inadequate unimodal representation** — existing models fail to capture fine-grained *dynamics* of emotional expressions
2. **Superficial cross-modal fusion** — concatenation / fixed metrics miss deep non-linear relationships between high-dimensional features

---

## Key Contributions

```
1. MDCNN  — lightweight Multi-granularity-attention Depthwise CNN
             parallel channel-spatial attention
             global + local pooling, DSC backbone
             refined multi-scale acoustic feature perception

2. Coarse-to-Fine visual frame selection
             optical flow motion gating  →  coarse filter
             MobileNetV3 expressiveness head  →  fine filter
             focuses computation on emotionally salient keyframes

3. Multi-aspect Consistency Discriminator
             f = [z_at ; z_v | |z_at - z_v| | z_at ⊙ z_v]  ∈ R^{4d}
             captures aggregation + conflict + synergy
             end-to-end MLP learns without explicit thresholds
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Video Segment (2–4 s)                     │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
  ┌───────────────┐             ┌───────────────┐
  │  AUDIO STREAM │             │  VISUAL STREAM│
  │  16 kHz       │             │  30 fps       │
  │  80-band      │             │  224×224      │
  │  log-Mel spec │             │  face-aligned │
  └───────┬───────┘             └───────┬───────┘
          │                             │
          │  ┌──────────────┐           │  ┌──────────────────────┐
          │  │  ASR/BERT    │           │  │  Coarse Filtering    │
          │  │  (frozen)    │           │  │  optical flow motion │
          │  │  → tokens    │           │  │  gate (ρ=80 pct)     │
          │  └──────┬───────┘           │  └──────────┬───────────┘
          │         │                   │             │
          ▼         ▼                   │  ┌──────────────────────┐
   ┌─────────────────────┐             │  │  Fine Filtering      │
   │     MDCNN           │             │  │  MobileNetV3 head    │
   │  4 DSC stages       │             │  │  Top-K (θ=0.7, K=8)  │
   │  [64,128,256,256]   │             │  └──────────┬───────────┘
   │  Channel-Spatial    │             │             │
   │  Attention (CBAM)   │             │             ▼
   └──────────┬──────────┘             │   ┌──────────────────┐
              │                        │   │  FV-LiteNet      │
              │  z_a ∈ R^{T×d}        │   │  GhostNet-based  │
              ▼                        │   │  spatial-channel │
   ┌──────────────────────┐            │   │  attention head  │
   │  Bidirectional       │            │   └──────────┬───────┘
   │  Cross-Attention     │            │              │
   │  A→T and T→A         │            │   z_v ∈ R^d  │
   │  h=4 heads, d=256    │            │              │
   └──────────┬───────────┘            └──────────────┘
              │                                │
              │   z_at ∈ R^d                   │  z_v ∈ R^d
              └──────────────┬─────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  Multi-aspect Fusion         │
              │  f = [z_at; z_v |            │
              │       |z_at−z_v| |           │
              │       z_at ⊙ z_v]  ∈ R^{4d} │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  MLP Discriminator           │
              │  512 → 128 → 1 (sigmoid)     │
              │  BN + Dropout(0.3)           │
              │  BCE loss, end-to-end        │
              └──────────────┬───────────────┘
                             ▼
                    p ∈ (0,1)  →  fake/real
```

**Default embedding dimension:** `d = 256`

---

## Module 1 — Speech–Text Emotion Feature Extractor

Produces `z_at ∈ R^d` — a compact emotion-sensitive joint acoustic-textual embedding.

### MDCNN Acoustic Branch

**Purpose:** Extract fine-grained emotional cues from log-Mel spectrograms efficiently.

**Input:** `X ∈ R^{F×T}` where F=80 (mel bands), T=time steps (25 ms window, 10 ms hop)

#### Architecture

```
log-Mel Spectrogram  X ∈ R^{80×T}
         │
         ▼
  3×3 DSConv (init)
         │
  ┌──────────────────────────────────────────────┐
  │         4 DSC Stages                         │
  │  filters: [64, 128, 256, 256]                │
  │  strides: [2,  2,   2,   1 ]                 │
  │  BatchNorm + ReLU throughout                 │
  └──────────────────┬───────────────────────────┘
                     │
            A ∈ R^{F×T×C}  (C=256)
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
  ┌─────────────┐        ┌─────────────────┐
  │  CHANNEL    │        │  SPATIAL        │
  │  ATTENTION  │        │  ATTENTION      │
  │             │        │                 │
  │  GAP + GMP  │        │  Mean + Max     │
  │  → [2C]     │        │  → concat U     │
  │  Shared MLP │        │  3×3 DSConv     │
  │  [2C→C/r→C] │        │  → score S      │
  │  r=8        │        │  → sigmoid      │
  │  → sigmoid  │        │  → w_t ∈ R^T    │
  │  → w_c ∈R^C │        └──────┬──────────┘
  └──────┬──────┘               │
         └───────────┬──────────┘
                     ▼
         Ã = (w_c ⊗ 1_F ⊗ 1_T) ⊙ (1_C ⊗ 1_T ⊗ w_t) ⊙ A
                     │
         Average along frequency axis
                     │
         F ∈ R^{T×C}  →  linear projection  →  z_a ∈ R^{T×d}
```

**Channel attention:** `w_c ∈ R^C` — highlights most emotion-relevant frequency channels (global tonal + transient local bursts via dual-pooling GAP+GMP, CBAM-inspired)

**Spatial attention:** `w_t ∈ R^T` — highlights time steps with strongest emotional evidence (3×3 DSC instead of large-kernel for efficiency)

**Key design choice:** DSC factorizes 2D conv into depthwise + pointwise → ≈1/9 MACs vs standard 3×3 conv.

---

### Bidirectional Cross-Modal Attention

**Purpose:** Deeply fuse acoustic tokens `z_a` with BERT text tokens `T_0` into joint embedding `z_at`.

**Config:** `d=256`, `h=4` heads, dropout=0.1, both streams projected to shared dimension.

#### Two parallel attention streams

```
Acoustic → Textual (A→T):
  Q_A = A·W^Q_A    (acoustic queries)
  K_T = T·W^K_T    (text keys)
  V_T = T·W^V_T    (text values)

  Attn_{A→T} = softmax(Q_A K_T^⊤ / √d_h + M_{A→T}) · V_T

Textual → Acoustic (T→A):
  Q_T = T·W^Q_T    (text queries)
  K_A = A·W^K_A    (acoustic keys)
  V_A = A·W^V_A    (acoustic values)

  Attn_{T→A} = softmax(Q_T K_A^⊤ / √d_h + M_{T→A}) · V_A
```

`M` = padding mask. `W^Q, W^K, W^V` = learned linear projections. Each direction: head outputs concatenated → linear projection → Residual + LayerNorm.

#### Fusion

```
ā  = mean-pool(Attn_{A→T}) over time
T̄  = mean-pool(Attn_{T→A}) over time
z_at = ā + T̄             (joint acoustic-textual embedding)
```

> **Frozen:** ASR/BERT backbone — only `W_A`, `b_A`, `W_T`, `b_T`, cross-attention params are trainable (stability + efficiency).

**Intuition:** "How it is said" (acoustics) + "what is said" (text) → holistic emotion representation.

---

## Module 2 — Dynamic–Temporal Facial Emotion Feature Extractor

Produces `z_v ∈ R^d` — emotionally salient visual representation.

**Key insight:** Emotional cues in video are *sparse and transient* — peak at expression change moments, not uniformly present. Brute-force processing all frames wastes compute and dilutes signal.

### Keyframe Selection (Coarse-to-Fine)

#### Stage 1 — Coarse: Motion Gating (optical flow)

```python
v_t(x) = Flow(I_{t-1}, I_t)[x]          # pixel-wise displacement at x
m_t = (1/|Ω|) · Σ_{x∈Ω} ||v_t(x)||_2   # mean magnitude over face region Ω

# Adaptive local threshold (sliding window w=0.5s, W=⌊w·fps⌋)
τ_t = Percentile_ρ({m_{t'} : |t'-t| ≤ W/2})   # ρ=80 (default)
G   = {t | m_t ≥ τ_t}                           # motion-gated candidate set
```

Prunes static / low-motion segments rapidly.

#### Stage 2 — Fine: Expressiveness Verification (MobileNetV3)

```python
c_t = σ(w^⊤ · g(I_t) + b)  ∈ (0, 1)    # expressiveness score
# g(·) = MobileNetV3 head, fixed scorer (no per-frame labels needed)

# Confidence filtering + Top-K ranking
θ = 0.7   # discard low-score frames
K = 8     # keyframes kept

K = Top-K({(t, c_t) : t ∈ G, c_t ≥ θ})
```

> The MobileNetV3 head is **not separately supervised** on the target datasets — used as a fixed scorer for emotional salience filtering.

#### Temporal alignment

Selected visual keyframes `K` mapped to nearest audio index `τ` via fps/audio frame rate ratio → ensures temporal consistency for multimodal fusion.

---

### Lightweight Spatiotemporal Feature Extraction (FV-LiteNet)

**Backbone:** GhostNet — generates feature maps via cheap linear operations.

**Modifications from stock GhostNet:**
- Keep shallow + mid layers (sensitive to local deformations)
- **Remove SE modules in last two stride-2 Ghost bottlenecks** (reduce params + latency)
- Replace classification head with **joint spatial-channel attention head**

```
Spatial Attention Head:                 Channel Attention Head:
  Final feature map                       Final feature map
       │                                       │
  Conv2D + BN ReLU                        Conv2D + BN ReLU
       │                                       │
  DWConv 3×3                                  MLP
       │                                       │
  Spatial Attention Map               Channel Attention Map
       │                                       │
  Sigmoid                                  Sigmoid
       └──────────────┬────────────────────────┘
                      ⊕ broadcasting
                      │
               Feature Embedding
```

#### Per-frame encoding + attention pooling

```python
f_t^v = h(I_t) ∈ R^{C_v}              # FV-LiteNet encoder, K=8 frames
F = [f_t^v]_{t∈K} ∈ R^{K×C_v}

# Attention weights from expressiveness scores c_t
α_t = exp(β·c_t) / Σ_{t'∈K} exp(β·c_{t'})   # β=5 (temperature)

# Weighted aggregation → project to shared dim d
z_v = (Σ_{t∈K} α_t · f_t^v) · W_v    W_v ∈ R^{C_v×d}
```

Higher expressiveness score → larger attention weight → more influence on final representation.

---

## Module 3 — Multimodal Consistency Discriminator

**Purpose:** Given `z_at` (speech-text) and `z_v` (visual), determine genuine/fake.

### Multi-aspect fusion vector

```
f = [z_at ; z_v | |z_at − z_v| | z_at ⊙ z_v]  ∈ R^{4d}
```

| Component | Operation | Captures |
|---|---|---|
| `[z_at ; z_v]` | Concatenation | Complete information from both modalities |
| `\|z_at − z_v\|` | Element-wise absolute difference | **Inter-modal conflict** — large = emotional mismatch |
| `z_at ⊙ z_v` | Element-wise product | **Synergy** — correlated activations reinforce in genuine pairs |

### MLP Discriminator (inverted triangle)

```python
h_1 = Drop(ReLU(BN(W_1 · f + b_1)))    W_1 ∈ R^{512×4d}
h_2 = Drop(ReLU(BN(W_2 · h_1 + b_2))) W_2 ∈ R^{128×512}
p   = σ(w_o^⊤ · h_2 + b_o)             p ∈ (0,1)
```

- `dim(h_1)=512`, `dim(h_2)=128`, Dropout=0.3 after each hidden layer
- Output `p` → probability of being **fake**
- Decision threshold: **0.5** (no external similarity threshold needed)

### Loss

```
L_BCE = −[y·log(p) + (1−y)·log(1−p)]    y ∈ {0,1}  (0=genuine, 1=fake)
```

Full end-to-end backprop: gradients flow through MLP → fusion → unimodal encoders. Upstream feature extractors refine representations for the forgery task.

---

## Training Strategy

**Two-stage decoupled training** — deliberately designed to learn *semantic* inconsistency, not low-level synthesis artifacts.

### Stage 1 — Unimodal Feature Learning

```
Train MDCNN + FV-LiteNet independently as emotion classifiers
Data: GENUINE only (CREMA-D, MELD, SAVEE)
Goal: robust emotion feature spaces grounded in authentic expressions

→ Freeze both feature extractors after stage 1
```

> **Critical:** Training exclusively on genuine data prevents any bias from synthetic data. The frozen feature space encodes authentic emotional expression patterns.

### Stage 2 — Consistency Discrimination

```
Train: fusion module + MLP discriminator (extractors frozen)
Data: balanced mixed dataset

Negative class (consistent, genuine):  same identity, same emotion (S same, A same)
Positive class (inconsistent, forged):
  ├── Emotional Tampering    (S same, emotion A ≠ B)    — 50%
  └── Cross-Identity Spliced (S1 ≠ S2, A ≠ B)          — 50%

Class balance: 1:1 genuine:forged, positive samples stratified evenly across subtypes
```

**Why this matters:** Forces model to learn semantic consistency across *identity and emotion*, not speaker-specific cues or synthesis artifacts.

---

## Computational Efficiency

| Component | Efficiency Mechanism | Params (M) | Analytical Reduction |
|---|---|---|---|
| MDCNN | 3×3 DSC; GAP+GMP (r=8) | 0.019 | ≈1/9 MACs vs standard 3×3 conv |
| Keyframe selection | K frames instead of all T | — | ×(K/T) ≈ 0.07–0.13 |
| FV-LiteNet | Truncated GhostNet; last 2 SE removed | 1.2 | — |
| Fusion MLP | Small inverted MLP on 4d-dim vector | 0.59 | — |

**Total:** ~1.8M parameters. Suitable for resource-constrained and real-time deployment.

---

## Datasets & Forgery Synthesis

### Emotion Corpora Used

| Dataset | Type | Split Strategy |
|---|---|---|
| CREMA-D | Audio-visual emotion (primary) | 80/10/10 holdout |
| MELD | Multi-party conversation | 80/10/10 holdout |
| SAVEE | Small British male speakers | 10-fold cross-validation |

### Forgery Synthesis (CREMA-D based)

Two paradigms designed to expose emotional consistency failures:

#### Paradigm 1 — Emotional Tampering

```
Goal: create audio-visual emotional conflict, preserve speaker identity

Input:  video of speaker S with face emotion A, transcript T
Step 1: synthesize new speech with emotion B ≠ A
        → Melotron TTS (emotion-controllable)
        → conditioned on target emotion B, original transcript T
Step 2: apply voice conversion to preserve identity
        → RVC (Retrieval-based Voice Conversion)
        → transfer speaker S timbre → â_{S,B}
Step 3: quality gate
        → x-vector cosine similarity (original vs converted) ≥ 0.75
Step 4: re-synchronize audio to original video at 30 fps

Result: face shows emotion A, voice says emotion B
```

#### Paradigm 2 — Cross-Identity Spliced Forgery

```
Goal: simulate real-world identity + emotion mismatch deepfakes

Input:  video of S1 expressing emotion A
        audio of S2 expressing emotion B  (A ≠ B)

Constraints:
  - No speaker overlap between audio and video streams
  - x-vector cosine(S2_audio, S2_ref) ≥ 0.75  (intra-identity match)
  - x-vector cosine(S2_audio, S1_video) ≤ 0.40 (confirmed identity mismatch)
  - audio-video sync: ≤ 2-frame offset at 30 fps
  - duration normalized to within 0.2s via time-stretching/trimming
```

### Data Preprocessing

**Visual:**
- MTCNN face detection → 5 landmarks → similarity transform → canonical alignment
- Bounding box expanded 25% → resized to 224×224
- Frames with >10% face detection failure → discarded
- Augmentation: random horizontal flip, brightness jitter

**Audio:**
- Resampled to 16 kHz
- 80-dim log-Mel spectrograms (25 ms Hanning window, 10 ms hop, 1024 FFT)
- Augmentation: random volume perturbation

**Text:**
- OpenAI Whisper 'base' (frozen ASR)
- Cleaned: lowercase, punctuation removed
- BERT tokenizer, max length L

---

## Results

### Emotion Recognition (Unimodal)

#### Speech–Text Branch (MDCNN) — Accuracy %

| Model | CREMA-D | SAVEE | MELD |
|---|---|---|---|
| CNN | 62.15 | 68.25 | 55.48 |
| CNN + GRU | 65.83 | 71.98 | 58.01 |
| CNN + BERT | 70.29 | 76.54 | 64.33 |
| **MDCNN (Ours)** | **74.67** | **80.25** | **68.92** |

**Notable:** Bidirectional cross-attention mechanism outperforms simple late-fusion (CNN+BERT) by 4–5% consistently.

#### Visual Branch (FV-LiteNet) — Accuracy %

| Model | CREMA-D | SAVEE | MELD |
|---|---|---|---|
| LBP + SVM | 88.10 | 82.45 | 54.18 |
| **FV-LiteNet (Ours)** | **86.83** | **90.15** | **71.77** |

FV-LiteNet slightly below LBP+SVM on CREMA-D (88.10 vs 86.83) but substantially better on SAVEE and MELD — better generalization.

---

### Forgery Detection by Type

Full cross-modal detection on all three datasets:

| Dataset | Pairing Type | ACC | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| SAVEE | Genuine Pairs | 91.2 | 90.8 | 92.0 | 91.4 | 0.95 |
| SAVEE | Emotion Tampering | 85.6 | 86.5 | 84.2 | 85.3 | 0.91 |
| SAVEE | Cross-Identity Spliced | 88.1 | 87.6 | 89.0 | 88.3 | 0.92 |
| CREMA-D | Genuine Pairs | 90.5 | 90.9 | 90.2 | 90.5 | 0.95 |
| CREMA-D | Emotion Tampering | 86.2 | 88.1 | 86.0 | 87.0 | 0.92 |
| CREMA-D | Cross-Identity Spliced | 88.9 | 88.5 | 89.4 | 88.9 | 0.94 |
| MELD | Genuine Pairs | 75.1 | 76.0 | 74.8 | 75.4 | 0.82 |
| MELD | Emotion Tampering | 70.4 | 71.2 | 70.0 | 70.6 | 0.77 |
| MELD | Cross-Identity Spliced | 73.2 | 72.8 | 73.8 | 73.3 | 0.80 |

**Key observations:**
- Cross-Identity forgeries detected better than Emotion Tampering — more pronounced inter-modal inconsistency
- MELD hardest dataset (complex multi-party conversations) — all metrics >70%
- Accuracy ≈ F1-score throughout: confirms balanced error profile (symmetric FP/FN), not biased toward either class

---

### Ablation Study

Effect of fusion operations on CREMA-D:

| Forgery Type | Fusion Method | ACC | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Emotion Tampering | **Concat + Diff + Product** | **87.2** | 88.1 | 86.0 | **87.0** | **0.92** |
| Emotion Tampering | Concat + Diff | 85.9 | 86.3 | 85.2 | 85.7 | 0.90 |
| Emotion Tampering | Concat only | 82.5 | 83.1 | 82.0 | 82.5 | 0.88 |
| Cross-Identity | **Concat + Diff + Product** | **90.3** | 89.8 | 91.2 | **90.5** | **0.94** |
| Cross-Identity | Concat + Diff | 89.1 | 89.6 | 88.3 | 88.9 | 0.93 |
| Cross-Identity | Concat only | 85.1 | 84.8 | 85.5 | 85.1 | 0.90 |

**Progressive improvement:** Each added operation contributes meaningfully:
- Diff term: +3.2% F1 (captures inter-modal conflict explicitly)
- Product term: +1.3% F1 (captures feature correlation/synergy)

---

### Comparison with SOTA on DFDC

| Category | Method | AUC |
|---|---|---|
| Visual — Artifact/Texture | MesoNet-4 | 0.753 |
| Visual — Artifact/Texture | Face X-ray | 0.809 |
| Visual — Artifact/Texture | Two-stream CNN | 0.614 |
| Audio — Acoustic Artifact | CQCC-GMM | 0.523 |
| Audio — Acoustic Artifact | RawNet2 | 0.718 |
| Multimodal — High-level Semantics | DeepRhythm | 0.745 |
| Multimodal — High-level Semantics | Siamese | 0.844 |
| Multimodal — High-level Semantics | MDS | 0.915 |
| **Multimodal — Emotional Consistency** | **ACE-Net (Ours)** | **0.921** |

**Takeaway:** Unimodal detectors peak at 0.809 (visual) — multimodal semantic methods consistently dominate. ACE-Net outperforms prior SOTA MDS by 0.006 AUC while being significantly more lightweight.

---

## Key Equations Reference

| Eq. | Formula | Description |
|---|---|---|
| (1) | `Ã = (w_c ⊗ 1_F ⊗ 1_T) ⊙ (1_C ⊗ 1_T ⊗ w_t) ⊙ A` | MDCNN channel-spatial reweighting |
| (2) | `Attn_{A→T} = softmax(Q_A K_T^⊤ / √d_h + M_{A→T}) V_T` | Acoustic-to-Text attention |
| (3) | `Attn_{T→A} = softmax(Q_T K_A^⊤ / √d_h + M_{T→A}) V_A` | Text-to-Acoustic attention |
| (4) | `v_t(x) = Flow(I_{t-1}, I_t)[x]` | Pixel-wise optical flow |
| (5) | `m_t = (1/\|Ω\|) Σ_{x∈Ω} \|\|v_t(x)\|\|_2` | Frame motion score |
| (6) | `τ_t = Percentile_ρ({m_{t'} : \|t'-t\| ≤ W/2})` | Adaptive motion threshold |
| (7) | `G = {t \| m_t ≥ τ_t}` | Motion-gated candidate set |
| (8) | `c_t = σ(w^⊤ g(I_t) + b) ∈ (0,1)` | Frame expressiveness score |
| (9) | `K = Top-K({(t,c_t) : t∈G, c_t≥θ})` | Final keyframe set |
| (10) | `f_t^v = h(I_t) ∈ R^{C_v}` | FV-LiteNet per-frame encoding |
| (11) | `F = [f_t^v]_{t∈K} ∈ R^{K×C_v}` | Temporal frame sequence |
| (12) | `α_t = exp(βc_t) / Σ exp(βc_{t'})` | Temperature-scaled attention (β=5) |
| (13) | `z_v = (Σ α_t f_t^v) W_v ∈ R^d` | Visual embedding via attention pooling |
| (14) | `f = [z_at; z_v \| \|z_at−z_v\| \| z_at⊙z_v] ∈ R^{4d}` | Multi-aspect fusion vector |
| (15) | `h_1 = Drop(ReLU(BN(W_1 f + b_1)))` | MLP layer 1 |
| (16) | `h_2 = Drop(ReLU(BN(W_2 h_1 + b_2)))` | MLP layer 2 |
| (17) | `p = σ(w_o^⊤ h_2 + b_o)` | Fake probability output |
| (18) | `L_BCE = −[y log p + (1−y) log(1−p)]` | Binary cross-entropy loss |

---

## Abbreviations

| Abbrev. | Full Form |
|---|---|
| ACE-Net | Affective Consistency Evaluation Network |
| ASR | Automatic Speech Recognition |
| AUC | Area Under the ROC Curve |
| BERT | Bidirectional Encoder Representations from Transformers |
| CBAM | Convolutional Block Attention Module |
| DSC | Depthwise Separable Convolution |
| DWConv | Depthwise Convolution |
| FV-LiteNet | Facial Visual Lite Network |
| GAP | Global Average Pooling |
| GMP | Global Max Pooling |
| MACs | Multiply-Accumulate Operations |
| MDCNN | Multi-granularity-attention Depthwise Convolutional Network |
| MLP | Multi-Layer Perceptron |
| PWConv | Pointwise Convolution |
| RVC | Retrieval-based Voice Conversion |
| SER | Speech Emotion Recognition |
| TTS | Text-to-Speech |
| ViT | Vision Transformer |

---

> **Implementation:** PyTorch 2.4.1, Python 3.8, NVIDIA RTX 3090 (24 GB VRAM)
> **Optimizer:** Adam, lr=1×10⁻⁴, weight_decay=1×10⁻⁵, batch=32, max 50 epochs
> **Early stopping:** patience=25 on validation loss
