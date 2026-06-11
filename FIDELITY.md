# ACE-Net Replication — Fidelity Notes

Status of this implementation vs the source paper (Yu et al., *Electronics* 2025,
14, 4420). See [docs/ACE-Net.md](docs/ACE-Net.md) for the spec.

## Leakage & bias controls

A dedicated audit (`scripts/bias_audit.py`) checks every shortcut a discriminator
could exploit instead of genuine emotional inconsistency. Findings and the
controls applied:

| Leak / bias | Symptom | Control |
|---|---|---|
| Duplicate manifest rows | each clip listed twice → same clip in train & test | dedup by `(split, file_id)` in the sample builders |
| **Forgery component leakage** | a P2 splice `A___B` reuses genuine visual clip `A`; 283/360 P2 test fakes had `A` in train as genuine | **actor-disjoint split** keyed on the source/visual actor |
| **Cross-stage familiarity** | frozen extractor trained (Stage-1) on the same genuine clips later in Stage-2 *test* → genuine = in-distribution, fake-audio = out-of-distribution; 593/737 test clips were seen, AUC ≈ 0.99 | **single shared actor partition** across both stages — the same held-out actors are excluded from Stage-1 training and used as the Stage-2 test set (verified 0 clip / 0 actor overlap) |
| Loudness fingerprint | genuine −51 dB, P1 −54, P2 −48 → class separable by level | **per-sample mel z-score** (verified all classes → mean 0, std 1) |
| Duration tell | mean T: genuine 250, P1 223, P2 259 → length tags class | **fixed-length crop** to 128 frames (all clips ≥134 ⇒ no padding ⇒ no length signal) |
| α-weight signature | P2 α exactly uniform, genuine/P1 slightly peaked → pooling pattern tags P2 | **uniform α for all** samples (verified α std = 0 for every class) |

All controls are verified by `scripts/bias_audit.py`; rerun it after any data or
split change.

## Faithful (matches paper spec)

| Component | Detail | Paper ref |
|---|---|---|
| MDCNN backbone | 4 DSC stages [64,128,256,256] strides [2,2,2,1], BN+ReLU | §3.2.1 |
| Channel attention | GAP+GMP concat -> shared MLP [2C->C/r->C], r=8, sigmoid | §3.2.1 |
| Spatial attention | mean+max channel-pool -> 3x3 DSC -> sigmoid -> freq-avg = w_t | §3.2.1 |
| MDCNN reweight | Eq.1; reweight -> average over frequency -> linear proj to d | §3.2.1 |
| Cross-attention | bidirectional A->T / T->A, h=4, dropout 0.1, Residual+LayerNorm | §3.2.2, Eq.2-3 |
| z_at fusion | mean-pool both directions over time, sum (per paper TEXT) | §3.2.2 |
| Frozen BERT | ASR/BERT backbone frozen; only proj + cross-attn trainable | §3.2.2 |
| Keyframe alpha | uses precomputed alpha = softmax(beta*expr), beta=5 — VERIFIED (err 5e-5) | Eq.12 |
| Attention pooling | z_v = (sum alpha_t * f_t) W_v | Eq.13 |
| FV-LiteNet GhostNet | Ghost modules, SE removed in last two stride-2 bottlenecks | §3.3.2, Fig.4a |
| FV-LiteNet final | GhostNet final 1x1 conv -> C_v=960 -> Fig.4b head | Fig.4b |
| Spatial-channel head | Conv2D+BN+ReLU -> spatial(DWConv3x3)+channel(MLP), residual-add+sigmoid gates, Add | Fig.4b |
| Multi-aspect fusion | f = [z_at; z_v \| \|z_at-z_v\| \| z_at(.)z_v] in R^{4d} | Eq.14 |
| MLP discriminator | 512 -> 128 -> 1, BN+ReLU+Dropout(0.3), sigmoid | Eq.15-17 |
| Loss | BCE (BCEWithLogitsLoss) | Eq.18 |
| Two-stage training | Stage-1 genuine-only emotion classifiers -> freeze -> Stage-2 fusion+MLP | §3.5 |
| Stage-1 per-dataset | each corpus trained in its NATIVE label space (CREMA 6-cls, MELD 7-cls) | Tables 2/3 |
| Stage-2 balance | 1:1 genuine:fake, positives stratified 50/50 across P1/P2 | §3.5 |
| Optimizer | Adam lr=1e-4, wd=1e-5, batch=32, max 50 ep, early-stop patience 25 | §4.2.1 |
| Eval protocol | fixed 80/10/10 holdout, test split only, no leakage (seeded) | §4.2.2 |

## Resolved ambiguities

- **Cross-attention z_at (text vs Fig.3):** Figure 3 shows Concat -> Modality
  Fusion -> GRU. The paper TEXT (§3.2.2) explicitly states "mean-pool the
  outputs of both directions ... and sum them to form z_at" and lists only
  projection + cross-attention params as trainable (no GRU params). We follow
  the text. If GRU fidelity is required, it would be an additive change.

## Known gaps (data / environment, not architecture)

1. **CREMA split is speaker-independent (deliberate deviation).** The paper uses
   a random 80/10/10 split for emotion recognition (speaker-dependent). We use an
   actor-disjoint partition for CREMA — more rigorous, and required to share a
   leakage-free holdout with Stage-2. Emotion accuracy is therefore expected to
   be lower than a speaker-dependent split would report. MELD keeps the paper's
   stratified random split (no speaker labels exposed, no forgery stage).

2. **No SAVEE.** Paper Tables 2/3 include SAVEE; not in our data. CREMA-D + MELD
   only.

3. **No DFDC (Table 6).** Not available. FakeAVCeleb is present but RAW mp4 —
   would need the full preprocessing pipeline (face/keyframe/melspec/ASR/BERT)
   before it can enter the vector pipeline for a cross-dataset AUC.

4. **Uniform keyframe weights (deliberate deviation from Eq. 13).** The
   preprocessing produced exactly-uniform α for P2 but slightly-peaked α for
   genuine/P1, which leaks the class through the visual pooling pattern. We
   therefore use uniform α for ALL samples (mean-pooling of keyframes), trading
   the paper's expressiveness weighting for a leakage-free visual branch.

5. **Hardware.** Paper used RTX 3090 (24GB, batch=32). Local GPU is RTX 4050
   (6GB). batch=32 may OOM for the speech_text branch (BERT). Use `--batch-size`
   to lower; this does not change architecture, only optimization noise.

## Not yet verified against ground truth

- Exact MDCNN tensor dimensions after stride sequence on 80-band input (the
  frequency axis collapses through 3 stride-2 stages); matches the paper's
  textual description but the paper gives no explicit intermediate shapes.
- FV-LiteNet GhostNet channel config uses the standard GhostNet-1.0 table; the
  paper does not publish FV-LiteNet's exact per-stage widths beyond "GhostNet
  backbone, SE removed in last two stride-2 bottlenecks, C_v=960 final".
