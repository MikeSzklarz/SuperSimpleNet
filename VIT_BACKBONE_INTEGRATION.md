# ViT / Foundation-Model Backbones in SuperSimpleNet

This document explains the integration of vision-foundation-model (ViT) backbones —
adapted from [AnomalyVFM](https://github.com/vicoslab/AnomalyVFM) (CVPR 2026) — into
SuperSimpleNet as drop-in alternatives to the default CNN feature extractor. It covers
the two architectures, what was borrowed and what was not, the geometry of the swap,
and what results to expect.

Companion analyses in this repo: `CNN_AND_VIT_EXPLAINED.md` (CNN vs ViT fundamentals)
and `RESOLUTION_AND_SMALL_DEFECT_ANALYSIS.md` (why input resolution, not backbone
quality, bounds small-defect detection).

---

## 1. The two architectures, side by side

### SuperSimpleNet (this repo)

A *discriminative* anomaly detector, trainable **unsupervised, weakly, mixed, or fully
supervised**. Forward pass with the default `wide_resnet50_2` at 256×256:

```
image [B, 3, 256, 256]
  └─ Feature extractor (FROZEN CNN)                    model/feature_extractor.py
       layer2 [B, 512, 32, 32] ─┐ upsample both to 64×64,
       layer3 [B, 1024, 16, 16] ─┘ concat → [B, 1536, 64, 64], 3×3 avg-pool
  └─ Feature adaptor (1×1 conv, trainable)             [B, 1536, 64, 64]
  └─ TRAINING ONLY: anomaly generator
       duplicates batch, adds Gaussian noise (std 0.015) inside binarized
       Perlin masks, updates GT mask/label               [2B, 1536, 64, 64]
  └─ Discriminator (trainable)
       seg head:  1×1 convs → anomaly map               [·, 1, 64, 64]
       cls head:  5×5 conv on (features ⊕ map) + pooling → image score  [·]
  └─ INFERENCE ONLY: upsample map to 256×256 + Gaussian blur (σ=4)
```

Losses: truncated-L1 + focal on the map, focal on the score. Real defect masks (when
supervision provides them) are downsampled to the 64×64 feature grid and combined
with the synthetic ones.

### AnomalyVFM (../AnomalyVFM)

A **zero-shot** detector: a frozen ViT foundation backbone with LoRA/DoRA adapters
injected into its attention layers, plus a small convolutional decoder and a linear
image-level predictor, trained *only on synthetic (Flux/Qwen-generated) anomalies* and
evaluated zero-shot on real datasets:

```
image [B, 3, 768, 768]
  └─ ViT backbone (frozen + LoRA/DoRA)  → patch tokens [B, 48·48, 1024], CLS summary
  └─ reshape tokens → [B, 1024, 48, 48]
  └─ SimpleDecoder (bottleneck convs + 1 bilinear-upsample block) → mask [B, 1, 96, 96]
  └─ SimplePredictor (linear on CLS) → image score
```

Trained with a confidence-weighted focal + L1 loss to tolerate noisy synthetic labels.

### What this integration borrows — and what it deliberately does not

| From AnomalyVFM | Taken? | Why |
|---|---|---|
| ViT backbone wrappers (DINOv2/v3, RADIO, CLIP, SigLIP2, TIPSv2) | **Yes** (ported, self-contained) | The point of the exercise: stronger frozen features |
| Frozen-backbone philosophy | **Yes** | Already identical in SuperSimpleNet |
| Token→spatial-grid reshape | **Yes** | Makes ViT output a drop-in CNN feature map |
| LoRA/DoRA PEFT fine-tuning | **No** (future work, §6) | Keeps the change minimal and the WRN50 comparison clean |
| SimpleDecoder / SimplePredictor heads | **No** | SuperSimpleNet's seg/cls heads + feature-space anomaly generation ARE the method being kept |
| Synthetic Flux training data, confidence-weighted loss | **No** | Part of AnomalyVFM's zero-shot recipe, orthogonal here |
| AnomalyVFM as a pip/code dependency | **No** | It pins torch 2.10/numpy 2.x/transformers 4.56 vs this repo's older pins; its wrappers hardcode `.cuda()` and PEFT imports. Wrappers were re-implemented cleanly in `model/vit_feature_extractor.py` |

The key insight making the swap clean: **everything downstream of SuperSimpleNet's
extractor is shape-driven**. The adaptor, discriminator, Perlin generator, and GT-mask
downsampling all derive from the `(channels, fh, fw)` tuple the extractor reports. A
ViT extractor that outputs `[B, C, H/p, W/p]` propagates automatically — the losses,
supervision modes, and training loop are untouched.

## 2. What was added

- **`model/vit_feature_extractor.py`** — `ViTFeatureExtractor` (same contract as the
  CNN `FeatureExtractor`: `forward → [B, C, fh, fw]`, `feature_dim` via dry-run,
  frozen via `requires_grad_(False)` + `eval()` + `no_grad()`), with per-backbone
  wrappers:
  - `dinov2_*` / `dinov3_*`: torch.hub, `get_intermediate_layers(reshape=True)` —
    supports multi-block extraction via `--vit-layers`.
  - `radio_*`: torch.hub NVlabs/RADIO (needs `timm`), last-layer tokens.
  - `clip_vitl14_336`, `siglip2_so400m`, `tipsv2_l14`: HuggingFace `AutoModel`
    (lazy `transformers` import), last-layer tokens. The SigLIP2 NaFlex patch-packing
    shim was ported device-aware (the AnomalyVFM original hardcoded `.cuda()`).
- **Renormalization**: the datamodules always produce ImageNet-normalized tensors;
  backbones expecting other statistics (CLIP stats, ±0.5 for SigLIP2, raw [0,1] for
  RADIO/TIPSv2) get a fused affine renormalization inside the extractor. DINOv2/v3
  expect ImageNet stats, so nothing changes for them.
- **`build_feature_extractor`** factory in `model/feature_extractor.py`, dispatching
  on the backbone-name prefix — every existing CNN config keeps working unchanged.
- **CLI**: `--vit-layers`, `--feat-scale`, `--dinov3-path`, `--dinov3-weights`; a
  fail-fast patch-divisibility check; example configs `configs/mvtec_dinov2.txt` and
  `configs/custom_unsup_dinov2.txt`; smoke test `tests/smoke_test_backbones.py`.

## 3. Grid geometry: the part that actually matters

A ViT's spatial resolution is `input / patch_size` — coarser than the CNN path at
equal input size:

| Configuration | Working feature grid | Effective stride |
|---|---|---|
| WRN50 @ 256² (default) | 64×64 (layer2 32² + layer3 16² upsampled ×2) | 8 (finest tap) |
| DINOv2 (p14) @ 252² | 18×18 | 14 |
| DINOv2 (p14) @ 448², `--feat-scale 1` | 32×32 | 14 |
| **DINOv2 (p14) @ 448², `--feat-scale 2`** | **64×64** | **7** |
| patch-16 backbones @ 512², `--feat-scale 2` | 64×64 | 8 |

So the recommended ViT configuration is **448² for patch-14 / 512² for patch-16 with
`--feat-scale 2`**: the same 64×64 working grid as the CNN baseline, with equal or
finer effective stride. Anything at ~256² input makes localization *coarser* than the
baseline, whatever the feature quality.

Constraint: the input size must be divisible by the patch size (14 → 252/448/518;
16 → 256/512). KSDD2's fixed 640×232 resolution is incompatible; Sensum's fixed
resolutions (320×192, 144×144) work with patch-16 backbones only.

## 4. Expectations

- **Where gains are plausible**: image-level detection (I-AUROC/AP-det) and the
  low-label supervised regimes (weak/mixed supervision) — foundation features are
  semantically far stronger than ImageNet-CNN features, which is what DINOv2-based
  detectors (AnomalyDINO, Dinomaly, AnomalyVFM) exploit.
- **Not a small-defect fix**: per `RESOLUTION_AND_SMALL_DEFECT_ANALYSIS.md`, defects
  destroyed by downsampling to the training resolution stay invisible to any backbone.
  The lever for small defects remains input resolution / tiling; this integration is
  orthogonal to that (though ViTs at 448/512 modestly raise the ceiling vs 256).
- **Compute**: ViT-L @ 448² is roughly 3–5× the backbone FLOPs of WRN50 @ 256². Use
  `--amp` and batch ≤16 (`--feat-scale 2` additionally ~4× the head/generator memory,
  and the anomaly generator doubles the batch at feature resolution during training).
- **Hyperparameters were tuned for WRN50**: `--noise-std 0.015` and the Perlin
  threshold target CNN feature statistics. ViT features have different magnitudes, so
  treat first runs as smoke tests and expect a small `--noise-std` sweep per backbone
  before judging results.
- **Untouched**: all supervision modes, losses, datamodules, optimizers, checkpoint
  format (backbone weights are excluded from checkpoints exactly as before — a saved
  ViT run needs only the matching `--backbone`/`--image-size`/`--feat-scale` at load
  time; see the architecture block in `eval.py`'s config).

## 5. Literature evidence: why this should (and might not) work

**For — the same recipe is published and works:**

- **[GeneralAD (ECCV 2024)](https://arxiv.org/abs/2407.12427)** is nearly this exact
  experiment: frozen **DINOv2 last-layer patch features** (518² input, LayerNorm-ed,
  CLS dropped) perturbed with **Gaussian noise** to create pseudo-anomalies for a
  discriminator. Result: **99.2% I-AUROC on MVTec-AD and 96.0% on VisA** — on par with
  CNN-based SimpleNet on MVTec (99.6%) and **+8.1 points over it on VisA** (87.9%).
  Their ablation found noise on a *random subset of patches* best for industrial data —
  which is functionally what SuperSimpleNet's Perlin-masked regional noise already does.
- **Noise scale confirms the retuning expectation**: GeneralAD's optimal noise was
  **σ = 0.25** on DINOv2 features vs SuperSimpleNet's 0.015 tuned for WRN50 — an
  order-of-magnitude difference, so expect the `--noise-std` sweep to matter.
- **DINOv2 features separate defects exceptionally well even with no training**:
  [AnomalyDINO (WACV 2025)](https://arxiv.org/abs/2405.14529) reports DINOv2 features
  give **≥ +4% AUROC over classification-pretrained ViT features** on MVTec-AD/VisA,
  and one-shot detection at 96.6% AUROC from raw patch-feature similarity alone.
- **Supervised mode should benefit most**: frozen DINOv2 + a small trainable
  segmentation head has been shown to beat fully-supervised CNNs with only
  **6–24 labeled defect images** in industrial settings
  ([injection-molding defect study, IJAMT 2026](https://link.springer.com/article/10.1007/s00170-026-17386-1)) —
  directly relevant to SuperSimpleNet's weakly/mixed-supervised regimes.
- **AnomalyVFM itself** (CVPR 2026) demonstrates a frozen VFM + tiny trainable head
  trained only on synthetic anomalies transfers zero-shot to real datasets.

**Against / caveats the literature also supports:**

- **A naive backbone swap can regress on MVTec**: GeneralAD reports SimpleNet at
  99.6% (WRN50) → **93.3% with a supervised ViT-B/16** → ~97.7% with DINOv2-B.
  MVTec's texture-centric defects favor CNN low-level features; the ViT payoff shows
  up on VisA/LOCO-style structured data, not necessarily on MVTec.
- **GeneralAD used an attention-based discriminator** (4-head MHA + MLP), not
  SimpleNet's pointwise scoring. SuperSimpleNet's seg head is 1×1 convs (pointwise) —
  the AvgPool neighborhood aggregation and 5×5 cls-head conv add some spatial context,
  but this is a structural difference that could cost accuracy on ViT features.
- **Middle layers beat the last layer for reconstruction-based AD**:
  [Dinomaly (CVPR 2025)](https://arxiv.org/abs/2405.14325) deliberately uses ViT
  layers 3–10 and avoids the final block. GeneralAD's success with the last layer
  shows discriminative methods differ, but `--vit-layers` is the first knob to try
  if last-block results disappoint.
- **Registers matter for dense maps**:
  [Vision Transformers Need Registers (ICLR 2024)](https://arxiv.org/abs/2309.16588)
  documents high-norm artifact tokens in DINOv2 that corrupt dense feature maps —
  use the `_reg` variants (the default here) for anomaly localization.

Net read: the mechanism (Gaussian-noise pseudo-anomalies on frozen foundation-model
patch features + discriminator) is validated at high resolution with regional noise;
the open questions are per-dataset (MVTec may not improve) and structural (pointwise
seg head, layer choice, noise scale).

## 6. How to run the comparison

```bash
# baseline
python train.py --dataset mvtec --backbone wide_resnet50_2 --image-size 256 256

# DINOv2, matched working grid
python train.py --config configs/mvtec_dinov2.txt

# no dataset needed — verify any backbone end-to-end:
python tests/smoke_test_backbones.py dinov2_vitl14_reg
```

Compare I-AUROC / AP-det (detection) and P-AUROC / AUPRO / AP-loc (localization) from
the per-category CSVs in the results directory.

## 7. Future work

- **LoRA/DoRA fine-tuning (phase 2)**: port AnomalyVFM's pure-torch `peft_local/`
  package, gate the extractor's `no_grad()` on PEFT being enabled, add a fourth
  optimizer parameter group, and change `save_model`'s exclusion filter from
  "everything under `feature_extractor`" to "only `requires_grad == False` params"
  so adapter weights are checkpointed.
- **Tiling / native-resolution inference** for small defects — the actual
  small-defect lever (see `RESOLUTION_AND_SMALL_DEFECT_ANALYSIS.md` §12).
