---
name: mask-encoder-current-stage
description: "Current stage of the UTE energy-guidance project as of 2026-06-22 — Phase 1 Step 1 mask-encoder retrain done, new-aug latest checkpoint chosen, moving to Step 2"
metadata: 
  node_type: memory
  type: project
  originSessionId: e73b4044-a5a9-419e-a9a3-06bb21952afe
---

**DECISION (2026-06-22): Phase 1 Step 1 retrain is done — go forward with the new-aug (anisotropic
scaling + elastic) `latest` checkpoint as the Phase-1 mask-encoder.** Two findings drove this, both from
the Phase-0 diagnostics in `phase0_plots/new_aug_mask_model/` (morph-sweep Spearman = energy↔shape-quality
monotonicity, target ≤ −0.7; the property that makes the energy usable as a guidance gradient):
- **Use `latest`, not `best`, for either run.** `best` (selected by val reconstruction L1) collapses the
  energy landscape: morph Spearman −0.136 (new-aug) / −0.300 (no-aug). The useful energy geometry peaks
  later in training, after the recon-loss minimum.
- **Among `latest` checkpoints, new-aug wins on monotonicity:** new-aug −0.682 (essentially reaches the
  bar) vs no-aug −0.600. New-aug is the only one of the four near ≤ −0.7.
Note on a red herring: no-aug latest has a larger GT–pred energy gap (0.0447 vs new-aug 0.0263), but
absolute separation doesn't matter for guidance — gradient *direction* (monotonicity) does. The smaller
gap is expected/desirable: augmentation stops the energy over-penalizing benign deformations.
Also chosen for **augmentation symmetry** — Phase 1 Step 2 image encoder will train under the same
anisotropic+elastic regime, keeping both encoders' manifolds consistent, which aids the source-free
CT→UTE generalization goal. **Caveat for go/no-go:** −0.682 is borderline, not comfortably past −0.7;
watch whether it holds once the image encoder is in the loop (see [[final-cross-modal-phase0]]).

Earlier context (2026-06-21), current stage of [[ute-energy-guidance-project]] = **Phase 1 Step 1**:
(re)ran the mask-encoder training with the fixes from the Phase-0 diagnostics (see [[phase0-1-2-plan]]).

Already wired on disk (verified):
- `dataset_mask_jepa.py`: `_clean_binary` (keep components ≥10% largest + fill holes 3D & per-axial-slice),
  `_aniso_zoom` (per-axis rescale → elongation, applied on native mask before fit), `_elastic`
  (smooth coarse-grid displacement → extent, applied after fit on 256³). Augmentation is **symmetric**
  (single returned mask feeds both context and EMA-target encoders). Constructor args: `clean_mask`,
  `clean_keep_frac`, `augment`, `scale_prob/scale_range`, `elastic_prob/elastic_alpha/elastic_ctrl`.
- `train_mask_jepa.py`: `CLEAN_MASKS=True`, `SCALE_RANGE=(0.80,1.25)`, `ELASTIC_ALPHA=12`, `ELASTIC_CTRL=8`,
  `augment=True` on train, `augment=False` (cleaned, no aug) on val. Saves `aug` provenance into ckpt.
  Outputs tagged `..._ct256_new_aug_optimised` → `save_models/best_mask_jepa_ct256_new_aug_optimised.pth`.
  Optimized dataloaders (pin_memory, persistent_workers, prefetch). **Note:** `VAL_CSV` is now
  `only_ute_1.25mm.csv` (UTE) — model selection (lowest val L1) is on UTE generalization, valid because
  this fresh CT-only encoder never saw any UTE.

Done this session: updated `phase0_energy_vs_dice.py` to mirror the encoder's cleaned training input.
It imports `_clean_binary` from `dataset_mask_jepa.py`; new `CLEAN_MASKS`/`CLEAN_KEEP_FRAC` flags
(default None → auto-read from ckpt `aug` block, fallback True/0.10); `maybe_clean()` applied to every
candidate (GT/prediction/morph/aug) BEFORE both its Dice and energy, so the two always describe one mask.
Default ckpt path bumped to `best_mask_jepa_ct256_new_aug_optimised.pth`; OUT_DIR `phase0_plots/new_aug_mask_model/`.
The other two diagnostics (`phase0_volume_vs_energy.py`, `phase0_shape_vs_energy.py`) import
`load_volume`/`compute_energy` from it but call them directly — NOT yet cleaned; thread cleaning in if needed.

Immediate next steps (in order): (1) ✅ retrain done; (2) ✅ Phase-0 diagnostics re-run — new-aug latest
chosen (UTE morph Spearman recovered to −0.682, see decision above); (3) ✅ Phase 1 Step 2 cross-modal
trainer code built + cross-modal Phase-0 eval script built (2026-06-22); (4) ⏳ USER is reviewing the code
and will run the Step-2 training (`python train_img_jepa.py`) themselves — handoff point as of 2026-06-22.
Next session likely resumes from training issues, codebase questions, or running `phase0_cross_modal_energy.py`
once a Step-2 checkpoint exists; (5) run final cross-modal Phase 0 [[final-cross-modal-phase0]]; (6) Phase 2.
Nothing has been trained or launched by me — Step-2/eval code is CPU-smoke-tested only, not yet run on GPU.

**Phase 1 Step 2 code (built 2026-06-22, smoke-tested on CPU):** three new files mirroring the Step-1
trio's style.
- `dataset_img_jepa.py` — `ImgJEPADataset` returns paired (image, mask, filename) at 256³. Reuses
  `datasets_causality` (normalise_hu/one_one, `_clean_binary`, PAIRED `RandomAnisoScale`/`RandomElastic`,
  `PadOrCrop`, `ToTensor`) + a new `MaskGaussianBlur` (σ=1.0, mask-only, last step). Image→[-1,1] (HU clip
  on CT train only); mask→clean→fit→blur so E_mask sees its Step-1 training distribution. Geometric aug
  identical to Step-1 params (scale 0.5/0.80–1.25, elastic 0.5/α12/ctrl8), applied to BOTH image+mask.
- `model_img_jepa.py` — `load_frozen_mask_encoder` (loads Step-1 `target_encoder`, freezes); `ImgPredictor`
  (deterministic full-token map, RoPE, defaults pred_dim384/depth6/heads6, ~10.9M); `ImgJEPA` wraps
  E_img (VJEPAEncoder, ~30.3M, scratch) + P + frozen E_mask (~30.3M). `forward`→(pred,target) with
  `F.layer_norm` target (matches phase0). `energy()` = ‖P(E_img(x))−E_mask(m)‖₁ kept on the model so
  train objective == deployed Phase-2 energy. Asserts E_img/E_mask 8³ grids match.
- `train_img_jepa.py` — CT-only train / UTE val (lowest UTE val L1 = honest cross-modal selection, E_img
  never sees UTE). GIN (`CausalityAugmentation3D`) image-only, GPU, no_grad, 80% (causality_train pattern).
  No EMA (E_mask frozen). BATCH 2 × ACCUM 8, 400 ep, cosine+15 warmup, AdamW 5e-4/wd0.04. Saves
  `{best,latest}_img_jepa_ct256_new_aug.pth` incl. frozen E_mask state for self-contained Phase-2 reload.
Defaults I chose (not dictated by consistency): E_img trained from scratch (not warm-started from E_mask);
predictor widened to encoder width + depth 6 (carries the full cross-modal map). Frozen E_mask =
`target_encoder` (EMA), matching `phase0_energy_vs_dice.py`.
