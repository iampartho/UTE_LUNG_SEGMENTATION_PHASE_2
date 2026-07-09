---
name: final-cross-modal-phase0
description: "How the final (decision-grade) cross-modal Phase 0 is evaluated for the UTE project, with go/no-go criteria gating Phase 2"
metadata: 
  node_type: memory
  type: project
  originSessionId: e73b4044-a5a9-419e-a9a3-06bb21952afe
---

The **decision-grade** Phase 0 for [[ute-energy-guidance-project]] (the one that gates Phase 2), run
AFTER Phase 1 Step 2, uses the cross-modal energy `E_x(x,m) = ‖P(E_img(x)) − E_mask(m)‖₁` instead of
the mask-only screening energy.

Mechanical differences from the mask-only screen:
- Energy takes TWO inputs: image x → `E_img` → `P` → "mask latent the scan implies"; mask m → frozen
  `E_mask` → "mask latent this mask has"; L1 between them. Mask side keeps same preprocessing
  (clean → fit 256³ → blur).
- **Deterministic** — no token-hiding, no K-pattern averaging (cross-modal predictor regresses the full
  mask-token set from the image).
- **Real images, no GIN** at eval (GIN is train-time only) → also the first honest test of whether the
  GIN-trained image branch transfers to true UTE appearance (the single biggest scientific risk).

Tests = same three (`raw_scatter`, `morph_sweep`, `random_aug_scatter`) plus two targeted extras:
- Re-check the high-energy CT/emphysema scans that inverted under the shape-only meter (`18065W_FRC`,
  `10993X_RV`) — image conditioning should rescue them.
- **Prediction-centered morph (new mode to add):** start from each UTE prediction m₀ and morph toward/away
  from its GT; check `E_x` decreases monotonically along the path to GT. Tests the EXACT regime Phase 2
  operates in (start from a real prediction in Dice 0.85–0.95 band, not from GT).

**GO to Phase 2** if, on UTE with `E_x`: (a) morph minimum at radius 0; (b) median per-scan Spearman
≲ −0.7; (c) prediction-centered sweep monotonically decreasing from m₀ toward GT in the 0.85–0.95 band.
Jointly establishes −∇E_x is a trustworthy "improve the mask" direction in Phase 2's operating regime.

**NO-GO / fix first** if minimum still off-GT on UTE or trend flat/inverted. Likely culprit = image branch
not transferring (real UTE appearance too far from GIN'd CT); fix lives in Phase 1 Step 2 (stronger/realistic
GIN, paired geometric aug), NOT in Phase 2.

Script work to add for this: a cross-modal energy fn (`E_img`+`P`+frozen `E_mask`, no masking), image-loading
path feeding `E_img`, and the prediction-centered morph mode. Most plumbing (`load_volume`, `predict_mask`,
morph/aug ops, cleaning) already exists in `phase0_energy_vs_dice.py`. See [[phase0-1-2-plan]], [[mask-encoder-current-stage]].

**Built 2026-06-22 — `phase0_cross_modal_energy.py`** (standalone sibling of `phase0_energy_vs_dice.py`,
which stays as the mask-only screen). Loads the Step-2 image-JEPA via `ImgJEPA.from_checkpoint` (self-contained:
rebuilds frozen E_mask from the bundled `mask_encoder`/`mask_config`, no dependency on the Step-1 file).
Energy = DETERMINISTIC `E_x = ||P(E_img(x)) − layernorm(E_mask(m))||₁` — no token hiding, no K-averaging;
image side computed ONCE per scan (`image_pred_latent`) and reused across candidate masks; real image, NO GIN.
Image+mask both fit 256³ via centre pad/crop (FIT_MODE default 'pad_crop', PADVAL −1 for image). Modes:
`raw_scatter` (annotates WATCH_SCANS 18065W_FRC/10993X_RV), `morph_sweep`, `random_aug_scatter`, and NEW
`pred_morph_sweep` (start from seg prediction m0, blend toward GT for t∈[0,1] + a little away for t<0 via
`blend_masks` smooth-field interp; per-scan Spearman over the toward-GT band). Smoke-tested on CPU with a
throwaway ckpt. Default CKPT `./save_models/latest_img_jepa_ct256_new_aug.pth`, OUT_DIR `./phase0_plots/cross_modal/`.
