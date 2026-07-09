---
name: phase0-1-2-plan
description: "Definitions of Phase 0 / Phase 1 (two steps) / Phase 2 for the UTE energy-guidance project, and the Phase-0 diagnostic journey/findings"
metadata: 
  node_type: memory
  type: project
  originSessionId: e73b4044-a5a9-419e-a9a3-06bb21952afe
---

Phase definitions for [[ute-energy-guidance-project]]:

**Phase 0 — validate the compass.** Phase 2 is gradient descent on energy; only works if
"lower energy = better mask." Cheaply test that: take a GT mask, manufacture worse masks
(erode/dilate/morph/random-aug, each with known Dice to GT), plot energy(y) vs Dice(x), per-scan,
faceted CT vs UTE. Script `phase0_energy_vs_dice.py`, modes `raw_scatter` / `morph_sweep` /
`random_aug_scatter`. GO if within-scan energy falls as Dice rises (median per-scan Spearman
≲ −0.7), morph minimum at radius 0, holds **on UTE**.

**Phase 1 Step 1 — mask encoder (E_mask).** Fresh V-JEPA mask encoder, **CT masks only**, fixed
**256³** (native UTE size → no resize/crop at Phase 2), **patch 32 → 8³=512 tokens**. Becomes the
**frozen target encoder** in Step 2. Files: `model_vjepa.py`, `dataset_mask_jepa.py`, `train_mask_jepa.py`.

**Phase 1 Step 2 — image encoder + predictor (NOT built yet).** Train `E_img` (context, heavy GIN
for appearance invariance) + predictor `P` so `P(E_img(GIN(CT_img)))` regresses the frozen
`E_mask(mask)` latent. Geometric shape-aug must be applied **identically to image AND mask**
(preserve correspondence); GIN applied to the (warped) image only. `E_x` is deterministic (full
token regression, no token-hiding/averaging).

**Phase 2 — test-time activation guidance.** Per MRI scan: freeze everything, add learnable δ
(init 0) to bottleneck `x4`, iterate decode→compute `E_x`→backprop to δ→step, minimizing
`E_x(x, σ(decode(x4+δ))) + λ_prox·‖δ‖`. Stop when `|E_t−E_{t−1}|<τ` or max-K. Guards: cap ‖δ‖,
monitor lung volume/components, fall back to m₀.

**Phase-0 diagnostic journey (findings that reshaped Step 1):**
1. Old joint (CT+UTE, 96³, resize) encoder screened well: UTE morph Spearman −0.827. GO.
2. Decided to start fresh: CT-only at native 256³ with `pad_crop`.
3. New encoder regressed: UTE morph −0.509, energy *fell* when dilating UTE masks.
4. `phase0_volume_vs_energy.py` killed the absolute-size hypothesis (CT vol↔energy ρ≈−0.03, flat);
   real signal = shape gap at matched volume (UTE ~0.04–0.06 higher energy).
5. `phase0_shape_vs_energy.py` localized it to a **compactness/inflation axis**: `elongation`
   (UTE more elongated) + `extent` (UTE fills bbox less); both strong energy correlates.
   (`solidity_close`/`n_components` were a speck/fragmentation artifact the energy ignores.)
6. `phase0_verify_anatomy.py` confirmed **genuine anatomy, not artifact** (cleaning left
   extent d −1.04→−1.02, elongation d 2.38→2.38 unchanged).

Fix adopted: **mask cleaning** (hygiene) + **SYMMETRIC** anisotropic-scale (→elongation) + elastic
(→extent) augmentation, train-only. Isotropic scale ruled out (moves neither descriptor).
**Asymmetric** context-deformed/target-clean was explicitly REJECTED here — it would push correct
UTE masks to high energy; it's a later tool for error-sensitivity, not the modality-gap fix.
See [[mask-encoder-current-stage]].
