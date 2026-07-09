---
name: ute-energy-guidance-project
description: Overall goal of the UTE segmentation phase-2 project — source-free domain generalization via a learned energy and test-time activation guidance
metadata: 
  node_type: memory
  type: project
  originSessionId: e73b4044-a5a9-419e-a9a3-06bb21952afe
---

Goal: a lung segmentation model (`basic_unet_disentangled.py`, `BasicUNet`) trained on
**COPDgene CT** (breath-hold, hyperinflated, diseased lungs) with GIN augmentation must
**generalize to UTE MRI** (free-breathing, non-COPD, different cohort) — **source-free**:
no MRI labels ever used in training (so "why not just do supervised" is defensible).

An earlier anatomical-constraint approach (ShapeAE / mask V-JEPA regularizer in
`causality_train_with_ACNN.py`) lost to plain GIN (`causality_train.py`) because it injected
a **CT-shape prior** rather than a modality-invariant constraint.

New approach (this project): build a learned **energy** = a frozen-network "wrongness meter"
scoring an (image, mask) pair (low = mask fits scan), then at **test time steer the U-Net's
internal activations** down that energy to fix a wrong MRI mask — **never touching weights**.
Mirrors §5 of Humayun et al. ICLR 2025 (reward-model gradient guidance of latents / "universal
guidance"). Baselines to beat on UTE: GIN-only, Tent, Karani et al. 2021 DAE-prior TTA.

Two energies (don't conflate):
- **`E_shape(m)`** — mask-only V-JEPA fill-in-the-blank loss; sees only the mask, can't localize
  to a scan. Used only for cheap *screening*.
- **`E_x(x,m) = ‖P(E_img(x)) − E_mask(m)‖₁`** — cross-modal "two-witnesses" energy; image side
  anchors *where/how big* lungs belong for this scan. The real meter Phase 2 steers.

Key files: `basic_unet_disentangled.py` (BasicUNet, bottleneck `x4` is the guidance site),
`model_vjepa.py` (size-agnostic encoder/predictor, per-forward RoPE), `dataset_mask_jepa.py`,
`train_mask_jepa.py`, `phase0_energy_vs_dice.py` (+ `phase0_volume_vs_energy.py`,
`phase0_shape_vs_energy.py`, `phase0_verify_anatomy.py`). Seg checkpoint:
`save_models/current_best_may_2026/best_bunet_causality_paper_ct_train_UTE_test_w_tversky_wo_kl_only_gin_roughness_enforced.pth`.
See [[phase0-1-2-plan]], [[mask-encoder-current-stage]], [[final-cross-modal-phase0]].
