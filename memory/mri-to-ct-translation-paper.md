---
name: mri-to-ct-translation-paper
description: "Separate paper idea — MRI→CT CycleGAN translation as domain adaptation for CT-only lung segmentor; report drafted, pipeline reorg pending"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6ed1d25c-6fbc-4f14-803a-05a33a2d8c5e
---

A distinct line of work from the [[ute-energy-guidance-project]]. Idea: instead of adapting the segmentor to MRI (augmentation / causal / roughness), adapt the *input* — at test time translate MRI→synthetic CT with a 3D CycleGAN, then segment with a CT-only BasicUNet (never fine-tuned on MRI).

Test-time pipeline lives in `eval_step_1_mri_to_ct.py` (CycleGAN) and `eval_step_1_mri_to_ct_fda.py` (FDA baseline). Translation training in `cyclegan/train.py`.

**Core problem (the paper's crux):** current CycleGAN training has MRI-label leakage in TWO places — (a) the seg-consistency Tversky loss uses the MRI GT mask `ute_mask`, and (b) the frozen segmentor it uses (`best_bunet_joint_train.pth`) is jointly trained on CT+MRI. To make the UDA claim clean, Method 1 removes BOTH and keeps only the CT GT mask + a CT-only segmentor. Risk: dropping `ute_mask` leaves the MRI→fakeCT direction with no lung-mask anchor (only cycle + adversarial) → anatomy may not survive translation.

**Plan:** Method 1 (label-free, COPD-Gene ~690 CT + ~110 Marissa MRI, scale to ~220) → if it fails, anatomy add-ons → Method 2 (paired MRI-CT, PI provides) → else abandon. Success ≈ Method 1 matches the leakage version within margin while beating no-adaptation and FDA.

**Status (2026-06-30):** Wrote scoping report for PI at `reports/mri_to_ct_translation_report.md` (hypothesis/method/data/experiments + go-no-go decision tree). User is reviewing it; will ping with major changes. Next step after review: reorganize the training pipeline per the report.
