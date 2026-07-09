---
name: phase2-guidance-built
description: Phase 2 test-time activation-guidance script built (phase2_guidance.py) + the GO-with-leashes decision and the Phase-0 cross-modal read that motivated it
metadata: 
  node_type: memory
  type: project
  originSessionId: 9bba2ca4-bddb-4f13-8917-001ce8590122
---

**Built 2026-06-23 — `phase2_guidance.py`** (the Phase-2 step of [[ute-energy-guidance-project]]; see [[phase0-1-2-plan]]).

Mechanism (per plan): freeze seg U-Net (`basic_unet_disentangled.BasicUNet`) AND the
cross-modal energy; add learnable `delta` (init 0) to bottleneck `x4` (after `conv_3_1`,
before `up_4`; shape (1,256,32,32,32) at 256^3); iterate
`logits=decode(skips, x4+delta) -> m=sigmoid -> E_x=||P(E_img(x)) - LN(E_mask(m))||_1`,
backprop to delta only. Image side `P(E_img(x))` computed ONCE per scan (constant wrt delta);
only the mask side carries grad (so it does NOT use `ImgJEPA.energy`, which no_grad's the mask
side). U-Net split done in-script via `unet_encode`/`unet_decode` (no model edits). For UTE
(native 256^3) the seg grid and energy grid coincide -> NO resample inside the loop.
Energies reported on CLEANED BINARY masks (fit256->blur sigma1->LN) so numbers match
phase0_cross_modal_energy.py; soft energy is only the optimisation/early-stop signal.

**Decision: GO to Phase 2, with leashes** — taken from the Phase-0 cross-modal plots run on the
still-training `best_img_jepa_ct256_new_aug` (in `phase0_plots/cross_modal/`):
- raw_scatter (n=110 UTE): clean negative E_x vs Dice trend -> energy transfers to true UTE
  (biggest scientific risk retired). BUT GT stars span 0.17-0.34 and mean GT E_x 0.236 only ~0.015
  below mean pred 0.251; many 0.94-0.98 preds sit BELOW their own GT -> GT is often NOT the per-scan
  energy minimum.
- morph_sweep (GT-centred, 5 scans): median Spearman -0.864 PASS, min ~radius 0 (faint dilation bias).
- pred_morph_sweep (5 scans): median toward-GT Spearman **-0.257 -> FAILS the <=-0.7 criterion**.
  Failure is entirely the high-Dice (>~0.93) scans: their basin is flat/slightly rising toward GT.
  Low-Dice scans descend correctly.
Read (user-led, agreed): energy is a strong correctly-signed compass for BAD predictions (where
Phase-2 value lives) and a near-flat basin for already-good ones. So "fix the bad, protect the good":
benign failure, not an image-branch transfer failure. NOT final — re-run pred_morph on the CONVERGED
ckpt over 15-20 scans before locking the go.

**Leashes = independent booleans (for ablation):** USE_PROX (lambda_prox*||delta||), USE_DELTA_CAP
(||delta||<=DELTA_MAX_NORM), USE_EARLY_STOP (rel |dE|<tol, PATIENCE — auto-protects good preds via
tiny grad), USE_VOLUME_GUARD, USE_COMPONENT_GUARD, USE_LCC_POST (keep N largest, matches
eval_step_1_resample), USE_ENERGY_FALLBACK (ship m0 unless binary E beats it). USE_GUIDANCE=False ->
baseline m0 (zero-leash anchor). I/O merges eval_step_1.py (1.25mm) + eval_step_1_resample_to_1mm.py
(1mm) under one switch `RESAMPLE_TO_1MM`. Writes per-scan metrics CSV (`log/phase2_guidance_metrics.csv`:
dice_m0/dice_refined, E0/E_final, iters, fell_back/reason, vol, n_comp).

Open caveats: (1) NOT yet run — 256^3 fwd+bwd per iter is GPU-heavy, will contend with/OOM the
running train_img_jepa job; run after it finishes or on a free GPU (add decode grad-checkpointing if
OOM). (2) SEG input pad value = -1 (phase0's validated preds), NOT eval_step_1's 0.5 — moot for
native-256 UTE (no padding). (3) LR/DELTA_MAX_NORM not yet calibrated to delta/activation scale.

**Also added 2026-06-23 — `gt_vs_pred` mode in `phase0_cross_modal_energy.py`** (the check I had
offered but not built): per UTE scan computes E_x(GT) vs E_x(seg prediction), prints the % of scans
where E_pred >= E_GT (GT is the per-scan energy minimum = safe), the inverted %, mean margin, and the
Dice of inverted vs safe scans (to confirm inversions cluster at high Dice as pred_morph_sweep
implied). Saves a 2-panel plot (E_GT vs E_pred coloured by Dice + margin histogram). Stricter lens
than pred_morph (asks if GT is the actual minimum, not just downhill-along-path). Read: high % (>~80)
strengthens GO; low % concentrated at high Dice just reconfirms the early-stop/leash design.

**NEXT-SESSION PLAN (agreed 2026-06-23), in order:**
1. Re-run `phase0_cross_modal_energy.py` on the (ideally converged) best img-JEPA: NEW `gt_vs_pred`
   mode + bump `MORPH_NUM_SCANS` and `PRED_MORPH_NUM_SCANS` to ~15-20 (the -0.257 pred_morph median
   was set by only 5 scans). Review whether morph/pred_morph/gt_vs_pred hold up at larger n.
2. User reviews `phase2_guidance.py`; will ask me to explain any unclear part (e.g. the x4/delta split,
   soft-vs-binary energy, the fallback ladder) and/or report an OOM. If OOM -> add gradient
   checkpointing on `unet_decode` (the heavy 256^3 path).
3. User runs `phase2_guidance.py` with the current best img-JEPA, reports results (mean Dice m0->refined,
   #improved/#worsened, fell_back counts, the metrics CSV) — then we iterate (likely LR/DELTA_MAX_NORM
   calibration first; watch the printed |delta| and E_soft trace).
