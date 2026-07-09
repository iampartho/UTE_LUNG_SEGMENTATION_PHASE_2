---
name: phase2-delta-capacity-ablation
description: "User's open concern that the delta perturbation may be too weak / nullified by leashes; 3-step ablation plan + pending verification against phase2_guidance_metrics.csv"
metadata: 
  node_type: memory
  type: project
  originSessionId: cf0ca90f-82b3-4c4e-a4de-91923e7c1828
---

**RESOLVED 2026-06-24** — first 4-set run came back; hypothesis confirmed. Capacity was fine; the real cost of leashes was missed-improvement (volume-guard hard-revert + zero-margin energy fallback), NOT below-baseline damage. Full results, diagnosis, and the two leash fixes in [[phase2-results-and-leash-tuning]]. Original thread below.

Open Phase-2 thread (raised 2026-06-23). User is uneasy that the test-time activation guidance in [[phase2-guidance-built]] won't actually help: worried `delta` is too small to change the mask much and that the many leashes will nullify whatever it does. He clarified he meant `delta` (the perturbation), not `LAMBDA_PROX`.

**My assessment (for when results come back):**
- `delta` is NOT small/scalar — it's the full bottleneck `x4` tensor `(1,256,32,32,32)` ≈ 8.4M free params, 20 Adam steps. Capacity is not the limiter.
- Real limiters: (a) fine detail is locked by FROZEN high-res skips x1,x2,x3 — delta only moves coarse/global structure (good for fixing bad scans, weak for nudging good boundaries); (b) optimization budget (LR=0.05 / MAX_ITERS=20).
- Key correction to his fear: with `USE_ENERGY_FALLBACK=True` the method **cannot underperform m0** on the proxy — deliverable reverts to m0 unless a candidate beat E0. So the true failure mode is **no-op / timid**, NOT below-baseline damage.

**Agreed 3-step plan (user saved to his notes; running Step 1 with defaults first):**
1. Capacity probe: USE_GUIDANCE=True, ALL leashes OFF (prox, delta_cap, early_stop, volume_guard, component_guard, energy_fallback, lcc_post), MAX_ITERS~50, on known-bad scans → does delta move the mask at all?
2. Add raw movement metric: Dice(m0, refined) + |refined−m0| voxel count to decouple "did it move" from "did it help GT". (NOT yet added to script — offered, user is running defaults first.)
3. Re-enable leashes one at a time; watch where improvement dies. If it survives unleashed but dies when energy_fallback turns on → proxy/Dice misalignment, not a leash problem.

**My hypothesis to verify:** capacity fine; main cost of leashes = missed improvements via fallback, not damage.

**Verification when user returns:** read `./log/phase2_guidance_metrics.csv` (METRICS_CSV). Check fell_back rate + reasons, dice_m0 vs dice_refined (esp. on low-dice_m0 scans), E0 vs E_final. High fell_back rate on bad scans ⇒ timid/no-op confirmed. Improvement concentrated on low-dice scans ⇒ working as designed.
