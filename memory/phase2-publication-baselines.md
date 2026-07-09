---
name: phase2-publication-baselines
description: What Phase-2 must compare against (TTA/SFDA SOTA baselines) + framing to be publishable at a top-tier venue
metadata: 
  node_type: memory
  type: project
  originSessionId: 7b2255a2-a355-4c8b-bb34-30776ee03798
---

**For top-tier publication (MICCAI / MedIA / IEEE-TMI tier) the current results are NOT enough on their own** (noted 2026-06-24). The novelty is real — source-free, weight-FROZEN test-time adaptation driven by a *learned cross-modal energy* used as a reward-gradient on bottleneck activations, for CT→UTE lung-seg generalization (closest prior: Humayun et al. ICLR'25 reward-gradient latent guidance, already cited). But:

**1. We have only beaten our OWN frozen baseline (m0), not literature SOTA.** Reviewers will require comparison against established test-time / source-free domain-adaptation methods run on the SAME UTE scans. Must add: TENT, CoTTA, SAR (TTA), plus a source-free domain adaptation (SFDA) baseline. Cannot claim "beats SOTA" until those are in the table. Current +0.0088 mean is "over a no-op", not "over SOTA".

**2. Frame around WORST-CASE robustness, not the pooled mean.** The +0.9% pooled headline undersells it; the story is +0.052 ΔDice on the failing tail (m0<0.88), double-digit per-scan rescues, zero source data, zero retraining, frozen network. See [[phase2-results-and-leash-tuning]].

**3. Need the full per-leash ablation table** (the script's leashes are independent booleans by design — see [[phase2-guidance-built]]) + the leash-tuning choices (VOLUME_GUARD_ENERGY_BYPASS, ENERGY_FALLBACK_MARGIN) justified on a VALIDATION split, not tuned-on-test.

Until 1–3 exist, this is a strong-venue candidate but not submittable. Part of [[ute-energy-guidance-project]].
