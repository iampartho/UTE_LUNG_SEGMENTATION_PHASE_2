# Memory Index

- [UTE energy-guidance project](ute-energy-guidance-project.md) — source-free CT→UTE generalization via learned energy + test-time activation guidance (goal, two energies, key files)
- [Phase 0/1/2 plan](phase0-1-2-plan.md) — definitions of all phases + the Phase-0 diagnostic journey that reshaped Step 1
- [Mask-encoder current stage](mask-encoder-current-stage.md) — Step 1 done (new-aug LATEST chosen); Step 2 image-encoder code built (dataset/model/train_img_jepa.py), ready to train (2026-06-22)
- [Final cross-modal Phase 0](final-cross-modal-phase0.md) — decision-grade Phase 0 design + go/no-go criteria gating Phase 2
- [Phase 2 guidance built](phase2-guidance-built.md) — phase2_guidance.py built (x4-delta activation guidance + toggleable leashes); GO-with-leashes decision from the cross-modal Phase-0 plots (2026-06-23)
- [Phase 2 delta-capacity ablation](phase2-delta-capacity-ablation.md) — user's open concern (delta too weak / nullified by leashes) + agreed 3-step ablation; RESOLVED — see results memory
- [Phase 2 results + leash tuning](phase2-results-and-leash-tuning.md) — first 4-set run (+0.0088 pooled, +0.052 on m0<0.88); volume-guard fallback diagnosis + the 2 leash changes (energy-gated guard, fallback margin) + plot_phase2_low_dice.py
- [Phase 2 publication baselines](phase2-publication-baselines.md) — must beat TENT/CoTTA/SAR + an SFDA baseline (not just m0); frame on worst-case robustness; tune leashes on validation split
- [GitHub push setup](github-push-setup.md) — dedicated SSH key + config for pushing this repo from Argon; .gitignore gotcha (only .py tracked, ._*.py sneak in)
- [MRI→CT translation paper](mri-to-ct-translation-paper.md) — separate idea: CycleGAN MRI→synthetic-CT as UDA for CT-only segmentor; PI report drafted, pipeline reorg pending
