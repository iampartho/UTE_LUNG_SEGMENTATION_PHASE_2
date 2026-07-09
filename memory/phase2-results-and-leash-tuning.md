---
name: phase2-results-and-leash-tuning
description: "Phase 2 first-run results across 4 UTE sets + the two leash changes (energy-gated volume guard, energy-fallback margin) and the analysis that motivated them"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7b2255a2-a355-4c8b-bb34-30776ee03798
---

**First full Phase-2 run results (2026-06-24)** — `phase2_guidance.py` ([[phase2-guidance-built]]) run on FOUR UTE sets; CSVs in `prediction/phase2_logs/` (`phase2_guidance_metrics{,_marissa_data,_KP_data_ILD,_prev_data}.csv`) + terminal-output .docx per run.

**Pooled (N=221): mean Dice 0.9314 → 0.9402 (+0.0088), paired-t p=7e-8, Wilcoxon p=2e-4.** The win is concentrated exactly as the "fix the bad, protect the good" design predicted:
- baseline m0 < 0.88 (n=23): **ΔDice +0.052** (per-scan rescues up to +0.133: CVD-XE-001C 0.733→0.867, HN-XE-07 0.674→0.800, prev `034` 0.855→0.942, Marissa `103-035` 0.789→0.903).
- m0 0.88–0.93: ~+0.015. m0 ≥ 0.93 (good): ~−0.001 (a wash, slight negative lean).
- Among improved scans mean gain +0.021; among worsened mean loss only −0.005; worst single regression −0.026 (no blow-ups → leashes work). BUT ~102/221 "worsened", nearly all in the m0≥0.93 bucket — the good-scan tail is the only weakness.

**User's "energy improved but prediction not taken" mystery = the VOLUME GUARD.** Three scans fell back: CVD-XE-002C (0.708, E 0.616→0.351 in-walk), CVD-XE-007A (0.779, E 0.447→0.215), OECLAD_004A (0.808, E 0.371→0.226). All had energy CRASHING (the compass said go) but lung volume moved >25% so the symmetric guard hard-broke the walk and reverted to m0 — `if guard_fired: refined_bin = m0_bin` discards the already-tracked lower-E `best_iter_bin`. Root cause: a bad baseline is bad BECAUSE it over/under-segments, so the correct fix REQUIRES a big volume change → the guard fires precisely on the highest-value scans.

**Two leash changes made (2026-06-24), both auditable (new knob = 0.0 reproduces first-run):**
1. **Energy-gated volume guard.** Moved `E_bin` computation ABOVE the guards in `guide_scan`; guard now fires only if `not (E_bin < E0*(1-VOLUME_GUARD_ENERGY_BYPASS))`. New config `VOLUME_GUARD_ENERGY_BYPASS = 0.15`. First-run default was hard-revert, no gate, `VOLUME_REL_TOL=0.25`.
2. **Energy-fallback relative margin.** Decision changed `best_iter_E > E0` → `best_iter_E > E0*(1 - ENERGY_FALLBACK_MARGIN)`. Motivated by: ALL 102 worsened scans had E_final<E0 (shallow-basin off-GT drift) but worsened scans dropped E only ~13% rel vs ~28% for genuine wins → a margin separates them.

   **Re-tuned 2026-06-25: `ENERGY_FALLBACK_MARGIN` 0.10 → 0.15** (file updated + Leash-7 comment now self-documents this). Full replay over the 4 CSVs (218 applied scans; the runs logged the kept refinement's energy with the fallback un-enforced — energy-drop runs to ~0%, no `energy_fallback` reason — so any margin M can be simulated exactly by zeroing every applied scan with `1−E_final/E0 ≤ M` back to m0). Delivered **pooled mean ΔDice**: 0.10→+0.00934, **0.15→+0.00984**, 0.18→+0.01004, 0.20→+0.00998, 0.25→+0.00961 — a **flat plateau 0.15–0.22, then a cliff at 0.25+** (margin starts vetoing genuine wins; #improve falls 87→64). #worsen collapses with margin: 63→30 (0.15)→17 (0.18)→13 (0.20). Per-dataset 0.10→0.15 raises/matches pooled gain on 3 of 4 and halves worseners everywhere (prev 12→1 at identical gain; KP_ILD best at 0.18: +0.0049→+0.0059; KP_OECLAD dips trivially but n=16=noise). Chose **0.15 = front of the plateau** (safest vs in-sample overfit) over the noisier 0.18–0.22 peak. CAVEAT unchanged: in-sample selection → lock on a validation split for the paper, where 0.15's plateau-front position is the safe bet ([[phase2-publication-baselines]]).

Other leashes (prox, delta_cap, early_stop, component_guard, LCC) left as-is — not the bottleneck. E_soft traces bounce (LR=0.05 slightly hot) but best-iter-on-binary-energy absorbs it.

**Pending verification:** re-run prev + OECLAD sets (the two with fallbacks; Marissa/KP-ILD had zero) with the new leashes. Check the 3 target scans clear their fallback Dice (0.708/0.779/0.808) and that no NEW large regression appears (gating the guard also removes its safety net for other scans).

**Also built today — `plot_phase2_low_dice.py`** (repo root): standalone. Takes one/more metrics CSVs + a Dice threshold; dumbbell figure, baseline→refined arrows, LEFT=Dice (higher better) RIGHT=energy (lower better), shared scan rows sorted worst-first; green=good move, red=bad, hollow-grey=fell_back (no action). Auto-detects 1mm vs 1.25mm Dice columns, concatenates multiple CSVs, echoes table to terminal. `python plot_phase2_low_dice.py CSV... -t 0.90 -o out.png`.

See [[phase2-publication-baselines]] for what we still must compare against to publish. Resolves the verification in [[phase2-delta-capacity-ablation]] (capacity was fine; the real cost of leashes was missed-improvement via fallback/guard, exactly as hypothesised).
