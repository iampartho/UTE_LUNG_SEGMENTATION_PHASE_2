# Draft Report — MRI Lung Segmentation via CT-Domain Image Translation

**Author:** Partho
**Date:** 2026-06-30
**Status:** Draft for PI review (scoping document, not the paper)

---

## 0. One-paragraph summary

We want to segment lungs in MRI (UTE/FRC) using a segmentation model that was **trained only on CT**. Instead of adapting the *segmentor* to MRI (augmentation / causal / roughness-enforced approaches we explored separately), we adapt the *input*: at test time we translate the MRI volume into a "synthetic CT" with an image-translation network, then hand that synthetic CT to the CT-only segmentor. The translation network is a 3D CycleGAN trained on **unpaired** CT and MRI. The central scientific concern is that our *current* translation network is trained with help from MRI ground-truth lung masks and an MRI-aware segmentor — which weakens the "domain adaptation" claim because target-domain labels leak into training. This report proposes how to remove that leakage, what data we need, what experiments establish the result, and the decision rule for whether the paper is viable.

---

## 1. Hypothesis

**Primary hypothesis (H1).**
A lung segmentation model trained *only on CT* generalizes poorly to MRI, but if we first translate the MRI into a synthetic CT using a CycleGAN trained on **unpaired CT/MRI without any MRI lung labels** (using only the **CT ground-truth lung mask** for anatomical consistency), the same CT-only segmentor will produce substantially better MRI lung segmentation — without ever fine-tuning the segmentor on MRI.

**Secondary hypotheses.**
- **H2 (baseline gap).** The CT-only segmentor applied directly to MRI performs poorly (establishes the problem is real).
- **H3 (beats a simple frequency baseline).** Our translation-based pipeline beats Fourier Domain Adaptation (FDA, `eval_step_1_mri_to_ct_fda.py`), which transfers only low-frequency amplitude and requires no training.
- **H4 (label-free is enough).** Removing the MRI GT mask and the MRI-aware (jointly trained) segmentor from translation training does **not** materially degrade downstream MRI Dice relative to the current leakage-containing pipeline. *(This is the make-or-break hypothesis for the paper's framing.)*
- **H5 (paired fallback).** If H4 fails, a translation network trained on a small set of **paired** MRI–CT scans recovers (or exceeds) the performance, and is still a clean domain-adaptation story.

---

## 2. Method

### 2.1 Overall pipeline (unchanged at test time)
Two stages, applied per MRI volume at inference:

1. **Translation (MRI → synthetic CT):** the trained generator `G_UTE→CT` maps a normalized MRI volume to a CT-like volume (sliding-window 256³ inference, Tanh output in [−1, 1]). Implemented in `eval_step_1_mri_to_ct.py`.
2. **Segmentation (synthetic CT → lung mask):** the frozen CT-only `BasicUNet` segments the synthetic CT exactly as it would a real CT.

The segmentor is never trained or fine-tuned on MRI. All adaptation happens in stage 1.

### 2.2 Translation network (CycleGAN), current vs. proposed

**Current training (`cyclegan/train.py`) — has target-label leakage:**
- Standard 3D CycleGAN: generators `G_UTE→CT`, `G_CT→UTE`; PatchGAN discriminators `D_CT`, `D_UTE`.
- Losses: adversarial (BCE) + cycle-consistency (L1) + **segmentation-consistency (Tversky, ×40)**.
- The segmentation-consistency term uses a **frozen segmentor jointly trained on CT *and* MRI** (`best_bunet_joint_train.pth`) and is computed against **both** the CT GT mask (`ct_mask`) **and** the MRI GT mask (`ute_mask`):
  - `seg_loss(seg(fake_ct_from_ute), ute_mask)` — requires the **MRI GT mask**.
  - `seg_loss(seg(fake_ute_from_ct), ct_mask)` — requires only the CT GT mask.
- **Leakage sources to remove:** (a) the MRI GT lung mask `ute_mask`; (b) the MRI-aware joint segmentor.

**Proposed training — label-free on the MRI side ("Method 1"):**
- Keep the CycleGAN backbone (adversarial + cycle-consistency).
- Replace the joint segmentor with a **CT-only segmentor** used purely to enforce **CT-side** anatomical consistency: `seg_loss(seg_ct_only(fake_ute_from_ct), ct_mask)` (uses only the CT GT mask).
- **Drop** the `ute_mask` term entirely. Consequence: the MRI→fake-CT direction is no longer anatomically anchored by a lung-mask loss; it is constrained only by cycle-consistency and the adversarial term. **Preserving lung anatomy through translation without an MRI mask is the core technical risk of Method 1**, and is exactly what H4 tests.
- Optional mitigations to evaluate if anatomy drifts: stronger cycle weight, identity loss, structure/edge or gradient-correlation losses, or a self-consistency term that segments the *reconstructed* MRI (cycle output) with the CT-only model.

**Fallback training — paired data ("Method 2"):**
- Use the newly collected **paired MRI–CT** scans. With registration/pairing available, train translation with a direct supervised image loss (e.g., L1/perceptual between fake-CT and the true paired CT) ± a paired segmentation-consistency loss using only CT labels. This removes the need for any MRI mask while restoring strong anatomical anchoring.

### 2.3 Comparators
- **Lower bound:** CT-only segmentor applied directly to MRI (no adaptation).
- **Training-free baseline:** FDA (`eval_step_1_mri_to_ct_fda.py`).
- **Upper-bound / leakage reference (not a publishable method):** current CycleGAN *with* MRI mask + joint segmentor — used internally to quantify how much performance we "give up" by removing leakage.

---

## 3. Data

> All volumes processed to the existing 1.25 mm isotropic `.npy` convention; evaluation resampled to 1 mm against reference NIfTIs where GT is available (matching the extended-mode path in `eval_step_1_mri_to_ct.py`).

### Method 1 (label-free, primary attempt)
| Role | Source | Count | Labels used |
|---|---|---|---|
| CT for translation + CT-only segmentor training | COPD-Gene | ~690 | CT GT lung masks **used** |
| MRI for translation (unpaired) | MRI (UTE/FRC) | ~110 (Marissa) | **No MRI masks used in training** |
| MRI for evaluation only | same MRI cohort | held-out subset | MRI GT masks used **only to score Dice** |
| MRI scale-up (if H1/H4 hold) | additional MRI | ~110 more (~220 total) | eval masks only |

Key point for the PI: in Method 1 the **only** masks entering training are CT masks. MRI masks are used **exclusively for evaluation**, which makes this a clean unsupervised domain-adaptation claim.

### Method 2 (paired fallback)
- **Paired MRI–CT** scans (PI to provide). Count TBD — need to confirm how many pairs and whether they are spatially registered.
- Same COPD-Gene CT pool can still augment the CT side.

### Data status / asks
- COPD-Gene CT (690) + masks: available.
- Marissa MRI (110): available; need to confirm how many have GT lung masks for evaluation and define a fixed train/val/test split.
- Additional ~110 MRI: availability/timeline?
- Paired MRI–CT: count, registration status, and timeline from PI.

---

## 4. Experiments

### 4.1 Experiments that *establish the result* (the paper's core table)
On the held-out MRI evaluation set, report Dice (primary), plus HD95/ASSD and lung-volume agreement (Bland–Altman / correlation, already supported by the validation helpers), for:

1. **E1 — No adaptation:** CT-only segmentor on raw MRI. *(Tests H2.)*
2. **E2 — FDA:** training-free Fourier translation + CT-only segmentor. *(Tests H3.)*
3. **E3 — Method 1 (label-free CycleGAN):** proposed clean pipeline. *(Tests H1.)*
4. **E4 — Leakage reference:** current CycleGAN (MRI mask + joint segmentor). *(Upper-bound context for H4; reported as ablation, not as our method.)*
5. **E5 — Method 2 (paired):** only if H4/Method 1 underperforms. *(Tests H5.)*

**Success looks like:** E3 ≫ E1, E3 > E2, and E3 ≈ E4 (within a small margin), i.e., we lose little by removing leakage.

### 4.2 Experiments that *evaluate / stress the hypothesis* (ablations & QC)
- **A1 — Leakage ablation (decisive for H4):** systematically remove components from the current pipeline and measure downstream MRI Dice: (i) full (mask+joint), (ii) drop MRI mask only, (iii) drop joint segmentor → CT-only segmentor, (iv) drop both (= Method 1). Quantifies which leakage source actually mattered.
- **A2 — Anatomy-preservation QC of synthetic CT:** because Method 1 has no MRI-mask anchor, verify the fake CT preserves lung boundaries: visual QC of `*_fake_ct` volumes, plus segmentor-agreement between fake-CT prediction and MRI GT, plus cycle-reconstruction error on MRI.
- **A3 — Anatomy-loss add-ons:** if A2 shows drift, test the mitigations in §2.2 (identity, gradient/edge, cycle-recon segmentation self-consistency) and report their effect.
- **A4 — Data-scale ablation:** Method 1 with 110 vs ~220 MRI (and CT subsets) to show how much unlabeled target data is needed.
- **A5 — Robustness/sensitivity:** translation checkpoint epoch, FDA beta sweep (fair tuning of the baseline), connected-component post-processing on/off, per-subject failure analysis.
- **A6 — (if pursued) Paired vs unpaired:** Method 2 vs Method 1 head-to-head on the same eval set.

---

## 5. Decision rule (when we proceed vs. abandon)

```
E1 (no adaptation) is poor on MRI?
  └─ no  → problem isn't real for this segmentor; reconsider framing.
  └─ yes → run Method 1 (E3) + ablation A1
       ├─ E3 ≫ E1, E3 > E2, and E3 ≈ E4  → STRONG result: clean UDA story. WRITE THE PAPER (Method 1).
       ├─ E3 ≫ E1, E3 > E2, but E3 ≪ E4  → leakage was doing real work; try anatomy add-ons (A3).
       │      ├─ recovers → write paper (Method 1 + add-ons).
       │      └─ still weak → fall back to PAIRED data (Method 2 / E5).
       │              ├─ Method 2 works → write paper (paired translation, still label-clean on MRI).
       │              └─ Method 2 fails → ABANDON this approach.
       └─ E3 ≈ E1 (label-free translation doesn't help) → go straight to Method 2.
              └─ same branch as above (works → paper; fails → abandon).
```

In all surviving branches the published method uses **no MRI lung labels in training** — labels remain evaluation-only — which is what makes the domain-adaptation claim defensible.

---

## 6. Open questions for the PI
1. How many of the 110 Marissa MRI scans have GT lung masks usable for evaluation, and can we fix a train/val/test split now?
2. For the paired fallback: how many MRI–CT pairs, and are they spatially registered (affects whether we can use a direct image-supervision loss)?
3. Target venue / deadline — this sets how many of the ablations (§4.2) we attempt before submission.
4. Is "E3 ≈ E4 within margin" the right framing of success, or do you want Method 1 to *beat* the leakage version outright?
5. Do we want lung-volume agreement (clinically meaningful) reported as a co-primary endpoint alongside Dice?
```

---

## 7. Risks / notes
- **Main risk:** without the MRI mask, lung anatomy may not survive translation (hallucinated or shrunken lungs in the fake CT). A2/A3 are designed to catch and fix this early.
- The current saved checkpoint (`Epoch_50_..._seg_loss_weighted_by_40_...`) reflects the *leakage* pipeline; Method 1 requires retraining the CycleGAN with the modified loss and a CT-only segmentor.
- Keep all three eval scripts separate (already the case) so baselines and the proposed method are never conflated.
