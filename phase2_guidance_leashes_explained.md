# Phase 2 Guidance — The 7 Leashes Explained

A reference for understanding all the "leashes" in `phase2_guidance.py`: what each one
is, when it fires, a numeric example, an analogy, and how it protects a good prediction.

---

## The big picture: a glitchy compass

Everything is frozen — the segmentation U-Net and the Phase-1 energy model. The only
thing you learn, per scan, is a small additive perturbation `delta` to the U-Net
bottleneck `x4`:

```python
delta = torch.zeros_like(x4, requires_grad=True)
...
for it in range(1, MAX_ITERS + 1):
    opt.zero_grad()
    logits = unet_decode(seg, x1, x2, x3, x4 + delta)
    prob = torch.sigmoid(logits)
    target = soft_mask_latent(energy_model, prob)
    E = energy_of(img_pred, target)
    loss = E + (LAMBDA_PROX * delta.norm() if USE_PROX else 0.0)
    loss.backward()
    opt.step()
```

The cross-modal **energy `E_x`** is your compass. You nudge `delta` to walk "downhill"
in energy, hoping the mask gets better.

The crucial Phase-0 finding is that **this compass is trustworthy for bad masks but
unreliable for good ones**:

- **Bad prediction** (e.g. Dice 0.70): energy has a steep, correctly-pointing slope
  toward the ground truth. Following `-∇E` genuinely fixes it.
- **Already-good prediction** (Dice ≳ 0.93): energy is a nearly-flat basin. `-∇E` can
  point *slightly off* the true answer, so blindly following it can drag a great mask
  *downhill in energy but downhill in Dice too*.

So the entire philosophy is: **"fix the bad, protect the good."** The compass does the
fixing; the leashes do the protecting.

> **Master analogy.** Think of the refinement as a **dog (the optimizer) on leashes,
> sniffing toward a scent (lower energy)**. When you're genuinely lost in the woods
> (bad mask), you *want* the dog to pull hard — it knows the way home. But when you're
> already standing on your front porch (good mask), the dog still smells *something*
> faintly interesting down the street and will happily drag you off the porch. The
> leashes are the rules that let the dog run when you're lost but keep you planted when
> you're already home.

One more distinction: the loop optimizes the **soft energy** `E` (smooth,
differentiable, good for gradients and the early-stop signal), but it **selects** the
deliverable by the **binary energy** `E_bin` of the *cleaned* mask — the thing that
matches the Phase-0 numbers and what you ship:

```python
E_soft = E.item()
cur_bin = maybe_clean(_binary_native(prob, hwd))
vol, ncomp = float(cur_bin.sum()), n_components(cur_bin)
...
E_bin = binary_energy(energy_model, img_pred, cur_bin)
if E_bin < best_iter_E:
    best_iter_E, best_iter_bin = E_bin, cur_bin
```

`best_iter_bin` = the single iteration with the lowest binary energy — not necessarily
the last step. Several leashes feed this selection.

---

## Leash 1 — `USE_PROX`: the elastic leash (soft pull toward home)

```python
USE_PROX = True
LAMBDA_PROX = 0.01
```

**Mechanism.** Adds `LAMBDA_PROX * ||delta||` to the loss. Not a discrete event — a
*continuous* spring force pulling `delta` back toward 0 (and the mask back toward `m0`).
Every gradient step balances "lower the energy" against "don't move far from start."

**When it 'fires'.** Continuously. Its influence grows as `||delta||` grows: the
further the dog strays, the harder the spring tugs back.

**Numeric example.** With `LAMBDA_PROX = 0.01`, if `||delta||` reaches 30, it contributes
`0.3` of extra loss whose gradient opposes further growth. On a bad mask (large energy
gradient) this term is negligible — the dog still runs. On a good mask (energy gradient
≈ 0) this tiny spring **dominates**, so `delta` stays near 0.

**Analogy.** A bungee cord tied to your porch. Sprint away (bad mask) and you still
cover ground; stand around (good mask) and it quietly reels you back in.

**How it protects the good.** On a flat energy basin there's no real signal to chase, so
the only consistent force left is the spring → the mask barely changes → a 0.95-Dice
mask stays a 0.95-Dice mask.

---

## Leash 2 — `USE_DELTA_CAP`: the hard leash length (trust region)

```python
USE_DELTA_CAP = True
DELTA_MAX_NORM = 50.0
```

```python
if USE_DELTA_CAP:
    with torch.no_grad():
        nrm = delta.norm()
        if nrm > DELTA_MAX_NORM:
            delta.mul_(DELTA_MAX_NORM / (nrm + 1e-12))
```

**Mechanism.** A hard projection: after each optimizer step, if `||delta|| > 50`,
rescale `delta` back down to exactly norm 50. Where Leash 1 is a soft spring, this is a
**wall**.

**When it fires.** Any step where the accumulated perturbation exceeds the L2 ball of
radius 50 — typically on aggressive walks where Adam keeps pushing one direction.

**Numeric example.** If a step makes `||delta|| = 73`, it's multiplied by
`50/73 ≈ 0.685` → back to 50, preserving direction but capping magnitude. The dog can
keep pulling, but it physically cannot get more than 50 units from the post.

**Analogy.** A fixed-length leash. The dog can strain in any direction, but past 50 feet
the leash is taut — no matter how good the scent smells.

**How it protects the good.** It bounds the *worst-case* damage. Even if every other
leash failed and the compass pointed somewhere terrible, the mask can only be perturbed
so far. It's the backstop that guarantees a bad bottleneck nudge can't explode into a
wildly different mask.

---

## Leash 3 — `USE_EARLY_STOP`: stop sniffing when the trail goes cold

```python
USE_EARLY_STOP = True
ENERGY_REL_TOL = 1e-3
PATIENCE = 3
```

```python
rel = abs(prev_E - E_soft) / max(prev_E, 1e-8) if prev_E is not None else 1.0
...
if USE_EARLY_STOP:
    patience = patience + 1 if rel < ENERGY_REL_TOL else 0
    if patience >= PATIENCE:
        reason = "early_stop"
        break
prev_E = E_soft
```

**Mechanism.** Track the *relative* change in soft energy between steps. If it's below
`1e-3` (<0.1% change) for `PATIENCE = 3` consecutive steps, stop the walk.

**When it fires.** When the energy plateaus — which, by the Phase-0 insight, is *exactly
the signature of an already-good mask* sitting in the flat basin. `patience` resets to 0
the moment a step moves energy meaningfully again, so it only triggers on a sustained
plateau, not a single quiet step.

**Numeric example.** Good mask: energies go `0.402 → 0.4018 → 0.4017 → 0.4017` → three
consecutive sub-0.1% changes → stop at iteration ~4 instead of grinding all 20. A bad
mask whose energy is still crashing `0.616 → 0.55 → 0.47 → 0.40` keeps `rel` large,
never accumulates patience, and runs to convergence.

**Analogy.** The dog has stopped pulling and is just sniffing the same patch of grass.
No new scent → you don't keep wandering; you head back. Why walk 20 blocks when the
trail went cold after 3?

**How it protects the good.** Fewer steps on a flat basin = fewer chances for the off-GT
drift to accumulate. A good mask "stops almost immediately," so it never wanders far
enough to lose Dice.

---

## Leash 4 — `USE_VOLUME_GUARD`: the energy-gated volume sanity check (the subtle one)

```python
USE_VOLUME_GUARD = True
VOLUME_REL_TOL = 0.25
VOLUME_GUARD_ENERGY_BYPASS = 0.15
```

```python
energy_improving = E_bin < E0 * (1.0 - VOLUME_GUARD_ENERGY_BYPASS)
if (USE_VOLUME_GUARD and not energy_improving and vol0 > 0
        and abs(vol - vol0) / vol0 > VOLUME_REL_TOL):
    guard_fired, reason = True, f"volume_guard({abs(vol-vol0)/vol0:.2f})"
    break
```

**Mechanism.** If the refined lung volume drifts more than ±25% from `m0`'s volume,
that's suspicious — *unless* the energy has genuinely dropped past 15% below `E0`. The
`energy_improving` flag is the gate.

**When it fires.** Big volume move **AND** energy is *not* clearly improving. If both
hold, abort the walk and fall back to `m0`.

**The cautionary tale.** In the first run this guard was *not* energy-gated — a plain
±25% band. It fired on exactly the scans guidance helped most and threw the good results
away:

- **CVD-XE-002C** (Dice 0.708), **CVD-XE-007A** (0.779), **OECLAD_004A** (0.808) — all
  bad baselines.
- The correct fix for a badly over/under-segmenting baseline *requires* a large volume
  change. The energy was crashing (e.g. `0.616 → 0.351`, a ~43% drop), meaning the
  compass was clearly right — but the volume guard vetoed it and reverted to the worse
  `m0`.

The fix: only treat a big volume move as "divergence" when the energy *isn't*
validating it. Here `0.351 < 0.616 × 0.85 = 0.524` → `energy_improving = True` → guard
bypassed → the genuine fix is kept.

**Analogy.** A guardian who panics if you stray far from home: "You've gone 3 miles,
come back!" The energy gate is the guardian *checking the map first*: "Oh, you're 3
miles out but standing right at the destination I can see on GPS. Carry on." It only
drags you back when you've wandered far *and* there's no evidence you're heading
anywhere good.

**How it protects the good (and the bad).** For a good mask, a large unexplained volume
change with no real energy gain is almost certainly the flat-basin drift corrupting it →
revert to `m0`. For a bad mask, the energy gate prevents this same guard from sabotaging
legitimate large corrections.

---

## Leash 5 — `USE_COMPONENT_GUARD`: don't sprout phantom lungs

```python
USE_COMPONENT_GUARD = True
MAX_NEW_COMPONENTS = 1
COMPONENT_FULL_CONN = True
```

```python
if USE_COMPONENT_GUARD and ncomp > nc0 + MAX_NEW_COMPONENTS:
    guard_fired, reason = True, f"component_guard({ncomp}>{nc0}+{MAX_NEW_COMPONENTS})"
    break
```

**Mechanism.** Count connected components (26-connectivity). If guidance increases the
blob count by more than `MAX_NEW_COMPONENTS = 1` over the baseline `nc0`, abort and fall
back to `m0`.

**When it fires.** When the walk produces spurious extra blobs. Example: `m0` has 2
components (two lungs, `nc0 = 2`); the refinement scatters speckle and now there are 4 →
`4 > 2 + 1` → fire.

**Numeric example.** `nc0 = 2`. Guidance yields `ncomp = 4` floating fragments → guard
fires, revert to `m0`. But `ncomp = 3` (e.g. a small legitimately-detached lobe) is
tolerated, since `3 = 2 + 1` is not `> 2 + 1`.

**Analogy.** You asked the dog to fetch *the ball*. It comes back with the ball plus
three random shoes. That's not refinement, that's mess — drop it and keep what you had.

**How it protects the good.** Real anatomy is a small, stable number of pieces. A sudden
burst of components is a hallmark of the optimizer hallucinating structure, not
improving it. Aborting preserves the clean `m0` instead of shipping a fragmented mask.

---

## Leash 6 — `USE_LCC_POST`: the final fraction-gated cleanup

```python
USE_LCC_POST = True
NUM_LARGEST_CC = 2
LCC_FULL_CONNECTIVITY = True
LCC_MIN_FRAC = 0.10
```

```python
def keep_largest_cc_capped(mask_bin, k=NUM_LARGEST_CC, keep_frac=LCC_MIN_FRAC,
                           full_connectivity=LCC_FULL_CONNECTIVITY):
    ...
    keep = [int(order[0]) + 1]          # always keep the largest
    for idx in order[1:k]:              # at most k total
        if sizes[idx] >= keep_frac * largest:
            keep.append(int(idx) + 1)
        else:
            break
    return np.isin(lbl, keep).astype(np.float32)
```

**Mechanism.** A *post-process* applied to the chosen deliverable (not a mid-walk
abort). Keep at most `k = 2` largest components, but only keep the 2nd if it's ≥ `10%`
of the largest's size.

**When it fires.** Always, at the end (on whatever mask was selected — refined or `m0`).

**The subtle case it solves.** A naive "keep top-2" *always* keeps two blobs. But when
both lungs are predicted as **one fused component**, the largest CC is the whole lungs
and the "2nd largest" is whatever tiny false positive happens to be next — promoted to
rank 2 purely to satisfy `k=2`.

**Numeric example.** Fused lungs: largest CC = 180,000 voxels; next blob = 800 voxels.
`800 < 0.10 × 180,000 = 18,000` → the loop `break`s and only the lungs are kept.
Contrast a real two-lung mask: left = 95,000, right = 88,000; `88,000 ≥ 9,500` → both
kept.

**Analogy.** "Keep your two biggest suitcases" sounds fine until you only brought one
suitcase and a keychain — you don't count the keychain as luggage. The 10% rule asks "is
the second thing actually suitcase-sized?" before keeping it.

**How it protects the good.** It strips trailing false positives from the final mask
without forcing a phantom second lung onto fused-lung predictions — improving both
precision and the visual deliverable.

---

## Leash 7 — `USE_ENERGY_FALLBACK`: the margin that separates real wins from lucky dips

```python
USE_ENERGY_FALLBACK = True
ENERGY_FALLBACK_MARGIN = 0.10
```

```python
elif USE_ENERGY_FALLBACK and best_iter_E > E0 * (1.0 - ENERGY_FALLBACK_MARGIN):
    # refinement didn't beat m0 by the required relative margin
    refined_bin = m0_bin
    fell_back = True
    reason = (reason + "+" if reason else "") + "energy_fallback"
else:
    refined_bin = best_iter_bin
```

**Mechanism.** After the walk, only accept the refined mask if its best binary energy
beat `m0` by a **relative margin** of 10%: `best_iter_E < E0 × (1 − 0.10)`. Otherwise
revert to `m0`. This is the **final gatekeeper**.

**When it fires.** When the refinement only achieved a *small* energy improvement — the
tell-tale signature of an already-good mask.

**The data behind the number.** Across the 4 first-run datasets, of the ~100 scans that
"worsened," essentially all were already-good baselines (Dice ≥ 0.93) whose energy still
dipped ~**13% relative** while Dice drifted *down*. Genuine wins, by contrast, drop
energy ~**28% relative**. So a **10% margin cleanly separates the two regimes**.

**Numeric example.**

- Good mask, `E0 = 0.40`. A ~13%-class dip lands near the threshold `0.40 × 0.90 = 0.36`
  and mostly gets rejected → revert to `m0`. Tiny dips are not trusted.
- Bad mask, `E0 = 0.616`, refinement `0.351` (43% drop). `0.351 < 0.616 × 0.90 = 0.554`?
  Easily → accept. Real wins "blow past any threshold."

**Analogy.** A return policy: you can swap your current mask for a new one *only if the
new one is meaningfully better*, not 1% better. A trivial improvement isn't worth the
risk that you've actually downgraded on the things the receipt (energy) doesn't measure
(Dice).

**How it protects the good.** This is the single most important protector for good masks.
The flat-basin drift produces *small* energy improvements while silently hurting Dice;
the margin says "a small energy gain isn't proof of a real win," so it keeps `m0`.
Meanwhile genuine fixes drop energy so dramatically they sail through.

---

## How the leashes act together (one trip through `guide_scan`)

Order matters. Lifecycle:

1. **Compute `m0`, `vol0`, `nc0`, `E0`** — the baseline to protect and the reference for
   every guard.
2. **Walk up to `MAX_ITERS = 20` steps.** Each step:
   - Take an Adam step on `loss = E + 0.01·||delta||` → **Leash 1** (soft pull) shapes
     every gradient.
   - **Leash 2** projects `||delta|| ≤ 50` (hard cap).
   - Clean the mask, compute `E_bin`, update `best_iter_bin` (the selection memory).
   - **Leash 4** (volume, energy-gated) and **Leash 5** (components) can `break` and mark
     `guard_fired`.
   - **Leash 3** (early stop) can `break` on an energy plateau.
3. **Decide the deliverable**, in priority order:
   - guard fired → **`m0`**
   - never found a candidate → **`m0`**
   - **Leash 7**: best energy didn't beat `m0` by 10% → **`m0`**
   - else → **`best_iter_bin`** (the refinement)
4. **Leash 6** (`keep_largest_cc_capped`) cleans whatever was chosen.

The asymmetry that encodes "fix the bad, protect the good":

- **Bad mask:** Leashes 1/3 stay quiet (big gradients, big energy moves), Leash 4 is
  bypassed by the energy gate, and Leash 7's margin is trivially cleared. The dog runs
  home. ✅
- **Good mask:** Leash 1 dominates (no real gradient), Leash 3 stops it in ~3–4 steps,
  and even if it produced a tiny dip, Leash 7 rejects it for failing the 10% margin → you
  ship the original `m0`. The dog stays on the porch. ✅

And the **fallbacks are conservative**: when a guard fires mid-walk, the code reverts to
`m0` — *not* to `best_iter_bin`. The guards treat divergence as a reason to distrust the
*entire* walk, not just to pick an earlier step.

---

## Quick reference

| # | Leash | Type | Fires when | On fire |
|---|-------|------|-----------|---------|
| 1 | `USE_PROX` | soft loss term | always (grows with `||delta||`) | pulls `delta`→0 |
| 2 | `USE_DELTA_CAP` | hard projection | `||delta|| > 50` | rescale to norm 50 |
| 3 | `USE_EARLY_STOP` | stop criterion | `|ΔE|/E < 1e-3` for 3 steps | break the walk |
| 4 | `USE_VOLUME_GUARD` | abort guard | vol drift > 25% **and** energy not improving | break → `m0` |
| 5 | `USE_COMPONENT_GUARD` | abort guard | `ncomp > nc0 + 1` | break → `m0` |
| 6 | `USE_LCC_POST` | post-process | always (end) | keep ≤2 CCs, 2nd ≥10% of largest |
| 7 | `USE_ENERGY_FALLBACK` | final gate | `best_iter_E > E0·(1−0.10)` | revert → `m0` |

Set `USE_GUIDANCE = False` to emit the pure baseline `m0` (the zero-leash ablation
anchor). Every leash is an independent boolean so you can ablate them one at a time.
