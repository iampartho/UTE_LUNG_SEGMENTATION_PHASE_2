"""
temp.py
=======
2D smoke test for target-aware Fourier Domain Adaptation (FDA).

Mirrors the 3D logic in visualize_gin_vs_fda.py but applied to 2D RGB road
images to verify the method works before applying it to 3D medical volumes.

  SOURCE : daytime  dashcam road (CC0, Wikimedia Commons / Unsplash pre-2017)
  TARGET : night-time dashcam road (CC0, Wikimedia Commons / Unsplash pre-2017)

What target-aware FDA does (Yang & Soatto, 2020)
-------------------------------------------------
  1. FFT both images per colour channel.
  2. Keep the SOURCE's Fourier *phase*  (structure / edges / anatomy).
  3. Replace the SOURCE's low-frequency *amplitude* (smooth, global "look")
     with the TARGET's low-frequency amplitude.
  4. Inverse FFT back to image space.

Expected results
----------------
  • Panel "FDA day→night" : road scene with DAY structure but NIGHT brightness /
    colour mood (dark sky, orange street-light glow).
  • Panel "FDA night→day" : road scene with NIGHT structure (light trails) but
    DAY-like colour and brightness.
  • Beta sweep            : small β transfers only DC/very-low-freq → subtle
    brightness shift; large β transfers more mid-freq texture → stronger
    stylistic change but possible structural bleed.

If panels 3 & 4 look as described, the method is working correctly and the
same approach (visualize_gin_vs_fda.py) should be transferring the MRI "look"
onto the CT anatomy in the 3D medical case.

Outputs saved to ./visualization/fda_2d_smoke_test/:
  fda_main.png       – 4-panel comparison
  fda_beta_sweep.png – effect of varying β on the day→night direction

Image credits (CC0 public domain):
  Day:   "Empty highway" – Jason Blackeye / Unsplash (pre-2017, CC0)
         https://upload.wikimedia.org/wikipedia/commons/9/9c/Empty_highway_%28Unsplash%29.jpg
  Night: "Highway light trains at night" – Jamie Street / Unsplash (pre-2017, CC0)
         https://upload.wikimedia.org/wikipedia/commons/1/10/Highway_light_trains_at_night_%28Unsplash%29.jpg
"""

import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ======================================================================
#  CONFIG
# ======================================================================

OUTPUT_DIR = "./visualization/fda_2d_smoke_test"

# Resize both images to this common square so their FFTs align voxel-for-voxel.
# 512×512 keeps enough detail and runs quickly.
RESIZE_TO = (512, 512)

# Default beta: half-width of the low-frequency square as a fraction of each
# image dimension.  0.08 keeps structure intact while shifting the "look".
FDA_BETA = 0.01

# Beta values to sweep for the diagnostic plot.
BETA_SWEEP = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]

# CC0 public-domain road images from Wikimedia Commons.
DAY_URL   = (
    "https://upload.wikimedia.org/wikipedia/commons/9/9c/"
    "Empty_highway_%28Unsplash%29.jpg"
)
NIGHT_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/1/10/"
    "Highway_light_trains_at_night_%28Unsplash%29.jpg"
)

DAY_PATH   = os.path.join(OUTPUT_DIR, "source_day.jpg")
NIGHT_PATH = os.path.join(OUTPUT_DIR, "target_night.jpg")


# ======================================================================
#  Helpers
# ======================================================================

def download_if_needed(url: str, path: str) -> None:
    """Download *url* to *path* if the file does not already exist."""
    if os.path.exists(path):
        print(f"  [cached] {os.path.basename(path)}")
        return
    print(f"  [download] {url}")
    urllib.request.urlretrieve(url, path)
    print(f"  [saved]    {path}")


def load_and_resize(path: str, size: tuple) -> np.ndarray:
    """Load a JPEG as float32 RGB in [0, 1], resized to (W, H) = size."""
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def fda_2d(source: np.ndarray, target: np.ndarray, beta: float) -> np.ndarray:
    """2D target-aware FDA applied independently per colour channel.

    Keeps the SOURCE's Fourier phase (structure) and replaces its
    low-frequency amplitude with the TARGET's, then inverse-FFTs back.

    Parameters
    ----------
    source, target : (H, W, 3) float32 arrays in [0, 1].
    beta           : Half-width of the low-freq square as a fraction of dim.
                     Corresponds to ``FDA_BETA`` in visualize_gin_vs_fda.py.

    Returns
    -------
    (H, W, 3) float32 array, each channel min-max normalised to [0, 1].
    """
    assert source.shape == target.shape, (
        f"FDA requires identical shapes; got {source.shape} vs {target.shape}"
    )
    H, W, C = source.shape
    cy, cx = H // 2, W // 2
    # Half-widths of the low-frequency square (at least 1 pixel).
    by = max(1, int(beta * H))
    bx = max(1, int(beta * W))

    result = np.zeros_like(source)

    for c in range(C):
        fft_src = np.fft.fft2(source[:, :, c])
        fft_tgt = np.fft.fft2(target[:, :, c])

        amp_src = np.abs(fft_src)
        pha_src = np.angle(fft_src)
        amp_tgt = np.abs(fft_tgt)

        # Shift DC to the array centre so the low-frequency region is a
        # centred square (same convention as visualize_gin_vs_fda.py).
        amp_src_sh = np.fft.fftshift(amp_src)
        amp_tgt_sh = np.fft.fftshift(amp_tgt)

        # Overwrite source low-freq amplitude with target's.
        amp_src_sh[cy - by : cy + by + 1,
                   cx - bx : cx + bx + 1] = \
        amp_tgt_sh[cy - by : cy + by + 1,
                   cx - bx : cx + bx + 1]

        # Un-shift, recombine with source phase, invert.
        amp_new = np.fft.ifftshift(amp_src_sh)
        fft_new = amp_new * np.exp(1j * pha_src)
        ch_out  = np.real(np.fft.ifft2(fft_new)).astype(np.float32)

        # Per-channel min-max normalisation so values are displayable.
        lo, hi = ch_out.min(), ch_out.max()
        result[:, :, c] = (ch_out - lo) / (hi - lo + 1e-8)

    return np.clip(result, 0.0, 1.0)


# ======================================================================
#  Main
# ======================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download images
    # ------------------------------------------------------------------
    # print("\n[1/4] Fetching images ...")
    # download_if_needed(DAY_URL,   DAY_PATH)
    # download_if_needed(NIGHT_URL, NIGHT_PATH)

    # ------------------------------------------------------------------
    # 2. Load & resize to a common shape
    # ------------------------------------------------------------------
    print("[2/4] Loading & resizing images ...")
    day   = load_and_resize(DAY_PATH,   RESIZE_TO)
    night = load_and_resize(NIGHT_PATH, RESIZE_TO)
    print(f"  day   shape: {day.shape}  range [{day.min():.2f}, {day.max():.2f}]")
    print(f"  night shape: {night.shape}  range [{night.min():.2f}, {night.max():.2f}]")

    # ------------------------------------------------------------------
    # 3. Apply FDA in both directions
    # ------------------------------------------------------------------
    print(f"[3/4] Applying FDA (β = {FDA_BETA}) ...")
    day_wearing_night = fda_2d(day,   night, FDA_BETA)
    night_wearing_day = fda_2d(night, day,   FDA_BETA)

    # ------------------------------------------------------------------
    # 4. Main 4-panel comparison figure
    # ------------------------------------------------------------------
    print("[4/4] Saving figures ...")

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        "2D FDA Smoke Test  —  day ↔ night dashcam road images\n"
        "(phase carries structure; amplitude carries 'look')",
        fontsize=13,
    )

    panels = [
        (day,               "Source\n(day — original)"),
        (night,             "Target\n(night — original)"),
        (day_wearing_night, f"FDA: day → night  (β = {FDA_BETA})\n"
                            "day phase + night amplitude"),
        (night_wearing_day, f"FDA: night → day  (β = {FDA_BETA})\n"
                            "night phase + day amplitude"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    main_path = os.path.join(OUTPUT_DIR, "fda_main.png")
    fig.savefig(main_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {main_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Beta sweep figure (day → night direction)
    # ------------------------------------------------------------------
    sweep_imgs = [fda_2d(day, night, b) for b in BETA_SWEEP]

    n_cols = len(BETA_SWEEP) + 1
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig2.suptitle(
        "FDA beta sweep  —  day phase + night amplitude\n"
        "β controls how large a low-frequency region is swapped",
        fontsize=13,
    )

    axes2[0].imshow(day)
    axes2[0].set_title("Source (day)\noriginal", fontsize=10)
    axes2[0].axis("off")

    for ax, img, b in zip(axes2[1:], sweep_imgs, BETA_SWEEP):
        ax.imshow(img)
        ax.set_title(f"β = {b}", fontsize=10)
        ax.axis("off")

    fig2.tight_layout()
    sweep_path = os.path.join(OUTPUT_DIR, "fda_beta_sweep.png")
    fig2.savefig(sweep_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {sweep_path}")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # 6. Print interpretation guide
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("WHAT TO CHECK IN THE OUTPUT IMAGES")
    print("=" * 65)
    print()
    print("fda_main.png")
    print("  Panel 3 (day wearing night):")
    print("    ✓ Road geometry / lane markings should still look like the")
    print("      daytime photo (phase = structure is preserved).")
    print("    ✓ Overall brightness / colour should shift toward the night")
    print("      image (dark sky, orange street-light tones).")
    print()
    print("  Panel 4 (night wearing day):")
    print("    ✓ Light trails and long-exposure blur structure from the")
    print("      night photo should still be visible.")
    print("    ✓ Colour / brightness should feel more like daytime.")
    print()
    print("fda_beta_sweep.png")
    print("  β = 0.01 → only DC (mean brightness) swapped → subtle shift.")
    print("  β = 0.08 → recommended default, good balance.")
    print("  β = 0.30+ → more mid-freq texture transferred; structure may")
    print("              start to distort (anatomy bleeds).")
    print()
    print("If the above holds true, the same mechanism in")
    print("visualize_gin_vs_fda.py should correctly transfer the MRI")
    print("'look' onto the CT anatomy in the 3D medical setting.")


if __name__ == "__main__":
    main()
