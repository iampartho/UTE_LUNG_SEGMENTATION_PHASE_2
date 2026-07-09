"""
render_lung_masks_html.py

Render every ``*_pred_1mm.nii.gz`` lung-mask in ``prediction/temp_pred125_eval1mm/``
as a static 3D PNG (marching-cubes + matplotlib) and emit a single
``index.html`` with a 2-column table:

    | Filename | 3D render |

All artefacts go into ``prediction/temp_pred125_eval1mm_html/``:

    prediction/temp_pred125_eval1mm_html/
        index.html
        images/
            <case>_pred_1mm.png

Run from the repo root:
    python render_lung_masks_html.py
"""

import glob
import html
import os
import time
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # no GUI; safe on headless servers

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import zoom
from skimage import measure

# =====================================================================
#  CONFIG
# =====================================================================

INPUT_DIR  = "./prediction/phase2_guidance_new_run/single_scan"
OUTPUT_DIR = "./prediction/phase2_guidance_new_run/single_scan_html"
IMG_SUBDIR = "images"

# Spatial downsample factor before marching-cubes.  1.0 = full resolution.
# 0.5 cuts voxel count ~8x and renders ~5–10× faster with negligible visual loss
# for a thumbnail-sized 3D view.
DOWNSAMPLE = 0.5

# Render geometry — anterior 3/4 view of the chest.
# Coordinate convention after sitk.DICOMOrient(img, 'LPS') and our axis remap:
#   +X = patient's Left,   +Y = Posterior,   +Z = Superior (apex up).
# An azim near -90° puts the camera in front of the patient looking back; a
# small offset (-75°) gives a slight 3/4 turn so both lungs are clearly visible.
FIG_SIZE_IN   = 4.5         # inches (square figure)
DPI           = 130
VIEW_ELEV     = 12          # degrees above horizontal
VIEW_AZIM     = -75         # degrees around vertical axis (anterior view)
THUMB_WIDTH   = 360         # px width in the HTML

# Skip re-rendering files that already have a PNG (set False to force redo)
SKIP_EXISTING = True


# =====================================================================
#  3D rendering
# =====================================================================

def _empty_placeholder(out_png):
    fig, ax = plt.subplots(figsize=(FIG_SIZE_IN, FIG_SIZE_IN))
    ax.text(0.5, 0.5, "Empty mask", ha="center", va="center",
            fontsize=14, color="#888")
    ax.set_axis_off()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_mask_3d(mask_path, out_png):
    """Render a single binary NIfTI mask as a shaded 3D mesh PNG.

    Loading is done with SimpleITK and the volume is reoriented to canonical
    LPS so that, regardless of how the file is stored, the rendered apex is
    at the top, posterior is +Y, and the patient's left is +X.
    """
    img = sitk.ReadImage(mask_path)
    try:
        # Reorient to a canonical LPS frame.  After this call:
        #   array axes (k, j, i) correspond to (Superior, Posterior, Left)
        #   GetSpacing() returns (sx_L, sy_P, sz_S) in the same order.
        img = sitk.DICOMOrient(img, "LPS")
    except Exception:                                       # noqa: BLE001
        # Some images (e.g. with degenerate Direction) cannot be reoriented;
        # fall back to whatever orientation is on disk.
        pass

    arr  = sitk.GetArrayFromImage(img)                      # (z, y, x)
    mask = (arr > 0.5).astype(np.uint8)

    if mask.sum() == 0:
        _empty_placeholder(out_png)
        return

    if DOWNSAMPLE != 1.0:
        mask = zoom(mask, DOWNSAMPLE, order=0)

    # 1-voxel zero pad so marching-cubes closes any boundary-touching surfaces.
    mask = np.pad(mask, 1, mode="constant", constant_values=0)

    # SimpleITK GetSpacing() is (sx, sy, sz) in physical/world order, but the
    # numpy array axes are (z, y, x).  Reverse so the spacing matches the
    # array, and undo the downsample so verts come out in true millimetres.
    spacing_xyz = np.abs(np.array(img.GetSpacing(), dtype=np.float32))
    spacing_zyx = (spacing_xyz[::-1] / max(DOWNSAMPLE, 1e-6)).tolist()

    verts, faces, _, _ = measure.marching_cubes(
        mask, level=0.5, spacing=tuple(spacing_zyx)
    )
    # `verts` columns are (z_mm, y_mm, x_mm).  Remap to (x, y, z) so matplotlib's
    # 3D axes correspond to (Left, Posterior, Superior).
    verts = verts[:, [2, 1, 0]]

    # Per-face shading: surface-normal Lambertian + a subtle anterior-posterior
    # depth gradient (anterior = bright, posterior = darker) for depth cues.
    tris    = verts[faces]                                  # (F, 3, 3)
    e1      = tris[:, 1] - tris[:, 0]
    e2      = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norms, 1e-9, None)

    # Light direction in world space — slightly above and to the side of the
    # camera, so contours read well from the chosen viewpoint.
    light_az = np.deg2rad(VIEW_AZIM + 25)
    light_el = np.deg2rad(VIEW_ELEV + 30)
    light_dir = np.array([
        np.cos(light_el) * np.cos(light_az),
        np.cos(light_el) * np.sin(light_az),
        np.sin(light_el),
    ], dtype=np.float32)
    shade = np.clip(np.abs(normals @ light_dir), 0.0, 1.0)  # (F,)

    y_centres = tris.mean(axis=1)[:, 1]                     # +Y = posterior
    y_norm    = (y_centres - y_centres.min()) / max(
        1e-6, float(y_centres.max() - y_centres.min())
    )
    base = plt.cm.RdPu(0.65 - 0.35 * y_norm)                # anterior brighter
    intensity = (0.40 + 0.60 * shade)[:, None]              # ambient + lambert
    face_colors = base.copy()
    face_colors[:, :3] *= intensity

    fig = plt.figure(figsize=(FIG_SIZE_IN, FIG_SIZE_IN))
    ax  = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(tris, linewidths=0, alpha=1.0)
    mesh.set_facecolor(face_colors)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(tuple(np.maximum(maxs - mins, 1e-3)))
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =====================================================================
#  HTML emission
# =====================================================================

_HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lung mask 3D renders &mdash; {src}</title>
<style>
  :root {{
    --bg: #fafafa;
    --card: #ffffff;
    --border: #e6e6e6;
    --hover: #fcf7e9;
    --text: #222;
    --muted: #666;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 32px; color: var(--text); background: var(--bg);
  }}
  h1 {{ font-weight: 600; margin-bottom: 4px; }}
  p.meta {{ color: var(--muted); margin-top: 0; }}
  table {{
    border-collapse: collapse;
    width: 100%; max-width: 1100px;
    background: var(--card);
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
  }}
  th, td {{
    border-bottom: 1px solid var(--border);
    padding: 10px 14px;
    vertical-align: middle;
    text-align: left;
  }}
  th {{ background: #f0f0f0; font-weight: 600; position: sticky; top: 0; }}
  tr:hover td {{ background: var(--hover); }}
  td.fname {{
    font-family: "JetBrains Mono", "Menlo", monospace;
    font-size: 13px;
    word-break: break-all;
    width: 40%;
  }}
  td.img img {{
    width: {thumb}px;
    height: auto;
    display: block;
    border-radius: 4px;
  }}
  .count-badge {{
    display: inline-block; background:#eef; color:#335;
    padding: 2px 8px; border-radius: 10px; font-size: 12px;
    margin-left: 6px;
  }}
</style>
</head>
<body>
<h1>3D renders of lung-mask predictions <span class="count-badge">{n} masks</span></h1>
<p class="meta">Source folder: <code>{src}</code></p>
<table>
<thead><tr><th>Filename</th><th>3D render</th></tr></thead>
<tbody>
"""

_HTML_TAIL = """</tbody>
</table>
</body>
</html>
"""


def write_html(rows, out_path, src):
    with open(out_path, "w") as f:
        f.write(_HTML_HEAD.format(src=html.escape(src),
                                  n=len(rows),
                                  thumb=THUMB_WIDTH))
        for fname, rel in rows:
            f.write(
                f'  <tr>'
                f'<td class="fname">{html.escape(fname)}</td>'
                f'<td class="img"><img src="{html.escape(rel)}" alt="{html.escape(fname)}"/></td>'
                f'</tr>\n'
            )
        f.write(_HTML_TAIL)


# =====================================================================
#  Main
# =====================================================================

def main():
    image_dir = os.path.join(OUTPUT_DIR, IMG_SUBDIR)
    os.makedirs(image_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nii.gz")))
    print(f"Found {len(paths)} mask files in {INPUT_DIR}")

    rows = []  # type: List[Tuple[str, str]]
    t_total = time.time()

    for i, p in enumerate(paths, 1):
        fname    = os.path.basename(p)
        png_name = fname.replace(".nii.gz", ".png")
        png_path = os.path.join(image_dir, png_name)
        rel_path = f"{IMG_SUBDIR}/{png_name}"

        if SKIP_EXISTING and os.path.exists(png_path):
            print(f"[{i:>3}/{len(paths)}] skip (cached)  {fname}")
            rows.append((fname, rel_path))
            continue

        t0 = time.time()
        try:
            render_mask_3d(p, png_path)
        except Exception as e:                              # noqa: BLE001
            print(f"[{i:>3}/{len(paths)}] FAILED         {fname}: {e}")
            continue
        dt = time.time() - t0
        print(f"[{i:>3}/{len(paths)}] rendered {dt:5.1f}s {fname}")
        rows.append((fname, rel_path))

    html_path = os.path.join(OUTPUT_DIR, "index.html")
    write_html(rows, html_path, INPUT_DIR)

    print(f"\nWrote {html_path}")
    print(f"Total elapsed: {time.time() - t_total:.1f}s "
          f"({len(rows)}/{len(paths)} rendered successfully)")


if __name__ == "__main__":
    main()
