"""Test 1 ($0 local): 3D CT-First Feasibility Kill Test across all 4 $50k First Letters winners.

Tests whether raw CT intensity / thin-stroke CT DoG / bright-pixel excess / local CT texture
(WITHOUT a neural net!) separates real Greek-letter tiles from blank/non-ink papyrus tiles
on the EXACT SAME winning scroll meshes:
  - PHerc0846B (9.362 um / 113 keV)
  - PHerc0211  (9.362 um / 113 keV)
  - PHerc0343  (8.640 um / 116 keV)
  - PHerc0306B (8.640 um / 116 keV)

For each winner, compares:
  (A) Known winning letter tile (from Step 0 blind top-1) vs. 3 non-letter tiles on the SAME mesh
  (B) Pixel-level correlation r(raw_CT_signal, reader_v3_ink_map) inside the winning letter tile!
      (If ink is visible directly as bright strokes in CT, pixel correlation r(CT_max/CT_DoG, ink)
       or bright-pixel contrast on ink vs non-ink pixels MUST be strongly positive!)
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr
from PIL import Image
from scipy.ndimage import gaussian_filter
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fl_radar_sweep as fl


def compute_ct_features(slab17: np.ndarray, mask: np.ndarray) -> dict:
    """Compute non-ML physical CT ink proxies on a 17-layer slab [-8..+8]."""
    # Center 5 layers [-2..+2] = sheet interior/recto; full 17 layers max/mean
    c5 = slab17[6:11].astype(np.float32)
    mid = slab17[8].astype(np.float32)
    max5 = c5.max(axis=0)
    mean5 = c5.mean(axis=0)
    # Bright anomaly above local 18px neighborhood (DoG on CT itself)
    v = mask.astype(bool)
    w = v.astype(np.float32)
    def norm_dog(img, s1, s2):
        im = np.where(v, img, 0.0)
        g1 = gaussian_filter(im, s1) / np.maximum(gaussian_filter(w, s1), 1e-4)
        g2 = gaussian_filter(im, s2) / np.maximum(gaussian_filter(w, s2), 1e-4)
        return np.where(v, g1 - g2, 0.0)

    dog_max5 = norm_dog(max5, 4.0, 22.0)
    dog_mean5 = norm_dog(mean5, 4.0, 22.0)
    # Also check high-frequency texture / crackle variance (std over 15px window)
    mu = gaussian_filter(np.where(v, mean5, 0.0), 8.0) / np.maximum(gaussian_filter(w, 8.0), 1e-4)
    mu2 = gaussian_filter(np.where(v, mean5 ** 2, 0.0), 8.0) / np.maximum(gaussian_filter(w, 8.0), 1e-4)
    local_std = np.sqrt(np.maximum(0.0, mu2 - mu ** 2))
    return {
        "mid": mid,
        "max5": max5,
        "mean5": mean5,
        "dog_max5": dog_max5,
        "dog_mean5": dog_mean5,
        "local_std": local_std,
    }


def main():
    out_dir = ROOT / "outputs" / "council_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)

    # Known winning letter tiles (from Step 0 blind top-1) & saved full maps
    cases = [
        {
            "name": "PHerc0846B_113keV",
            "scroll": "PHerc0846B",
            "seg": ROOT / "cache/killtest1_846b/PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750",
            "auto_up": False,
            "win_yx": (3584, 3072, 1024, 1024),
            "ink_map": ROOT / "outputs/step0/full_maps/PHerc0846B_FULL_UNBOUNDED_rev.png",
            "alt_ink_map": ROOT / "outputs/step0/full_maps/PHerc0846B_FULL_UNBOUNDED_rev.jpg",
        },
        {
            "name": "PHerc0211_113keV",
            "scroll": "PHerc0211",
            "seg": ROOT / "cache/winners/PHerc0211",
            "auto_up": True,
            "win_yx": (696, 3584, 1024, 1024),
            "ink_map": ROOT / "outputs/step0/full_maps/WIN_PHerc0211_113keV_fwd.jpg",
            "alt_ink_map": ROOT / "outputs/step0/full_maps/WIN_PHerc0211_113keV_fwd.png",
        },
        {
            "name": "PHerc0343_116keV",
            "scroll": "PHerc0343",
            "seg": ROOT / "cache/winners/PHerc0343",
            "auto_up": True,
            "win_yx": (1024, 1536, 1024, 1024),
            "ink_map": ROOT / "outputs/step0/full_maps/WIN_PHerc0343_116keV_fwd.jpg",
            "alt_ink_map": ROOT / "outputs/step0/full_maps/WIN_PHerc0343_116keV_fwd.png",
        },
        {
            "name": "PHerc0306B_116keV",
            "scroll": "PHerc0306B",
            "seg": ROOT / "cache/winners/PHerc0306B",
            "auto_up": True,
            "win_yx": (1536, 2048, 1024, 1024),
            "ink_map": ROOT / "outputs/step0/full_maps/WIN_PHerc0306B_116keV_fwd.jpg",
            "alt_ink_map": ROOT / "outputs/step0/full_maps/WIN_PHerc0306B_116keV_fwd.png",
        },
    ]

    results = []
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 3.8 * len(cases)))

    for row_i, c in enumerate(cases):
        vpath, vum = fl.SCROLL_VOLUMES[c["scroll"]]
        arr = zarr.open_group(FsspecStore(fs, path=vpath, read_only=True), mode="r")["0"]
        cvol = fl.ChunkCachedVolume(arr, max_chunks=512)
        surf = fl.UprightTifXYZSurface(c["seg"], auto_upright=c["auto_up"])
        H_full, W_full = surf.full_resolution_shape
        y0, x0, wh, ww = c["win_yx"]
        bounds = (y0, y0 + wh, x0, x0 + ww)
        print(f"\n[{c['name']}] rendering winning tile {bounds}...", flush=True)
        slab, mask, dt = fl.render_surface_cached(surf, cvol, bounds=bounds, layers=17, workers=16)
        feats = compute_ct_features(slab, mask)

        # Load saved reader-v3 ink map (2x or 4x downsampled) and upsample to exact 1024x1024 tile
        imp = c["ink_map"] if c["ink_map"].exists() else c["alt_ink_map"]
        ink_full = np.asarray(Image.open(imp).convert("L"), dtype=np.float32) / 255.0
        scale_y = H_full / ink_full.shape[0]
        scale_x = W_full / ink_full.shape[1]
        iy0, iy1 = int(round(y0 / scale_y)), int(round((y0 + wh) / scale_y))
        ix0, ix1 = int(round(x0 / scale_x)), int(round((x0 + ww) / scale_x))
        ink_sub = ink_full[max(0, iy0):min(ink_full.shape[0], iy1), max(0, ix0):min(ink_full.shape[1], ix1)]
        ink_tile = np.asarray(Image.fromarray((ink_sub * 255).astype(np.uint8)).resize((ww, wh), Image.BILINEAR), dtype=np.float32) / 255.0

        # Pixel-level contrast between ink pixels (ink >= 0.50) and background papyrus (ink <= 0.15)
        ink_m = mask & (ink_tile >= 0.50)
        bg_m = mask & (ink_tile <= 0.15)
        rec = {"name": c["name"], "ink_px": int(ink_m.sum()), "bg_px": int(bg_m.sum()), "metrics": {}}
        for fk in ("mid", "max5", "mean5", "dog_max5", "dog_mean5", "local_std"):
            f = feats[fk]
            r_val = float(np.corrcoef(f[mask], ink_tile[mask])[0, 1]) if mask.any() else 0.0
            m_ink = float(f[ink_m].mean()) if ink_m.any() else 0.0
            m_bg = float(f[bg_m].mean()) if bg_m.any() else 0.0
            s_bg = float(f[bg_m].std()) if bg_m.any() else 1.0
            cohen_d = (m_ink - m_bg) / max(1e-5, s_bg)
            rec["metrics"][fk] = {"r_vs_ink": round(r_val, 4), "mean_ink": round(m_ink, 2), "mean_bg": round(m_bg, 2), "cohen_d": round(cohen_d, 3)}
        print(f"  ink_px={rec['ink_px']} | " + " | ".join(
            f"{k}: r={v['r_vs_ink']:+.3f} d={v['cohen_d']:+.2f}" for k, v in rec["metrics"].items()
        ), flush=True)
        results.append(rec)

        # Visualize: (0) reader-v3 ink ground truth, (1) CT max5, (2) CT DoG max5, (3) CT local_std
        axes[row_i, 0].imshow(ink_tile, cmap="gray", vmin=0, vmax=0.8)
        axes[row_i, 0].set_title(f"{c['name']}\nreader-v3 ink (target)", fontsize=9)
        for col_i, (fk, ttl) in enumerate([("max5", "Raw CT max(±2 vox)"), ("dog_max5", "CT DoG(max5)"), ("local_std", "CT local std (8px)")], 1):
            img = feats[fk]
            lo, hi = np.percentile(img[mask], [2, 98])
            axes[row_i, col_i].imshow(img, cmap="gray", vmin=lo, vmax=hi)
            m = rec["metrics"][fk]
            axes[row_i, col_i].set_title(f"{ttl}\nr={m['r_vs_ink']:+.3f}, d={m['cohen_d']:+.2f}", fontsize=9)
        for col_i in range(4):
            axes[row_i, col_i].set_xticks([]); axes[row_i, col_i].set_yticks([])

    fig.suptitle("Test 1 ($0 Local): Can Raw 3D CT / Non-ML Proxies See Ink Directly on the 4 $50k Winners?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "test1_ct_first_feasibility.png", dpi=130)
    plt.close(fig)
    (out_dir / "test1_ct_first_feasibility.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out_dir / 'test1_ct_first_feasibility.png'}", flush=True)


if __name__ == "__main__":
    main()
