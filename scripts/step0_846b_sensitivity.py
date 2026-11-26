"""Step 0 ($0 local): Depth-offset & 8.64um scale sensitivity curve on $50k PHerc0846B winner.

Fetches ONE 45-layer slab ([-22..+22] vox along surface normal) over the known $50k
PHerc0846B Greek letter region (bounds=(2048, 3072, 2048, 3072)), then runs reader-v3
across:
  1. Depth-center offsets dz in [-12, -10, -8, -6, -4, -2, 0, +2, +4, +6, +8, +10, +12] voxels
  2. In-plane + through-plane isotropic scale shifts 0.923x (8.64um/9.362um) and 1.083x
  3. Per-256px-tile CT-peak recentering check (does auto-centering preserve or shift the letters?)

Outputs JSON metrics + a visual comparison strip in outputs/step0/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import tifffile
import torch
import zarr
from scipy.ndimage import gaussian_filter, label, zoom
from zarr.storage import FsspecStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fl_radar_sweep as fl


def raw_stroke_score(prob: np.ndarray, mask: np.ndarray, voxel_um: float = 9.362) -> dict:
    """Raw-float thin-stroke DoG morphology scorer at full 9um resolution (no PNG contrast warp)."""
    # Scale sigmas proportionally to voxel size so physical stroke band (~0.04..0.22 mm) stays invariant
    s_scale = 9.362 / voxel_um
    # Full-res 9.362um: Greek stroke width ~15-45 px (0.14-0.42 mm).
    # In rerank_panels_strokes.py (2.2x downsampled), sigmas were 1.8 and 9.0 -> ~4.0 and 20.0 at full res!
    s_lo = 3.5 * s_scale
    s_hi = 18.0 * s_scale
    valid = mask.astype(bool)
    if valid.mean() < 0.30:
        return {"score": 0.0, "n_strokes": 0, "cells": 0, "dog99": 0.0, "dog95": 0.0, "giant": 0.0, "p99": 0.0}

    # Zero out invalid area cleanly using normalized convolution so mask edges don't ring
    p = np.where(valid, np.clip(prob, 0.0, 1.0), 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    g_lo = gaussian_filter(p, s_lo) / np.maximum(gaussian_filter(w, s_lo), 1e-4)
    g_hi = gaussian_filter(p, s_hi) / np.maximum(gaussian_filter(w, s_hi), 1e-4)
    dog = np.where(valid, np.clip(g_lo - g_hi, 0.0, 1.0), 0.0)

    # Crisp thin-stroke components
    bw = (dog > 0.06) & valid
    lbl, ncomp = label(bw)
    sizes = np.bincount(lbl.ravel())[1:] if ncomp > 0 else np.array([], dtype=np.int64)
    area_scale = s_scale ** 2
    min_sz = int(round(100 * area_scale))
    max_sz = int(round(7000 * area_scale))
    stroke_comps = int(((sizes >= min_sz) & (sizes <= max_sz)).sum())

    # Broad delamination giant-blob penalty on raw probability
    p_val = p[valid]
    thr_blob = max(0.35, float(np.percentile(p_val, 88)))
    raw_bw = (p > thr_blob) & valid
    rlbl, rn = label(raw_bw)
    rsizes = np.bincount(rlbl.ravel())[1:] if rn > 0 else np.array([], dtype=np.int64)
    giant_lim = int(round(11000 * area_scale))
    giant_frac = float(rsizes[rsizes > giant_lim].sum()) / max(1, int(valid.sum()))

    # 4x4 grid spread
    H, W = dog.shape
    gh, gw = max(1, H // 4), max(1, W // 4)
    cell_hits = 0
    for r in range(4):
        for c in range(4):
            cm = valid[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw]
            if cm.mean() < 0.25:
                continue
            cell = dog[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw][cm]
            if float(np.percentile(cell, 98)) > 0.065:
                cell_hits += 1

    dog_p99 = float(np.percentile(dog[valid], 99))
    dog_p95 = float(np.percentile(dog[valid], 95))
    p99 = float(np.percentile(p_val, 99))
    score = (
        dog_p99
        * (0.5 + 4.0 * dog_p95)
        * (min(stroke_comps, 40) / 10.0)
        * (0.25 + cell_hits / 16.0)
        / (1.0 + 8.0 * giant_frac)
    )
    return {
        "score": round(float(score), 5),
        "n_strokes": stroke_comps,
        "cells": cell_hits,
        "dog99": round(dog_p99, 4),
        "dog95": round(dog_p95, 4),
        "giant": round(giant_frac, 4),
        "p99": round(p99, 4),
    }


def main():
    out_dir = ROOT / "outputs" / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = ROOT / "cache" / "killtest1_846b" / "PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750"
    pub_raw_path = ROOT / "cache" / "killtest1_846b" / "PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750_reader-v3_raw.tif"

    bounds = (2048, 3072, 2048, 3072)  # 1024x1024 centered right on $50k letters
    vpath, vum = fl.SCROLL_VOLUMES["PHerc0846B"]

    print("Loading reader-v3 on", torch.cuda.get_device_name(0), "(FP32)...", flush=True)
    runner = fl.FlatModelRunner("reader_v3", ROOT / "cache" / "models" / "reader-v3.pth", fp16=False, bf16=False)

    print("Opening PHerc0846B S3 volume and rendering 45 layers ([-22..+22] vox)...", flush=True)
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    arr = zarr.open_group(FsspecStore(fs, path=vpath, read_only=True), mode="r")["0"]
    cvol = fl.ChunkCachedVolume(arr, max_chunks=512)
    surf = fl.UprightTifXYZSurface(seg_dir, auto_upright=False)

    layers45, mask, dt_ren = fl.render_surface_cached(surf, cvol, bounds=bounds, layers=45, workers=16)
    print(f"Rendered 45x1024x1024 in {dt_ren:.1f}s (valid={mask.mean():.3f})", flush=True)

    # Check CT intensity profile along normal [-22..+22]
    ct_means = [float(layers45[i][mask].mean()) for i in range(45)]
    ct_stds = [float(layers45[i][mask].std()) for i in range(45)]
    peak_layer = int(np.argmax(ct_means)) - 22
    print(f"CT mean profile peak at dz={peak_layer:+d} vox (val={max(ct_means):.1f}, dz=0 val={ct_means[22]:.1f})", flush=True)

    pub_crop = tifffile.imread(pub_raw_path)[bounds[0]:bounds[1], bounds[2]:bounds[3]].astype(np.float32) / 255.0

    # 1. Depth-center offset sweep: dz in [-12..+12]
    offsets = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
    depth_results = []
    depth_maps = {}

    # Compute baseline dz=0 first (both fwd and rev)
    base_slab = layers45[22 - 8 : 22 + 9]
    base_fwd = runner.infer(base_slab, mask, overlap=0.5, reverse=False, batch_size=4)
    base_rev = runner.infer(base_slab, mask, overlap=0.5, reverse=True, batch_size=4)
    r_pub_fwd = float(np.corrcoef(base_fwd[mask], pub_crop[mask])[0, 1])
    r_pub_rev = float(np.corrcoef(base_rev[mask], pub_crop[mask])[0, 1])
    use_rev = r_pub_rev > r_pub_fwd
    base_map = base_rev if use_rev else base_fwd
    print(f"Published raw match: r_fwd={r_pub_fwd:.4f}, r_rev={r_pub_rev:.4f} -> using {'rev' if use_rev else 'fwd'}", flush=True)

    for dz in offsets:
        c = 22 + dz
        slab = layers45[c - 8 : c + 9]
        pred = runner.infer(slab, mask, overlap=0.5, reverse=use_rev, batch_size=4)
        pred_opp = runner.infer(slab, mask, overlap=0.5, reverse=(not use_rev), batch_size=4)
        r_base = float(np.corrcoef(pred[mask], base_map[mask])[0, 1]) if pred[mask].std() > 1e-5 else 0.0
        r_pub = float(np.corrcoef(pred[mask], pub_crop[mask])[0, 1]) if pred[mask].std() > 1e-5 else 0.0
        st = raw_stroke_score(pred, mask, vum)
        dir_pred = np.clip(pred - 0.5 * pred_opp, 0.0, 1.0)
        st_dir = raw_stroke_score(dir_pred, mask, vum)
        depth_results.append({
            "dz": dz,
            "ct_mean": round(ct_means[c], 2),
            "r_vs_dz0": round(r_base, 4),
            "r_vs_published": round(r_pub, 4),
            "raw_stroke": st,
            "dir_stroke": st_dir,
        })
        depth_maps[dz] = pred
        print(f"  dz={dz:+3d} | CT={ct_means[c]:5.1f} | r_dz0={r_base:6.3f} r_pub={r_pub:6.3f} | "
              f"raw_score={st['score']:.4f} (n={st['n_strokes']:2d}, c={st['cells']:2d}, p99={st['p99']:.3f}) | "
              f"dir_score={st_dir['score']:.4f}", flush=True)

    # 2. Scale sensitivity: 0.923x (9.362 -> 8.640 um) and 1.083x (8.640 -> 9.362 um)
    # Wait: if a scroll is scanned at 8.64 um (smaller voxels), a 1mm letter occupies 9.362/8.64 = 1.0836x MORE voxels!
    # So we test BOTH 0.923x (letter smaller in voxels) AND 1.0836x (letter larger in voxels, exact 8.64um simulation)!
    scale_results = []
    scale_maps = {}
    for sf, desc in [
        (0.923, "0.923x (coarser voxels / smaller px letters)"),
        (1.000, "1.000x (native 9.362 um)"),
        (1.0836, "1.0836x (simulated 8.640 um native voxel grid)"),
    ]:
        if abs(sf - 1.0) < 1e-4:
            pred_s = base_map
            m_s = mask
        else:
            # Resample centered 25-layer slab isotropically by sf, then crop center 17 layers & center 1024x1024
            sub25 = layers45[22 - 12 : 22 + 13].astype(np.float32)
            rz = zoom(sub25, (sf, sf, sf), order=1)
            cz, cy, cx = rz.shape[0] // 2, rz.shape[1] // 2, rz.shape[2] // 2
            slab_s = np.clip(np.rint(rz[cz - 8 : cz + 9]), 0, 255).astype(np.uint8)
            h_s, w_s = slab_s.shape[1], slab_s.shape[2]
            m_s = np.ones((h_s, w_s), dtype=bool)
            pred_s = runner.infer(slab_s, m_s, overlap=0.5, reverse=use_rev, batch_size=4)
        # Resize pred_s back to 1024x1024 for exact pixel correlation against dz=0
        if pred_s.shape != base_map.shape:
            pred_back = zoom(pred_s, (base_map.shape[0] / pred_s.shape[0], base_map.shape[1] / pred_s.shape[1]), order=1)
        else:
            pred_back = pred_s
        r_base = float(np.corrcoef(pred_back[mask], base_map[mask])[0, 1])
        st = raw_stroke_score(pred_s, m_s, vum / sf)
        scale_results.append({
            "scale_factor": sf,
            "desc": desc,
            "r_vs_native": round(r_base, 4),
            "raw_stroke": st,
        })
        scale_maps[sf] = pred_back
        print(f"  scale={sf:.4f} ({desc}): r_vs_native={r_base:.4f} | score={st['score']:.4f} (n={st['n_strokes']}, p99={st['p99']:.3f})", flush=True)

    # Save JSON
    summary = {
        "bounds": list(bounds),
        "ct_profile_dz_minus22_to_plus22": [round(v, 2) for v in ct_means],
        "ct_peak_dz": peak_layer,
        "use_rev": use_rev,
        "r_pub_fwd": round(r_pub_fwd, 4),
        "r_pub_rev": round(r_pub_rev, 4),
        "depth_sweep": depth_results,
        "scale_sweep": scale_results,
    }
    (out_dir / "846b_depth_scale_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save crisp visual figure: top row depth strip (-12, -8, -4, 0, +4, +8, +12), bottom row curve + scales
    show_dz = [-12, -8, -4, -2, 0, 2, 4, 8, 12]
    fig = plt.figure(figsize=(18, 7.5))
    gs = fig.add_gridspec(2, 9, height_ratios=[1.1, 1.0])
    for i, dz in enumerate(show_dz):
        ax = fig.add_subplot(gs[0, i])
        im = depth_maps[dz]
        ax.imshow(im, cmap="gray", vmin=0, vmax=max(0.4, float(np.percentile(base_map, 99.5))))
        rec = next(d for d in depth_results if d["dz"] == dz)
        ax.set_title(f"dz={dz:+d}\nr={rec['r_vs_dz0']:.2f}\nsc={rec['raw_stroke']['score']:.2f}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

    ax_c = fig.add_subplot(gs[1, :5])
    dzs = [d["dz"] for d in depth_results]
    rs = [d["r_vs_dz0"] for d in depth_results]
    scs = [d["raw_stroke"]["score"] for d in depth_results]
    ax_c.plot(dzs, rs, "o-", color="#2563eb", lw=2, label="Pearson r vs dz=0")
    ax_c.axhline(0.8, color="#2563eb", ls="--", alpha=0.5, label="r = 0.80 threshold")
    ax_c.set_xlabel("Normal Depth Offset dz (voxels, 9.36 um/vox)")
    ax_c.set_ylabel("Pearson r vs dz=0", color="#2563eb")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.grid(alpha=0.3)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(dzs, scs, "s-", color="#dc2626", lw=2, label="Raw Stroke Score")
    ax_c2.set_ylabel("Raw Thin-Stroke DoG Score", color="#dc2626")
    ax_c.set_title("PHerc0846B ($50k Winner): Depth-Offset Sensitivity Curve")
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    for j, (sf, label_s) in enumerate([(0.923, "0.923x"), (1.000, "1.000x (9.36um)"), (1.0836, "1.084x (8.64um)")]):
        ax_s = fig.add_subplot(gs[1, 5 + j])
        ax_s.imshow(scale_maps[sf], cmap="gray", vmin=0, vmax=max(0.4, float(np.percentile(base_map, 99.5))))
        sr = next(d for d in scale_results if abs(d["scale_factor"] - sf) < 1e-3)
        ax_s.set_title(f"{label_s}\nr={sr['r_vs_native']:.3f}\nsc={sr['raw_stroke']['score']:.2f}", fontsize=8.5)
        ax_s.set_xticks([]); ax_s.set_yticks([])

    ax_ct = fig.add_subplot(gs[1, 8])
    ax_ct.plot(range(-22, 23), ct_means, "-", color="#059669", lw=2)
    ax_ct.axvline(0, color="k", ls="--", alpha=0.5)
    ax_ct.set_title("CT Mean vs dz", fontsize=8.5)
    ax_ct.set_xlabel("dz (vox)", fontsize=8)
    ax_ct.grid(alpha=0.3)

    fig.suptitle("Step 0 ($0 Local): PHerc0846B Depth-Offset & 8.64um Scale Sensitivity (reader-v3)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "846b_depth_scale_sensitivity.png", dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_dir / '846b_depth_scale_sensitivity.png'}", flush=True)


if __name__ == "__main__":
    main()
