"""Kill Test 2: Opus 5.5's 1-hour PHerc1451 116 keV teacher-label feasibility test.

Part A (2.4 um positive control):
  Verify our local ink_canonical_2um (ResNet3D-152) on a known PHerc0139 letter crop.

Part B (PHerc1451 2.399 um / 78 keV teacher check):
  Stream 4 clean upright 1024x1024 L0 (~0.25x0.25 cm each, 4x4=16 tiles) mid-radius patches at
  4 well-separated heights (z_L0 ~ 16k, 26k, 36k, 46k) using PHerc1451's precomputed
  20260319101107 surface-m7 L2 volume -> render 96 layers along inward normals from L0 2.399 um CT
  -> run ink_canonical_2um at 3 depth windows (dz=-14, 0, +14) x BOTH layer orders (forward & reversed).

Part C (PHerc1451 8.640 um / 116 keV quick reader-v3 check):
  Also render one 1536x1536 patch from PHerc1451's existing S3 GrowPatch mesh z_dbg_gen_04971 on
  20250521151225-8.640um and run reader-v3 forward & reversed.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import tifffile
import torch
import zarr
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from zarr.storage import FsspecStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from killtest1a_846b_repro import TifXYZSurface, infer_reader_v3, load_reader_v3, read_bbox_padded
from killtest1b_0175a_patch import build_isometric_upright_grid, find_clean_polar_arc, track_sheet_3d
from segment_qa.data import OUT_DIR, SEGMENTS, BoundedCache, LayersVolume, read_crop
from segment_qa.ink_tolerance import END, START, InkModel


def infer_2um_grid(model: InkModel, slab_zyx: np.ndarray, dz: int = 0, reverse: bool = False) -> np.ndarray:
    """Run ink_canonical_2um on (D, H, W) uint8 slab (H, W multiples of 256) at depth offset dz."""
    vol = slab_zyx[::-1] if reverse else slab_zyx
    D, H, W = vol.shape
    mid = D // 2
    z0 = mid - 31 + dz
    z1 = z0 + 62
    assert 0 <= z0 and z1 <= D, f"window [{z0},{z1}) outside depth {D}"
    out = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, 256):
        for x in range(0, W, 256):
            out[y : y + 256, x : x + 256] = model.predict(vol[z0:z1, y : y + 256, x : x + 256])
    return out


def main():
    t_all = time.perf_counter()
    out_dir = ROOT / "outputs" / "killtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs_a = s3fs.S3FileSystem(anon=True, asynchronous=True)
    fs_s = s3fs.S3FileSystem(anon=True)

    # --- PART A: Positive control on PHerc0139 (known crisp letters) ---
    print("=== Part A: ink_canonical_2um positive control on PHerc0139 ===", flush=True)
    ink_model = InkModel()
    cache = BoundedCache()
    crops_meta = json.loads((OUT_DIR / "ink_tolerance_crops.json").read_text(encoding="utf-8"))
    # Pick top-scoring crop
    ctrl = max(crops_meta["crops"], key=lambda c: c.get("score", 0))
    vol0139 = LayersVolume(SEGMENTS["PHerc0139"])
    ctrl_slab = read_crop(vol0139, ctrl["y0"], ctrl["x0"], 256, level=0, cache=cache)
    vol0139.close()
    ctrl_fwd = ink_model.predict(ctrl_slab[START:END])
    ctrl_rev = ink_model.predict(ctrl_slab[::-1][START:END])
    print(f"  PHerc0139 control ({ctrl['y0']},{ctrl['x0']}): fwd p99={np.percentile(ctrl_fwd,99):.3f} fg(>0.5)={(ctrl_fwd>0.5).mean():.3f} | "
          f"rev p99={np.percentile(ctrl_rev,99):.3f} fg={(ctrl_rev>0.5).mean():.3f}", flush=True)

    # --- PART B: PHerc1451 2.399 um teacher check on 4 well-separated mid-radius windows ---
    print("\n=== Part B: PHerc1451 2.399 um ink_canonical_2um check (4 windows) ===", flush=True)
    ct_path = "vesuvius-challenge-open-data/PHerc1451/volumes/20260319101107-2.399um-0.2m-78keV-masked.zarr"
    sf_path = "vesuvius-challenge-open-data/PHerc1451/representations/predictions/surfaces/20260319101107-surface-20260413222639-surface-m7-L2-th0.2.zarr"
    ct_g = zarr.open_group(FsspecStore(fs_a, path=ct_path, read_only=True), mode="r")
    sf_g = zarr.open_group(FsspecStore(fs_a, path=sf_path, read_only=True), mode="r")
    ct_l0, ct_l2 = ct_g["0"], ct_g["2"]
    sf_l2 = sf_g["0"]  # surface-m7-L2 level '0' IS 2.399um level 2! shape (14986, 5203, 5203)
    print(f"  PHerc1451 2.4um L0={ct_l0.shape}, L2={ct_l2.shape}, sf_L2={sf_l2.shape}", flush=True)

    SIZE_L0 = 1024   # 4x4 = 16 tiles of 256px (~2.46 x 2.46 mm) per window
    NZ_L2 = SIZE_L0 // 4  # 256 L2 slices
    LAYERS = 96      # allows dz=-14, 0, +14 with 62-layer window (96/2=48; 48-31-14=3 >=0, 48+31+14=93 <=96)
    offsets = np.arange(LAYERS, dtype=np.float64) - LAYERS // 2

    z_targets_l2 = [4000, 6500, 9000, 11500]  # z_L0 = 16k, 26k, 36k, 46k
    win_results = []
    win_panels = []

    for wi, z_mid_l2 in enumerate(z_targets_l2, 1):
        tw = time.perf_counter()
        print(f"\n  --- Window {wi}/4 at z_L0={z_mid_l2*4} (z_L2={z_mid_l2}) ---", flush=True)
        sf_mid = np.asarray(sf_l2[z_mid_l2, :, :])
        ct_mid = np.asarray(ct_l2[z_mid_l2, :, :])
        yy, xx = np.nonzero(ct_mid > 30)
        cy, cx = float(np.median(yy)), float(np.median(xx))
        # Mid-radius wraps on 5203x5203 L2: r = 350..1100 L2 px (~3.4..10.5 mm from center)
        t_init, r_init, score = find_clean_polar_arc(sf_mid, cy, cx, r_min=350, r_max=1100, arc_px=300)
        z0_l2 = z_mid_l2 - NZ_L2 // 2
        R_l2 = track_sheet_3d(sf_l2, z0_l2, NZ_L2, cy, cx, t_init, r_init)
        xyz, nrm = build_isometric_upright_grid(z0_l2 * 4.0, SIZE_L0, SIZE_L0, cy * 4.0, cx * 4.0, t_init, R_l2, l2_scale=4.0)

        # Render (96, 1024, 1024) from L0 2.399 um CT in 256x256 tiles
        slab = np.zeros((LAYERS, SIZE_L0, SIZE_L0), dtype=np.uint8)
        tiles = [(r, r + 256, c, c + 256) for r in range(0, SIZE_L0, 256) for c in range(0, SIZE_L0, 256)]

        def render_tile(b):
            r0, r1, c0, c1 = b
            p = xyz[r0:r1, c0:c1]
            n = nrm[r0:r1, c0:c1]
            pts = p[None] + offsets[:, None, None, None] * n[None]
            zyx = pts[..., ::-1].reshape(-1, 3)
            lo = np.floor(zyx.min(0)).astype(int) - 1
            hi = np.ceil(zyx.max(0)).astype(int) + 2
            blk = read_bbox_padded(ct_l0, lo, hi)
            s = map_coordinates(blk, (zyx - lo).T, order=1, mode="constant", cval=0.0)
            return b, np.clip(np.rint(s), 0, 255).astype(np.uint8).reshape(LAYERS, 256, 256)

        with ThreadPoolExecutor(max_workers=6) as ex:
            for b, s in ex.map(render_tile, tiles):
                r0, r1, c0, c1 = b
                slab[:, r0:r1, c0:c1] = s

        # Check where peak CT band sits in the 96-layer slab (auto-center if surface-m7 L2 had small radial bias!)
        zprof = slab.mean(axis=(1, 2))
        peak_z = int(np.argmax(ndi.gaussian_filter1d(zprof, 2.0)))
        center_shift = peak_z - (LAYERS // 2)
        print(f"    rendered 96x1024x1024 in {time.perf_counter()-tw:.1f}s | CT peak at z={peak_z} (shift {center_shift:+d} vx from mid=48)", flush=True)

        # Run 3 depth windows centered around peak_z (clamped) x 2 layer orders
        base_dz = int(np.clip(center_shift, -12, 12))
        fwd_list, rev_list = [], []
        for d in (-10, 0, 10):
            dz_use = int(np.clip(base_dz + d, -16, 16))
            fwd_list.append(infer_2um_grid(ink_model, slab, dz=dz_use, reverse=False))
            rev_list.append(infer_2um_grid(ink_model, slab, dz=-dz_use, reverse=True))
        fwd_max = np.max(fwd_list, axis=0)
        rev_max = np.max(rev_list, axis=0)

        rec = {
            "window": wi,
            "z_l0": int(z_mid_l2 * 4),
            "r_l2_range": [round(float(r_init.min()), 1), round(float(r_init.max()), 1)],
            "ct_peak_layer": peak_z,
            "ct_mid_mean_std": [round(float(slab[peak_z].mean()), 1), round(float(slab[peak_z].std()), 1)],
            "fwd_max_p50_p95_p99_max": [round(float(v), 4) for v in np.percentile(fwd_max, [50, 95, 99, 100])],
            "fwd_fg_gt_0.5": round(float((fwd_max > 0.5).mean()), 4),
            "rev_max_p50_p95_p99_max": [round(float(v), 4) for v in np.percentile(rev_max, [50, 95, 99, 100])],
            "rev_fg_gt_0.5": round(float((rev_max > 0.5).mean()), 4),
            "seconds": round(time.perf_counter() - tw, 1),
        }
        print(f"    -> fwd p99={rec['fwd_max_p50_p95_p99_max'][2]:.3f} fg={rec['fwd_fg_gt_0.5']:.3f} | "
              f"rev p99={rec['rev_max_p50_p95_p99_max'][2]:.3f} fg={rec['rev_fg_gt_0.5']:.3f}", flush=True)
        win_results.append(rec)
        win_panels.append((ct_mid, cy, cx, xyz[SIZE_L0 // 2] / 4.0, slab[peak_z], fwd_max, rev_max, rec))

    # Free 2um model before loading reader-v3
    del ink_model
    torch.cuda.empty_cache()

    # --- PART C: Quick 8.64 um reader-v3 check on PHerc1451 z_dbg_gen_04971 ---
    print("\n=== Part C: PHerc1451 8.64 um reader-v3 check on z_dbg_gen_04971 ===", flush=True)
    seg_cache = ROOT / "cache" / "killtest2_pherc1451_04971"
    seg_cache.mkdir(parents=True, exist_ok=True)
    for fn in ("meta.json", "x.tif", "y.tif", "z.tif"):
        dst = seg_cache / fn
        if not dst.exists():
            fs_s.get(f"vesuvius-challenge-open-data/PHerc1451/segments/raw/z_dbg_gen_04971/{fn}", str(dst))
    surf86 = TifXYZSurface(seg_cache)
    ct86 = zarr.open_group(
        FsspecStore(fs_a, path="vesuvius-challenge-open-data/PHerc1451/volumes/20250521151225-8.640um-1.2m-116keV-masked.zarr", read_only=True),
        mode="r",
    )["0"]
    # Pick a clean 1536x1536 valid interior block in 04971
    vg = surf86.valid_grid
    # Find largest valid interior region in grid coords (each grid cell = 20 L0 px)
    gh, gw = vg.shape
    best_r, best_c, best_frac = 0, 0, -1.0
    for r in range(0, max(1, gh - 77), 20):
        for c in range(0, max(1, gw - 77), 40):
            fr = float(vg[r : r + 77, c : c + 77].mean())
            if fr > best_frac:
                best_frac, best_r, best_c = fr, r, c
    R0, C0, S86 = best_r * 20, best_c * 20, 1536
    print(f"  04971 full={surf86.full_resolution_shape}, rendering 1536x1536 at ({R0},{C0}) valid_grid={best_frac:.2f}", flush=True)
    xyz_c, nrm_c, val_c = surf86.tile_geometry((R0, R0 + S86, C0, C0 + S86))
    umb = (ct86.shape[2] / 2.0, ct86.shape[1] / 2.0)
    towards = ((np.asarray(umb) - xyz_c[val_c, :2]) * nrm_c[val_c, :2]).sum(1) > 0
    sign = 1.0 if towards.mean() >= 0.5 else -1.0
    offs17 = np.arange(17, dtype=np.float64) - 8.0
    layers86 = np.zeros((17, S86, S86), dtype=np.uint8)
    mask86 = np.zeros((S86, S86), dtype=bool)
    tiles86 = [(r, min(r + 256, R0 + S86), c, min(c + 256, C0 + S86))
               for r in range(R0, R0 + S86, 256) for c in range(C0, C0 + S86, 256)]

    def render86(b):
        r0, r1, c0, c1 = b
        x_, n_, v_ = surf86.tile_geometry(b)
        if not v_.any():
            return b, None, v_
        pts = x_[v_][None] + sign * offs17[:, None, None] * n_[v_][None]
        zyx = pts[..., ::-1].reshape(-1, 3)
        lo = np.floor(zyx.min(0)).astype(int) - 1
        hi = np.ceil(zyx.max(0)).astype(int) + 2
        blk = read_bbox_padded(ct86, lo, hi)
        s = map_coordinates(blk, (zyx - lo).T, order=1, mode="constant", cval=0.0)
        return b, np.clip(np.rint(s), 0, 255).astype(np.uint8).reshape(17, -1), v_

    with ThreadPoolExecutor(max_workers=6) as ex:
        for b, s, v_ in ex.map(render86, tiles86):
            r0, r1, c0, c1 = b
            lr0, lr1, lc0, lc1 = r0 - R0, r1 - R0, c0 - C0, c1 - C0
            mask86[lr0:lr1, lc0:lc1] = v_
            if s is not None:
                layers86[:, lr0:lr1, lc0:lc1][:, v_] = s

    r3 = load_reader_v3()
    p86_fwd, _ = infer_reader_v3(r3, layers86, mask86, reverse=False)
    p86_rev, _ = infer_reader_v3(r3, layers86, mask86, reverse=True)
    part_c = {
        "segment": "z_dbg_gen_04971",
        "crop_rc_size": [R0, C0, S86],
        "fwd_p50_p95_p99_max": [round(float(v), 4) for v in np.percentile(p86_fwd[mask86], [50, 95, 99, 100])],
        "rev_p50_p95_p99_max": [round(float(v), 4) for v in np.percentile(p86_rev[mask86], [50, 95, 99, 100])],
    }
    print("  PHerc1451 8.64um reader-v3:", json.dumps(part_c), flush=True)

    summary = {
        "positive_control_pherc0139": {
            "y0": ctrl["y0"], "x0": ctrl["x0"],
            "fwd_p99": round(float(np.percentile(ctrl_fwd, 99)), 4),
            "fwd_fg_gt_0.5": round(float((ctrl_fwd > 0.5).mean()), 4),
            "rev_p99": round(float(np.percentile(ctrl_rev, 99)), 4),
            "rev_fg_gt_0.5": round(float((ctrl_rev > 0.5).mean()), 4),
        },
        "pherc1451_2p4um_windows": win_results,
        "pherc1451_8p6um_04971": part_c,
        "total_seconds": round(time.perf_counter() - t_all, 1),
    }
    (out_dir / "killtest2_pherc1451_teacher.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save visual sheet for Part A + Part B (blind inspection)
    fig, axes = plt.subplots(5, 4, figsize=(15, 18.5))
    # Row 0: Positive control PHerc0139
    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.5, f"Positive Control\nPHerc0139 w023\n({ctrl['y0']},{ctrl['x0']})\n256x256 L0",
                    ha="center", va="center", fontsize=11, fontweight="bold")
    axes[0, 1].imshow(ctrl_slab[54], cmap="gray")
    axes[0, 1].set_title("Control 2.4 um CT (z=54)")
    axes[0, 2].imshow(ctrl_fwd, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Control FWD fg={(ctrl_fwd>0.5).mean():.2f}")
    axes[0, 3].imshow(ctrl_rev, cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title(f"Control REV fg={(ctrl_rev>0.5).mean():.2f}")

    for row, (ct_mid, cy, cx, trace_l2, ct_surf, fm, rm, rec) in enumerate(win_panels, 1):
        yl, yh = max(0, int(cy - 1400)), min(ct_mid.shape[0], int(cy + 1400))
        xl, xh = max(0, int(cx - 1400)), min(ct_mid.shape[1], int(cx + 1400))
        axes[row, 0].imshow(ct_mid[yl:yh, xl:xh], cmap="gray")
        axes[row, 0].plot(trace_l2[:, 0] - xl, trace_l2[:, 1] - yl, "r-", lw=2.0)
        axes[row, 0].set_title(f"W{rec['window']} z_L0={rec['z_l0']} CT trace")

        lo_c, hi_c = np.percentile(ct_surf, [2, 98])
        axes[row, 1].imshow(ct_surf, cmap="gray", vmin=lo_c, vmax=hi_c)
        axes[row, 1].set_title(f"2.4 um Surface CT (peak z={rec['ct_peak_layer']})")

        axes[row, 2].imshow(fm, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"FWD max3 p99={rec['fwd_max_p50_p95_p99_max'][2]:.2f} fg={rec['fwd_fg_gt_0.5']:.3f}")

        axes[row, 3].imshow(rm, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title(f"REV max3 p99={rec['rev_max_p50_p95_p99_max'][2]:.2f} fg={rec['rev_fg_gt_0.5']:.3f}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Kill Test 2: PHerc1451 2.399 um ink_canonical_2um Teacher Check (4 Mid-Radius Windows + Control)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "killtest2_pherc1451_teacher.png", dpi=150)
    plt.close(fig)

    # Save Part C 8.64 um visual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lo_c, hi_c = np.percentile(layers86[8][mask86], [2, 98])
    axes[0].imshow(layers86[8], cmap="gray", vmin=lo_c, vmax=hi_c)
    axes[0].set_title("PHerc1451 8.64 um CT (z_dbg_gen_04971)")
    axes[1].imshow(p86_fwd, cmap="gray", vmin=0, vmax=max(0.5, float(np.percentile(p86_fwd[mask86], 99.5))))
    axes[1].set_title(f"reader-v3 FWD p99={part_c['fwd_p50_p95_p99_max'][2]:.3f}")
    axes[2].imshow(p86_rev, cmap="gray", vmin=0, vmax=max(0.5, float(np.percentile(p86_rev[mask86], 99.5))))
    axes[2].set_title(f"reader-v3 REV p99={part_c['rev_p50_p95_p99_max'][2]:.3f}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Kill Test 2C: PHerc1451 8.64 um (116 keV) reader-v3 on z_dbg_gen_04971", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "killtest2_pherc1451_8p6um_reader_v3.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved Kill Test 2 outputs in {time.perf_counter()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
