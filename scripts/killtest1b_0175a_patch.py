"""Kill Test 1B: First clean upright streaming patch + reader-v3 sweep on neglected scroll PHerc0175A (8.64 um / 116 keV).

Finds the cleanest, best-separated mid-radius spiral arc in PHerc0175A's precomputed surface-m7 L2
volume, tracks r(z, theta) smoothly with zero wrap-jumps over 1536 L0 slices (~13.3 mm tall),
resamples to an isometric upright 1536x1536 patch (~1.76 cm^2) with horizontal papyrus fibers,
streams 19 layers along inward normals from S3 8.640 um CT, and runs reader-v3 in BOTH directions
at depth offsets -1, 0, +1.
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
import torch
import zarr
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from zarr.storage import FsspecStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from killtest1a_846b_repro import infer_reader_v3, load_reader_v3, read_bbox_padded


def find_clean_polar_arc(surf_slice: np.ndarray, cy: float, cx: float,
                         r_min: float, r_max: float, n_theta: int = 720,
                         arc_px: int = 384) -> tuple[np.ndarray, np.ndarray, float]:
    """Unwrap one XY slice of surface-m7 into (theta, r), find the cleanest well-separated arc."""
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    radii = np.arange(r_min, r_max, 1.0)
    tt, rr = np.meshgrid(thetas, radii, indexing="ij")  # (T, R)
    yy = cy + rr * np.sin(tt)
    xx = cx + rr * np.cos(tt)
    polar = map_coordinates(surf_slice.astype(np.float32), [yy, xx], order=1, mode="constant", cval=0.0)
    # Smooth slightly along theta to emphasize continuous wraps
    polar_s = ndi.gaussian_filter(polar, sigma=(2.0, 0.8))

    # Pick candidate radii at theta steps with strong surface peak AND wide air gaps on both sides
    best = (-1.0, None, None)
    for t0 in range(0, n_theta, 30):
        # Arc length in L2 px ~ r * dtheta. We want ~arc_px L2 pixels -> dtheta ~ arc_px / r
        col = polar_s[t0]
        # Local maxima along radius
        peaks = np.where((col[1:-1] > col[:-2]) & (col[1:-1] >= col[2:]) & (col[1:-1] > 100))[0] + 1
        for pk in peaks:
            r0 = radii[pk]
            # Check radial isolation (valley depth within +/-8 L2 px)
            lo_win = col[max(0, pk - 8) : max(1, pk - 2)]
            hi_win = col[min(len(col) - 1, pk + 3) : min(len(col), pk + 9)]
            if lo_win.size == 0 or hi_win.size == 0:
                continue
            gap = float(col[pk] - max(lo_win.min(), hi_win.min()))
            if gap < 60:
                continue
            n_steps = int(round((arc_px / max(r0, 50.0)) / (2 * np.pi / n_theta)))
            if n_steps < 20 or n_steps > n_theta // 2:
                continue
            # Track forward from (t0, pk) with tight +/-2 px step
            r_track = np.empty(n_steps, dtype=np.float64)
            strengths = np.empty(n_steps, dtype=np.float64)
            cur = pk
            ok = True
            for k in range(n_steps):
                ti = (t0 + k) % n_theta
                w0, w1 = max(0, cur - 2), min(len(radii), cur + 3)
                sub = polar_s[ti, w0:w1]
                cur = w0 + int(np.argmax(sub))
                val = float(polar_s[ti, cur])
                if val < 50:
                    ok = False
                    break
                r_track[k] = radii[cur]
                strengths[k] = val
            if not ok:
                continue
            score = float(strengths.mean()) + 0.5 * gap - 2.0 * float(np.std(np.diff(r_track)))
            if score > best[0]:
                t_arr = thetas[t0] + np.arange(n_steps) * (2 * np.pi / n_theta)
                best = (score, t_arr, r_track)
    if best[1] is None:
        raise RuntimeError("Could not find a clean continuous spiral arc")
    return best[1], best[2], best[0]


def track_sheet_3d(surf_l2, z0_l2: int, nz_l2: int, cy: float, cx: float,
                   t_init: np.ndarray, r_init: np.ndarray, r_pad: int = 3) -> np.ndarray:
    """Track r(z, k) smoothly up and down from middle slice using local peak-following on surface-m7 L2."""
    # Read tight bounding box of surface-m7 L2 needed for this arc
    y_est = cy + r_init * np.sin(t_init)
    x_est = cx + r_init * np.cos(t_init)
    margin = 45
    y0 = max(0, int(np.floor(y_est.min())) - margin)
    y1 = min(surf_l2.shape[1], int(np.ceil(y_est.max())) + margin)
    x0 = max(0, int(np.floor(x_est.min())) - margin)
    x1 = min(surf_l2.shape[2], int(np.ceil(x_est.max())) + margin)
    print(f"  streaming surface-m7 L2 box z=[{z0_l2},{z0_l2+nz_l2}) y=[{y0},{y1}) x=[{x0},{x1}) ...", flush=True)
    sub = np.asarray(surf_l2[z0_l2 : z0_l2 + nz_l2, y0:y1, x0:x1], dtype=np.float32)

    R = np.zeros((nz_l2, len(t_init)), dtype=np.float64)
    mid = nz_l2 // 2
    R[mid] = ndi.gaussian_filter1d(r_init, 2.0)
    dr_grid = np.linspace(-r_pad, r_pad, 2 * r_pad + 1)

    def step_slice(z_idx: int, prev_r: np.ndarray) -> np.ndarray:
        sl = ndi.gaussian_filter(sub[z_idx], sigma=0.8)
        rr = prev_r[:, None] + dr_grid[None, :]
        yy = (cy + rr * np.sin(t_init[:, None])) - y0
        xx = (cx + rr * np.cos(t_init[:, None])) - x0
        vals = map_coordinates(sl, [yy, xx], order=1, mode="constant", cval=0.0)
        # Prefer small radial shift via mild quadratic penalty
        pen = vals - 4.0 * (dr_grid[None, :] ** 2)
        best_i = np.argmax(pen, axis=1)
        best_v = vals[np.arange(len(prev_r)), best_i]
        new_r = prev_r + np.where(best_v > 35.0, dr_grid[best_i], 0.0)
        return ndi.gaussian_filter1d(new_r, 3.0)

    for zi in range(mid + 1, nz_l2):
        R[zi] = step_slice(zi, R[zi - 1])
    for zi in range(mid - 1, -1, -1):
        R[zi] = step_slice(zi, R[zi + 1])

    # Gentle 2D smoothing along (z, theta) to remove voxel staircase
    return ndi.gaussian_filter(R, sigma=(2.0, 2.0))


def build_isometric_upright_grid(z0_l0: float, nz_l0: int, width_l0: int,
                                 cy_l0: float, cx_l0: float,
                                 t_init: np.ndarray, R_l2: np.ndarray, l2_scale: float = 4.0):
    """Build (nz_l0, width_l0, 3) XYZ coordinates (L0 voxels) and inward unit normals."""
    nz_l2, nt = R_l2.shape
    R_l0 = R_l2 * l2_scale
    # Arc length along middle row to get isometric horizontal sampling
    mid_r = R_l0[nz_l2 // 2]
    x_mid = cx_l0 + mid_r * np.cos(t_init)
    y_mid = cy_l0 + mid_r * np.sin(t_init)
    ds = np.hypot(np.diff(x_mid), np.diff(y_mid))
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    total_s = float(s_cum[-1])
    s_start = max(0.0, 0.5 * (total_s - width_l0))
    s_target = np.linspace(s_start, min(total_s, s_start + width_l0), width_l0)
    k_target = np.interp(s_target, s_cum, np.arange(nt))
    t_target = np.interp(k_target, np.arange(nt), t_init)

    z_rows_l2 = np.linspace(0, nz_l2 - 1, nz_l0)
    zz_g, kk_g = np.meshgrid(z_rows_l2, k_target, indexing="ij")
    R_full = map_coordinates(R_l0, [zz_g, kk_g], order=1, mode="nearest")

    Z_full = z0_l0 + np.arange(nz_l0, dtype=np.float64)[:, None] * np.ones((1, width_l0))
    X_full = cx_l0 + R_full * np.cos(t_target[None, :])
    Y_full = cy_l0 + R_full * np.sin(t_target[None, :])
    xyz = np.stack([X_full, Y_full, Z_full], axis=-1)

    # Smooth tangents & inward normals (towards umbilicus)
    du = np.gradient(xyz, axis=1)
    dv = np.gradient(xyz, axis=0)
    nrm = np.cross(du, dv)
    norm = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 1e-6)
    inward = ((cx_l0 - X_full) * nrm[..., 0] + (cy_l0 - Y_full) * nrm[..., 1]).mean()
    if inward < 0:
        nrm = -nrm
    return xyz, nrm


def main():
    t0 = time.perf_counter()
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    ct_path = "vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr"
    sf_path = "vesuvius-challenge-open-data/PHerc0175A/representations/predictions/surfaces/20250521115057-surface-20260413222639-surface-m7-L0-th0.2.zarr"
    ct_g = zarr.open_group(FsspecStore(fs, path=ct_path, read_only=True), mode="r")
    sf_g = zarr.open_group(FsspecStore(fs, path=sf_path, read_only=True), mode="r")
    ct_l0, ct_l2 = ct_g["0"], ct_g["2"]
    sf_l2 = sf_g["2"]
    print(f"PHerc0175A CT L0={ct_l0.shape}, surface-m7 L2={sf_l2.shape}", flush=True)

    # Mid-scroll slice at L2
    z_mid_l2 = sf_l2.shape[0] // 2  # ~1593 -> z_L0 ~ 6372
    sf_mid = np.asarray(sf_l2[z_mid_l2, :, :])
    ct_mid = np.asarray(ct_l2[z_mid_l2, :, :])
    yy, xx = np.nonzero(ct_mid > 30)
    cy, cx = float(np.median(yy)), float(np.median(xx))
    print(f"  scroll center L2=(y={cy:.1f}, x={cx:.1f}), searching open mid-radius wraps ...", flush=True)

    # Search mid-radius open wraps (r = 160..420 L2 px -> 5.5..14.5 mm from center)
    t_init, r_init, arc_score = find_clean_polar_arc(sf_mid, cy, cx, r_min=160, r_max=420, arc_px=420)
    print(f"  found clean arc: n_theta={len(t_init)}, r=[{r_init.min():.1f},{r_init.max():.1f}] L2 px, score={arc_score:.1f}", flush=True)

    # Track 384 L2 slices = 1536 L0 rows (~13.3 mm tall) x 1536 L0 cols (~13.3 mm wide) = 1.76 cm^2
    NZ_L2 = 384
    SIZE_L0 = 1536
    z0_l2 = z_mid_l2 - NZ_L2 // 2
    R_l2 = track_sheet_3d(sf_l2, z0_l2, NZ_L2, cy, cx, t_init, r_init)
    xyz, nrm = build_isometric_upright_grid(z0_l2 * 4.0, SIZE_L0, SIZE_L0, cy * 4.0, cx * 4.0, t_init, R_l2, l2_scale=4.0)

    # Render 19 layers (offsets -9..+9) from L0 8.640 um CT so we can run depth offsets -1, 0, +1
    LAYERS = 19
    offsets = np.arange(LAYERS, dtype=np.float64) - LAYERS // 2
    layers = np.zeros((LAYERS, SIZE_L0, SIZE_L0), dtype=np.uint8)
    TILE = 256
    tiles = [(r, min(r + TILE, SIZE_L0), c, min(c + TILE, SIZE_L0))
             for r in range(0, SIZE_L0, TILE) for c in range(0, SIZE_L0, TILE)]

    t_ren = time.perf_counter()
    def render_tile(b):
        r0, r1, c0, c1 = b
        p = xyz[r0:r1, c0:c1]  # (h, w, 3)
        n = nrm[r0:r1, c0:c1]
        pts = p[None] + offsets[:, None, None, None] * n[None]  # (LAYERS, h, w, 3)
        zyx = pts[..., ::-1].reshape(-1, 3)
        lo = np.floor(zyx.min(0)).astype(int) - 1
        hi = np.ceil(zyx.max(0)).astype(int) + 2
        blk = read_bbox_padded(ct_l0, lo, hi)
        s = map_coordinates(blk, (zyx - lo).T, order=1, mode="constant", cval=0.0)
        return b, np.clip(np.rint(s), 0, 255).astype(np.uint8).reshape(LAYERS, r1 - r0, c1 - c0)

    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, (b, slab) in enumerate(ex.map(render_tile, tiles), 1):
            r0, r1, c0, c1 = b
            layers[:, r0:r1, c0:c1] = slab
            if i % 12 == 0 or i == len(tiles):
                print(f"  rendered {i}/{len(tiles)} L0 tiles ({time.perf_counter()-t_ren:.1f}s)", flush=True)
    dt_ren = time.perf_counter() - t_ren

    # Quick raw-CT sanity check: does the papyrus sheet sit centered at layer 9 (mid)?
    z_profile = layers.mean(axis=(1, 2))
    print(f"  mean CT across 19 layers (-9..+9): {np.round(z_profile, 1).tolist()}", flush=True)

    # Run reader-v3 in BOTH orientations across depth offsets -1, 0, +1
    model = load_reader_v3()
    fwd_preds, rev_preds = [], []
    t_gpu = time.perf_counter()
    for dz in (-1, 0, 1):
        sub17 = layers[1 + dz : 18 + dz]
        pf, _ = infer_reader_v3(model, sub17, reverse=False)
        pr, _ = infer_reader_v3(model, sub17, reverse=True)
        fwd_preds.append(pf)
        rev_preds.append(pr)
    dt_gpu = time.perf_counter() - t_gpu
    fwd_mean = np.mean(fwd_preds, axis=0)
    rev_mean = np.mean(rev_preds, axis=0)

    out_dir = ROOT / "outputs" / "killtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scroll": "PHerc0175A",
        "volume": "20250521115057-8.640um-1.2m-116keV",
        "patch_shape_l0": [SIZE_L0, SIZE_L0],
        "area_cm2": round((SIZE_L0 * 8.640 / 1e4) ** 2, 3),
        "z_range_l0": [int(z0_l2 * 4), int((z0_l2 + NZ_L2) * 4)],
        "render_seconds": round(dt_ren, 1),
        "gpu_6pass_seconds": round(dt_gpu, 1),
        "mean_ct_z_profile_19": [round(float(v), 1) for v in z_profile],
        "fwd_mean_p50_p95_p99_max": [round(float(v), 4) for v in np.percentile(fwd_mean, [50, 95, 99, 100])],
        "rev_mean_p50_p95_p99_max": [round(float(v), 4) for v in np.percentile(rev_mean, [50, 95, 99, 100])],
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    print("\n=== KILL TEST 1B (PHerc0175A) SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    (out_dir / "killtest1b_0175a_patch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Visual verification figure: CT cross-section trace, surface CT, reader-v3 fwd & rev
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    # Panel 1: Cross-section at mid-Z with traced sheet in red (zoomed around scroll)
    pad = 120
    y_lo, y_hi = max(0, int(cy - 520)), min(ct_mid.shape[0], int(cy + 520))
    x_lo, x_hi = max(0, int(cx - 520)), min(ct_mid.shape[1], int(cx + 520))
    axes[0, 0].imshow(ct_mid[y_lo:y_hi, x_lo:x_hi], cmap="gray")
    mid_xyz = xyz[SIZE_L0 // 2] / 4.0
    axes[0, 0].plot(mid_xyz[:, 0] - x_lo, mid_xyz[:, 1] - y_lo, "r-", lw=2.0)
    axes[0, 0].set_title(f"PHerc0175A CT L2 cross-section (z={z_mid_l2*4}) + traced upright sheet")

    # Panel 2: Rendered surface CT mid-layer (fiber check!)
    mid_ct = layers[LAYERS // 2]
    lo_c, hi_c = np.percentile(mid_ct, [2, 98])
    axes[0, 1].imshow(mid_ct, cmap="gray", vmin=lo_c, vmax=hi_c)
    axes[0, 1].set_title(f"Rendered 8.64 um Surface CT ({summary['area_cm2']:.2f} cm2, upright)")

    # Panel 3: reader-v3 forward (mean of dz=-1,0,+1)
    axes[1, 0].imshow(fwd_mean, cmap="gray", vmin=0, vmax=max(0.5, float(np.percentile(fwd_mean, 99.5))))
    axes[1, 0].set_title(f"reader-v3 FORWARD (dz=-1,0,+1) p99={summary['fwd_mean_p50_p95_p99_max'][2]:.3f} max={summary['fwd_mean_p50_p95_p99_max'][3]:.3f}")

    # Panel 4: reader-v3 reversed (mean of dz=-1,0,+1)
    axes[1, 1].imshow(rev_mean, cmap="gray", vmin=0, vmax=max(0.5, float(np.percentile(rev_mean, 99.5))))
    axes[1, 1].set_title(f"reader-v3 REVERSED (dz=-1,0,+1) p99={summary['rev_mean_p50_p95_p99_max'][2]:.3f} max={summary['rev_mean_p50_p95_p99_max'][3]:.3f}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Kill Test 1B: PHerc0175A (116 keV / 8.64 um) — First Upright Streaming Patch + reader-v3", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "killtest1b_0175a_patch.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'killtest1b_0175a_patch.png'}", flush=True)


if __name__ == "__main__":
    main()
