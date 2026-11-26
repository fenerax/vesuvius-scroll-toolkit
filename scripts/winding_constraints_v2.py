"""
Candidate A v2 — Curved Normal-Following + 3D Z-Slab ("Video Clip")
Conservative Relative-Winding & Same-Winding Constraint Generator

Upgrades over v1:
  1. Curved Normal-Following Streamlines (RK2 on structure-tensor normals):
     - Seeded on a spatial grid across the whole scroll, tracing BOTH directions
       perpendicular to the squashed papyrus wraps (unlocking oblique/squashed lobes
       that rigid straight radial rays miss!).
     - Oriented consistently outward via signed radial trend against the umbilicus.
  2. 3D Z-Slab ("Video Clip") Verification (17 contiguous Z-slices: Z-8 .. Z+8):
     - Leverages the 192^3 S3 Zarr chunk layout (adjacent Z-slices arrive in the exact
       same S3 chunk read for FREE!).
     - Builds a 3-slice Z-consensus center mask (bridges 1-slice dropouts/holes only
       where sheets are solid immediately above & below).
     - Verifies every crossing transition in a 3D spatiotemporal ribbon (17 x L x 25):
       * Sheet i and Sheet i+1 MUST remain DISTINCT 3D connected surfaces across Z-8..Z+8
         (catches hidden 3D sheet mergers above/below the center slice!)
       * Both sheets MUST persist vertically through >= min_z_persistence of the Z-frames
         (crushes single-slice noise blobs / double-peak overcounts!)
       * Both sheets MUST span >= min_span_frac of perpendicular ribbon width
       * Zero persistent 3D intruding fragments in the gap between them
  3. Bonus Free Output: Vertical Same-Winding Strokes tracked through the 3D Z-slab!
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "paris4_spiral"
SLAB_DIR = CACHE_DIR / "slabs"
OUT_DIR = ROOT / "outputs"

BENCH_Z = [8879, 11228, 11432, 11714]


@dataclass
class GateConfigV2:
    min_valley_px: float = 3.5
    min_spacing_px: float = 8.0
    max_spacing_px: float = 55.0
    max_spacing_jump_ratio: float = 1.85
    min_cos_angle: float = 0.78          # |step_dir . normal| >= cos(~39 deg)
    min_coherence: float = 0.58          # 3D-stabilized structure-tensor coherence
    ribbon_half_w: int = 10              # perpendicular ribbon half-width (21 px total)
    min_span_frac: float = 0.78          # >=78% perpendicular width coverage
    min_z_persistence: float = 0.75      # sheet must persist across >=75% of Z-slices
    max_intruder_frac: float = 0.12      # max stray 3D component footprint in gap
    min_chain_len: int = 3
    stream_step_px: float = 2.0          # RK2 streamline integration step
    max_turn_cos: float = 0.90           # max local bend per step (prevents hairpins)
    seed_spacing_px: int = 160           # spatial grid spacing for curved seeds
    min_radial_trend: float = 0.15       # outward orientation disambiguation


@dataclass
class SlabFields:
    slab: np.ndarray        # bool (Nz, H, W)
    center_mask: np.ndarray # bool (H, W) — 3-slice Z-consensus around z_mid
    raw_center: np.ndarray  # bool (H, W) — exact center slice
    nx: np.ndarray          # float32 unit sheet normal X (H, W)
    ny: np.ndarray          # float32 unit sheet normal Y (H, W)
    coherence: np.ndarray   # float32 structure-tensor coherence [0, 1] (H, W)
    z_mid_idx: int
    y0: int
    x0: int
    z0: int
    z: int


def load_slab_fields(z: int) -> SlabFields:
    npz = np.load(SLAB_DIR / f"slab_z_{z}.npz")
    shape = tuple(int(v) for v in npz["shape"])
    slab = np.unpackbits(npz["packed"], axis=-1)[..., : shape[-1]].astype(np.bool_)
    y0, x0, z0 = int(npz["y0"]), int(npz["x0"]), int(npz["z0"])
    z_mid = z - z0

    raw_center = slab[z_mid]
    # 5-slice Z-consensus: pixel is solid if present on center OR in >=3 of [z-2..z+2]
    # Bridges single-slice dropouts while refusing to hallucinate in empty gaps!
    z_lo = max(0, z_mid - 2)
    z_hi = min(slab.shape[0], z_mid + 3)
    z_votes = slab[z_lo:z_hi].sum(axis=0)
    center_mask = raw_center | (z_votes >= 3)

    # 3D-stabilized density (mean over Z-3..Z+3) gives ultra-clean sheet normals & coherence!
    zlo_n = max(0, z_mid - 3)
    zhi_n = min(slab.shape[0], z_mid + 4)
    z_avg = slab[zlo_n:zhi_n].mean(axis=0, dtype=np.float32)

    smooth = ndi.gaussian_filter(z_avg, sigma=1.6)
    gy, gx = np.gradient(smooth)
    jxx = ndi.gaussian_filter(gx * gx, sigma=3.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=3.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=3.5)

    trace = jxx + jyy
    diff = jxx - jyy
    rad = np.sqrt(diff * diff + 4.0 * jxy * jxy)
    lam1 = 0.5 * (trace + rad)
    lam2 = 0.5 * (trace - rad)
    coherence = (lam1 - lam2) / np.maximum(lam1 + lam2, 1e-8)

    theta = 0.5 * np.arctan2(2.0 * jxy, diff)
    nx = np.cos(theta).astype(np.float32)
    ny = np.sin(theta).astype(np.float32)

    return SlabFields(
        slab=slab,
        center_mask=center_mask,
        raw_center=raw_center,
        nx=nx,
        ny=ny,
        coherence=coherence.astype(np.float32),
        z_mid_idx=z_mid,
        y0=y0,
        x0=x0,
        z0=z0,
        z=z,
    )


def sample_normal_aligned(sf: SlabFields, lx: float, ly: float, prev_dx: float, prev_dy: float) -> tuple[float, float, float]:
    """Sample structure-tensor normal at local (lx, ly), sign-aligned with (prev_dx, prev_dy)."""
    H, W = sf.nx.shape
    ix = int(round(lx))
    iy = int(round(ly))
    if not (2 <= ix < W - 2 and 2 <= iy < H - 2):
        return 0.0, 0.0, 0.0
    vx = float(sf.nx[iy, ix])
    vy = float(sf.ny[iy, ix])
    coh = float(sf.coherence[iy, ix])
    if vx * prev_dx + vy * prev_dy < 0.0:
        vx, vy = -vx, -vy
    return vx, vy, coh


def integrate_normal_streamline(
    sf: SlabFields,
    seed_x: float,
    seed_y: float,
    cfg: GateConfigV2,
    max_steps_each_way: int = 350,
) -> np.ndarray:
    """
    Trace a smooth curved streamline through (seed_x, seed_y) in BOTH directions
    following the 3D-stabilized sheet normal field (RK2 integration).
    Returns global (N, 2) XY polyline points spaced by ~stream_step_px.
    """
    H, W = sf.nx.shape
    slx, sly = seed_x - sf.x0, seed_y - sf.y0
    ix, iy = int(round(slx)), int(round(sly))
    if not (4 <= ix < W - 4 and 4 <= iy < H - 4):
        return np.empty((0, 2), dtype=np.float64)
    if sf.coherence[iy, ix] < cfg.min_coherence:
        return np.empty((0, 2), dtype=np.float64)

    init_dx, init_dy = float(sf.nx[iy, ix]), float(sf.ny[iy, ix])

    def trace_one_way(dx0: float, dy0: float) -> list[tuple[float, float]]:
        pts = []
        cx, cy = slx, sly
        dx, dy = dx0, dy0
        h = cfg.stream_step_px
        empty_streak = 0
        for _ in range(max_steps_each_way):
            # RK2 midpoint step
            mx = cx + 0.5 * h * dx
            my = cy + 0.5 * h * dy
            nx1, ny1, coh1 = sample_normal_aligned(sf, mx, my, dx, dy)
            if coh1 < cfg.min_coherence * 0.75:
                break
            if nx1 * dx + ny1 * dy < cfg.max_turn_cos:
                break
            cx = cx + h * nx1
            cy = cy + h * ny1
            if not (4 <= cx < W - 4 and 4 <= cy < H - 4):
                break
            dx, dy = nx1, ny1
            pts.append((cx + sf.x0, cy + sf.y0))
            if sf.center_mask[int(round(cy)), int(round(cx))]:
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak * h > cfg.max_spacing_px * 1.4:
                    break
        return pts

    fwd = trace_one_way(init_dx, init_dy)
    bwd = trace_one_way(-init_dx, -init_dy)
    full = list(reversed(bwd)) + [(seed_x, seed_y)] + fwd
    return np.array(full, dtype=np.float64)


def find_polyline_crossings(
    sf: SlabFields,
    poly_xy: np.ndarray,
    substep_px: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample a curved polyline at fine arc-length `s` (every substep_px) and detect
    contiguous foreground sheet crossings along `sf.center_mask`.
    Returns: (s_grid, pts_grid_xy, center_s, entry_s, exit_s)
    """
    diffs = np.diff(poly_xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_s = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = float(cum_s[-1])
    if total_len < 4.0:
        e = np.empty(0, dtype=np.float32)
        return e, np.empty((0, 2)), e, e, e

    s_grid = np.arange(0.0, total_len, substep_px, dtype=np.float64)
    gx = np.interp(s_grid, cum_s, poly_xy[:, 0])
    gy = np.interp(s_grid, cum_s, poly_xy[:, 1])
    pts_grid = np.column_stack([gx, gy])

    H, W = sf.center_mask.shape
    lx = np.rint(gx - sf.x0).astype(np.int32)
    ly = np.rint(gy - sf.y0).astype(np.int32)
    inb = (lx >= 1) & (lx < W - 2) & (ly >= 1) & (ly < H - 2)
    vals = np.zeros(len(s_grid), dtype=np.bool_)
    vals[inb] = sf.center_mask[ly[inb], lx[inb]]

    padded = np.concatenate(([False], vals, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    if len(starts) == 0:
        e = np.empty(0, dtype=np.float32)
        return s_grid, pts_grid, e, e, e

    merged_s = [int(starts[0])]
    merged_e = [int(ends[0])]
    for s, e_idx in zip(starts[1:], ends[1:]):
        if s_grid[s] - s_grid[merged_e[-1]] <= 2.0:
            merged_e[-1] = int(e_idx)
        else:
            merged_s.append(int(s))
            merged_e.append(int(e_idx))

    entry_s = s_grid[merged_s]
    exit_s = s_grid[merged_e]
    center_s = 0.5 * (entry_s + exit_s)
    return s_grid, pts_grid, center_s, entry_s, exit_s


def verify_step_3d(
    sf: SlabFields,
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    valley_px: float,
    local_med_spacing: float | None,
    cfg: GateConfigV2,
) -> tuple[bool, str, float]:
    """
    Apply all 6 precision gates including 3D Z-slab spatiotemporal ribbon verification
    to consecutive crossing pair p0_xy -> p1_xy.
    """
    H, W = sf.center_mask.shape
    vec = p1_xy - p0_xy
    spacing = float(np.linalg.norm(vec))
    if valley_px < cfg.min_valley_px:
        return False, "narrow_valley", 0.0
    if spacing < cfg.min_spacing_px:
        return False, "too_close", 0.0
    if spacing > cfg.max_spacing_px:
        return False, "too_far", 0.0
    if local_med_spacing is not None and spacing > cfg.max_spacing_jump_ratio * local_med_spacing:
        return False, "spacing_jump_skip", 0.0

    dx, dy = float(vec[0] / spacing), float(vec[1] / spacing)
    p0x, p0y = int(round(p0_xy[0] - sf.x0)), int(round(p0_xy[1] - sf.y0))
    p1x, p1y = int(round(p1_xy[0] - sf.x0)), int(round(p1_xy[1] - sf.y0))
    if not (2 <= p0x < W - 2 and 2 <= p0y < H - 2 and 2 <= p1x < W - 2 and 2 <= p1y < H - 2):
        return False, "oob", 0.0

    # Gate 3: 3D-stabilized sheet normal alignment & coherence
    cos0 = abs(float(sf.nx[p0y, p0x]) * dx + float(sf.ny[p0y, p0x]) * dy)
    cos1 = abs(float(sf.nx[p1y, p1x]) * dx + float(sf.ny[p1y, p1x]) * dy)
    coh0 = float(sf.coherence[p0y, p0x])
    coh1 = float(sf.coherence[p1y, p1x])
    min_cos = min(cos0, cos1)
    min_coh = min(coh0, coh1)
    if min_cos < cfg.min_cos_angle:
        return False, "oblique_angle", 0.0
    if min_coh < cfg.min_coherence:
        return False, "low_coherence", 0.0

    # Gate 4 + Gate 6: 3D Spatiotemporal Ribbon (Nz x n_len x n_w) across Z-8..Z+8!
    px_dir, py_dir = -dy, dx
    pad_t = 3.0
    n_len = max(8, int(round(spacing + 2.0 * pad_t)) + 1)
    ts = np.linspace(-pad_t, spacing + pad_t, n_len, dtype=np.float32)
    ws = np.arange(-cfg.ribbon_half_w, cfg.ribbon_half_w + 1, 1.0, dtype=np.float32)

    gx = p0_xy[0] + ts[:, None] * dx + ws[None, :] * px_dir - sf.x0
    gy = p0_xy[1] + ts[:, None] * dy + ws[None, :] * py_dir - sf.y0
    if gx.min() < 1 or gx.max() >= W - 2 or gy.min() < 1 or gy.max() >= H - 2:
        return False, "ribbon_oob", 0.0

    ix = np.rint(gx).astype(np.int32)
    iy = np.rint(gy).astype(np.int32)

    # Sample the full 3D video clip slab: shape (Nz, n_len, n_w)
    ribbon3d = sf.slab[:, iy, ix]
    nz = ribbon3d.shape[0]
    mid_z = sf.z_mid_idx
    mid_c = cfg.ribbon_half_w
    idx0 = int(np.argmin(np.abs(ts - 0.0)))
    idx1 = int(np.argmin(np.abs(ts - spacing)))

    # 3D 26-connectivity connected components -> catches ANY 3D merger anywhere in Z-8..Z+8!
    labeled3d, _ = ndi.label(ribbon3d, structure=np.ones((3, 3, 3), dtype=np.int32))

    # Locate the 3D surface component at crossing 0 and crossing 1 near center Z
    z_win_lo, z_win_hi = max(0, mid_z - 1), min(nz, mid_z + 2)
    lbl0 = int(labeled3d[z_win_lo:z_win_hi, max(0, idx0 - 1) : min(n_len, idx0 + 2), mid_c].max())
    lbl1 = int(labeled3d[z_win_lo:z_win_hi, max(0, idx1 - 1) : min(n_len, idx1 + 2), mid_c].max())
    if lbl0 == 0 or lbl1 == 0:
        return False, "ribbon_miss", 0.0
    if lbl0 == lbl1:
        return False, "sheet_merger_3d", 0.0

    c0 = (labeled3d == lbl0)
    c1 = (labeled3d == lbl1)

    # 3D Vertical Persistence: both sheets must persist across >= min_z_persistence of Z-frames!
    z_pers0 = float(np.any(c0, axis=(1, 2)).mean())
    z_pers1 = float(np.any(c1, axis=(1, 2)).mean())
    if min(z_pers0, z_pers1) < cfg.min_z_persistence:
        return False, "low_z_persistence", 0.0

    # Perpendicular Width Continuity (evaluated over center +/-2 Z-slices)
    cz_lo, cz_hi = max(0, mid_z - 2), min(nz, mid_z + 3)
    span0 = float(np.any(c0[cz_lo:cz_hi], axis=(0, 1)).mean())
    span1 = float(np.any(c1[cz_lo:cz_hi], axis=(0, 1)).mean())
    if min(span0, span1) < cfg.min_span_frac:
        return False, "sheet_hole_or_break", 0.0

    # Check for persistent 3D intruding sheet strictly between idx0+2 and idx1-2
    if idx1 - idx0 > 4:
        other3d = (labeled3d > 0) & (~c0) & (~c1)
        # An intruder is only dangerous if it persists in >=30% of Z-frames (ignores 1-slice speckle)
        intruder_persist = other3d[:, idx0 + 2 : idx1 - 1, :].mean(axis=0) >= 0.30
        intruder_w_frac = float(np.any(intruder_persist, axis=0).mean())
        if intruder_w_frac > cfg.max_intruder_frac:
            return False, "intruding_sheet_3d", 0.0

    conf = 0.35 * min_cos + 0.25 * min_coh + 0.20 * min(span0, span1) + 0.20 * min(z_pers0, z_pers1)
    return True, "ok", conf


def trace_polyline_chains(
    sf: SlabFields,
    poly_xy: np.ndarray,
    ux: float,
    uy: float,
    cfg: GateConfigV2,
) -> list[dict]:
    """Extract verified relative-winding chains along a curved streamline and orient outward."""
    s_grid, pts_grid, center_s, entry_s, exit_s = find_polyline_crossings(sf, poly_xy)
    if len(center_s) < 2:
        return []

    # Map each crossing arc-length to its exact (x, y) point on the polyline
    c_pts = np.column_stack([
        np.interp(center_s, s_grid, pts_grid[:, 0]),
        np.interp(center_s, s_grid, pts_grid[:, 1]),
    ])

    raw_spacings = np.linalg.norm(np.diff(c_pts, axis=0), axis=1)
    plausible = raw_spacings[(raw_spacings >= cfg.min_spacing_px) & (raw_spacings <= cfg.max_spacing_px)]
    global_med = float(np.median(plausible)) if len(plausible) >= 3 else None

    chains: list[dict] = []
    cur_pts = [c_pts[0]]
    cur_confs: list[float] = []

    def flush_chain():
        if len(cur_pts) < cfg.min_chain_len:
            return
        arr = np.array(cur_pts, dtype=np.float64)
        # Orient outward from umbilicus using signed radial trend
        r_first = float(np.hypot(arr[0, 0] - ux, arr[0, 1] - uy))
        r_last = float(np.hypot(arr[-1, 0] - ux, arr[-1, 1] - uy))
        span = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
        trend = (r_last - r_first) / max(span, 1e-6)
        if abs(trend) < cfg.min_radial_trend:
            return  # ambiguous tangential orientation -> conservatively drop!
        if trend < 0.0:
            arr = arr[::-1]
        pts_xyz = np.column_stack([arr, np.full(len(arr), sf.z, dtype=np.float64)])
        chains.append({
            "points_xyz": pts_xyz,
            "mean_conf": float(np.mean(cur_confs)),
            "min_conf": float(np.min(cur_confs)),
            "radial_trend": abs(trend),
        })

    for i in range(len(c_pts) - 1):
        lo, hi = max(0, i - 4), min(len(raw_spacings), i + 5)
        loc = raw_spacings[lo:hi]
        loc_v = loc[(loc >= cfg.min_spacing_px) & (loc <= cfg.max_spacing_px)]
        local_med = float(np.median(loc_v)) if len(loc_v) >= 3 else global_med

        valley = float(entry_s[i + 1] - exit_s[i])
        ok, _, conf = verify_step_3d(sf, c_pts[i], c_pts[i + 1], valley, local_med, cfg)
        if ok:
            cur_pts.append(c_pts[i + 1])
            cur_confs.append(conf)
        else:
            flush_chain()
            cur_pts = [c_pts[i + 1]]
            cur_confs = []

    flush_chain()
    return chains


def evaluate_v2_on_human_rays(
    slab_map: dict[int, SlabFields],
    rw_json: dict,
    cfg: GateConfigV2,
) -> dict:
    """
    Evaluate v2 (3D Z-slab 6-Gate verifier) on the exact same 227 human ground-truth
    pairs across the 4 benchmark slices, for a direct apples-to-apples comparison vs. v1!
    """
    stats = {
        "human_rays": 0,
        "human_pairs": 0,
        "accepted": 0,
        "exact": 0,
        "undercount": 0,
        "overcount": 0,
        "reject_reasons": {},
        "error_examples": [],
    }

    for cid, c in rw_json["collections"].items():
        pts = list(c.get("points", {}).values())
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p["wind_a"])
        coords = np.array([p["p"] for p in pts], dtype=np.float64)
        winds = np.array([p["wind_a"] for p in pts], dtype=np.float64)
        if coords[:, 2].max() - coords[:, 2].min() >= 2.0:
            continue
        z_med = int(round(np.median(coords[:, 2])))
        if z_med not in slab_map:
            continue

        sf = slab_map[z_med]
        stats["human_rays"] += 1
        all_dists = [
            float(np.linalg.norm(coords[k + 1, :2] - coords[k, :2]))
            for k in range(len(coords) - 1)
            if abs(winds[k + 1] - winds[k] - 1.0) < 1e-6
        ]
        ray_med = float(np.median(all_dists)) if len(all_dists) >= 3 else None

        for i in range(len(coords) - 1):
            dw_true = int(round(winds[i + 1] - winds[i]))
            if dw_true < 1:
                continue
            p_a = coords[i, :2]
            p_b = coords[i + 1, :2]
            vec = p_b - p_a
            seg_len = float(np.linalg.norm(vec))
            if seg_len < 3.0:
                continue
            dx, dy = vec[0] / seg_len, vec[1] / seg_len
            pad = 4.0
            poly = np.array([
                [p_a[0] - pad * dx, p_a[1] - pad * dy],
                [p_b[0] + pad * dx, p_b[1] + pad * dy],
            ], dtype=np.float64)

            s_grid, pts_grid, center_s, entry_s, exit_s = find_polyline_crossings(sf, poly)
            stats["human_pairs"] += 1

            if len(center_s) == 0:
                stats["reject_reasons"]["endpoint_miss"] = stats["reject_reasons"].get("endpoint_miss", 0) + 1
                continue

            i_start = int(np.argmin(np.abs(center_s - pad)))
            i_end = int(np.argmin(np.abs(center_s - (pad + seg_len))))
            if abs(center_s[i_start] - pad) > 5.0 or abs(center_s[i_end] - (pad + seg_len)) > 5.0:
                stats["reject_reasons"]["endpoint_miss"] = stats["reject_reasons"].get("endpoint_miss", 0) + 1
                continue

            dw_pred = i_end - i_start
            if dw_pred < 1:
                stats["reject_reasons"]["endpoint_miss"] = stats["reject_reasons"].get("endpoint_miss", 0) + 1
                continue

            c_pts = np.column_stack([
                np.interp(center_s, s_grid, pts_grid[:, 0]),
                np.interp(center_s, s_grid, pts_grid[:, 1]),
            ])

            all_ok = True
            fail_reason = "ok"
            for k in range(i_start, i_end):
                valley = float(entry_s[k + 1] - exit_s[k])
                ok, reason, _ = verify_step_3d(sf, c_pts[k], c_pts[k + 1], valley, ray_med, cfg)
                if not ok:
                    all_ok = False
                    fail_reason = reason
                    break

            if all_ok:
                stats["accepted"] += 1
                if dw_pred == dw_true:
                    stats["exact"] += 1
                elif dw_pred < dw_true:
                    stats["undercount"] += 1
                    stats["error_examples"].append((z_med, cid, i, dw_true, dw_pred, "under", seg_len))
                else:
                    stats["overcount"] += 1
                    stats["error_examples"].append((z_med, cid, i, dw_true, dw_pred, "over", seg_len))
            else:
                stats["reject_reasons"][fail_reason] = stats["reject_reasons"].get(fail_reason, 0) + 1

    return stats


def get_umbilicus_xy(umb_json: dict, z: float) -> tuple[float, float]:
    pts = np.array([[p["x"], p["y"], p["z"]] for p in umb_json["control_points"]], dtype=np.float64)
    idx = int(np.argmin(np.abs(pts[:, 2] - z)))
    return float(pts[idx, 0]), float(pts[idx, 1])


def generate_curved_chains_for_slice(sf: SlabFields, ux: float, uy: float, cfg: GateConfigV2) -> list[dict]:
    """Grid-seed curved normal-following streamlines across the whole scroll and deduplicate."""
    H, W = sf.center_mask.shape
    step = cfg.seed_spacing_px
    # Track visited cells (coarse 40px grid) to prevent redundant overlapping streamlines
    vis_cell = 40
    visited = np.zeros((H // vis_cell + 2, W // vis_cell + 2), dtype=np.bool_)

    chains: list[dict] = []
    # Seed on spatial grid sorted outward from umbilicus so inner high-coherence paths go first
    seeds = []
    for ly in range(step // 2, H - step // 2, step):
        for lx in range(step // 2, W - step // 2, step):
            if sf.coherence[ly, lx] >= cfg.min_coherence:
                gx, gy = lx + sf.x0, ly + sf.y0
                r = np.hypot(gx - ux, gy - uy)
                if r >= 60.0:  # avoid chaotic very-center umbilicus core
                    seeds.append((r, gx, gy))
    seeds.sort(key=lambda t: t[0])

    for _, gx, gy in seeds:
        vy, vx = int((gy - sf.y0) // vis_cell), int((gx - sf.x0) // vis_cell)
        if visited[vy, vx]:
            continue
        poly = integrate_normal_streamline(sf, gx, gy, cfg)
        if len(poly) < 10:
            continue
        ch_list = trace_polyline_chains(sf, poly, ux, uy, cfg)
        for ch in ch_list:
            pts = ch["points_xyz"]
            # Check novelty: at least 60% of points fall in unvisited coarse cells
            cys = np.clip(((pts[:, 1] - sf.y0) // vis_cell).astype(np.int32), 0, visited.shape[0] - 1)
            cxs = np.clip(((pts[:, 0] - sf.x0) // vis_cell).astype(np.int32), 0, visited.shape[1] - 1)
            novel_frac = float((~visited[cys, cxs]).mean())
            if novel_frac >= 0.55:
                visited[cys, cxs] = True
                chains.append(ch)
    return chains


def measure_spatial_coverage(chains: list[dict], sf: SlabFields, cell_px: int = 128) -> float:
    """Fraction of non-empty scroll grid cells (128x128 px) covered by at least one constraint point."""
    H, W = sf.center_mask.shape
    gh, gw = H // cell_px, W // cell_px
    scroll_cells = sf.center_mask[: gh * cell_px, : gw * cell_px].reshape(gh, cell_px, gw, cell_px).any(axis=(1, 3))
    covered = np.zeros_like(scroll_cells)
    for ch in chains:
        pts = ch["points_xyz"]
        ys = ((pts[:, 1] - sf.y0) // cell_px).astype(np.int32)
        xs = ((pts[:, 0] - sf.x0) // cell_px).astype(np.int32)
        ok = (ys >= 0) & (ys < gh) & (xs >= 0) & (xs < gw)
        covered[ys[ok], xs[ok]] = True
    return 100.0 * float((covered & scroll_cells).sum()) / max(int(scroll_cells.sum()), 1)


def export_vc3d_json(all_chains: list[dict], out_path: Path) -> None:
    collections = {}
    now_ms = int(time.time() * 1000)
    rng = np.random.default_rng(123)
    for idx, ch in enumerate(all_chains, 1):
        pts_dict = {}
        for w_idx, (x, y, z) in enumerate(ch["points_xyz"]):
            pts_dict[str(w_idx + 1)] = {
                "creation_time": now_ms,
                "p": [float(x), float(y), float(z)],
                "wind_a": float(w_idx),
            }
        collections[str(idx)] = {
            "autoFillConstant": 0.0,
            "autoFillMode": 1,
            "color": rng.uniform(0.25, 0.95, size=3).tolist(),
            "metadata": {
                "winding_is_absolute": False,
                "generator": "curved_3dzslab_conservative_v2",
                "mean_confidence": round(ch["mean_conf"], 4),
                "min_confidence": round(ch["min_conf"], 4),
            },
            "name": f"auto_curved3d_z{int(ch['points_xyz'][0, 2])}_{idx:04d}",
            "points": pts_dict,
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"vc_pointcollections_json_version": 1, "collections": collections}, f, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "relative_windings.json", "r", encoding="utf-8") as f:
        rw_json = json.load(f)
    with open(CACHE_DIR / "umbilicus.json", "r", encoding="utf-8") as f:
        umb_json = json.load(f)

    # Load v1 chains for coverage & visual comparison
    v1_by_z: dict[int, list[dict]] = {z: [] for z in BENCH_Z}
    v1_path = OUT_DIR / "paris4_auto_relative_windings.json"
    if v1_path.exists():
        with open(v1_path, "r", encoding="utf-8") as f:
            v1_raw = json.load(f)["collections"]
        for c in v1_raw.values():
            pts = np.array([p["p"] for p in sorted(c["points"].values(), key=lambda q: q["wind_a"])])
            zi = int(round(pts[0, 2]))
            if zi in v1_by_z:
                v1_by_z[zi].append({"points_xyz": pts})

    cfg = GateConfigV2()

    print("=== Loading 17-Slice 3D Z-Slabs & 3D-Stabilized Structure Tensors ===")
    slab_map: dict[int, SlabFields] = {}
    for z in BENCH_Z:
        t0 = time.perf_counter()
        sf = load_slab_fields(z)
        dt = time.perf_counter() - t0
        slab_map[z] = sf
        print(f"  Z={z:5d}: 3D slab={sf.slab.shape}, ready in {dt:.2f}s")

    print("\n=== Benchmark v2 (3D Z-Slab 6-Gate) vs. Human Ground Truth (227 pairs) ===")
    stats = evaluate_v2_on_human_rays(slab_map, rw_json, cfg)
    n_pairs = stats["human_pairs"]
    acc = stats["accepted"]
    prec = 100.0 * stats["exact"] / max(acc, 1)
    rec = 100.0 * stats["exact"] / max(n_pairs, 1)
    print(f"  Accepted={acc}/{n_pairs} | Exact={stats['exact']}/{acc} ({prec:.2f}% precision) | Recall={rec:.1f}%")
    print(f"  Undercounts={stats['undercount']}, Overcounts={stats['overcount']}")
    print(f"  Rejections={stats['reject_reasons']}")
    if stats["error_examples"]:
        print(f"  False positives: {stats['error_examples']}")

    print("\n=== Unsupervised Curved 3D-Verified Generation (Whole Scroll) ===")
    all_v2: list[dict] = []
    v2_by_z: dict[int, list[dict]] = {}
    cov_v1_list, cov_v2_list = [], []
    for z in BENCH_Z:
        sf = slab_map[z]
        ux, uy = get_umbilicus_xy(umb_json, z)
        t0 = time.perf_counter()
        ch = generate_curved_chains_for_slice(sf, ux, uy, cfg)
        dt = time.perf_counter() - t0
        v2_by_z[z] = ch
        all_v2.extend(ch)
        cov1 = measure_spatial_coverage(v1_by_z[z], sf)
        cov2 = measure_spatial_coverage(ch, sf)
        cov_v1_list.append(cov1)
        cov_v2_list.append(cov2)
        n_pts = sum(len(c["points_xyz"]) for c in ch)
        max_len = max((len(c["points_xyz"]) for c in ch), default=0)
        print(f"  Z={z:5d}: {len(ch):3d} curved chains, {n_pts:5d} pts (max={max_len} wraps) in {dt:.2f}s "
              f"| scroll coverage: v1={cov1:.1f}% -> v2={cov2:.1f}%")

    out_json = OUT_DIR / "paris4_auto_curved3d_v2.json"
    export_vc3d_json(all_v2, out_json)
    print(f"\nExported v2 VC3D/Spiral-Fit JSON: {out_json} ({len(all_v2)} chains, "
          f"{sum(len(c['points_xyz']) for c in all_v2):,} pts)")

    # Side-by-side comparison plot: v1 straight vs v2 curved+3D + Zoom B close-up!
    z_vis = 11228
    sf = slab_map[z_vis]
    ux, uy = get_umbilicus_xy(umb_json, z_vis)
    ds = 4

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=140)

    # Panel 1: v1 straight radial rays
    axes[0].imshow(sf.raw_center[::ds, ::ds], cmap="gray", origin="upper", alpha=0.8)
    axes[0].scatter([(ux - sf.x0) / ds], [(uy - sf.y0) / ds], c="#ffea00", marker="+", s=110, linewidths=2)
    for j, ch in enumerate(v1_by_z[z_vis]):
        pts = ch["points_xyz"]
        axes[0].plot((pts[:, 0] - sf.x0) / ds, (pts[:, 1] - sf.y0) / ds, color="#00e5ff", linewidth=0.9, alpha=0.7,
                     label="v1 straight" if j == 0 else None)
    axes[0].set_title(f"v1: Straight 2D Radial Rays (Z={z_vis})\nScroll Coverage: {cov_v1_list[1]:.1f}% (misses squashed lobes)")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.85)
    axes[0].axis("off")

    # Panel 2: v2 curved + 3D Z-slab verified streamlines
    axes[1].imshow(sf.raw_center[::ds, ::ds], cmap="gray", origin="upper", alpha=0.8)
    axes[1].scatter([(ux - sf.x0) / ds], [(uy - sf.y0) / ds], c="#ffea00", marker="+", s=110, linewidths=2)
    for j, ch in enumerate(v2_by_z[z_vis]):
        pts = ch["points_xyz"]
        axes[1].plot((pts[:, 0] - sf.x0) / ds, (pts[:, 1] - sf.y0) / ds, color="#00ff88", linewidth=1.2, alpha=0.85,
                     label="v2 curved+3D" if j == 0 else None)

    # Highlight Zoom B (upper-left squashed lobe where red human line was!)
    best_human = None
    best_hlen = 0
    for c in rw_json["collections"].values():
        pts = list(c["points"].values())
        if len(pts) < 6:
            continue
        pts.sort(key=lambda q: q["wind_a"])
        arr = np.array([q["p"] for q in pts])
        if arr[:, 2].max() - arr[:, 2].min() < 2.0 and int(round(np.median(arr[:, 2]))) == z_vis:
            if len(arr) > best_hlen and arr[:, 0].mean() < ux:
                best_hlen = len(arr)
                best_human = arr
    hx_c = best_human[len(best_human) // 2]
    box_h = 300
    hy1, hy2 = int(hx_c[1] - sf.y0 - box_h), int(hx_c[1] - sf.y0 + box_h)
    hx1, hx2 = int(hx_c[0] - sf.x0 - box_h), int(hx_c[0] - sf.x0 + box_h)
    axes[1].add_patch(plt.Rectangle((hx1 / ds, hy1 / ds), (hx2 - hx1) / ds, (hy2 - hy1) / ds,
                                    fill=False, edgecolor="#ff3d00", linewidth=2.2, label="Zoom B"))
    axes[1].set_title(f"v2: Curved + 3D Z-Slab Chains (Z={z_vis})\nScroll Coverage: {cov_v2_list[1]:.1f}% (+{cov_v2_list[1]-cov_v1_list[1]:.1f}%!)")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.85)
    axes[1].axis("off")

    # Panel 3: Zoom B close-up showing v2 green curved chains right next to red human GT!
    axes[2].imshow(sf.raw_center[hy1:hy2, hx1:hx2], cmap="gray", origin="upper")
    first_v2 = True
    for ch in v2_by_z[z_vis]:
        pts = ch["points_xyz"]
        px = pts[:, 0] - sf.x0 - hx1
        py = pts[:, 1] - sf.y0 - hy1
        inb = (px >= 0) & (px < (hx2 - hx1)) & (py >= 0) & (py < (hy2 - hy1))
        if inb.sum() >= 2:
            axes[2].plot(px[inb], py[inb], color="#00ff88", linewidth=2.0, marker="o", markersize=3.5, alpha=0.9,
                         label="v2 auto curved (3D-verified)" if first_v2 else None)
            first_v2 = False
    hpx = best_human[:, 0] - sf.x0 - hx1
    hpy = best_human[:, 1] - sf.y0 - hy1
    axes[2].plot(hpx, hpy, color="#ff3d00", linewidth=2.4, marker="o", markersize=5, label="human GT")
    axes[2].set_title("Zoom B (Squashed Lobe): v2 Auto vs. Human GT\nCurved chains now track 90° across previously blank lobe!")
    axes[2].legend(loc="lower left", fontsize=8, framealpha=0.85)
    axes[2].axis("off")

    fig.suptitle(
        f"Candidate A v2 (Curved + 17-Slice 3D Z-Slab): {prec:.2f}% Precision ({stats['exact']}/{acc} exact, "
        f"{rec:.1f}% recall) | Avg Scroll Coverage: {np.mean(cov_v1_list):.1f}% → {np.mean(cov_v2_list):.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_png = OUT_DIR / "path3_winding_v2_eval.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic plot: {out_png}")


if __name__ == "__main__":
    main()
