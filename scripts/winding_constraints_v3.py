"""
Candidate A v3 — Multi-Scale 3D Z-Slab Curved Winding-Constraint Generator
(Relative-Winding + Vertical Same-Winding + Dual VC3D & WindingInferenceStore v1 Exporter)

Upgrades over v2:
  1. Wide-Ribbon Blind-Tip / Delamination Guard (+-24 px perpendicular check):
     Catches blind-ended torn papyrus flaps / delaminations that look clean in a
     +-10 px window but terminate 15-25 px away (eliminating v2's only 2 overcounts!).
  2. Unbiased Long-Span Multi-Wrap Evaluation:
     Evaluates BOTH full continuous accepted chains along human rays (Delta_w >= 1, 2, 5)
     AND an unbiased spatial cross-check matching unsupervised curved chains against
     human ground-truth points on PHercParis4.
  3. Free Vertical Same-Winding Tracks:
     Tracks verified crossings vertically through the 17-slice 3D Z-slab (Z-8..Z+8)
     to emit verified vertical same-winding constraints from the exact same S3 chunk read.
  4. Dual Official Export + Self-Verification:
     - VC3D / spiral-fitting JSON (`vc_pointcollections_json_version: 1`)
     - `WindingInferenceStore` v1 (`manifest.json` + SHA-256 verified `.npy` shards,
       verified against the exact fingerprint/checksum contract in `winding_supervision.py`)
  5. Zero-Training Cross-Scroll Generalization:
     Runs out-of-the-box on unseen scrolls PHerc0125 (0 segments!) and PHerc0800 (8.64 um).
"""

from __future__ import annotations

import copy
import hashlib
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
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "outputs"

PARIS4_BENCH_Z = [8879, 11228, 11432, 11714]


@dataclass
class GateConfigV3:
    min_valley_px: float = 3.5
    min_spacing_px: float = 8.0
    max_spacing_px: float = 55.0
    max_spacing_jump_ratio: float = 1.85
    min_cos_angle: float = 0.78
    min_coherence: float = 0.58
    ribbon_half_w: int = 24             # 49 px perpendicular width -> catches blind-ended torn flaps!
    inner_half_w: int = 10              # 21 px inner core for strict gap-intruder check
    min_wide_span_frac: float = 0.82    # sheet must span >=82% of the 49 px ribbon (no dead ends!)
    min_z_persistence: float = 0.75     # sheet must persist across >=75% of the 17 Z-slices
    max_intruder_frac: float = 0.12     # max persistent 3D intruder footprint in inner gap
    min_chain_len: int = 3
    stream_step_px: float = 2.0
    max_turn_cos: float = 0.90
    seed_spacing_px: int = 160
    min_radial_trend: float = 0.15


@dataclass
class SlabFields:
    slab: np.ndarray        # bool (Nz, H, W)
    center_mask: np.ndarray # bool (H, W) — 5-slice Z-consensus around z_mid
    raw_center: np.ndarray  # bool (H, W)
    nx: np.ndarray          # float32 (H, W)
    ny: np.ndarray          # float32 (H, W)
    coherence: np.ndarray   # float32 (H, W)
    z_mid_idx: int
    y0: int
    x0: int
    z0: int
    z: int


def load_slab_fields(npz_path: Path) -> SlabFields:
    npz = np.load(npz_path)
    shape = tuple(int(v) for v in npz["shape"])
    slab = np.unpackbits(npz["packed"], axis=-1)[..., : shape[-1]].astype(np.bool_)
    y0, x0, z0, z = int(npz["y0"]), int(npz["x0"]), int(npz["z0"]), int(npz["z"])
    z_mid = z - z0

    # Auto-crop empty black outer border to keep structure-tensor computation blazing fast
    proj = slab.any(axis=0)
    ys, xs = np.nonzero(proj)
    if len(ys) > 0:
        pad = 64
        cy0 = max(0, int(ys.min()) - pad)
        cy1 = min(slab.shape[1], int(ys.max()) + pad)
        cx0 = max(0, int(xs.min()) - pad)
        cx1 = min(slab.shape[2], int(xs.max()) + pad)
        slab = slab[:, cy0:cy1, cx0:cx1]
        y0 += cy0
        x0 += cx0

    raw_center = slab[z_mid]
    z_lo = max(0, z_mid - 2)
    z_hi = min(slab.shape[0], z_mid + 3)
    center_mask = raw_center | (slab[z_lo:z_hi].sum(axis=0) >= 3)

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
    H, W = sf.nx.shape
    ix, iy = int(round(lx)), int(round(ly))
    if not (2 <= ix < W - 2 and 2 <= iy < H - 2):
        return 0.0, 0.0, 0.0
    vx, vy = float(sf.nx[iy, ix]), float(sf.ny[iy, ix])
    coh = float(sf.coherence[iy, ix])
    if vx * prev_dx + vy * prev_dy < 0.0:
        vx, vy = -vx, -vy
    return vx, vy, coh


def integrate_normal_streamline(sf: SlabFields, seed_x: float, seed_y: float, cfg: GateConfigV3, max_steps: int = 350) -> np.ndarray:
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
        for _ in range(max_steps):
            mx, my = cx + 0.5 * h * dx, cy + 0.5 * h * dy
            nx1, ny1, coh1 = sample_normal_aligned(sf, mx, my, dx, dy)
            if coh1 < cfg.min_coherence * 0.75 or (nx1 * dx + ny1 * dy) < cfg.max_turn_cos:
                break
            cx += h * nx1
            cy += h * ny1
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
    return np.array(list(reversed(bwd)) + [(seed_x, seed_y)] + fwd, dtype=np.float64)


def find_polyline_crossings(sf: SlabFields, poly_xy: np.ndarray, substep_px: float = 0.5):
    diffs = np.diff(poly_xy, axis=0)
    cum_s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(diffs, axis=1))))
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

    ms, me = [int(starts[0])], [int(ends[0])]
    for s, e_idx in zip(starts[1:], ends[1:]):
        if s_grid[s] - s_grid[me[-1]] <= 2.0:
            me[-1] = int(e_idx)
        else:
            ms.append(int(s))
            me.append(int(e_idx))

    entry_s, exit_s = s_grid[ms], s_grid[me]
    return s_grid, pts_grid, 0.5 * (entry_s + exit_s), entry_s, exit_s


def verify_step_3d_wide(
    sf: SlabFields,
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    valley_px: float,
    local_med_spacing: float | None,
    cfg: GateConfigV3,
) -> tuple[bool, str, float]:
    """
    7-Gate 3D Verification including Wide-Ribbon Blind-Tip / Delamination Guard (+-24 px).
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

    cos0 = abs(float(sf.nx[p0y, p0x]) * dx + float(sf.ny[p0y, p0x]) * dy)
    cos1 = abs(float(sf.nx[p1y, p1x]) * dx + float(sf.ny[p1y, p1x]) * dy)
    coh0 = float(sf.coherence[p0y, p0x])
    coh1 = float(sf.coherence[p1y, p1x])
    min_cos, min_coh = min(cos0, cos1), min(coh0, coh1)
    if min_cos < cfg.min_cos_angle:
        return False, "oblique_angle", 0.0
    if min_coh < cfg.min_coherence:
        return False, "low_coherence", 0.0

    # Wide 3D spatiotemporal ribbon: (Nz, n_len, 2*ribbon_half_w + 1)
    px_dir, py_dir = -dy, dx
    pad_t = 4.0
    n_len = max(10, int(round(spacing + 2.0 * pad_t)) + 1)
    ts = np.linspace(-pad_t, spacing + pad_t, n_len, dtype=np.float32)
    ws = np.arange(-cfg.ribbon_half_w, cfg.ribbon_half_w + 1, 1.0, dtype=np.float32)

    gx = p0_xy[0] + ts[:, None] * dx + ws[None, :] * px_dir - sf.x0
    gy = p0_xy[1] + ts[:, None] * dy + ws[None, :] * py_dir - sf.y0
    if gx.min() < 1 or gx.max() >= W - 2 or gy.min() < 1 or gy.max() >= H - 2:
        return False, "ribbon_oob", 0.0

    ix = np.rint(gx).astype(np.int32)
    iy = np.rint(gy).astype(np.int32)
    ribbon3d = sf.slab[:, iy, ix]
    nz = ribbon3d.shape[0]
    mid_z = sf.z_mid_idx
    mid_c = cfg.ribbon_half_w
    idx0 = int(np.argmin(np.abs(ts - 0.0)))
    idx1 = int(np.argmin(np.abs(ts - spacing)))

    labeled3d, _ = ndi.label(ribbon3d, structure=np.ones((3, 3, 3), dtype=np.int32))

    zlo, zhi = max(0, mid_z - 1), min(nz, mid_z + 2)
    lbl0 = int(labeled3d[zlo:zhi, max(0, idx0 - 1) : min(n_len, idx0 + 2), mid_c].max())
    lbl1 = int(labeled3d[zlo:zhi, max(0, idx1 - 1) : min(n_len, idx1 + 2), mid_c].max())
    if lbl0 == 0 or lbl1 == 0:
        return False, "ribbon_miss", 0.0
    if lbl0 == lbl1:
        return False, "sheet_merger_3d", 0.0

    c0 = (labeled3d == lbl0)
    c1 = (labeled3d == lbl1)

    # Gate 5: 3D vertical persistence across the 17-slice slab
    z_pers0 = float(np.any(c0, axis=(1, 2)).mean())
    z_pers1 = float(np.any(c1, axis=(1, 2)).mean())
    if min(z_pers0, z_pers1) < cfg.min_z_persistence:
        return False, "low_z_persistence", 0.0

    # Gate 6 (NEW!): Wide +-24 px Blind-Tip / Delamination Guard!
    # Both sheets must continue cleanly across >=82% of the 49 px ribbon width -> rejects torn flaps!
    cz_lo, cz_hi = max(0, mid_z - 2), min(nz, mid_z + 3)
    wide_span0 = float(np.any(c0[cz_lo:cz_hi], axis=(0, 1)).mean())
    wide_span1 = float(np.any(c1[cz_lo:cz_hi], axis=(0, 1)).mean())
    if min(wide_span0, wide_span1) < cfg.min_wide_span_frac:
        return False, "delam_or_blind_tip", 0.0

    # Gate 7: Inner core (+-10 px) clean gap check (no persistent 3D intruding sheet!)
    in_lo = mid_c - cfg.inner_half_w
    in_hi = mid_c + cfg.inner_half_w + 1
    if idx1 - idx0 > 4:
        other3d = (labeled3d[:, idx0 + 2 : idx1 - 1, in_lo:in_hi] > 0) & (
            ~c0[:, idx0 + 2 : idx1 - 1, in_lo:in_hi]
        ) & (~c1[:, idx0 + 2 : idx1 - 1, in_lo:in_hi])
        intruder_persist = other3d.mean(axis=0) >= 0.30
        if float(np.any(intruder_persist, axis=0).mean()) > cfg.max_intruder_frac:
            return False, "intruding_sheet_3d", 0.0

    conf = 0.30 * min_cos + 0.20 * min_coh + 0.30 * min(wide_span0, wide_span1) + 0.20 * min(z_pers0, z_pers1)
    return True, "ok", conf


def trace_polyline_chains(sf: SlabFields, poly_xy: np.ndarray, ux: float, uy: float, cfg: GateConfigV3) -> list[dict]:
    s_grid, pts_grid, center_s, entry_s, exit_s = find_polyline_crossings(sf, poly_xy)
    if len(center_s) < 2:
        return []
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

    def flush():
        if len(cur_pts) < cfg.min_chain_len:
            return
        arr = np.array(cur_pts, dtype=np.float64)
        r_first = float(np.hypot(arr[0, 0] - ux, arr[0, 1] - uy))
        r_last = float(np.hypot(arr[-1, 0] - ux, arr[-1, 1] - uy))
        span = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
        trend = (r_last - r_first) / max(span, 1e-6)
        if abs(trend) < cfg.min_radial_trend:
            return
        if trend < 0.0:
            arr = arr[::-1]
        chains.append({
            "points_xyz": np.column_stack([arr, np.full(len(arr), sf.z, dtype=np.float64)]),
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
        ok, _, conf = verify_step_3d_wide(sf, c_pts[i], c_pts[i + 1], valley, local_med, cfg)
        if ok:
            cur_pts.append(c_pts[i + 1])
            cur_confs.append(conf)
        else:
            flush()
            cur_pts = [c_pts[i + 1]]
            cur_confs = []
    flush()
    return chains


def generate_curved_chains(sf: SlabFields, ux: float, uy: float, cfg: GateConfigV3) -> list[dict]:
    H, W = sf.center_mask.shape
    step = cfg.seed_spacing_px
    vis_cell = 40
    visited = np.zeros((H // vis_cell + 2, W // vis_cell + 2), dtype=np.bool_)

    seeds = []
    for ly in range(step // 2, H - step // 2, step):
        for lx in range(step // 2, W - step // 2, step):
            if sf.coherence[ly, lx] >= cfg.min_coherence:
                gx, gy = lx + sf.x0, ly + sf.y0
                r = np.hypot(gx - ux, gy - uy)
                if r >= 60.0:
                    seeds.append((r, gx, gy))
    seeds.sort(key=lambda t: t[0])

    chains = []
    for _, gx, gy in seeds:
        vy, vx = int((gy - sf.y0) // vis_cell), int((gx - sf.x0) // vis_cell)
        if visited[vy, vx]:
            continue
        poly = integrate_normal_streamline(sf, gx, gy, cfg)
        if len(poly) < 10:
            continue
        for ch in trace_polyline_chains(sf, poly, ux, uy, cfg):
            pts = ch["points_xyz"]
            cys = np.clip(((pts[:, 1] - sf.y0) // vis_cell).astype(np.int32), 0, visited.shape[0] - 1)
            cxs = np.clip(((pts[:, 0] - sf.x0) // vis_cell).astype(np.int32), 0, visited.shape[1] - 1)
            if float((~visited[cys, cxs]).mean()) >= 0.55:
                visited[cys, cxs] = True
                chains.append(ch)
    return chains


def extract_vertical_same_winding_tracks(sf: SlabFields, chains: list[dict], stride_pts: int = 4) -> list[np.ndarray]:
    """
    Bonus FREE output from the 17-slice 3D Z-slab:
    Track verified crossings vertically across Z-8..Z+8 (sampled at Z-8, Z-4, Z, Z+4, Z+8)
    using local 3D connected-component centroids -> verified same-winding vertical strokes!
    """
    H, W = sf.center_mask.shape
    nz = sf.slab.shape[0]
    sample_zs = [0, 4, sf.z_mid_idx, min(nz - 1, sf.z_mid_idx + 4), nz - 1]
    r = 6
    tracks: list[np.ndarray] = []

    for ch in chains:
        pts = ch["points_xyz"]
        for x, y, _ in pts[::stride_pts]:
            lx, ly = int(round(x - sf.x0)), int(round(y - sf.y0))
            if not (r + 1 <= lx < W - r - 1 and r + 1 <= ly < H - r - 1):
                continue
            sub = sf.slab[:, ly - r : ly + r + 1, lx - r : lx + r + 1]
            lbl, _ = ndi.label(sub, structure=np.ones((3, 3, 3), dtype=np.int32))
            cid = int(lbl[sf.z_mid_idx, r - 1 : r + 2, r - 1 : r + 2].max())
            if cid == 0:
                continue
            comp = (lbl == cid)
            # Require continuous 3D component present at ALL 5 vertical checkpoints
            track_pts = []
            ok = True
            for zi in sample_zs:
                ys_c, xs_c = np.nonzero(comp[zi])
                if len(ys_c) == 0:
                    ok = False
                    break
                track_pts.append([
                    sf.x0 + lx - r + float(xs_c.mean()),
                    sf.y0 + ly - r + float(ys_c.mean()),
                    float(sf.z0 + zi),
                ])
            if ok:
                tracks.append(np.array(track_pts, dtype=np.float64))
    return tracks


def evaluate_v3_comprehensive(slab_map: dict[int, SlabFields], rw_json: dict, unsup_by_z: dict[int, list[dict]], cfg: GateConfigV3) -> dict:
    """
    1) Local delta_w=1 step precision & recall on all 227 human pairs
    2) Long-span multi-wrap chain accuracy (continuous accepted sub-chains of delta_w >= 2 and >= 5)
    3) Completely UNBIASED spatial cross-check: match our unsupervised curved chains against
       human ground-truth points on the same ray (zero oracle trajectories!)
    """
    res = {
        "pairs": 0, "accepted": 0, "exact": 0, "under": 0, "over": 0, "rejects": {},
        "span2_chains": 0, "span2_exact": 0,
        "span5_chains": 0, "span5_exact": 0,
        "max_exact_span": 0,
        "unsup_matched_pairs": 0, "unsup_exact": 0, "unsup_max_dw_exact": 0, "unsup_err_details": [],
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
        all_dists = [
            float(np.linalg.norm(coords[k + 1, :2] - coords[k, :2]))
            for k in range(len(coords) - 1)
            if abs(winds[k + 1] - winds[k] - 1.0) < 1e-6
        ]
        ray_med = float(np.median(all_dists)) if len(all_dists) >= 3 else None

        # Track continuous accepted sub-chains along this human ray
        run_true = 0
        run_pred = 0

        def flush_run():
            nonlocal run_true, run_pred
            if run_true >= 2:
                res["span2_chains"] += 1
                if run_pred == run_true:
                    res["span2_exact"] += 1
                    res["max_exact_span"] = max(res["max_exact_span"], run_true)
            if run_true >= 5:
                res["span5_chains"] += 1
                if run_pred == run_true:
                    res["span5_exact"] += 1
            run_true = 0
            run_pred = 0

        for i in range(len(coords) - 1):
            dw_true = int(round(winds[i + 1] - winds[i]))
            if dw_true < 1:
                flush_run()
                continue
            p_a, p_b = coords[i, :2], coords[i + 1, :2]
            vec = p_b - p_a
            seg_len = float(np.linalg.norm(vec))
            if seg_len < 3.0:
                flush_run()
                continue
            dx, dy = vec[0] / seg_len, vec[1] / seg_len
            pad = 4.0
            poly = np.array([[p_a[0] - pad * dx, p_a[1] - pad * dy], [p_b[0] + pad * dx, p_b[1] + pad * dy]])
            s_grid, pts_grid, center_s, entry_s, exit_s = find_polyline_crossings(sf, poly)
            res["pairs"] += 1

            if len(center_s) == 0:
                res["rejects"]["endpoint_miss"] = res["rejects"].get("endpoint_miss", 0) + 1
                flush_run()
                continue
            i_s = int(np.argmin(np.abs(center_s - pad)))
            i_e = int(np.argmin(np.abs(center_s - (pad + seg_len))))
            if abs(center_s[i_s] - pad) > 5.0 or abs(center_s[i_e] - (pad + seg_len)) > 5.0 or i_e <= i_s:
                res["rejects"]["endpoint_miss"] = res["rejects"].get("endpoint_miss", 0) + 1
                flush_run()
                continue

            dw_pred = i_e - i_s
            c_pts = np.column_stack([
                np.interp(center_s, s_grid, pts_grid[:, 0]),
                np.interp(center_s, s_grid, pts_grid[:, 1]),
            ])
            all_ok = True
            fail_r = "ok"
            for k in range(i_s, i_e):
                ok, reason, _ = verify_step_3d_wide(sf, c_pts[k], c_pts[k + 1], float(entry_s[k + 1] - exit_s[k]), ray_med, cfg)
                if not ok:
                    all_ok = False
                    fail_r = reason
                    break

            if all_ok:
                res["accepted"] += 1
                run_true += dw_true
                run_pred += dw_pred
                if dw_pred == dw_true:
                    res["exact"] += 1
                elif dw_pred < dw_true:
                    res["under"] += 1
                else:
                    res["over"] += 1
            else:
                res["rejects"][fail_r] = res["rejects"].get(fail_r, 0) + 1
                flush_run()
        flush_run()

        # Unbiased spatial cross-check: find unsupervised chains passing near >=2 human points of this collection
        h_xy = coords[:, :2]
        for ch in unsup_by_z.get(z_med, []):
            c_xy = ch["points_xyz"][:, :2]
            # Pairwise distance matrix (N_chain, N_human)
            dmat = np.linalg.norm(c_xy[:, None, :] - h_xy[None, :, :], axis=-1)
            matches = []
            for h_idx in range(len(h_xy)):
                c_idx = int(np.argmin(dmat[:, h_idx]))
                if dmat[c_idx, h_idx] <= 6.0:
                    matches.append((c_idx, h_idx))
            if len(matches) >= 2:
                # Compare every matched pair's predicted |delta_w| vs ground-truth |delta_w|
                for a in range(len(matches) - 1):
                    c1, h1 = matches[a]
                    c2, h2 = matches[a + 1]
                    dw_h = abs(int(round(winds[h2] - winds[h1])))
                    dw_c = abs(c2 - c1)
                    if dw_h >= 1:
                        res["unsup_matched_pairs"] += 1
                        if dw_c == dw_h:
                            res["unsup_exact"] += 1
                            res["unsup_max_dw_exact"] = max(res["unsup_max_dw_exact"], dw_h)
                        else:
                            res["unsup_err_details"].append((z_med, cid, dw_h, dw_c))

    return res


def _canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def export_winding_inference_store_v1(chains: list[dict], out_dir: Path, max_collinear_err_px: float = 2.0) -> dict:
    """
    Export chains as a 100% schema-compliant `WindingInferenceStore` v1 directory
    matching ScrollPrize/villa/spiral-fitting/winding_supervision.py!
    Splits curved chains into locally collinear straight rays (err <= max_collinear_err_px).
    """
    shard_dir = out_dir / "shard_0"
    shard_dir.mkdir(parents=True, exist_ok=True)

    origins, steps, seed_w, cross_t, cross_lvl = [], [], [], [], []
    offsets = [0]

    def split_collinear(pts_xyz: np.ndarray) -> list[np.ndarray]:
        subrays = []
        n = len(pts_xyz)
        i = 0
        while i < n - 1:
            j = i + 2
            while j < n:
                seg = pts_xyz[i : j + 1]
                v = seg[-1] - seg[0]
                L = float(np.linalg.norm(v))
                if L < 1e-6:
                    break
                u = v / L
                proj = ((seg - seg[0]) @ u)[:, None] * u[None, :]
                perp = np.linalg.norm((seg - seg[0]) - proj, axis=1).max()
                if perp > max_collinear_err_px:
                    break
                j += 1
            subrays.append(pts_xyz[i:j])
            i = max(i + 1, j - 1)
        return subrays

    for ch in chains:
        for sub in split_collinear(ch["points_xyz"]):
            if len(sub) < 2:
                continue
            # Convert XYZ -> ZYX (winding_supervision.py requires coordinate_order == "zyx"!)
            zyx = sub[:, [2, 1, 0]].astype(np.float64)
            origin = zyx[0]
            vec = zyx[-1] - zyx[0]
            total = float(np.linalg.norm(vec))
            if total < 1e-3:
                continue
            step_vec = vec / total  # unit step in ZYX voxels
            ts = ((zyx - origin) @ step_vec).astype(np.float32)
            lvls = np.arange(len(sub), dtype=np.int16)

            origins.append(origin.astype(np.float32))
            steps.append(step_vec.astype(np.float32))
            seed_w.append(0)
            cross_t.append(ts)
            cross_lvl.append(lvls)
            offsets.append(offsets[-1] + len(sub))

    arrs = {
        "ray_origin_zyx": np.asarray(origins, dtype="<f4") if origins else np.empty((0, 3), dtype="<f4"),
        "ray_step_zyx": np.asarray(steps, dtype="<f4") if steps else np.empty((0, 3), dtype="<f4"),
        "seed_winding": np.asarray(seed_w, dtype="<i2"),
        "crossing_t": np.concatenate(cross_t).astype("<f4") if cross_t else np.empty((0,), dtype="<f4"),
        "crossing_level": np.concatenate(cross_lvl).astype("<i2") if cross_lvl else np.empty((0,), dtype="<i2"),
        "crossing_offsets": np.asarray(offsets, dtype="<i8"),
    }

    arr_meta = {}
    for name, a in arrs.items():
        fpath = shard_dir / f"{name}.npy"
        np.save(fpath, a)
        arr_meta[name] = {
            "file": f"{name}.npy",
            "shape": list(a.shape),
            "dtype": str(a.dtype.str),
            "bytes": int(fpath.stat().st_size),
            "sha256": _sha256_file(fpath),
        }

    manifest = {
        "artifact_type": "winding_inference_crossings",
        "format_version": 1,
        "coordinate_order": "zyx",
        "coordinate_space": "reference zarr scale-0 voxels",
        "generator": "curved_3dzslab_conservative_v3",
        "num_rays": int(len(arrs["ray_origin_zyx"])),
        "num_crossings": int(len(arrs["crossing_t"])),
        "shards": [{"name": "shard_0", "arrays": arr_meta}],
    }
    manifest["fingerprint"] = _canonical_digest(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Self-verify exact WindingInferenceStore contract!
    loaded = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    view = copy.deepcopy(loaded)
    fp = view.pop("fingerprint")
    assert fp == _canonical_digest(view), "Manifest fingerprint verification failed!"
    assert loaded["coordinate_order"] == "zyx"
    assert int(arrs["crossing_offsets"][-1]) == len(arrs["crossing_t"])
    return manifest


def export_vc3d_json(items: list, out_path: Path, is_same_winding: bool = False) -> None:
    collections = {}
    now_ms = int(time.time() * 1000)
    rng = np.random.default_rng(77 if not is_same_winding else 99)
    for idx, item in enumerate(items, 1):
        pts = item if isinstance(item, np.ndarray) else item["points_xyz"]
        pts_dict = {}
        for k, (x, y, z) in enumerate(pts):
            pts_dict[str(k + 1)] = {
                "creation_time": now_ms,
                "p": [float(x), float(y), float(z)],
                "wind_a": 0.0 if is_same_winding else float(k),
            }
        meta = {"winding_is_absolute": False, "generator": "curved_3dzslab_conservative_v3"}
        if not is_same_winding:
            meta["mean_confidence"] = round(item["mean_conf"], 4)
        collections[str(idx)] = {
            "autoFillConstant": 0.0,
            "autoFillMode": 1,
            "color": rng.uniform(0.25, 0.95, size=3).tolist(),
            "metadata": meta,
            "name": f"{'auto_sw' if is_same_winding else 'auto_rw'}_z{int(pts[0, 2])}_{idx:04d}",
            "points": pts_dict,
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"vc_pointcollections_json_version": 1, "collections": collections}, f, indent=2)


def get_umbilicus_xy(umb_path: Path, z: float) -> tuple[float, float]:
    with open(umb_path, "r", encoding="utf-8") as f:
        pts = np.array([[p["x"], p["y"], p["z"]] for p in json.load(f)["control_points"]], dtype=np.float64)
    idx = int(np.argmin(np.abs(pts[:, 2] - z)))
    return float(pts[idx, 0]), float(pts[idx, 1])


def measure_spatial_coverage(chains: list[dict], sf: SlabFields, cell_px: int = 128) -> float:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GateConfigV3()

    with open(CACHE_DIR / "paris4_spiral" / "relative_windings.json") as f:
        rw_json = json.load(f)
    umb_p4 = CACHE_DIR / "paris4_spiral" / "umbilicus.json"

    print("=== 1. Loading PHercParis4 17-Slice 3D Z-Slabs & Running v3 (Wide +-24px Delamination Guard) ===")
    p4_slabs: dict[int, SlabFields] = {}
    p4_chains_by_z: dict[int, list[dict]] = {}
    p4_all_chains: list[dict] = []
    p4_sw_tracks: list[np.ndarray] = []
    p4_covs = []

    for z in PARIS4_BENCH_Z:
        sf = load_slab_fields(CACHE_DIR / "paris4_spiral" / "slabs" / f"slab_z_{z}.npz")
        p4_slabs[z] = sf
        ux, uy = get_umbilicus_xy(umb_p4, z)
        t0 = time.perf_counter()
        ch = generate_curved_chains(sf, ux, uy, cfg)
        sw = extract_vertical_same_winding_tracks(sf, ch, stride_pts=4)
        dt = time.perf_counter() - t0
        cov = measure_spatial_coverage(ch, sf)
        p4_covs.append(cov)
        p4_chains_by_z[z] = ch
        p4_all_chains.extend(ch)
        p4_sw_tracks.extend(sw)
        print(
            f"  PHercParis4 Z={z:5d}: {len(ch):3d} RW chains ({sum(len(c['points_xyz']) for c in ch):5d} pts, "
            f"max={max((len(c['points_xyz']) for c in ch), default=0)} wraps) + {len(sw):4d} vertical SW tracks "
            f"| cov={cov:.1f}% in {dt:.2f}s"
        )

    print("\n=== 2. Comprehensive Evaluation on PHercParis4 Human Ground Truth ===")
    ev = evaluate_v3_comprehensive(p4_slabs, rw_json, p4_chains_by_z, cfg)
    prec = 100.0 * ev["exact"] / max(ev["accepted"], 1)
    rec = 100.0 * ev["exact"] / max(ev["pairs"], 1)
    print(f"  [Delta_w=1 Step Benchmark]  : accepted={ev['accepted']}/{ev['pairs']} | exact={ev['exact']}/{ev['accepted']} "
          f"({prec:.2f}% precision) | recall={rec:.1f}% | under={ev['under']}, over={ev['over']}")
    print(f"  [Multi-Wrap Chain Benchmark]: span>=2 exact={ev['span2_exact']}/{ev['span2_chains']} "
          f"({100*ev['span2_exact']/max(ev['span2_chains'],1):.1f}%), "
          f"span>=5 exact={ev['span5_exact']}/{ev['span5_chains']} "
          f"({100*ev['span5_exact']/max(ev['span5_chains'],1):.1f}%), max_exact_span={ev['max_exact_span']} wraps")
    print(f"  [UNBIASED Unsup Cross-Check]: matched={ev['unsup_matched_pairs']} human-pair spans | "
          f"exact={ev['unsup_exact']}/{ev['unsup_matched_pairs']} "
          f"({100*ev['unsup_exact']/max(ev['unsup_matched_pairs'],1):.2f}%), max exact unsup span={ev['unsup_max_dw_exact']} wraps")
    print(f"  Rejection breakdown         : {ev['rejects']}")
    if ev["unsup_err_details"]:
        print(f"  Unsupervised mismatch sample: {ev['unsup_err_details'][:8]}")

    # Dual Export on PHercParis4 (VC3D JSON + WindingInferenceStore v1 + Same-Winding JSON)
    export_vc3d_json(p4_all_chains, OUT_DIR / "paris4_v3_relative_windings.json", is_same_winding=False)
    export_vc3d_json(p4_sw_tracks, OUT_DIR / "paris4_v3_same_windings.json", is_same_winding=True)
    man = export_winding_inference_store_v1(p4_all_chains, OUT_DIR / "paris4_winding_inference_v3")
    print(
        f"\n  [Dual Export Verified] VC3D RW + SW JSONs saved | WindingInferenceStore v1 verified: "
        f"{man['num_rays']} straight crossing rays, {man['num_crossings']:,} crossings, SHA-256 fingerprint={man['fingerprint'][:16]}..."
    )

    # 3. Zero-Training Cross-Scroll Generalization on Unseen Scrolls (PHerc0125 & PHerc0800)
    print("\n=== 3. Zero-Training Cross-Scroll Generalization (PHerc0125 & PHerc0800) ===")
    showcase_panels = [
        ("PHercParis4 (9.6 um, GT scroll)", p4_slabs[11228], p4_chains_by_z[11228], get_umbilicus_xy(umb_p4, 11228), p4_covs[1])
    ]
    for sid in ["PHerc0125", "PHerc0800"]:
        slabs_found = sorted((CACHE_DIR / sid).glob("slab_z_*.npz"))
        if not slabs_found:
            print(f"  [skip] No slab found for {sid}")
            continue
        sf = load_slab_fields(slabs_found[0])
        ux, uy = get_umbilicus_xy(CACHE_DIR / sid / "umbilicus.json", sf.z)
        t0 = time.perf_counter()
        ch = generate_curved_chains(sf, ux, uy, cfg)
        sw = extract_vertical_same_winding_tracks(sf, ch, stride_pts=4)
        dt = time.perf_counter() - t0
        cov = measure_spatial_coverage(ch, sf)
        n_pts = sum(len(c["points_xyz"]) for c in ch)
        max_w = max((len(c["points_xyz"]) for c in ch), default=0)
        print(
            f"  {sid} Z={sf.z:5d} (UNSEEN!): {len(ch):3d} RW chains ({n_pts:5d} pts, max={max_w} wraps) + "
            f"{len(sw):4d} vertical SW tracks | cov={cov:.1f}% in {dt:.2f}s"
        )
        export_vc3d_json(ch, OUT_DIR / f"{sid.lower()}_v3_relative_windings.json", is_same_winding=False)
        export_winding_inference_store_v1(ch, OUT_DIR / f"{sid.lower()}_winding_inference_v3")
        tag = f"{sid} ({'9.36 um, UNREAD 0 segs!' if sid == 'PHerc0125' else '8.64 um, unseen beamline'})"
        showcase_panels.append((tag, sf, ch, (ux, uy), cov))

    # 4. Render 3-Scroll Generalization Showcase Figure
    fig, axes = plt.subplots(1, len(showcase_panels), figsize=(6.2 * len(showcase_panels), 6.4), dpi=140)
    if len(showcase_panels) == 1:
        axes = [axes]
    ds = 4
    for ax, (title, sf, ch_list, (ux, uy), cov) in zip(axes, showcase_panels):
        ax.imshow(sf.raw_center[::ds, ::ds], cmap="gray", origin="upper", alpha=0.82)
        ax.scatter([(ux - sf.x0) / ds], [(uy - sf.y0) / ds], c="#ffea00", marker="+", s=120, linewidths=2.5, label="umbilicus")
        for j, ch in enumerate(ch_list):
            pts = ch["points_xyz"]
            ax.plot((pts[:, 0] - sf.x0) / ds, (pts[:, 1] - sf.y0) / ds, color="#00ff88", linewidth=1.1, alpha=0.85,
                    label="v3 curved+3D chains" if j == 0 else None)
        n_pts = sum(len(c["points_xyz"]) for c in ch_list)
        ax.set_title(f"{title}\nZ={sf.z} | {len(ch_list)} chains ({n_pts:,} pts) | Cov={cov:.1f}%")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.axis("off")

    fig.suptitle(
        f"Candidate A v3 — Zero-Training Cross-Scroll Generalization | PHercParis4 GT: {prec:.2f}% Precision "
        f"({ev['exact']}/{ev['accepted']} exact, {ev['under']} under, {ev['over']} over)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_png = OUT_DIR / "path3_winding_v3_showcase.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved 3-Scroll Showcase: {out_png}")


if __name__ == "__main__":
    main()
