"""Core 3D-stabilized curved streamline & 7-Gate Z-slab winding-constraint tracer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config import GateConfig


@dataclass
class SlabFields:
    slab: np.ndarray        # bool (Nz, H, W)
    center_mask: np.ndarray # bool (H, W) — 5-slice Z-consensus around z_mid
    raw_center: np.ndarray  # bool (H, W)
    nx: np.ndarray          # float32 unit sheet normal X (H, W)
    ny: np.ndarray          # float32 unit sheet normal Y (H, W)
    coherence: np.ndarray   # float32 structure-tensor anisotropy [0, 1] (H, W)
    z_mid_idx: int
    y0: int
    x0: int
    z0: int
    z: int


def build_slab_fields(slab: np.ndarray, y0: int, x0: int, z0: int, z: int) -> SlabFields:
    """Auto-crop empty outer border and compute 3D-stabilized structure-tensor normals."""
    z_mid = z - z0
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


def _sample_normal_aligned(sf: SlabFields, lx: float, ly: float, prev_dx: float, prev_dy: float) -> tuple[float, float, float]:
    H, W = sf.nx.shape
    ix, iy = int(round(lx)), int(round(ly))
    if not (2 <= ix < W - 2 and 2 <= iy < H - 2):
        return 0.0, 0.0, 0.0
    vx, vy = float(sf.nx[iy, ix]), float(sf.ny[iy, ix])
    coh = float(sf.coherence[iy, ix])
    if vx * prev_dx + vy * prev_dy < 0.0:
        vx, vy = -vx, -vy
    return vx, vy, coh


def integrate_normal_streamline(
    sf: SlabFields,
    seed_x: float,
    seed_y: float,
    cfg: GateConfig,
    max_steps: int = 350,
) -> np.ndarray:
    """Integrate a smooth curved streamline perpendicular to papyrus wraps using RK2."""
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
            nx1, ny1, coh1 = _sample_normal_aligned(sf, mx, my, dx, dy)
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
    """Detect contiguous foreground papyrus sheet crossings along a curved polyline."""
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


def verify_step_7gate(
    sf: SlabFields,
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    valley_px: float,
    local_med_spacing: float | None,
    cfg: GateConfig,
) -> tuple[bool, str, float]:
    """
    Apply all 7 precision gates to consecutive crossing pair p0_xy -> p1_xy:
      1. Valley separation
      2. Physical spacing bounds + local spacing-jump skip detector
      3. Structure-tensor normal alignment + coherence
      4. 3D Z-slab distinct connected surfaces (catches hidden Z-mergers)
      5. 3D vertical persistence across Z-frames
      6. Wide perpendicular ribbon span (catches blind-ended torn flaps / delaminations)
      7. Inner-core 3D gap cleanliness (no persistent intruding third sheet)
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

    z_pers0 = float(np.any(c0, axis=(1, 2)).mean())
    z_pers1 = float(np.any(c1, axis=(1, 2)).mean())
    if min(z_pers0, z_pers1) < cfg.min_z_persistence:
        return False, "low_z_persistence", 0.0

    cz_lo, cz_hi = max(0, mid_z - 2), min(nz, mid_z + 3)
    wide_span0 = float(np.any(c0[cz_lo:cz_hi], axis=(0, 1)).mean())
    wide_span1 = float(np.any(c1[cz_lo:cz_hi], axis=(0, 1)).mean())
    if min(wide_span0, wide_span1) < cfg.min_wide_span_frac:
        return False, "delam_or_blind_tip", 0.0

    in_lo = max(0, mid_c - cfg.inner_half_w)
    in_hi = min(len(ws), mid_c + cfg.inner_half_w + 1)
    if idx1 - idx0 > 4:
        other3d = (
            (labeled3d[:, idx0 + 2 : idx1 - 1, in_lo:in_hi] > 0)
            & (~c0[:, idx0 + 2 : idx1 - 1, in_lo:in_hi])
            & (~c1[:, idx0 + 2 : idx1 - 1, in_lo:in_hi])
        )
        if float(np.any(other3d.mean(axis=0) >= 0.30, axis=0).mean()) > cfg.max_intruder_frac:
            return False, "intruding_sheet_3d", 0.0

    conf = 0.30 * min_cos + 0.20 * min_coh + 0.30 * min(wide_span0, wide_span1) + 0.20 * min(z_pers0, z_pers1)
    return True, "ok", conf


def trace_polyline_chains(sf: SlabFields, poly_xy: np.ndarray, ux: float, uy: float, cfg: GateConfig) -> list[dict]:
    """Split a curved streamline into verified, outward-oriented relative-winding chains."""
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
        ok, _, conf = verify_step_7gate(sf, c_pts[i], c_pts[i + 1], valley, local_med, cfg)
        if ok:
            cur_pts.append(c_pts[i + 1])
            cur_confs.append(conf)
        else:
            flush()
            cur_pts = [c_pts[i + 1]]
            cur_confs = []
    flush()
    return chains


def generate_curved_chains(sf: SlabFields, ux: float, uy: float, cfg: GateConfig) -> list[dict]:
    """Grid-seed curved streamlines across a slice and deduplicate overlapping paths."""
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
    """Track verified crossings vertically across the 3D Z-slab to emit free same-winding strokes."""
    H, W = sf.center_mask.shape
    nz = sf.slab.shape[0]
    sample_zs = sorted(set([0, max(0, sf.z_mid_idx - 4), sf.z_mid_idx, min(nz - 1, sf.z_mid_idx + 4), nz - 1]))
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


def measure_spatial_coverage(chains: list[dict], sf: SlabFields, cell_px: int = 128) -> float:
    """Fraction of non-empty scroll 128x128 cells covered by at least one constraint point."""
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
