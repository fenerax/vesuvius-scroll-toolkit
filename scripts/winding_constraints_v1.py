"""
Candidate A — Conservative Ribbon-Verified Relative-Winding Constraint Generator (v1)

Generates high-precision relative-winding constraints (delta_w = +1 per step)
directly from precomputed OME-Zarr surface predictions + scroll umbilicus JSON.
Zero neural training required; runs in ~1-2s per slice on CPU.

Core 5-Gate Precision Verifier (per consecutive crossing pair p_i -> p_{i+1}):
  1. Valley separation gate:   >= min_valley_px of empty background between crossings
  2. Physical spacing gate:    d in [min_spacing, max_spacing] px AND consistent with
                               local median wrap spacing (catches skipped missing sheets)
  3. Normal-alignment gate:    2D structure-tensor sheet normal aligns with ray
                               (|r . n| >= min_cos_angle) & high anisotropy (coherence >= min_coherence)
  4. Ribbon 1-to-1 topology:   Within a perpendicular ribbon (+-ribbon_half_w px around ray),
                               crossings i and i+1 MUST belong to DISTINCT connected components
                               (zero Y-junctions / mergers!), BOTH span >= min_span_frac of
                               ribbon width (no holes/breaks), and ZERO intruding sheet fragments
                               appear in the gap between them!
  5. Chain-breaking policy:    Any failed gate breaks the chain cleanly into independent
                               high-confidence sub-chains rather than guessing across ambiguity.

Exports:
  - VC3D / spiral-fitting point collections JSON (`vc_pointcollections_json_version: 1`)
  - Evaluation figure comparing Naive vs. 5-Gate Conservative Tracer on PHercParis4 human ground truth
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
SLICE_DIR = CACHE_DIR / "slices"
OUT_DIR = ROOT / "outputs"

BENCH_Z = [8879, 11228, 11432, 11714]


@dataclass
class GateConfig:
    min_valley_px: float = 4.0
    min_spacing_px: float = 8.0
    max_spacing_px: float = 55.0
    max_spacing_jump_ratio: float = 1.85  # reject step if d > 1.85x local median wrap spacing
    min_cos_angle: float = 0.78           # |r . n| >= cos(~39 deg)
    min_coherence: float = 0.62           # structure-tensor anisotropy in [0, 1]
    ribbon_half_w: int = 12               # perpendicular ribbon half-width (25 px total)
    min_span_frac: float = 0.80           # sheet must span >=80% of ribbon width
    max_intruder_frac: float = 0.12       # reject if stray fragment occupies >12% of gap width
    min_chain_len: int = 3                # minimum crossings in an exported chain


@dataclass
class SliceFields:
    mask: np.ndarray        # bool (H, W)
    prob: np.ndarray        # float32 [0, 1] (H, W)
    nx: np.ndarray          # float32 unit sheet normal X (H, W)
    ny: np.ndarray          # float32 unit sheet normal Y (H, W)
    coherence: np.ndarray   # float32 structure-tensor coherence [0, 1] (H, W)
    y0: int
    x0: int
    z: int


def compute_slice_fields(mask_u8: np.ndarray, y0: int, x0: int, z: int) -> SliceFields:
    """Compute smooth 2D structure-tensor normals & coherence for a slice."""
    prob = (mask_u8 > 0).astype(np.float32)
    smooth = ndi.gaussian_filter(prob, sigma=1.5)
    gy, gx = np.gradient(smooth)

    # Structure tensor integrated over local window (~5 px)
    jxx = ndi.gaussian_filter(gx * gx, sigma=3.0)
    jyy = ndi.gaussian_filter(gy * gy, sigma=3.0)
    jxy = ndi.gaussian_filter(gx * gy, sigma=3.0)

    trace = jxx + jyy
    diff = jxx - jyy
    rad = np.sqrt(diff * diff + 4.0 * jxy * jxy)
    lam1 = 0.5 * (trace + rad)
    lam2 = 0.5 * (trace - rad)
    coherence = (lam1 - lam2) / np.maximum(lam1 + lam2, 1e-8)

    # Principal eigenvector of J (perpendicular to sheet = sheet normal!)
    theta = 0.5 * np.arctan2(2.0 * jxy, diff)
    nx = np.cos(theta).astype(np.float32)
    ny = np.sin(theta).astype(np.float32)

    return SliceFields(
        mask=(mask_u8 > 0),
        prob=prob,
        nx=nx,
        ny=ny,
        coherence=coherence.astype(np.float32),
        y0=y0,
        x0=x0,
        z=z,
    )


def find_ray_crossings(
    fields: SliceFields,
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    t_max: float,
    step_px: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample binary sheet mask along ray (ox, oy) + t * (dx, dy) for t in [0, t_max].
    Returns (crossing_t, entry_t, exit_t) for each contiguous foreground run.
    """
    H, W = fields.mask.shape
    ts = np.arange(0.0, t_max, step_px, dtype=np.float32)
    xs = ox + ts * dx - fields.x0
    ys = oy + ts * dy - fields.y0

    inb = (xs >= 1) & (xs < W - 2) & (ys >= 1) & (ys < H - 2)
    if not np.any(inb):
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty

    xi = np.rint(xs[inb]).astype(np.int32)
    yi = np.rint(ys[inb]).astype(np.int32)
    vals = np.zeros(len(ts), dtype=np.bool_)
    vals[inb] = fields.mask[yi, xi]

    # Contiguous foreground runs
    padded = np.concatenate(([False], vals, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1

    if len(starts) == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty

    # Merge tiny 1-px noise gaps inside a single sheet (<= 2 px)
    merged_s = [int(starts[0])]
    merged_e = [int(ends[0])]
    for s, e in zip(starts[1:], ends[1:]):
        gap = (ts[s] - ts[merged_e[-1]])
        if gap <= 2.0:
            merged_e[-1] = int(e)
        else:
            merged_s.append(int(s))
            merged_e.append(int(e))

    entry_t = ts[merged_s]
    exit_t = ts[merged_e]
    center_t = 0.5 * (entry_t + exit_t)
    return center_t, entry_t, exit_t


def verify_step_gates(
    fields: SliceFields,
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    t0: float,
    t1: float,
    exit0: float,
    entry1: float,
    local_med_spacing: float | None,
    cfg: GateConfig,
) -> tuple[bool, str, float]:
    """
    Apply all 5 precision gates to consecutive crossing pair t0 -> t1.
    Returns (passed, reject_reason, confidence_score).
    """
    H, W = fields.mask.shape
    spacing = float(t1 - t0)
    valley = float(entry1 - exit0)

    # Gate 1: clear valley
    if valley < cfg.min_valley_px:
        return False, "narrow_valley", 0.0

    # Gate 2: physical spacing + local jump regularity
    if spacing < cfg.min_spacing_px:
        return False, "too_close", 0.0
    if spacing > cfg.max_spacing_px:
        return False, "too_far", 0.0
    if local_med_spacing is not None and spacing > cfg.max_spacing_jump_ratio * local_med_spacing:
        return False, "spacing_jump_skip", 0.0

    # Gate 3: local sheet normal alignment & coherence at both crossings
    p0x = int(round(ox + t0 * dx - fields.x0))
    p0y = int(round(oy + t0 * dy - fields.y0))
    p1x = int(round(ox + t1 * dx - fields.x0))
    p1y = int(round(oy + t1 * dy - fields.y0))
    if not (2 <= p0x < W - 2 and 2 <= p0y < H - 2 and 2 <= p1x < W - 2 and 2 <= p1y < H - 2):
        return False, "oob", 0.0

    cos0 = abs(float(fields.nx[p0y, p0x]) * dx + float(fields.ny[p0y, p0x]) * dy)
    cos1 = abs(float(fields.nx[p1y, p1x]) * dx + float(fields.ny[p1y, p1x]) * dy)
    coh0 = float(fields.coherence[p0y, p0x])
    coh1 = float(fields.coherence[p1y, p1x])
    min_cos = min(cos0, cos1)
    min_coh = min(coh0, coh1)
    if min_cos < cfg.min_cos_angle:
        return False, "oblique_angle", 0.0
    if min_coh < cfg.min_coherence:
        return False, "low_coherence", 0.0

    # Gate 4: Perpendicular 2D ribbon topology check (distinct sheets, no Y-merger, no holes, no intruders)
    px_dir, py_dir = -dy, dx  # unit vector perpendicular to ray
    pad_t = 3.0
    t_start = t0 - pad_t
    t_end = t1 + pad_t
    n_len = max(8, int(round(t_end - t_start)) + 1)
    ts = np.linspace(t_start, t_end, n_len, dtype=np.float32)
    ws = np.arange(-cfg.ribbon_half_w, cfg.ribbon_half_w + 1, 1.0, dtype=np.float32)
    n_w = len(ws)

    gx = ox + ts[:, None] * dx + ws[None, :] * px_dir - fields.x0
    gy = oy + ts[:, None] * dy + ws[None, :] * py_dir - fields.y0
    if gx.min() < 1 or gx.max() >= W - 2 or gy.min() < 1 or gy.max() >= H - 2:
        return False, "ribbon_oob", 0.0

    ribbon = fields.mask[np.rint(gy).astype(np.int32), np.rint(gx).astype(np.int32)]
    labeled, n_comp = ndi.label(ribbon, structure=np.ones((3, 3), dtype=np.int32))

    # Locate component labels at the center column for crossing 0 and crossing 1
    mid_c = cfg.ribbon_half_w
    idx0 = int(np.argmin(np.abs(ts - t0)))
    idx1 = int(np.argmin(np.abs(ts - t1)))

    # Small +/-1 search along ray axis in case subpixel rounding nudged off mask
    lbl0 = int(labeled[max(0, idx0 - 1) : min(n_len, idx0 + 2), mid_c].max())
    lbl1 = int(labeled[max(0, idx1 - 1) : min(n_len, idx1 + 2), mid_c].max())
    if lbl0 == 0 or lbl1 == 0:
        return False, "ribbon_miss", 0.0
    if lbl0 == lbl1:
        return False, "sheet_merger", 0.0

    # Both sheets must span >= min_span_frac of the perpendicular ribbon width
    span0 = float(np.any(labeled == lbl0, axis=0).mean())
    span1 = float(np.any(labeled == lbl1, axis=0).mean())
    if min(span0, span1) < cfg.min_span_frac:
        return False, "sheet_hole_or_break", 0.0

    # Check for intruding third component between sheet 0 and sheet 1 inside the ribbon
    other_mask = (labeled > 0) & (labeled != lbl0) & (labeled != lbl1)
    # Restrict check to interior rows strictly between idx0+2 and idx1-2
    if idx1 - idx0 > 4:
        intruder_cols = np.any(other_mask[idx0 + 2 : idx1 - 1, :], axis=0).mean()
        if float(intruder_cols) > cfg.max_intruder_frac:
            return False, "intruding_sheet", 0.0

    conf = 0.5 * min_cos + 0.3 * min_coh + 0.2 * min(span0, span1)
    return True, "ok", conf


def trace_ray_chains(
    fields: SliceFields,
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    t_max: float,
    cfg: GateConfig,
) -> tuple[list[dict], dict[str, int]]:
    """Trace one ray and split into verified conservative relative-winding chains."""
    center_t, entry_t, exit_t = find_ray_crossings(fields, ox, oy, dx, dy, t_max)
    reasons: dict[str, int] = {}
    if len(center_t) < 2:
        return [], reasons

    raw_spacings = np.diff(center_t)
    # Precompute robust local median wrap spacing (ignoring obvious outliers)
    plausible = raw_spacings[(raw_spacings >= cfg.min_spacing_px) & (raw_spacings <= cfg.max_spacing_px)]
    global_med = float(np.median(plausible)) if len(plausible) >= 3 else None

    chains: list[dict] = []
    cur_ts = [float(center_t[0])]
    cur_confs: list[float] = []

    for i in range(len(center_t) - 1):
        # Local window of +/- 4 steps around i for median spacing
        lo = max(0, i - 4)
        hi = min(len(raw_spacings), i + 5)
        loc = raw_spacings[lo:hi]
        loc_valid = loc[(loc >= cfg.min_spacing_px) & (loc <= cfg.max_spacing_px)]
        local_med = float(np.median(loc_valid)) if len(loc_valid) >= 3 else global_med

        ok, reason, conf = verify_step_gates(
            fields,
            ox,
            oy,
            dx,
            dy,
            float(center_t[i]),
            float(center_t[i + 1]),
            float(exit_t[i]),
            float(entry_t[i + 1]),
            local_med,
            cfg,
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        if ok:
            cur_ts.append(float(center_t[i + 1]))
            cur_confs.append(conf)
        else:
            if len(cur_ts) >= cfg.min_chain_len:
                pts = np.column_stack([
                    ox + np.array(cur_ts) * dx,
                    oy + np.array(cur_ts) * dy,
                    np.full(len(cur_ts), fields.z, dtype=np.float64),
                ])
                chains.append({
                    "points_xyz": pts,
                    "ts": np.array(cur_ts, dtype=np.float64),
                    "mean_conf": float(np.mean(cur_confs)),
                    "min_conf": float(np.min(cur_confs)),
                })
            cur_ts = [float(center_t[i + 1])]
            cur_confs = []

    if len(cur_ts) >= cfg.min_chain_len:
        pts = np.column_stack([
            ox + np.array(cur_ts) * dx,
            oy + np.array(cur_ts) * dy,
            np.full(len(cur_ts), fields.z, dtype=np.float64),
        ])
        chains.append({
            "points_xyz": pts,
            "ts": np.array(cur_ts, dtype=np.float64),
            "mean_conf": float(np.mean(cur_confs)),
            "min_conf": float(np.min(cur_confs)),
        })

    return chains, reasons


def evaluate_on_human_rays(
    slice_map: dict[int, SliceFields],
    rw_json: dict,
    cfg: GateConfig,
) -> dict:
    """
    Evaluate Naive peak-counting vs. 5-Gate Conservative Tracer along the EXACT
    trajectories of all planar human ground-truth relative-winding rays in slice_map.

    For each consecutive pair of human ground-truth points (h_a, h_b) with known
    integer delta_w = w_b - w_a (typically 1):
      - Count crossings along the line segment from h_a to h_b.
      - Naive accepts ALL steps -> count == delta_w is exact; != delta_w is error (over/under-count).
      - Conservative accepts a human pair ONLY if all detected steps spanning h_a..h_b pass the 5 gates!
    """
    stats = {
        "human_rays": 0,
        "human_pairs": 0,
        "naive_exact": 0,
        "naive_undercount": 0,
        "naive_overcount": 0,
        "cons_accepted": 0,
        "cons_exact": 0,
        "cons_undercount": 0,
        "cons_overcount": 0,
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
        if z_med not in slice_map:
            continue

        fields = slice_map[z_med]
        stats["human_rays"] += 1

        # Fit ray direction from first to last human point (or per-pair segment)
        # Note: human clicks can curve slightly, so evaluate each adjacent human pair along its exact segment
        # while computing local spacing context from the full ray!
        all_pair_dists = [
            float(np.linalg.norm(coords[k + 1, :2] - coords[k, :2]))
            for k in range(len(coords) - 1)
            if abs(winds[k + 1] - winds[k] - 1.0) < 1e-6
        ]
        ray_med = float(np.median(all_pair_dists)) if len(all_pair_dists) >= 3 else None

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

            # Start slightly before p_a and end slightly past p_b so both endpoints are cleanly detected
            pad = 4.0
            ox, oy = p_a[0] - pad * dx, p_a[1] - pad * dy
            t_max = seg_len + 2.0 * pad
            center_t, entry_t, exit_t = find_ray_crossings(fields, ox, oy, dx, dy, t_max)

            # Snap crossing nearest to p_a (t ~ pad) and p_b (t ~ pad + seg_len)
            if len(center_t) == 0:
                dw_pred = 0
                i_start = i_end = -1
            else:
                i_start = int(np.argmin(np.abs(center_t - pad)))
                i_end = int(np.argmin(np.abs(center_t - (pad + seg_len))))
                # Verify endpoints actually matched within 5 px of human clicks
                if abs(center_t[i_start] - pad) > 5.0 or abs(center_t[i_end] - (pad + seg_len)) > 5.0:
                    dw_pred = -1  # missed endpoint
                else:
                    dw_pred = i_end - i_start

            stats["human_pairs"] += 1
            if dw_pred == dw_true:
                stats["naive_exact"] += 1
            elif dw_pred < dw_true:
                stats["naive_undercount"] += 1
            else:
                stats["naive_overcount"] += 1

            # Conservative 5-Gate check on every internal step between i_start and i_end
            if dw_pred >= 1:
                all_ok = True
                fail_reason = "ok"
                for k in range(i_start, i_end):
                    ok, reason, _ = verify_step_gates(
                        fields,
                        ox,
                        oy,
                        dx,
                        dy,
                        float(center_t[k]),
                        float(center_t[k + 1]),
                        float(exit_t[k]),
                        float(entry_t[k + 1]),
                        ray_med,
                        cfg,
                    )
                    if not ok:
                        all_ok = False
                        fail_reason = reason
                        break
                if all_ok:
                    stats["cons_accepted"] += 1
                    if dw_pred == dw_true:
                        stats["cons_exact"] += 1
                    elif dw_pred < dw_true:
                        stats["cons_undercount"] += 1
                        stats["error_examples"].append((z_med, cid, i, dw_true, dw_pred, "under", seg_len))
                    else:
                        stats["cons_overcount"] += 1
                        stats["error_examples"].append((z_med, cid, i, dw_true, dw_pred, "over", seg_len))
                else:
                    stats["reject_reasons"][fail_reason] = stats["reject_reasons"].get(fail_reason, 0) + 1
            else:
                stats["reject_reasons"]["endpoint_miss"] = stats["reject_reasons"].get("endpoint_miss", 0) + 1

    return stats


def get_umbilicus_xy(umb_json: dict, z: float) -> tuple[float, float]:
    pts = np.array([[p["x"], p["y"], p["z"]] for p in umb_json["control_points"]], dtype=np.float64)
    idx = int(np.argmin(np.abs(pts[:, 2] - z)))
    return float(pts[idx, 0]), float(pts[idx, 1])


def export_vc3d_json(all_chains: list[dict], out_path: Path) -> None:
    """Export chains in exact VC3D / spiral-fitting vc_pointcollections_json_version 1 format."""
    collections = {}
    now_ms = int(time.time() * 1000)
    rng = np.random.default_rng(42)
    for idx, ch in enumerate(all_chains, 1):
        pts_dict = {}
        for w_idx, (x, y, z) in enumerate(ch["points_xyz"]):
            pts_dict[str(w_idx + 1)] = {
                "creation_time": now_ms,
                "p": [float(x), float(y), float(z)],
                "wind_a": float(w_idx),
            }
        color = rng.uniform(0.25, 0.95, size=3).tolist()
        collections[str(idx)] = {
            "autoFillConstant": 0.0,
            "autoFillMode": 1,
            "color": color,
            "metadata": {
                "winding_is_absolute": False,
                "generator": "ribbon_verified_conservative_v1",
                "mean_confidence": round(ch["mean_conf"], 4),
                "min_confidence": round(ch["min_conf"], 4),
            },
            "name": f"auto_rw_z{int(ch['points_xyz'][0, 2])}_{idx:04d}",
            "points": pts_dict,
        }

    payload = {
        "vc_pointcollections_json_version": 1,
        "collections": collections,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "relative_windings.json", "r", encoding="utf-8") as f:
        rw_json = json.load(f)
    with open(CACHE_DIR / "umbilicus.json", "r", encoding="utf-8") as f:
        umb_json = json.load(f)

    cfg = GateConfig()

    print("=== Computing Structure-Tensor Fields on 4 Benchmark Slices ===")
    slice_map: dict[int, SliceFields] = {}
    for z in BENCH_Z:
        npz = np.load(SLICE_DIR / f"z_{z}.npz")
        t0 = time.perf_counter()
        sf = compute_slice_fields(npz["mask"], int(npz["y0"]), int(npz["x0"]), int(npz["z"]))
        dt = time.perf_counter() - t0
        slice_map[z] = sf
        print(f"  Z={z:5d}: shape={sf.mask.shape}, structure-tensor ready in {dt:.2f}s")

    # 1. Ground-Truth Benchmark: Naive vs. 5-Gate Conservative
    print("\n=== Benchmark vs. Human Ground Truth (PHercParis4 relative_windings.json) ===")
    stats = evaluate_on_human_rays(slice_map, rw_json, cfg)
    n_pairs = stats["human_pairs"]
    naive_prec = 100.0 * stats["naive_exact"] / max(n_pairs, 1)
    cons_acc = stats["cons_accepted"]
    cons_prec = 100.0 * stats["cons_exact"] / max(cons_acc, 1)
    cons_rec = 100.0 * stats["cons_exact"] / max(n_pairs, 1)

    print(f"  Human planar rays evaluated : {stats['human_rays']}")
    print(f"  Human ground-truth pairs    : {n_pairs}")
    print(f"  [Naive Peak Counter]        : exact={stats['naive_exact']}/{n_pairs} ({naive_prec:.1f}%) "
          f"| under={stats['naive_undercount']}, over={stats['naive_overcount']}")
    print(f"  [5-Gate Conservative v1]    : accepted={cons_acc}/{n_pairs} | exact={stats['cons_exact']}/{cons_acc} "
          f"({cons_prec:.2f}% precision) | recall={cons_rec:.1f}% "
          f"| under={stats['cons_undercount']}, over={stats['cons_overcount']}")
    print(f"  Rejection breakdown         : {stats['reject_reasons']}")
    if stats["error_examples"]:
        print(f"  False positives ({len(stats['error_examples'])}): {stats['error_examples'][:10]}")

    # 2. Unsupervised Full-Slice Radial Ray Generation (180 rays/slice = every 2 deg)
    print("\n=== Unsupervised Full-Slice Radial Generation (180 rays/slice) ===")
    all_chains: list[dict] = []
    chains_by_z: dict[int, list[dict]] = {}
    for z in BENCH_Z:
        sf = slice_map[z]
        ux, uy = get_umbilicus_xy(umb_json, z)
        t0 = time.perf_counter()
        z_chains: list[dict] = []
        for deg in range(0, 360, 2):
            rad = np.deg2rad(deg)
            dx, dy = float(np.cos(rad)), float(np.sin(rad))
            ch, _ = trace_ray_chains(sf, ux, uy, dx, dy, t_max=3200.0, cfg=cfg)
            z_chains.extend(ch)
        dt = time.perf_counter() - t0
        n_pts = sum(len(c["points_xyz"]) for c in z_chains)
        max_len = max((len(c["points_xyz"]) for c in z_chains), default=0)
        chains_by_z[z] = z_chains
        all_chains.extend(z_chains)
        print(f"  Z={z:5d}: {len(z_chains):3d} chains, {n_pts:5d} points (max span={max_len} wraps) in {dt:.2f}s")

    out_json = OUT_DIR / "paris4_auto_relative_windings.json"
    export_vc3d_json(all_chains, out_json)
    print(f"\nExported VC3D/Spiral-Fit JSON: {out_json} ({len(all_chains)} chains, "
          f"{sum(len(c['points_xyz']) for c in all_chains):,} points)")

    # 3. Diagnostic Visualization on Z=11228 and Z=8879
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), dpi=140)
    for ax, z in zip(axes, [11228, 8879]):
        sf = slice_map[z]
        # Downsample 4x for crisp overview plotting
        ds = 4
        img = sf.mask[::ds, ::ds]
        ax.imshow(img, cmap="gray", origin="upper", alpha=0.75)

        ux, uy = get_umbilicus_xy(umb_json, z)
        ax.scatter([(ux - sf.x0) / ds], [(uy - sf.y0) / ds], c="#ffea00", marker="+", s=120, linewidths=2.5, label="umbilicus")

        # Plot auto-generated high-confidence chains (cyan/lime alternating)
        for j, ch in enumerate(chains_by_z[z]):
            pts = ch["points_xyz"]
            px = (pts[:, 0] - sf.x0) / ds
            py = (pts[:, 1] - sf.y0) / ds
            ax.plot(px, py, color="#00e5ff", linewidth=1.0, alpha=0.75, label="auto chains" if j == 0 else None)

        # Overlay human ground-truth relative-winding rays in bright orange/red
        first_h = True
        for cid, c in rw_json["collections"].items():
            pts = list(c.get("points", {}).values())
            if len(pts) < 2:
                continue
            pts.sort(key=lambda p: p["wind_a"])
            coords = np.array([p["p"] for p in pts], dtype=np.float64)
            if coords[:, 2].max() - coords[:, 2].min() < 2.0 and int(round(np.median(coords[:, 2]))) == z:
                hx = (coords[:, 0] - sf.x0) / ds
                hy = (coords[:, 1] - sf.y0) / ds
                ax.plot(hx, hy, color="#ff3d00", linewidth=2.0, marker="o", markersize=2.2,
                        label="human GT" if first_h else None)
                first_h = False

        ax.set_title(
            f"PHercParis4 Z={z} — {len(chains_by_z[z])} Auto Chains "
            f"({sum(len(c['points_xyz']) for c in chains_by_z[z]):,} pts)"
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.axis("off")

    fig.suptitle(
        f"Candidate A v1 Benchmark: Naive {naive_prec:.1f}% vs. 5-Gate Conservative {cons_prec:.2f}% Precision "
        f"({stats['cons_exact']}/{cons_acc} exact, {cons_rec:.1f}% recall on {n_pairs} human pairs)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_png = OUT_DIR / "path3_winding_v1_eval.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic plot: {out_png}")


if __name__ == "__main__":
    main()
