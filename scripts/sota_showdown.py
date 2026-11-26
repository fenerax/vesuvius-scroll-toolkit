"""
Honest 5-Minute SOTA Showdown:
  Upstream Neural `PHercParis4/winding_inference/shard_1` (217,829 rays, 4.6M crossings)
  vs.
  Our Autonomous Curved + 3D Z-Slab Tracer (`scroll_winding_tracer`, `balanced` preset)

Zero oracle endpoints, zero GT spacing leakage, STRICTLY SIGNED delta_w (no abs()!).

Answers 4 decisive questions:
  Q1. How accurate is upstream neural `winding_inference` (shard_1) on ALL human GT pairs?
  Q2. Does our 7-Gate 3D verifier CATCH real errors in upstream neural rays (error enrichment)?
  Q3. How accurate are OUR autonomous curved chains (SIGNED) vs. Human GT AND vs. Neural rays (thousands of pairs)?
  Q4. Do our 3D Z-slab gates actually fire on autonomous streamlines (vs. human chords)?
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scroll_winding_tracer.config import PRESETS
from scroll_winding_tracer.streaming import interpolate_umbilicus_xy, load_umbilicus_points, stream_z_slab
from scroll_winding_tracer.tracer import (
    SlabFields,
    build_slab_fields,
    find_polyline_crossings,
    generate_curved_chains,
    integrate_normal_streamline,
    verify_step_7gate,
)

ROOT = Path(__file__).resolve().parent.parent
NEURAL_DIR = ROOT / "cache" / "paris4_neural" / "shard_1"
PARIS4_DIR = ROOT / "cache" / "paris4_spiral"
SLAB_DIR = PARIS4_DIR / "slabs"


def load_neural_shard(umb_pts: np.ndarray):
    origin = np.load(NEURAL_DIR / "ray_origin_zyx.npy").astype(np.float64)
    step = np.load(NEURAL_DIR / "ray_step_zyx.npy").astype(np.float64)
    ct = np.load(NEURAL_DIR / "crossing_t.npy").astype(np.float64)
    lvl = np.load(NEURAL_DIR / "crossing_level.npy").astype(np.int32)
    off = np.load(NEURAL_DIR / "crossing_offsets.npy").astype(np.int64)

    # Exact upstream ray_orientation() from winding_supervision.py (INWARD_RAY_TREND = -0.5)
    order = np.argsort(umb_pts[:, 2])
    u_sorted = umb_pts[order]

    def z_to_umb_yx(z_arr):
        return np.column_stack([
            np.interp(z_arr, u_sorted[:, 2], u_sorted[:, 1]),
            np.interp(z_arr, u_sorted[:, 2], u_sorted[:, 0]),
        ])

    sign = np.ones(len(origin), dtype=np.int32)
    first = off[:-1]
    last = off[1:] - 1
    rays = np.flatnonzero(last > first)
    p_first = origin[rays] + ct[first[rays], None] * step[rays]
    p_last = origin[rays] + ct[last[rays], None] * step[rays]
    r_first = np.linalg.norm(p_first[:, 1:] - z_to_umb_yx(p_first[:, 0]), axis=-1)
    r_last = np.linalg.norm(p_last[:, 1:] - z_to_umb_yx(p_last[:, 0]), axis=-1)
    span = (ct[last[rays]] - ct[first[rays]]) * np.linalg.norm(step[rays], axis=-1)
    trend = (r_last - r_first) / np.maximum(span, 1e-9)
    sign[rays[trend < -0.5]] = -1
    return origin, step, ct, lvl, off, sign


def build_human_collections(rw_path: Path, umb_pts: np.ndarray):
    """Load ALL human collections (>=2 pts), orienting each human collection outward so wind_a increases outward."""
    with open(rw_path) as f:
        rw = json.load(f)["collections"]
    order = np.argsort(umb_pts[:, 2])
    u_sorted = umb_pts[order]

    cols = []
    for cid, c in rw.items():
        pts = list(c.get("points", {}).values())
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p["wind_a"])
        xyz = np.array([p["p"] for p in pts], dtype=np.float64)
        winds = np.array([p["wind_a"] for p in pts], dtype=np.float64)
        # Check whether human wind_a increases outward or inward relative to umbilicus
        ux = np.interp(xyz[:, 2], u_sorted[:, 2], u_sorted[:, 0])
        uy = np.interp(xyz[:, 2], u_sorted[:, 2], u_sorted[:, 1])
        radii = np.hypot(xyz[:, 0] - ux, xyz[:, 1] - uy)
        # Signed correlation between radius and wind_a
        dw_tot = winds[-1] - winds[0]
        dr_tot = radii[-1] - radii[0]
        # Canonicalize human winding to increase outward (+1 = outward)
        h_outward_sign = 1.0 if (dw_tot * dr_tot >= 0) else -1.0
        winds_out = winds * h_outward_sign
        cols.append({
            "cid": cid,
            "xyz": xyz,
            "zyx": xyz[:, [2, 1, 0]],
            "winds_out": winds_out,
            "is_planar": bool(xyz[:, 2].max() - xyz[:, 2].min() < 2.0),
            "z_med": int(round(np.median(xyz[:, 2]))),
        })
    return cols


def main() -> None:
    t_start = time.perf_counter()
    umb_pts = load_umbilicus_points(PARIS4_DIR / "umbilicus.json")
    human_cols = build_human_collections(PARIS4_DIR / "relative_windings.json", umb_pts)
    print(f"Loaded {len(human_cols)} human collections ({sum(len(c['xyz']) for c in human_cols):,} pts; "
          f"{sum(c['is_planar'] for c in human_cols)} planar, {sum(not c['is_planar'] for c in human_cols)} non-planar)")

    origin, step, ct, lvl, off, sign = load_neural_shard(umb_pts)
    n_inward = int((sign < 0).sum())
    print(f"Loaded neural shard_1: {len(origin):,} rays, {len(ct):,} crossings | "
          f"upstream ray_orientation() flagged inward rays: {n_inward:,} ({100*n_inward/len(origin):.2f}%)")

    # Compute ray midpoint ZYX for fast spatial lookup near human collections
    first_idx = off[:-1]
    last_idx = np.maximum(off[1:] - 1, first_idx)
    valid_rays = np.flatnonzero(off[1:] - off[:-1] >= 2)
    p0_all = origin[valid_rays] + ct[first_idx[valid_rays], None] * step[valid_rays]
    p1_all = origin[valid_rays] + ct[last_idx[valid_rays], None] * step[valid_rays]
    mid_all = 0.5 * (p0_all + p1_all)
    half_len_all = 0.5 * np.linalg.norm(p1_all - p0_all, axis=-1)
    max_half = float(half_len_all.max())
    ray_tree = cKDTree(mid_all)

    # =========================================================================
    # Q1: Benchmark Upstream Neural shard_1 (SIGNED!) vs ALL Human GT Rays
    # =========================================================================
    print("\n=== Q1: Upstream Neural `winding_inference` (shard_1) vs. ALL Human GT (SIGNED!) ===")
    q1 = Counter()
    q1_by_span = {"1": Counter(), "2-4": Counter(), "5+": Counter()}
    neural_matched_records = []  # store for Q2 3D-verifier enrichment test!

    match_tol_px = 6.0
    for col in human_cols:
        h_zyx = col["zyx"]
        h_w = col["winds_out"]
        h_center = h_zyx.mean(axis=0)
        h_rad = float(np.linalg.norm(h_zyx - h_center, axis=-1).max())
        cand_local = ray_tree.query_ball_point(h_center, r=h_rad + max_half + match_tol_px)
        if not cand_local:
            continue
        cand_rays = valid_rays[cand_local]

        for r_idx in cand_rays:
            s_i, e_i = int(off[r_idx]), int(off[r_idx + 1])
            r_pts = origin[r_idx] + ct[s_i:e_i, None] * step[r_idx]  # (K, 3) ZYX
            # Quick Z bounding box filter
            if r_pts[:, 0].max() < h_zyx[:, 0].min() - match_tol_px or r_pts[:, 0].min() > h_zyx[:, 0].max() + match_tol_px:
                continue
            # 3D distance matrix (K_ray, N_human)
            dmat = np.linalg.norm(r_pts[:, None, :] - h_zyx[None, :, :], axis=-1)
            # Strict 1-to-1 greedy closest matches <= match_tol_px
            matches = []
            used_r = set()
            for h_i in range(len(h_zyx)):
                k_best = int(np.argmin(dmat[:, h_i]))
                if dmat[k_best, h_i] <= match_tol_px and k_best not in used_r:
                    used_r.add(k_best)
                    matches.append((k_best, h_i))
            if len(matches) < 2:
                continue
            matches.sort(key=lambda t: t[0])
            r_lvls = lvl[s_i:e_i]
            r_sign = int(sign[r_idx])

            for m_i in range(len(matches) - 1):
                k1, h1 = matches[m_i]
                k2, h2 = matches[m_i + 1]
                dw_true = int(round(h_w[h2] - h_w[h1]))
                if dw_true == 0:
                    continue
                dw_pred = r_sign * int(r_lvls[k2] - r_lvls[k1])
                q1["total"] += 1
                bucket = "1" if abs(dw_true) == 1 else ("2-4" if abs(dw_true) <= 4 else "5+")
                q1_by_span[bucket]["total"] += 1

                if dw_pred == dw_true:
                    q1["exact"] += 1
                    q1_by_span[bucket]["exact"] += 1
                    outcome = "exact"
                elif dw_pred == -dw_true:
                    q1["sign_flip"] += 1
                    q1_by_span[bucket]["sign_flip"] += 1
                    outcome = "sign_flip"
                elif abs(dw_pred) < abs(dw_true):
                    q1["undercount"] += 1
                    q1_by_span[bucket]["undercount"] += 1
                    outcome = "under"
                else:
                    q1["overcount"] += 1
                    q1_by_span[bucket]["overcount"] += 1
                    outcome = "over"

                neural_matched_records.append({
                    "r_idx": int(r_idx),
                    "z_mid": float(0.5 * (r_pts[k1, 0] + r_pts[k2, 0])),
                    "pts_zyx": r_pts[k1 : k2 + 1],
                    "dw_true": dw_true,
                    "dw_pred": dw_pred,
                    "outcome": outcome,
                    "is_planar": col["is_planar"],
                })

    tot1 = max(q1["total"], 1)
    print(f"  Matched neural-to-human pairs (3D <=6px, SIGNED): {q1['total']:,}")
    print(f"  Neural Exact Signed : {q1['exact']:,}/{q1['total']:,} ({100*q1['exact']/tot1:.2f}%) | Error rate: {100*(1-q1['exact']/tot1):.2f}%")
    print(f"    Breakdown         : sign_flip={q1['sign_flip']}, undercount={q1['undercount']}, overcount={q1['overcount']}")
    for bname in ("1", "2-4", "5+"):
        b = q1_by_span[bname]
        if b["total"]:
            print(f"    Span |dw|={bname:<3}     : exact={b['exact']}/{b['total']} ({100*b['exact']/b['total']:.2f}%) "
                  f"[flip={b['sign_flip']}, under={b['undercount']}, over={b['overcount']}]")

    # =========================================================================
    # Load our 12 cached PHercParis4 3D Z-slabs for Q2, Q3, Q4
    # =========================================================================
    slab_files = sorted(SLAB_DIR.glob("slab_z_*.npz"))
    slabs_by_z: dict[int, SlabFields] = {}
    for sf_path in slab_files:
        slab, y0, x0, z0 = stream_z_slab(None, 0, cache_file=sf_path)
        z_val = int(sf_path.stem.split("_")[-1])
        slabs_by_z[z_val] = build_slab_fields(slab, y0, x0, z0, z_val)
    cached_zs = sorted(slabs_by_z.keys())
    print(f"\nLoaded {len(cached_zs)} cached 3D Z-slabs ({cached_zs[0]}..{cached_zs[-1]})")

    # =========================================================================
    # Q2: Error Enrichment — Does our 7-Gate 3D Verifier Catch Neural Errors?
    # =========================================================================
    print("\n=== Q2: 3D Verifier as QA Filter on Upstream Neural Rays (Error Enrichment) ===")
    cfg_bal = PRESETS["balanced"]
    q2 = Counter()
    q2_reasons = Counter()

    for rec in neural_matched_records:
        # Check if this neural pair lies within +/-3 slices of one of our 12 cached Z-slabs
        nearest_z = min(cached_zs, key=lambda z: abs(z - rec["z_mid"]))
        if abs(nearest_z - rec["z_mid"]) > 3.0:
            continue
        sf = slabs_by_z[nearest_z]
        pts_zyx = rec["pts_zyx"]
        # Verify every consecutive neural crossing step using our 7-gate 3D verifier
        all_ok = True
        fail_r = "ok"
        for k in range(len(pts_zyx) - 1):
            p0_xy = np.array([pts_zyx[k, 2], pts_zyx[k, 1]])
            p1_xy = np.array([pts_zyx[k + 1, 2], pts_zyx[k + 1, 1]])
            dist = float(np.linalg.norm(p1_xy - p0_xy))
            # Estimate valley directly along the segment on sf.center_mask
            ok, reason, _ = verify_step_7gate(sf, p0_xy, p1_xy, valley_px=max(0.0, dist - 4.0), local_med_spacing=None, cfg=cfg_bal)
            if not ok:
                all_ok = False
                fail_r = reason
                break

        is_err = (rec["outcome"] != "exact")
        if all_ok:
            q2["acc_total"] += 1
            if is_err:
                q2["acc_err"] += 1
        else:
            q2["rej_total"] += 1
            q2_reasons[fail_r] += 1
            if is_err:
                q2["rej_err"] += 1

    acc_n, acc_e = q2["acc_total"], q2["acc_err"]
    rej_n, rej_e = q2["rej_total"], q2["rej_err"]
    acc_rate = 100.0 * acc_e / max(acc_n, 1)
    rej_rate = 100.0 * rej_e / max(rej_n, 1)
    enrich = rej_rate / max(acc_rate, 1e-6)
    print(f"  Neural pairs evaluated on cached slabs : {acc_n + rej_n} (total neural errors={acc_e + rej_e})")
    print(f"  ACCEPTED by 3D verifier                : {acc_n:4d} pairs -> {acc_e:3d} errors ({acc_rate:.2f}% error rate | {100-acc_rate:.2f}% prec)")
    print(f"  FLAGGED/REJECTED by 3D verifier        : {rej_n:4d} pairs -> {rej_e:3d} errors ({rej_rate:.2f}% error rate)")
    print(f"  Error Enrichment Ratio (flagged/acc)   : {enrich:.2f}x | Reject reasons: {dict(q2_reasons)}")

    # =========================================================================
    # Q3: Honest End-to-End Autonomous Audit of OUR Curved Chains (balanced)
    #     (A) vs. Human GT (SIGNED), (B) vs. Neural shard_1 (THOUSANDS of SIGNED pairs!),
    #     (C) Spatial coverage overlap vs. neural shard_1
    # =========================================================================
    print("\n=== Q3: Honest End-to-End Autonomous Audit of OUR Curved Chains (`balanced`, SIGNED!) ===")
    ours_by_z: dict[int, list[dict]] = {}
    for z in cached_zs:
        sf = slabs_by_z[z]
        ux, uy = interpolate_umbilicus_xy(umb_pts, z)
        ours_by_z[z] = generate_curved_chains(sf, ux, uy, cfg_bal)

    # (A) Autonomous Ours vs. Human GT (SIGNED!)
    q3_h = Counter()
    for col in human_cols:
        if not col["is_planar"] or col["z_med"] not in ours_by_z:
            continue
        h_xy = col["xyz"][:, :2]
        h_w = col["winds_out"]
        for ch in ours_by_z[col["z_med"]]:
            c_xy = ch["points_xyz"][:, :2]
            dmat = np.linalg.norm(c_xy[:, None, :] - h_xy[None, :, :], axis=-1)
            matches = []
            used_c = set()
            for h_i in range(len(h_xy)):
                c_i = int(np.argmin(dmat[:, h_i]))
                if dmat[c_i, h_i] <= 6.0 and c_i not in used_c:
                    used_c.add(c_i)
                    matches.append((c_i, h_i))
            if len(matches) < 2:
                continue
            matches.sort(key=lambda t: t[0])
            for m_i in range(len(matches) - 1):
                c1, h1 = matches[m_i]
                c2, h2 = matches[m_i + 1]
                dw_h = int(round(h_w[h2] - h_w[h1]))
                if dw_h == 0:
                    continue
                dw_c = c2 - c1  # SIGNED! (our chains are ordered outward: wind_a = 0, 1, 2, ...)
                q3_h["total"] += 1
                if dw_c == dw_h:
                    q3_h["exact"] += 1
                elif dw_c == -dw_h:
                    q3_h["sign_flip"] += 1
                elif abs(dw_c) < abs(dw_h):
                    q3_h["undercount"] += 1
                else:
                    q3_h["overcount"] += 1

    print(f"  [Q3-A] Autonomous Ours vs Human GT (SIGNED): exact={q3_h['exact']}/{q3_h['total']} "
          f"({100*q3_h['exact']/max(q3_h['total'],1):.2f}%) | flip={q3_h['sign_flip']}, under={q3_h['undercount']}, over={q3_h['overcount']}")

    # (B) Autonomous Ours vs. Upstream Neural shard_1 (THOUSANDS of SIGNED pairs!)
    q3_n = Counter()
    q3_n_span = {"1": Counter(), "2-4": Counter(), "5+": Counter()}
    # Check coverage overlap per 128px cell across all 12 slabs
    cov_both = cov_ours_only = cov_neural_only = cov_neither = 0

    for z in cached_zs:
        sf = slabs_by_z[z]
        H, W = sf.center_mask.shape
        gh, gw = H // 128, W // 128
        scroll_cells = sf.center_mask[: gh * 128, : gw * 128].reshape(gh, 128, gw, 128).any(axis=(1, 3))
        ours_cells = np.zeros_like(scroll_cells)
        neural_cells = np.zeros_like(scroll_cells)

        for ch in ours_by_z[z]:
            pts = ch["points_xyz"]
            ys = ((pts[:, 1] - sf.y0) // 128).astype(np.int32)
            xs = ((pts[:, 0] - sf.x0) // 128).astype(np.int32)
            ok = (ys >= 0) & (ys < gh) & (xs >= 0) & (xs < gw)
            ours_cells[ys[ok], xs[ok]] = True

        # Find all neural rays in shard_1 crossing within +/-3 slices of z
        z_mask = (np.minimum(p0_all[:, 0], p1_all[:, 0]) <= z + 3.0) & (np.maximum(p0_all[:, 0], p1_all[:, 0]) >= z - 3.0)
        z_rays = valid_rays[z_mask]

        # Build KD-tree of all our chain points on slice z
        our_pts_xy = []
        our_meta = []  # (chain_idx, pt_idx)
        for ci, ch in enumerate(ours_by_z[z]):
            for pi, (x, y, _) in enumerate(ch["points_xyz"]):
                our_pts_xy.append((x, y))
                our_meta.append((ci, pi))
        if not our_pts_xy:
            continue
        our_pts_xy = np.array(our_pts_xy, dtype=np.float64)
        our_tree = cKDTree(our_pts_xy)

        for r_idx in z_rays:
            s_i, e_i = int(off[r_idx]), int(off[r_idx + 1])
            r_pts = origin[r_idx] + ct[s_i:e_i, None] * step[r_idx]  # (K, 3) ZYX
            near_z = np.flatnonzero(np.abs(r_pts[:, 0] - z) <= 3.0)
            if len(near_z) == 0:
                continue
            # Mark neural coverage cells
            ny = ((r_pts[near_z, 1] - sf.y0) // 128).astype(np.int32)
            nx = ((r_pts[near_z, 2] - sf.x0) // 128).astype(np.int32)
            ok = (ny >= 0) & (ny < gh) & (nx >= 0) & (nx < gw)
            neural_cells[ny[ok], nx[ok]] = True

            if len(near_z) < 2:
                continue
            r_xy = r_pts[near_z][:, [2, 1]]
            dists, idxs = our_tree.query(r_xy, distance_upper_bound=5.0)
            # Group hits by our chain id
            by_chain: dict[int, list[tuple[int, int]]] = {}
            r_lvls = lvl[s_i:e_i]
            r_sign = int(sign[r_idx])
            for loc_i, (d, oi) in enumerate(zip(dists, idxs)):
                if d <= 5.0 and oi < len(our_meta):
                    ci, pi = our_meta[oi]
                    by_chain.setdefault(ci, []).append((int(near_z[loc_i]), pi))
            for ci, pairs in by_chain.items():
                if len(pairs) < 2:
                    continue
                pairs.sort(key=lambda t: t[0])
                for m_i in range(len(pairs) - 1):
                    k1, p1 = pairs[m_i]
                    k2, p2 = pairs[m_i + 1]
                    dw_n = r_sign * int(r_lvls[k2] - r_lvls[k1])
                    dw_o = p2 - p1
                    if dw_n == 0 or dw_o == 0:
                        continue
                    q3_n["total"] += 1
                    bname = "1" if abs(dw_n) == 1 else ("2-4" if abs(dw_n) <= 4 else "5+")
                    q3_n_span[bname]["total"] += 1
                    if dw_o == dw_n:
                        q3_n["exact"] += 1
                        q3_n_span[bname]["exact"] += 1
                    elif dw_o == -dw_n:
                        q3_n["sign_flip"] += 1
                        q3_n_span[bname]["sign_flip"] += 1
                    else:
                        q3_n["mag_diff"] += 1
                        q3_n_span[bname]["mag_diff"] += 1

        cov_both += int((scroll_cells & ours_cells & neural_cells).sum())
        cov_ours_only += int((scroll_cells & ours_cells & ~neural_cells).sum())
        cov_neural_only += int((scroll_cells & ~ours_cells & neural_cells).sum())
        cov_neither += int((scroll_cells & ~ours_cells & ~neural_cells).sum())

    tot_n = max(q3_n["total"], 1)
    print(f"  [Q3-B] Autonomous Ours vs Neural shard_1 (SIGNED): exact={q3_n['exact']:,}/{q3_n['total']:,} "
          f"({100*q3_n['exact']/tot_n:.2f}%) | sign_flip={q3_n['sign_flip']:,} ({100*q3_n['sign_flip']/tot_n:.2f}%), "
          f"mag_diff={q3_n['mag_diff']:,} ({100*q3_n['mag_diff']/tot_n:.2f}%)")
    for bname in ("1", "2-4", "5+"):
        b = q3_n_span[bname]
        if b["total"]:
            print(f"         Span |dw|={bname:<3}: exact={b['exact']:,}/{b['total']:,} ({100*b['exact']/b['total']:.2f}%) "
                  f"[flip={b['sign_flip']}, mag_diff={b['mag_diff']}]")

    tot_cells = max(cov_both + cov_ours_only + cov_neural_only + cov_neither, 1)
    print(f"  [Q3-C] 128px Cell Coverage (12 slabs): both={100*cov_both/tot_cells:.1f}%, "
          f"OURS_ONLY={100*cov_ours_only/tot_cells:.1f}%, NEURAL_SHARD1_ONLY={100*cov_neural_only/tot_cells:.1f}%, "
          f"neither={100*cov_neither/tot_cells:.1f}%")

    # =========================================================================
    # Q4: Real Gate Trigger Histogram on Autonomous Streamlines (Z=11228)
    # =========================================================================
    print("\n=== Q4: Real Gate Trigger Histogram During Autonomous Tracing (Z=11228, `balanced`) ===")
    sf = slabs_by_z[11228]
    ux, uy = interpolate_umbilicus_xy(umb_pts, 11228)
    gate_hist = Counter()
    H, W = sf.center_mask.shape
    for ly in range(80, H - 80, 160):
        for lx in range(80, W - 80, 160):
            if sf.coherence[ly, lx] < cfg_bal.min_coherence:
                continue
            gx, gy = lx + sf.x0, ly + sf.y0
            if np.hypot(gx - ux, gy - uy) < 60.0:
                continue
            poly = integrate_normal_streamline(sf, gx, gy, cfg_bal)
            if len(poly) < 10:
                continue
            sg, pg, cs, es, xs = find_polyline_crossings(sf, poly)
            if len(cs) < 2:
                continue
            c_pts = np.column_stack([np.interp(cs, sg, pg[:, 0]), np.interp(cs, sg, pg[:, 1])])
            sp = np.linalg.norm(np.diff(c_pts, axis=0), axis=1)
            pl = sp[(sp >= cfg_bal.min_spacing_px) & (sp <= cfg_bal.max_spacing_px)]
            gmed = float(np.median(pl)) if len(pl) >= 3 else None
            for i in range(len(c_pts) - 1):
                _, reason, _ = verify_step_7gate(sf, c_pts[i], c_pts[i + 1], float(es[i + 1] - xs[i]), gmed, cfg_bal)
                gate_hist[reason] += 1

    tot_g = sum(gate_hist.values())
    print(f"  Total autonomous steps evaluated: {tot_g:,}")
    for r, cnt in gate_hist.most_common():
        print(f"    {r:<22}: {cnt:6,} ({100*cnt/max(tot_g,1):5.1f}%)")
    print(f"\nShowdown finished in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
