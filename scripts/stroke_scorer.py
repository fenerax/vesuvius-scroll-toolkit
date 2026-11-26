"""Full-mesh raw-float thin-stroke DoG tile scorer & 2x-downsampled map archiver.

Fixes all blind spots identified by the 6-seat Fancy Council:
  1. Covers 100% of every mesh including trailing right/bottom edges (exact edge-aligned starts).
  2. Accepts tiles down to valid_frac >= 0.30 (fixes 45-deg rotated diamond edge exclusion).
  3. Operates directly on raw float [0, 1] reader-v3 probabilities (zero PNG contrast warp,
     zero ink_9um geometric-mean veto, zero p99 giant-blob bias).
  4. Computes normalized-convolution DoG ONCE over the full mesh (at 2x area-averaged scale,
     ~18.7 um/px), then scores every overlapping tile on both directional (fwd - 0.5*rev)
     and raw single-side maps, returning NMS-deduplicated top-K tiles per mesh.
  5. Reports both valid papyrus area (cm2) and bounding-box area (cm2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label


def tile_starts(dim: int, win: int, stride: int) -> list[int]:
    """Exact edge-aligned tile start indices covering [0, dim) with 100% coverage."""
    if dim <= win:
        return [0]
    starts = list(range(0, dim - win, stride))
    if not starts or starts[-1] != dim - win:
        starts.append(dim - win)
    return starts


def downsample2x_masked(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact 2x area-averaged masked downsample preserving calibrated float probabilities."""
    H, W = arr.shape
    h2, w2 = H // 2, W // 2
    if h2 == 0 or w2 == 0:
        return arr.astype(np.float32), mask.astype(bool)
    a = arr[: 2 * h2, : 2 * w2].astype(np.float32).reshape(h2, 2, w2, 2)
    m = mask[: 2 * h2, : 2 * w2].astype(np.float32).reshape(h2, 2, w2, 2)
    wsum = m.sum(axis=(1, 3))
    asum = (a * m).sum(axis=(1, 3))
    out = np.divide(asum, wsum, out=np.zeros_like(asum), where=wsum > 0)
    out_m = wsum >= 2.0  # at least 2 of 4 subpixels valid
    return out, out_m


def _normalized_dog(prob: np.ndarray, valid: np.ndarray, s_lo: float, s_hi: float) -> np.ndarray:
    """Normalized-convolution DoG so valid-mask edges never create ringing artifacts."""
    p = np.where(valid, np.clip(prob, 0.0, 1.0), 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    g_lo = gaussian_filter(p, s_lo) / np.maximum(gaussian_filter(w, s_lo), 1e-4)
    g_hi = gaussian_filter(p, s_hi) / np.maximum(gaussian_filter(w, s_hi), 1e-4)
    return np.where(valid, np.clip(g_lo - g_hi, 0.0, 1.0), 0.0)


def _score_tile_2x(
    prob_2x: np.ndarray,
    dog_2x: np.ndarray,
    valid_2x: np.ndarray,
    s_scale: float,
) -> dict:
    """Score one 2x-downsampled tile (~18.7 um/px) by thin-stroke Greek-letter morphology."""
    v = valid_2x
    v_sum = int(v.sum())
    if v_sum == 0 or float(v.mean()) < 0.30:
        return {"score": 0.0, "n_strokes": 0, "cells": 0, "dog99": 0.0, "dog95": 0.0, "giant": 0.0, "p99": 0.0}

    dog_v = dog_2x[v]
    dog_p99 = float(np.percentile(dog_v, 99))
    # Fast early-out for completely blank papyrus tiles (saves >85% CPU time on large meshes!)
    if dog_p99 < 0.035:
        return {
            "score": 0.0,
            "n_strokes": 0,
            "cells": 0,
            "dog99": round(dog_p99, 4),
            "dog95": round(float(np.percentile(dog_v, 95)), 4),
            "giant": 0.0,
            "p99": round(float(np.percentile(prob_2x[v], 99)), 4),
        }

    area_scale = s_scale ** 2
    # Crisp thin-stroke connected components at ~18.7 um/px (matched to rerank_panels_strokes.py)
    bw = (dog_2x > 0.055) & v
    lbl, ncomp = label(bw)
    sizes = np.bincount(lbl.ravel())[1:] if ncomp > 0 else np.array([], dtype=np.int64)
    min_sz = max(12, int(round(25 * area_scale)))
    max_sz = max(400, int(round(1600 * area_scale)))
    stroke_comps = int(((sizes >= min_sz) & (sizes <= max_sz)).sum())

    # Broad delamination giant-blob penalty on raw calibrated probability
    p_v = prob_2x[v]
    p99 = float(np.percentile(p_v, 99))
    thr_blob = max(0.35, float(np.percentile(p_v, 88)))
    raw_bw = (prob_2x > thr_blob) & v
    rlbl, rn = label(raw_bw)
    rsizes = np.bincount(rlbl.ravel())[1:] if rn > 0 else np.array([], dtype=np.int64)
    giant_lim = max(800, int(round(2400 * area_scale)))
    giant_frac = float(rsizes[rsizes > giant_lim].sum()) / max(1, v_sum)

    # 4x4 spatial cell spread (multi-letter / multi-line clusters beat isolated specks!)
    H, W = dog_2x.shape
    gh, gw = max(1, H // 4), max(1, W // 4)
    cell_hits = 0
    for r in range(4):
        for c in range(4):
            cm = v[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw]
            if cm.mean() < 0.25:
                continue
            cell = dog_2x[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw][cm]
            if float(np.percentile(cell, 98)) > 0.06:
                cell_hits += 1

    dog_p95 = float(np.percentile(dog_v, 95))
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


def _box_iou(a: list[int], b: list[int]) -> float:
    ay0, ax0, ah, aw = a
    by0, bx0, bh, bw = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay0 + ah, by0 + bh), min(ax0 + aw, bx0 + bw)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    if inter <= 0:
        return 0.0
    union = ah * aw + bh * bw - inter
    return inter / max(1, union)


def score_full_mesh_tiles(
    prob_fwd: np.ndarray,
    prob_rev: np.ndarray | None,
    mask: np.ndarray,
    voxel_um: float,
    win_px: int = 768,
    stride_px: int = 384,
    top_k: int = 8,
    nms_iou: float = 0.35,
    extra_fwd: dict[str, np.ndarray] | None = None,
    extra_rev: dict[str, np.ndarray] | None = None,
    return_all_tiles: bool = False,
) -> dict:
    """Score 100% of overlapping tiles across a full mesh using raw-float thin-stroke DoG.

    Works with reader-v3 alone (fast screening mode) or with additional confirmation models
    (argus_v3, ink_9um) whose spatial correlations are logged on top tiles without vetoing hits.
    """
    H, W = mask.shape
    s_scale = 9.362 / voxel_um  # scales pixel sizes proportionally for 8.64 um vs 9.362 um
    s_lo = 1.8 * s_scale
    s_hi = 9.0 * s_scale

    # Exact 2x masked downsample (preserves raw float [0, 1] probabilities)
    f2, m2 = downsample2x_masked(prob_fwd, mask)
    dog_f2 = _normalized_dog(f2, m2, s_lo, s_hi)

    sides: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    if prob_rev is not None:
        r2, _ = downsample2x_masked(prob_rev, mask)
        dog_r2 = _normalized_dog(r2, m2, s_lo, s_hi)
        df2 = np.clip(f2 - 0.5 * r2, 0.0, 1.0)
        dr2 = np.clip(r2 - 0.5 * f2, 0.0, 1.0)
        dog_df2 = _normalized_dog(df2, m2, s_lo, s_hi)
        dog_dr2 = _normalized_dog(dr2, m2, s_lo, s_hi)
        sides.append(("fwd", f2, dog_f2, df2, dog_df2))
        sides.append(("rev", r2, dog_r2, dr2, dog_dr2))
    else:
        sides.append(("fwd", f2, dog_f2, f2, dog_f2))

    # Full-res tile grid with exact trailing-edge alignment
    wh = min(win_px, H)
    ww = min(win_px, W)
    st_h = max(128, min(stride_px, wh // 2))
    st_w = max(128, min(stride_px, ww // 2))
    ys = tile_starts(H, wh, st_h)
    xs = tile_starts(W, ww, st_w)

    h2, w2 = m2.shape
    all_tiles: list[dict] = []
    evaluated = 0
    for y0 in ys:
        y1 = y0 + wh
        y0_2, y1_2 = y0 // 2, min(h2, max(y0 // 2 + 1, y1 // 2))
        for x0 in xs:
            x1 = x0 + ww
            x0_2, x1_2 = x0 // 2, min(w2, max(x0 // 2 + 1, x1 // 2))
            vm = m2[y0_2:y1_2, x0_2:x1_2]
            vfrac = float(vm.mean())
            if vfrac < 0.30:
                continue
            evaluated += 1
            best_side_rec = None
            for side, raw_p, raw_d, dir_p, dir_d in sides:
                st_dir = _score_tile_2x(dir_p[y0_2:y1_2, x0_2:x1_2], dir_d[y0_2:y1_2, x0_2:x1_2], vm, s_scale)
                st_raw = _score_tile_2x(raw_p[y0_2:y1_2, x0_2:x1_2], raw_d[y0_2:y1_2, x0_2:x1_2], vm, s_scale)
                # Directional score rejects two-sided delamination glow; 0.75*raw rescues faint ink with mild rev bleed
                combined = max(st_dir["score"], 0.75 * st_raw["score"])
                rec = {
                    "side": side,
                    "win_yx": [int(y0), int(x0), int(wh), int(ww)],
                    "valid_frac": round(vfrac, 3),
                    "stroke_score": round(float(combined), 5),
                    "dir_score": st_dir["score"],
                    "raw_score": st_raw["score"],
                    "n_strokes": st_dir["n_strokes"] if st_dir["score"] >= 0.75 * st_raw["score"] else st_raw["n_strokes"],
                    "cells": st_dir["cells"] if st_dir["score"] >= 0.75 * st_raw["score"] else st_raw["cells"],
                    "dog99": st_dir["dog99"] if st_dir["score"] >= 0.75 * st_raw["score"] else st_raw["dog99"],
                    "giant": st_raw["giant"],
                    "p99": st_raw["p99"],
                }
                if best_side_rec is None or rec["stroke_score"] > best_side_rec["stroke_score"]:
                    best_side_rec = rec
            if best_side_rec is not None:
                all_tiles.append(best_side_rec)

    all_tiles.sort(key=lambda t: t["stroke_score"], reverse=True)

    # Spatial NMS so top_k preserves distinct regions across a large mesh
    selected: list[dict] = []
    for t in all_tiles:
        if any(_box_iou(t["win_yx"], s["win_yx"]) > nms_iou for s in selected):
            continue
        # Optionally compute confirmation model correlations on selected top tiles only (fast!)
        if extra_fwd is not None:
            y0, x0, th, tw = t["win_yx"]
            sl = (slice(y0, y0 + th), slice(x0, x0 + tw))
            m_sub = mask[sl]
            p_base = (prob_fwd if t["side"] == "fwd" else prob_rev)[sl][m_sub]
            p_ext_map = extra_fwd if t["side"] == "fwd" else (extra_rev or {})
            corrs = {}
            for nm, arr in p_ext_map.items():
                p_other = arr[sl][m_sub]
                r = float(np.corrcoef(p_base, p_other)[0, 1]) if (p_base.std() > 1e-5 and p_other.std() > 1e-5) else 0.0
                corrs[f"r_reader3_{nm}"] = round(r, 4)
            t = {**t, **corrs}
        selected.append(t)
        if len(selected) >= top_k:
            break

    bbox_cm2 = (H * W) * ((voxel_um / 1e4) ** 2)
    valid_cm2 = float(mask.sum()) * ((voxel_um / 1e4) ** 2)
    res = {
        "n_tiles_evaluated": evaluated,
        "tile_grid_hw": [len(ys), len(xs)],
        "bbox_area_cm2": round(bbox_cm2, 3),
        "valid_area_cm2": round(valid_cm2, 3),
        "best_tile": selected[0] if selected else None,
        "top_tiles": selected,
    }
    if return_all_tiles:
        res["all_tiles"] = all_tiles
    return res


def save_full_mesh_maps(
    maps_dir: Path,
    mesh_id: str,
    ct_mid: np.ndarray,
    prob_fwd: np.ndarray,
    prob_rev: np.ndarray | None,
    mask: np.ndarray,
    scale_div: int = 4,
) -> None:
    """Save calibrated downsampled full-mesh maps (fwd, rev, ct) for $0 local browsing & re-ranking.

    At 4x downsample (~37 um/px), a 4 mm Greek letter is ~108 px tall (fully readable by eye and
    DoG-rerankable), while each full mesh takes only ~25-45 KB as JPEG (q=88) for fast download.
    """
    maps_dir.mkdir(parents=True, exist_ok=True)
    f_ds, m_ds = downsample2x_masked(prob_fwd, mask)
    r_ds = downsample2x_masked(prob_rev, mask)[0] if prob_rev is not None else None
    ct_ds, _ = downsample2x_masked(ct_mid.astype(np.float32), mask)
    if scale_div >= 4 and f_ds.shape[0] >= 4 and f_ds.shape[1] >= 4:
        f_ds, m_ds2 = downsample2x_masked(f_ds, m_ds)
        if r_ds is not None:
            r_ds, _ = downsample2x_masked(r_ds, m_ds)
        ct_ds, _ = downsample2x_masked(ct_ds, m_ds)
        m_ds = m_ds2

    Image.fromarray(np.clip(np.rint(np.where(m_ds, f_ds, 0.0) * 255.0), 0, 255).astype(np.uint8)).save(
        maps_dir / f"{mesh_id}_fwd.jpg", quality=88
    )
    if r_ds is not None:
        Image.fromarray(np.clip(np.rint(np.where(m_ds, r_ds, 0.0) * 255.0), 0, 255).astype(np.uint8)).save(
            maps_dir / f"{mesh_id}_rev.jpg", quality=88
        )
    if m_ds.any() and (ct_ds[m_ds] > 0).any():
        lo, hi = np.percentile(ct_ds[m_ds & (ct_ds > 0)], [2, 98])
        ct_u8 = np.clip((ct_ds - lo) / max(1e-3, hi - lo) * 255.0, 0, 255)
        ct_u8 = np.where(m_ds, ct_u8, 0.0).astype(np.uint8)
    else:
        ct_u8 = np.zeros_like(ct_ds, dtype=np.uint8)
    Image.fromarray(ct_u8).save(maps_dir / f"{mesh_id}_ct.jpg", quality=85)
