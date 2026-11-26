"""Streaming audit runs: targeted crops, full-segment pyramid sweep, three-scroll verification."""

from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .auditor import (CLASS_NAMES, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH, INVALID_PAD, OK, WARN_DRIFT,
                      AuditResult, Thresholds, audit_field, find_hotspots, restrict_valid, save_audit)
from .band_detector import GRID_FIELDS, BandField, BandParams, detect_bands, pool_all, with_steps
from .data import (OUT_DIR, REFERENCE_1UM_VOLUME, SEGMENTS, BoundedCache, LayersVolume, SegmentRef, check_crop,
                   output_tag, read_crop, segment_record)

CLASSES = (OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH, INVALID_PAD)
FAILS = (FAIL_OFF_SHEET, FAIL_SHEET_SWITCH)


def load_thresholds(calibration: str | Path | None, **overrides) -> Thresholds:
    if calibration and Path(calibration).exists():
        return Thresholds.from_calibration(calibration, **overrides)
    if calibration:
        raise FileNotFoundError(calibration)
    kw = {k: v for k, v in overrides.items() if v is not None}
    return Thresholds(**kw)


def audit_crop(vol: LayersVolume, y0: int, x0: int, thr: Thresholds, size: int = 256, level: int = 0,
               cell: int | None = None, cache: BoundedCache | None = None) -> tuple[AuditResult, np.ndarray]:
    """Audit one crop. ``y0``, ``x0``, ``size`` and ``cell`` are in pixels of ``level`` (default cell: 8 L0 px).

    Bounds are checked before any read. At ``level > 0`` the valid mask is the exact level-0
    support (``all`` over each cell's L0 pixels), read from the matching L0 crop, so mean-pooled
    padding never leaks into a cell that the level-0 audit would call invalid."""
    f = 2 ** level
    cell = cell or max(1, 8 // f)
    check_crop(vol.shape(level), y0, x0, size, cell)
    slab = read_crop(vol, y0, x0, size, level=level, cache=cache)
    field = detect_bands(slab, vol.pixel_size_um, BandParams(cell_size=cell))
    if level > 0:
        s0 = read_crop(vol, y0 * f, x0 * f, size * f, level=0, cache=cache)
        exact = pool_all((s0 != 0).any(axis=0), cell * f)[:field.shape[0], :field.shape[1]]
        field = restrict_valid(field, exact)
    meta = {"sample": vol.ref.sample, "segment": segment_record(vol.ref)["long_id"], "volume": vol.ref.volume,
            "level": level, "crop_origin_level_yx": [y0, x0], "crop_size_level_px": size,
            "pixel_size_um": vol.pixel_size_um, "valid_mask": "exact level-0 support (all pixels of the cell)"}
    res = audit_field(field, thr, origin_l0=(y0 * f, x0 * f), cell_l0=cell * f, meta=meta)
    return res, slab


def _stitch(parts: list[tuple[int, int, BandField]], shape: tuple[int, int], template: BandField) -> BandField:
    """Assemble block fields (already trimmed to their cores) into one field and recompute the steps."""
    full = {}
    for n in GRID_FIELDS:
        a = getattr(template, n)
        fill = np.nan if a.dtype.kind == "f" else 0
        full[n] = np.full(shape, fill, a.dtype)
    sp, spc = [np.zeros(0)], [np.zeros(0, np.int64)]
    for gy, gx, fld in parts:
        h, w = fld.shape
        for n in GRID_FIELDS:
            full[n][gy:gy + h, gx:gx + w] = getattr(fld, n)
        iy, ix = np.divmod(fld.band_spacing_cell, w)
        sp.append(fld.band_spacing_um)
        spc.append((iy + gy) * shape[1] + ix + gx)
    f = replace(template, band_spacing_um=np.concatenate(sp), band_spacing_cell=np.concatenate(spc), **full)
    return with_steps(f)


def l0_exact_validity(vol: LayersVolume, valid: np.ndarray, level: int, workers: int = 6) -> tuple[np.ndarray, dict]:
    """Exact level-0 validity (``all`` over each cell's L0 pixels) of a level-``level`` grid.

    A mean-pooled level pixel is non-zero as soon as one of its L0 pixels is, so the pyramid alone
    over-reports validity at the padding boundary. tifxyz meshes (scale 0.05) are valid on a 20 px
    lattice: every invalid region is a union of 20 x 20 L0 quads. For ``f = 2 ** level <= 10``
    (levels 1..3), every 20 px run contains at least one whole aligned ``f``-px cell, so any cell
    that holds an invalid L0 pixel is either fully zero or 8-adjacent to one; only those boundary
    candidates need their L0 chunks read. For ``f > 10`` (level >= 4), a 20 x 20 invalid quad can
    fit inside one pooled cell or straddle two cells without creating a fully zero neighbor, so all
    level-valid cells are checked against L0. The assumption is checked empirically by
    ``l0_cross_check``."""
    f = 2 ** level
    boundary_only = f <= 10
    info = {"rule": "valid = all L0 pixels of the cell have non-zero CT in some layer",
            "candidates": ("level-valid cells 8-adjacent to a level-invalid cell or the grid edge"
                           if boundary_only else "all level-valid cells (cell size exceeds 20 px quad guarantee)")}
    if f == 1:
        return valid.copy(), {**info, "regions_read": 0, "cells_removed": 0}
    _, H0, W0 = vol.shape(0)
    gh, gw = valid.shape
    cand = (valid & ndi.binary_dilation(~valid, structure=np.ones((3, 3), bool), border_value=1)
            if boundary_only else valid.copy())
    cs = int(vol.array(0).chunks[1])
    region = cs if cs % f == 0 else -(-cs // f) * f
    cpr = region // f
    iy, ix = np.nonzero(cand)
    regions = sorted(set(zip((iy // cpr).tolist(), (ix // cpr).tolist())))
    exact = valid.copy()
    t0 = time.perf_counter()

    def load(r):
        ry, rx = r
        y0, x0 = ry * region, rx * region
        a = vol.read(0, y0, x0, min(region, H0 - y0), min(region, W0 - x0))
        return r, pool_all((a != 0).any(axis=0), f)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, ((ry, rx), pooled) in enumerate(ex.map(load, regions), 1):
            gy, gx = ry * cpr, rx * cpr
            ph, pw = min(pooled.shape[0], gh - gy), min(pooled.shape[1], gw - gx)
            exact[gy:gy + ph, gx:gx + pw] &= pooled[:ph, :pw]
            if i % 200 == 0 or i == len(regions):
                print(f"  exact L0 validity: {i}/{len(regions)} chunks, {time.perf_counter() - t0:.0f}s", flush=True)
    return exact, {**info, "l0_region_px": region, "candidate_cells": int(cand.sum()), "regions_read": len(regions),
                   "cells_removed": int((valid & ~exact).sum()), "seconds": round(time.perf_counter() - t0, 1)}


def sweep_level(vol: LayersVolume, level: int, block: int = 512, workers: int = 3,
                exact_validity: bool = True) -> tuple[BandField, dict]:
    """Band field of the whole segment at pyramid ``level`` with one cell per level pixel.

    Pyramid levels are mean-pooled in Y and X only (all 109 layers kept), so a level-3 pixel is
    the mean profile of an 8 x 8 L0 cell. Each block is read with a one-pixel halo on its bottom
    and right edge, so the exact sheet-continuity flags across block seams are computed from both
    cells' own band lists. The valid mask is then replaced by the exact level-0 support
    (``l0_exact_validity``) before any lateral step is evaluated. An all-padding level returns an
    all-invalid field."""
    D, H, W = vol.shape(level)
    tiles = [(y, x) for y in range(0, H, block) for x in range(0, W, block)]
    params = BandParams(cell_size=1)
    parts = []
    t0 = time.perf_counter()

    def load(t):
        y, x = t
        return y, x, vol.read(level, y, x, min(block + 1, H - y), min(block + 1, W - x))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (y, x, slab) in enumerate(ex.map(load, tiles), 1):
            core = slab[:, :min(block, H - y), :min(block, W - x)]
            if core.any():
                fld = detect_bands(slab, vol.pixel_size_um, params)
                parts.append((y, x, fld.crop(slice(0, core.shape[1]), slice(0, core.shape[2]))))
            if i % 8 == 0 or i == len(tiles):
                print(f"  level {level}: {i}/{len(tiles)} blocks, {time.perf_counter() - t0:.0f}s", flush=True)
    if not parts:
        return BandField.empty((H, W), vol.pixel_size_um, 1, D), {"empty": True, "cells_removed": 0}
    field = _stitch(parts, (H, W), parts[0][2])
    info = {"empty": False}
    if exact_validity:
        exact, info = l0_exact_validity(vol, field.valid, level)
        field = restrict_valid(field, exact)
    return field, info


def concordance(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement between two class maps over cells valid in both."""
    m = (a != INVALID_PAD) & (b != INVALID_PAD)
    a, b = a[m], b[m]
    n = int(m.sum())
    if n == 0:
        return {"cells": 0}
    labels = [OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH]
    conf = np.array([[int(((a == i) & (b == j)).sum()) for j in labels] for i in labels])
    fail_a, fail_b = np.isin(a, labels[2:]), np.isin(b, labels[2:])
    return {"cells": n, "exact_agreement": round(float(np.trace(conf) / n), 4),
            "cohen_kappa": round(kappa(conf), 4),
            "fail_vs_nonfail_agreement": round(float((fail_a == fail_b).mean()), 4),
            "confusion_rows_first_cols_second": {"labels": [CLASS_NAMES[c] for c in labels],
                                                 "matrix": conf.tolist()}}


def kappa(conf: np.ndarray) -> float:
    n = conf.sum()
    if n == 0:
        return float("nan")
    po = np.trace(conf) / n
    pe = float((conf.sum(1) * conf.sum(0)).sum()) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def confusion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    labels = [OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH]
    m = (a != INVALID_PAD) & (b != INVALID_PAD)
    return np.array([[int(((a == i) & (b == j) & m).sum()) for j in labels] for i in labels])


@dataclass
class FullRun:
    result: AuditResult
    hotspots: list
    summary: dict


def calibration_crops_for(ref: SegmentRef, calibration: dict | None) -> list[tuple[int, int, int]]:
    """L0 ``(y0, x0, size)`` of the calibration crops when the calibration used this segment."""
    if not calibration:
        return []
    seg = calibration.get("segment", {})
    if seg.get("sample") != ref.sample or seg.get("volume") != ref.volume:
        return []
    size = int(calibration.get("crop_size", 256))
    return [(int(c["y0"]), int(c["x0"]), size) for c in calibration.get("crops", [])]


def audit_full(ref: SegmentRef, thr: Thresholds, out_dir: Path = OUT_DIR, level: int = 3,
               l0_check_tiles: int = 24, cache: BoundedCache | None = None, preview: bool = True,
               mark_points: list[tuple[int, int, str]] | None = None, calibration: dict | None = None) -> FullRun:
    out_dir = Path(out_dir)
    cache = cache or BoundedCache()
    vol = LayersVolume(ref)
    tag = f"{output_tag(ref)}_full"
    try:
        t0 = time.perf_counter()
        print(f"full sweep {ref.sample} level {level} {vol.shape(level)} ...", flush=True)
        field, validity = sweep_level(vol, level)
        f = 2 ** level
        meta = {"sample": ref.sample, "segment": segment_record(ref)["long_id"], "volume": ref.volume,
                "asset": vol.cache_tag, "level": level, "lateral_cell_um": vol.lateral_um(level),
                "pixel_size_um": vol.pixel_size_um, "sweep_seconds": round(time.perf_counter() - t0, 1),
                "mode": "full pyramid sweep, 1 cell per level px, 1 px block halo, exact L0 validity",
                "exact_l0_validity": validity}
        res = audit_field(field, thr, origin_l0=(0, 0), cell_l0=f, meta=meta)
        hot = find_hotspots(res, top=500, min_cells=2)
        summary = save_audit(res, out_dir, tag, hotspots=hot)

        check = {}
        if l0_check_tiles:
            check = l0_cross_check(vol, res, thr, l0_check_tiles)
            summary["l0_tile_grid_cross_check"] = check
        points = []
        for y, x, label in mark_points or []:
            rep = res.cell_report(y, x)
            rep["label"] = label
            points.append(rep)
        summary["marked_points"] = points
        summary["top_hotspots"] = hot[:15]
        summary["hotspot_ranking"] = "connected regions of each FAIL class ranked by summed per-cell severity"
        (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        ink = None
        if preview:
            from .data import ink_preview_ds8
            ink = ink_preview_ds8(ref, cache)
        if ink is not None and ink.shape == res.cls.shape and ref.sample in REFERENCE_1UM_VOLUME and level == 3:
            from .data import reference_1um_on_grid
            ref8 = reference_1um_on_grid(ref, cache, res.cls.shape)
            summary["ink_reference_concordance"] = reference_concordance(
                res.cls, ink, ref8, exclude_l0=calibration_crops_for(ref, calibration))
            (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        if preview:
            from .viz import plot_full_overlay
            plot_full_overlay(res, ink, hot, out_dir / f"{tag}_qa_overlay.png", marks=mark_points or [],
                              calibration=calibration, check=check,
                              ref_conc=summary.get("ink_reference_concordance"))
        print(json.dumps({"class_counts": summary["class_counts"], "exact_l0_validity": validity,
                          "cross_check": {k: v for k, v in check.items() if k != "tiles"}}, indent=1), flush=True)
        return FullRun(res, hot, summary)
    finally:
        vol.close()


def reference_concordance(cls: np.ndarray, ink8: np.ndarray, ref8: np.ndarray, tile: int = 32, block_tiles: int = 4,
                          exclude_l0: list[tuple[int, int, int]] | None = None, n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """Does the QA map predict where the published 2.4 um ink map disagrees with the 1.129 um one?

    Both ink maps and ``cls`` are on the ds8 grid (1 px = 8 L0 px). For every fully valid
    ``tile`` x ``tile`` tile (256 L0 px) inside the reference coverage, Pearson r between the two
    ink maps is grouped by the tile's FAIL share (OFF_SHEET + SWITCH).

    Held out from the calibration: tiles are grouped into spatial blocks of ``block_tiles`` x
    ``block_tiles`` tiles (1024 L0 px), and every block that touches a calibration crop
    (``exclude_l0``: L0 ``(y0, x0, size)``) is dropped. CIs resample whole blocks (block bootstrap),
    so neighbouring, spatially correlated tiles never count as independent evidence."""
    from scipy.stats import spearmanr

    exclude_l0 = exclude_l0 or []
    bpx = tile * block_tiles
    gh, gw = cls.shape
    touched = set()
    direct = np.zeros((gh // tile + 1, gw // tile + 1), bool)
    for y0, x0, s in exclude_l0:
        y8a, y8b, x8a, x8b = y0 // 8, (y0 + s - 1) // 8, x0 // 8, (x0 + s - 1) // 8
        for by in range(y8a // bpx, y8b // bpx + 1):
            for bx in range(x8a // bpx, x8b // bpx + 1):
                touched.add((by, bx))
        direct[y8a // tile:y8b // tile + 1, x8a // tile:x8b // tile + 1] = True

    rows, blocks = [], []
    n_direct = n_block_excluded = 0
    for y in range(0, gh - tile + 1, tile):
        for x in range(0, gw - tile + 1, tile):
            c = cls[y:y + tile, x:x + tile]
            a = ink8[y:y + tile, x:x + tile].astype(float)
            b = ref8[y:y + tile, x:x + tile].astype(float)
            if (c == INVALID_PAD).any() or (b == 0).any() or a.std() < 5 or b.std() < 5:
                continue
            if direct[y // tile, x // tile]:
                n_direct += 1
            if (y // bpx, x // bpx) in touched:
                n_block_excluded += 1
                continue
            rows.append((np.corrcoef(a.ravel(), b.ravel())[0, 1], np.isin(c, FAILS).mean(),
                         (c == FAIL_OFF_SHEET).mean(), (c == WARN_DRIFT).mean()))
            blocks.append((y // bpx) * (gw // bpx + 1) + x // bpx)
    if len(rows) < 20:
        return {"skipped": f"only {len(rows)} held-out tiles inside the reference coverage"}
    R = np.array(rows)
    blk = np.array(blocks)
    ub, inv = np.unique(blk, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ub))]
    rng = np.random.default_rng(seed)
    boot_idx = [np.concatenate([members[k] for k in rng.integers(0, len(ub), len(ub))]) for _ in range(n_boot)]

    bins = [(0.0, 1e-9, "no FAIL cell"), (1e-9, 0.05, "FAIL < 5%"), (0.05, 0.20, "FAIL 5-20%"),
            (0.20, 1.01, "FAIL >= 20%")]

    def in_bin(rr, lo, hi):
        return (rr[:, 1] >= lo) & (rr[:, 1] < hi)

    def ci(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        return np.percentile(v, [2.5, 97.5]).round(4).tolist() if v.size else None

    out_bins = []
    for lo, hi, name in bins:
        m = in_bin(R, lo, hi)
        if not m.any():
            continue
        bm = [R[i][in_bin(R[i], lo, hi), 0].mean() if in_bin(R[i], lo, hi).any() else np.nan for i in boot_idx]
        out_bins.append({"bin": name, "n_tiles": int(m.sum()), "n_blocks": int(np.unique(blk[m]).size),
                         "mean_r": round(float(R[m, 0].mean()), 4), "ci95": ci(bm),
                         "median_r": round(float(np.median(R[m, 0])), 4)})

    def rho(rr, col):
        with warnings.catch_warnings():          # constant share column (e.g. no WARN tile): rho = NaN
            warnings.simplefilter("ignore")
            return float(spearmanr(rr[:, 0], rr[:, col]).statistic)

    def rho_entry(col):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = spearmanr(R[:, 0], R[:, col])
        st = float(s.statistic)
        return {"rho": round(st, 4) if np.isfinite(st) else None,
                "ci95_block_bootstrap": ci([rho(R[i], col) for i in boot_idx]),
                "p_iid_tiles": float(f"{s.pvalue:.3g}") if np.isfinite(s.pvalue) else None}

    diff = None
    clean, worst = R[:, 1] == 0, R[:, 1] >= 0.2
    if clean.any() and worst.any():
        d = []
        for i in boot_idx:
            rr = R[i]
            c_, w_ = rr[rr[:, 1] == 0, 0], rr[rr[:, 1] >= 0.2, 0]
            d.append(c_.mean() - w_.mean() if c_.size and w_.size else np.nan)
        diff = {"mean": round(float(R[clean, 0].mean() - R[worst, 0].mean()), 4), "ci95": ci(d)}
    return {"tile_l0_px": tile * 8, "block_l0_px": bpx * 8, "n_tiles": int(len(R)), "n_blocks": int(len(ub)),
            "held_out": {"calibration_crops": len(exclude_l0), "tiles_overlapping_calibration_crops": n_direct,
                         "tiles_excluded_by_block": n_block_excluded, "blocks_excluded": len(touched),
                         "rule": "drop every block that touches a calibration crop"},
            "ci_method": f"block bootstrap ({n_boot} draws of whole {bpx * 8} px blocks)",
            "by_fail_share": out_bins,
            "spearman_r_vs_fail_share": rho_entry(1),
            "spearman_r_vs_offsheet_share": rho_entry(2),
            "spearman_r_vs_warn_share": rho_entry(3),
            "mean_r_no_fail_minus_fail_ge_20pct": diff}


def l0_cross_check(vol: LayersVolume, coarse: AuditResult, thr: Thresholds, n_tiles: int, size: int = 256,
                   halo: int = 128, n_boundary: int = 8, lattice: int = 16, n_boot: int = 2000, seed: int = 0) -> dict:
    """Level-0 audits of a deterministic tile sample compared with the coarse (level-3) map.

    Sampling: tile origins on a ``lattice``-cell grid in raster order, the haloed crop inside the
    level; ``n_tiles`` interior tiles (fully valid in the coarse map) and ``n_boundary`` padding-
    boundary tiles (partly valid) at evenly spaced ranks (``np.linspace``) of each list, so reruns
    pick the same tiles. Each tile is audited at L0 on a ``size + 2 * halo`` crop and only its
    central ``size`` px are compared, so the L0 audit has the same lateral context (drift window,
    switch groups) as the coarse map. Interior tiles give the class agreement (kappa with a
    tile-bootstrap CI); boundary tiles give the padding agreement (exact L0 validity vs. the
    coarse valid mask), reported separately."""
    f = coarse.cell_l0
    if f != 8:
        return {"skipped": "cross-check needs an 8-px coarse cell (level 3)"}
    w, hw = size // 8, halo // 8
    gh, gw = coarse.cls.shape
    valid = coarse.cls != INVALID_PAD
    grid = [(y, x) for y in range(hw, gh - w - hw + 1, lattice) for x in range(hw, gw - w - hw + 1, lattice)]
    nvalid = {t: int(valid[t[0]:t[0] + w, t[1]:t[1] + w].sum()) for t in grid}
    interior = [t for t in grid if nvalid[t] == w * w]
    boundary = [t for t in grid if 0 < nvalid[t] < w * w]

    def pick(lst, k):
        return [lst[i] for i in np.unique(np.linspace(0, len(lst) - 1, min(k, len(lst))).round().astype(int))] if lst else []

    def run(t):
        cy, cx = t
        res0, slab = audit_crop(vol, cy * 8 - halo, cx * 8 - halo, thr, size=size + 2 * halo, cache=None)
        sl = (slice(hw, hw + w), slice(hw, hw + w))
        l0_valid = pool_all((slab != 0).any(axis=0), 8)[sl]
        return res0.cls[sl], l0_valid, coarse.cls[cy:cy + w, cx:cx + w], valid[cy:cy + w, cx:cx + w]

    tiles, confs = [], []
    pad_violations = 0
    for t in pick(interior, n_tiles):
        a, l0v, b, cv = run(t)
        cf = confusion(a, b)
        confs.append(cf)
        pad_violations += int((cv & ~l0v).sum())
        tiles.append({"origin_l0_yx": [t[0] * 8, t[1] * 8], "kind": "interior", "kappa": round(kappa(cf), 4),
                      "exact_agreement": round(float(np.trace(cf) / max(cf.sum(), 1)), 4),
                      "l0_fail_share": round(float(np.isin(a, FAILS).mean()), 4),
                      "coarse_fail_share": round(float(np.isin(b, FAILS).mean()), 4)})
    if not confs:
        return {"skipped": "no fully valid tiles"}
    C = np.sum(confs, axis=0)
    out = concordance_from(C)
    rng = np.random.default_rng(seed)
    kb = [kappa(np.sum([confs[i] for i in rng.integers(0, len(confs), len(confs))], axis=0)) for _ in range(n_boot)]
    out["cohen_kappa_ci95_tile_bootstrap"] = np.percentile(kb, [2.5, 97.5]).round(4).tolist()
    out["n_tiles"] = len(confs)
    out["interior_cells_coarse_valid_but_l0_padding"] = pad_violations

    pad = np.zeros((2, 2), int)          # rows: L0 exact valid (no, yes); cols: coarse valid (no, yes)
    bconfs = []
    for t in pick(boundary, n_boundary):
        a, l0v, b, cv = run(t)
        for i in (0, 1):
            for j in (0, 1):
                pad[i, j] += int(((l0v == bool(i)) & (cv == bool(j))).sum())
        bconfs.append(confusion(a, b))
        tiles.append({"origin_l0_yx": [t[0] * 8, t[1] * 8], "kind": "padding_boundary",
                      "coarse_valid_cells": int(cv.sum()), "l0_valid_cells": int(l0v.sum()),
                      "valid_mask_disagreement": int((l0v != cv).sum())})
    out["padding_boundary"] = {
        "n_tiles": len(bconfs), "cells": int(pad.sum()),
        "valid_mask_confusion": {"rows_l0_exact_valid": [False, True], "cols_coarse_valid": [False, True],
                                 "matrix": pad.tolist()},
        "valid_mask_agreement": round(float(np.trace(pad) / max(pad.sum(), 1)), 4),
        "class_agreement_on_cells_valid_in_both": concordance_from(np.sum(bconfs, axis=0)) if bconfs else None,
    }
    out["sampling"] = (f"deterministic: tile origins every {lattice} cells in raster order, haloed crop inside the "
                       f"level; evenly spaced ranks of the {len(interior)} interior and {len(boundary)} boundary "
                       "candidates")
    out["note"] = (f"rows: L0 {size + 2 * halo} px crop audited at 8x8 px cells, central {size} px compared "
                   f"({halo} px halo trimmed); columns: level-3 full sweep (1 px = 8x8 L0)")
    out["tiles"] = tiles
    return out


def concordance_from(conf: np.ndarray) -> dict:
    labels = [OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH]
    n = int(conf.sum())
    if n == 0:
        return {"cells": 0}
    fail_agree = conf[:2, :2].sum() + conf[2:, 2:].sum()
    return {"cells": n, "exact_agreement": round(float(np.trace(conf) / n), 4), "cohen_kappa": round(kappa(conf), 4),
            "fail_vs_nonfail_agreement": round(float(fail_agree / n), 4),
            "confusion_rows_first_cols_second": {"labels": [CLASS_NAMES[c] for c in labels],
                                                 "matrix": conf.tolist()}}


# ------------------------------------------------------------------------------- verification
@dataclass(frozen=True)
class Case:
    sample: str
    row: str
    hotspot_yx: tuple[int, int]
    crop_yx: tuple[int, int]
    section_axis: str            # "x": XZ cut through the hotspot row; "y": YZ cut through its column
    note: str
    expected: tuple[int, ...]                 # acceptable classes of the hotspot cell
    expected_reasons: tuple[str, ...] = ()    # reason bits the hotspot cell must carry
    pair: tuple[tuple[int, int], tuple[int, int]] | None = None   # adjacent L0 points of a documented step
    pair_kind: str | None = None              # expected step type: "switch" or "flip"
    priority_controls: tuple[tuple[int, int], ...] = ()


CASES = (
    Case("PHerc0139", "B", (12620, 15708), (12544, 15616), "x",
         "Sol Row B: suspected sheet switch, midpoint 66.24 -> 24.63 (41.61 vx) across 8 X px",
         (FAIL_SHEET_SWITCH, FAIL_OFF_SHEET), ("LATERAL_JUMP",),
         pair=((12620, 15708), (12620, 15716)), pair_kind="switch", priority_controls=((20608, 19328),)),
    Case("PHerc1667", "G", (18980, 15740), (18944, 15616), "x",
         "Sol Row G: competing bands, midpoint 32.53 -> 83.78 (51.25 vx) across 8 X px",
         (WARN_DRIFT,), ("COMPETING_BANDS",),
         pair=((18980, 15740), (18980, 15748)), pair_kind="flip"),
    Case("PHercParis4", "I", (24484, 93396), (24320, 93312), "y",
         "Sol Row I: band ends 39.85 vx (95.64 um) short of z=54, 62-layer window empty; "
         "76.51 vx handoff at (24468, 93388) -> (24476, 93388)",
         (FAIL_OFF_SHEET,), ("WINDOW_EMPTY",),
         pair=((24468, 93388), (24476, 93388)), pair_kind="flip"),
)
CONTROL_MAX_FAIL = 0.02


def fail_share(res: AuditResult) -> float:
    v = res.cls != INVALID_PAD
    return float(np.isin(res.cls[v], FAILS).mean()) if v.any() else float("nan")


def pair_check(res: AuditResult, a: tuple[int, int], b: tuple[int, int], thr: Thresholds,
               expected_kind: str | None = None) -> dict:
    """Explicit test of a documented step between two 4-adjacent cells.

    ``kind``: "flip" when the two selected bands are disjoint and both continue across the edge
    (exact interval test and prefilter), "switch" when they are disjoint and at least one ends,
    "same_sheet" when they overlap. ``flagged``: the step reaches ``switch_jump_um`` across
    disjoint bands and both cells carry the matching reason (COMPETING_BANDS or LATERAL_JUMP)."""
    ia, ib = res.cell_of(*a), res.cell_of(*b)
    if abs(ia[0] - ib[0]) + abs(ia[1] - ib[1]) != 1:
        raise ValueError(f"{a} and {b} are not 4-adjacent cells")
    f = res.field
    lo, hi = (ia, ib) if ia <= ib else (ib, ia)
    edge = "down" if lo[0] != hi[0] else "right"
    cont = bool(getattr(f, f"cont_{edge}")[lo])
    pre = bool(getattr(f, f"pre_{edge}")[lo])
    da, db = f.delta_z_um[ia], f.delta_z_um[ib]
    step = abs(float(da - db)) if np.isfinite(da) and np.isfinite(db) else None
    overlap = float(min(f.z_bot[ia], f.z_bot[ib]) - max(f.z_top[ia], f.z_top[ib]))
    disjoint = bool(np.isfinite(overlap) and overlap < 0)
    kind = "same_sheet" if not disjoint else ("flip" if cont and pre else "switch")
    ra, rb = res.cell_report(*a), res.cell_report(*b)
    reason = "COMPETING_BANDS" if kind == "flip" else "LATERAL_JUMP"
    flagged = bool(step is not None and step >= thr.switch_jump_um and disjoint
                   and reason in ra["reasons"] and reason in rb["reasons"])
    return {"a_l0_yx": list(a), "b_l0_yx": list(b), "step_um": None if step is None else round(step, 2),
            "switch_jump_um": thr.switch_jump_um, "bands_disjoint": disjoint,
            "band_overlap_vx": round(overlap, 2) if np.isfinite(overlap) else None,
            "both_continue_exact": cont, "both_continue_prefilter": pre, "sheet_id_uncertain": bool(pre and not cont),
            "kind": kind, "expected_kind": expected_kind, "flagged": flagged,
            "passed": flagged and (expected_kind is None or kind == expected_kind),
            "a": {k: ra[k] for k in ("class_name", "reasons", "z_top", "z_bot", "delta_z_um")},
            "b": {k: rb[k] for k in ("class_name", "reasons", "z_top", "z_bot", "delta_z_um")}}


def control_candidates(ref: SegmentRef, vol: LayersVolume, hotspot: tuple[int, int], cache: BoundedCache,
                       radius: int | None, n: int, size: int = 256, exclude: int = 1024) -> list:
    """Text-rich crops of the published ds8 ink preview (legible text implies an on-sheet surface),
    ranked a priori by text richness and independent of the auditor; ``radius=None``: whole segment."""
    from .data import ink_preview_ds8
    from .ink_tolerance import select_crops

    ink = ink_preview_ds8(ref, cache).astype(np.float32)
    _, H, W = vol.shape(0)
    if radius is None:
        ys = xs = 0
        win = ink[:H // 8, :W // 8]
    else:
        y8, x8, r8 = hotspot[0] // 8, hotspot[1] // 8, radius // 8
        ys, xs = max(0, y8 - r8) // 16 * 16, max(0, x8 - r8) // 16 * 16
        win = ink[ys:min(H // 8, y8 + r8), xs:min(W // 8, x8 + r8)]
    cands = select_crops(win, ndi.binary_erosion(win > 0, iterations=2), n * 3, size=size, min_sep=size * 2)
    out = []
    for c in cands:
        y0, x0 = ys * 8 + c.y0, xs * 8 + c.x0
        if max(abs(y0 - hotspot[0]), abs(x0 - hotspot[1])) < exclude:
            continue
        out.append(((y0, x0), c))
        if len(out) >= n:
            break
    return out


def pick_control(ref: SegmentRef, vol: LayersVolume, hotspot: tuple[int, int], thr: Thresholds,
                 cache: BoundedCache, priority=(), radius: int = 8192, tries: int = 24, size: int = 256,
                 max_fail: float = CONTROL_MAX_FAIL):
    """First candidate, in a fixed a-priori order, whose CT audit is clean (FAIL share < ``max_fail``).

    Order: ``priority`` crops (e.g. Sol's centred Row D control), then the text-richest preview crops
    within ``radius`` of the hotspot, then over the whole segment. The CT audit is the acceptance
    check, so a control certifies "clean region exists and the auditor calls it clean", not an
    independent specificity estimate. Returns ``(yx, info, result, slab)``."""
    tried = []
    order = [(tuple(p), "priority (a-priori control)", None) for p in priority]
    order += [(yx, f"preview text rank within {radius} px", c)
              for yx, c in control_candidates(ref, vol, hotspot, cache, radius, tries, size)]
    order += [(yx, "preview text rank, whole segment", c)
              for yx, c in control_candidates(ref, vol, hotspot, cache, None, tries, size)]
    seen = set()
    best = None
    for yx, src, c in order:
        if yx in seen:
            continue
        seen.add(yx)
        try:
            res, slab = audit_crop(vol, *yx, thr, size=size, cache=None)
        except ValueError:
            continue
        if not (res.cls != INVALID_PAD).all():
            continue
        share = fail_share(res)
        tried.append({"origin_l0": list(yx), "source": src, "fail_share": round(share, 4),
                      "ink_text_score": None if c is None else round(c.score, 5)})
        if best is None or share < best[0]:
            best = (share, yx, res, slab)
        if share < max_fail:
            break
    if best is None:
        raise RuntimeError(f"no valid control crop near {hotspot}")
    share, yx, res, slab = best
    read_crop(vol, *yx, size, cache=cache)       # keep the chosen control's CT for the figure / reruns
    info = {"criterion": (f"first crop in a fixed a-priori order (priority controls, then published ds8 ink-preview "
                          f"text richness within {radius} px, then the whole segment; fully valid CT, >= 1024 px "
                          f"from the hotspot) whose L0 CT audit has FAIL < {max_fail:.0%}"),
            "clean": bool(share < max_fail), "rank": len(tried), "candidates_tried": tried}
    return yx, info, res, slab


def verify(thr: Thresholds, out_dir: Path = OUT_DIR, cache: BoundedCache | None = None) -> dict:
    from .viz import plot_hotspot_verification

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = cache or BoundedCache()
    report = {"thresholds": thr.to_dict(), "control_max_fail_share": CONTROL_MAX_FAIL, "cases": []}
    panels = []
    for case in CASES:
        ref = SEGMENTS[case.sample]
        vol = LayersVolume(ref)
        try:
            hot, hot_slab = audit_crop(vol, *case.crop_yx, thr, cache=cache)
            ctrl_yx, ctrl_info, ctrl, ctrl_slab = pick_control(ref, vol, case.hotspot_yx, thr, cache,
                                                               priority=case.priority_controls)
            # coarse (level-3) audit of the same areas for pyramid concordance (exact L0 validity)
            l3_hot, _ = audit_crop(vol, case.crop_yx[0] // 8, case.crop_yx[1] // 8, thr, size=32, level=3,
                                   cell=1, cache=cache)
            l3_ctrl, _ = audit_crop(vol, ctrl_yx[0] // 8, ctrl_yx[1] // 8, thr, size=32, level=3, cell=1,
                                    cache=None)
        finally:
            vol.close()
        tag = f"verify_{output_tag(ref)}_{case.row}"
        hot_hs = find_hotspots(hot, top=20)
        save_audit(hot, out_dir / "verification", f"{tag}_hotspot", hotspots=hot_hs)
        save_audit(ctrl, out_dir / "verification", f"{tag}_control")
        cell = hot.cell_report(*case.hotspot_yx)
        pair = pair_check(hot, *case.pair, thr, case.pair_kind) if case.pair else None
        missing = [r for r in case.expected_reasons if r not in cell["reasons"]]
        ctrl_counts = ctrl.class_counts()
        hot_counts = hot.class_counts()
        flagged = cell["class"] in case.expected and not missing
        entry = {
            "sample": case.sample, "row": case.row, "note": case.note, "segment": hot.meta["segment"],
            "asset": vol.cache_tag,
            "hotspot_crop_origin_l0": list(case.crop_yx), "hotspot_cell": cell,
            "hotspot_flagged": flagged,
            "expected_classes": [CLASS_NAMES[c] for c in case.expected],
            "expected_reasons": list(case.expected_reasons), "missing_reasons": missing,
            "pair_check": pair,
            "passed": bool(flagged and (pair is None or pair["passed"]) and ctrl_info["clean"]),
            "hotspot_crop_class_counts": hot_counts, "hotspot_crop_top_hotspots": hot_hs[:5],
            "control_origin_l0": list(ctrl_yx), "control_selection": ctrl_info,
            "control_class_counts": ctrl_counts,
            "control_fail_share": round(fail_share(ctrl), 4),
            "hotspot_crop_fail_share": round(fail_share(hot), 4),
            "level3_concordance": {"hotspot_crop": concordance(hot.cls, l3_hot.cls),
                                   "control_crop": concordance(ctrl.cls, l3_ctrl.cls),
                                   "hotspot_cell_level3_class": CLASS_NAMES[int(l3_hot.cls[hot.cell_of(*case.hotspot_yx)])]},
        }
        report["cases"].append(entry)
        panels.append((case, hot, hot_slab, ctrl, ctrl_slab, entry))
        print(f"{case.sample} {case.row}: hotspot {cell['class_name']} {cell['reasons']} | pair "
              f"{None if pair is None else (pair['kind'], pair['step_um'], pair['passed'])} | crop fail "
              f"{entry['hotspot_crop_fail_share']:.1%} | control {ctrl_yx} fail {entry['control_fail_share']:.1%} "
              f"(rank {ctrl_info['rank']}) | passed {entry['passed']}", flush=True)
    report["all_passed"] = all(c["passed"] for c in report["cases"])
    (out_dir / "hotspot_verification.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    plot_hotspot_verification(panels, thr, out_dir / "hotspot_verification.png")
    return report
