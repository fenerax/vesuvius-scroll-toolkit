"""Empirical ``ink_canonical_2um`` drift-tolerance calibration.

For each of N text-bearing 256 x 256 crops of PHerc0139 w023 (2.399 um layers-zarr), the
official model (ResNet3D-152 + 3D decoder, ``new_canon_autoresearch_recipe``: 62 layers
``[23 + dz, 85 + dz)``, clip to 200 and divide by 200, FP32, batch 1) is run once per
window shift ``dz``. Each prediction is scored against

* the independent published 1.129 um-scan ink map (different scan, different model
  ``mrg20736``), resized onto the 2.4 um ds8 grid: Pearson r on the crop's 32 x 32 ds8 cells;
* the crop's own ``dz = 0`` prediction (self-consistency r at full resolution);
* foreground retention: share of the ``dz = 0`` pixels with p > 0.5 that stay above 0.5.

Crops are chosen only from the independent reference's text richness (not from the model
being tested), so the ``dz = 0`` optimum is not built into the selection.

Thresholds (rule fixed before the run):

* **safe**: contiguous shifts around 0 whose bootstrap 95 % lower bound of the agreement
  retention ``mean_c r_c(dz) / mean_c r_c(0)`` stays >= ``safe_retention`` (0.90);
* **fail**: the shift, linearly interpolated between grid points, where the point estimate
  of the retention first falls below ``fail_retention`` (0.70) on each side. A side whose
  retention never falls below 0.70 inside the measured support is **censored**: its edge is
  the end of the data, flagged ``crossed = False``, and the auditor treats drift beyond it as a
  warning (``DRIFT_BEYOND_MEASURED``), not as an empirical failure;
* **warn**: everything between the safe edge and the fail threshold.

All thresholds are **provisional** outside the calibration's own scope: one PHerc0139 segment,
one scan, and coherent shifts of whole 256 px crops. Their transfer to other scrolls and to
8 px cells (5 x 5-cell medians) is assumed, not measured (``thresholds.provisional_scope``).

Offsets are measured from the nominal anchor ``z_nom = 54`` (the mesh layer), not from the
window midpoint 53.5 of ``[23, 85)``; see ``band_detector``.

Window shift ``dz`` and band drift are mirror images: moving the window by ``+dz`` places the
band at ``-dz`` relative to the window centre. The crops' own band offsets at ``dz = 0`` spread
over tens of micrometres, so the auditor's thresholds (``thresholds.band_frame``) come from
``band_offset_curve``: agreement against the band position each prediction actually saw
(crop offset minus ``dz``), with a per-crop fixed effect. The mirrored window ranges are kept
for comparison as ``band_frame_window_mirror``.

``calibrate`` runs the GPU measurement and writes ``ink_tolerance_raw.npz`` and
``ink_tolerance_crops.json``; ``reanalyze`` recomputes every statistic and the figure from them.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .band_detector import BandParams, detect_bands
from .data import (MODELS_DIR, OUT_DIR, ROOT, SEGMENTS, BoundedCache, LayersVolume, read_crop,
                   reference_1um_on_grid, segment_record)

CKPT_PATH = MODELS_DIR / "r152_3ddec_v2_l5_epoch13.ckpt"
VENDOR = ROOT / "scripts" / "vendor"
START, END = 23, 85
DEPTH = 109
MAX_CLIP = 200.0
TILE = 256
BIN_WIDTHS_UM = (7.2, 9.6, 12.0)       # band-offset bin-width sensitivity (9.6 um = 4 layers is the primary)
PROVISIONAL_SCOPE = [
    "cross-scroll: measured on one PHerc0139 segment (one 2.399 um scan, one ink model); applying the "
    "thresholds to PHerc1667, PHercParis4 or other scans assumes the same tolerance and is untested",
    "crop-to-cell: measured for a coherent shift of whole 256 px (614 um) crops; the auditor applies them "
    "to same-sheet 5 x 5-cell (96 um) medians of 8 px cells, a scale the calibration did not test",
]


def validate_shifts(shifts, depth: int = DEPTH) -> list[int]:
    """Integer window shifts: strictly increasing (sorted, unique), containing 0, inside the stack."""
    out = []
    for s in shifts:
        fs = float(s)
        if not np.isfinite(fs) or fs != int(fs):
            raise ValueError(f"shift {s!r} is not a finite integer number of layers")
        out.append(int(fs))
    if not out:
        raise ValueError("no shifts given")
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(f"shifts must be sorted and unique, got {out}")
    if 0 not in out:
        raise ValueError("shifts must include 0")
    if out[0] + START < 0 or out[-1] + END > depth:
        raise ValueError(f"shifted window leaves the {depth}-layer stack: allowed {-START}..{depth - END}")
    return out


# the requested grid -16..+16 plus -20/+20/+24 to locate the cliff (the stack allows -23..+24)
SHIFTS = (-20, -16, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12, 16, 20, 24)
SAFE_RETENTION = 0.90
FAIL_RETENTION = 0.70
# detector-noise rules, fixed from physical scale before any hotspot was audited:
# a disjoint-band handoff of >= 40 um (~0.63 of the 63.5 um median wrap pitch), in a group of
# >= 4 cells (a 2-cell-long edge seen from both sides), and drift as the median of 5 x 5 cells
# (40 L0 px = 96 um, the scale of a letter stroke group)
SWITCH_JUMP_UM = 40.0
SWITCH_MIN_CELLS = 4
DRIFT_WINDOW_CELLS = 5


class InkModel:
    """Local ``ink_canonical_2um`` (FP32, batch 1). One 256 x 256 tile -> 256 x 256 probabilities."""

    def __init__(self, ckpt: Path = CKPT_PATH):
        import torch

        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from model_resnet3d_3d_decoder import load_model

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wrapper = load_model(str(ckpt), device=self.device, num_frames=END - START)
        self.wrapper.eval()
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu"

    def predict(self, vol62: np.ndarray) -> np.ndarray:
        import torch.nn.functional as F

        torch = self.torch
        if vol62.shape != (END - START, TILE, TILE):
            raise ValueError(f"expected (62, 256, 256), got {vol62.shape}")
        x = np.clip(vol62.astype(np.float32), 0, MAX_CLIP) / MAX_CLIP
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(self.device, torch.float32)
            p = torch.sigmoid(self.wrapper.forward(t))
            p = F.interpolate(p.float(), size=(TILE, TILE), mode="bilinear", align_corners=False)
        return p[0, 0].cpu().numpy()


def ds8(p: np.ndarray) -> np.ndarray:
    h, w = p.shape
    return p.reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Crop:
    y0: int
    x0: int
    score: float
    ref_mean: float
    ref_std: float


def select_crops(ref8: np.ndarray, valid8: np.ndarray, n: int, size: int = TILE, min_sep: int = 1024,
                 align: int = 128) -> list[Crop]:
    """Greedy, spatially diverse text-rich crops chosen from the *independent* reference only.

    ``ref8`` is the 1.129 um ink map on the 2.4 um ds8 grid (0..255); ``valid8`` marks ds8
    pixels with valid CT. Text richness = std x mean absolute gradient of the 32 x 32 patch,
    with the patch mean restricted to 0.08..0.65 (strokes on background, not blank or saturated)."""
    w = size // 8
    step = align // 8
    a = ref8 / 255.0
    gy = np.abs(np.diff(a, axis=0, append=a[-1:]))
    gx = np.abs(np.diff(a, axis=1, append=a[:, -1:]))
    cands = []
    h8, w8 = a.shape
    for r in range(0, h8 - w + 1, step):
        for c in range(0, w8 - w + 1, step):
            if not valid8[r:r + w, c:c + w].all():
                continue
            patch = a[r:r + w, c:c + w]
            if (patch <= 2 / 255).mean() > 0.0:        # outside the reference's coverage
                continue
            m, s = float(patch.mean()), float(patch.std())
            if not (0.08 <= m <= 0.65):
                continue
            g = float(gy[r:r + w, c:c + w].mean() + gx[r:r + w, c:c + w].mean())
            cands.append(Crop(r * 8, c * 8, s * g, m, s))
    cands.sort(key=lambda k: -k.score)
    chosen: list[Crop] = []
    for k in cands:
        if all(max(abs(k.y0 - o.y0), abs(k.x0 - o.x0)) >= min_sep for o in chosen):
            chosen.append(k)
        if len(chosen) >= n:
            break
    return chosen


def bootstrap_mean_ci(x: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    m = np.nanmean(x[idx], axis=1)
    return float(np.nanpercentile(m, 2.5)), float(np.nanpercentile(m, 97.5))


def retention_stats(r: np.ndarray, shifts, n_boot: int = 4000, seed: int = 0) -> dict:
    """Agreement retention ``mean_c r(dz) / mean_c r(0)`` with a paired crop bootstrap."""
    shifts = list(shifts)
    k0 = shifts.index(0)
    rng = np.random.default_rng(seed)
    c = r.shape[0]
    idx = rng.integers(0, c, size=(n_boot, c))
    means = np.nanmean(r[idx], axis=1)            # (B, K)
    ratio = means / means[:, [k0]]
    point = np.nanmean(r, axis=0) / np.nanmean(r[:, k0])
    return {
        "point": point,
        "lo": np.nanpercentile(ratio, 2.5, axis=0),
        "hi": np.nanpercentile(ratio, 97.5, axis=0),
    }


def _crossing(shifts, values, level, k0, direction):
    """Interpolated shift where ``values`` first drops below ``level`` walking from k0."""
    k = k0
    while 0 <= k + direction < len(shifts):
        nk = k + direction
        if values[nk] < level:
            v0, v1 = values[k], values[nk]
            t = (v0 - level) / (v0 - v1) if v0 != v1 else 0.0
            return shifts[k] + t * (shifts[nk] - shifts[k]), True
        k = nk
    return float(shifts[-1] if direction > 0 else shifts[0]), False


def derive_thresholds(shifts, point, lo, pixel_size_um: float, reference_offset_um: float,
                      safe_retention: float = SAFE_RETENTION, fail_retention: float = FAIL_RETENTION) -> dict:
    """Window-frame safe/warn/fail shifts and their mirrored band-frame ranges (um)."""
    shifts = [float(s) for s in shifts]
    k0 = shifts.index(0.0)
    safe = [0.0, 0.0]
    for direction, slot in ((-1, 0), (1, 1)):
        k = k0
        while 0 <= k + direction < len(shifts) and lo[k + direction] >= safe_retention:
            k += direction
        safe[slot] = shifts[k]
    fail_neg, reached_neg = _crossing(shifts, point, fail_retention, k0, -1)
    fail_pos, reached_pos = _crossing(shifts, point, fail_retention, k0, +1)
    fail_neg = min(fail_neg, safe[0])
    fail_pos = max(fail_pos, safe[1])
    px = pixel_size_um
    win = {
        "safe_dz_vx": [safe[0], safe[1]],
        "warn_dz_vx": [round(fail_neg, 3), round(fail_pos, 3)],
        "fail_dz_vx": {"neg": round(fail_neg, 3), "pos": round(fail_pos, 3),
                       "neg_reached_within_sweep": reached_neg, "pos_reached_within_sweep": reached_pos},
    }
    win_um = {
        "safe_dz_range_um": [round(safe[0] * px, 2), round(safe[1] * px, 2)],
        "warn_dz_range_um": [round(fail_neg * px, 2), round(fail_pos * px, 2)],
        "fail_dz_um": {"neg": round(fail_neg * px, 2), "pos": round(fail_pos * px, 2)},
    }
    band = {
        "safe_drift_um": [round(-safe[1] * px, 2), round(-safe[0] * px, 2)],
        "fail_drift_um": [round(-fail_pos * px, 2), round(-fail_neg * px, 2)],
        # nominal slice in air: the nearest band face is at least this far away -> off-sheet
        "air_gap_fail_um": round(min(abs(fail_neg), abs(fail_pos)) * px, 2),
    }
    return {
        "rule": {"safe": f"bootstrap 95% lower bound of agreement retention >= {safe_retention}",
                 "fail": f"point retention < {fail_retention} (linear interpolation between shifts)"},
        "window_frame": {**win, **win_um},
        "band_frame": band,
        "reference_offset_um": round(float(reference_offset_um), 2),
        "frame_note": ("window shift dz reads layers [23+dz, 85+dz); the band then sits at -dz relative "
                       "to the window centre, so band_frame = -window_frame. drift_um = delta_z_um - "
                       "reference_offset_um."),
    }


def fit_offset_curve(r: np.ndarray, b: np.ndarray, nb: int, min_pairs: int, iters: int = 50):
    """Crop fixed effect + offset-bin effect by alternating means, guarded against sparse bins.

    Pairs in bins with fewer than ``min_pairs`` finite pairs do not enter the fit at all, so a
    sparse edge bin cannot pull the crop effects; those bins come back as NaN. Returns
    ``(curve, counts, used)``; ``curve = mean(a) + g`` (invariant to the a/g constant)."""
    fin = np.isfinite(r)
    cnt = np.bincount(b[fin], minlength=nb)
    use = cnt >= min_pairs
    w = fin & use[b]
    if not use.any():
        return np.full(nb, np.nan), cnt, use
    rows = w.any(axis=1)
    wf = w.astype(float)
    rr = np.where(w, r, 0.0)
    g = np.zeros(nb)
    a = np.zeros(r.shape[0])
    for _ in range(iters):
        a = np.where(rows, ((rr - g[b]) * wf).sum(1) / np.maximum(wf.sum(1), 1), np.nan)
        res = np.where(w, rr - np.nan_to_num(a)[:, None], 0.0)
        num = np.bincount(b.ravel(), weights=res.ravel(), minlength=nb)
        den = np.bincount(b.ravel(), weights=wf.ravel(), minlength=nb)
        g = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return np.where(use, np.nanmean(a) + g, np.nan), cnt, use


def support_run(use: np.ndarray, k: int) -> np.ndarray:
    """Indices of the contiguous run of supported bins containing bin ``k``."""
    lo = hi = k
    while lo - 1 >= 0 and use[lo - 1]:
        lo -= 1
    while hi + 1 < len(use) and use[hi + 1]:
        hi += 1
    return np.arange(lo, hi + 1)


def band_offset_curve(r: np.ndarray, offsets_um: np.ndarray, shifts, pixel_size_um: float,
                      bin_um: float = 9.6, min_pairs: int = 20, n_boot: int = 2000, seed: int = 0,
                      safe_retention: float = SAFE_RETENTION, fail_retention: float = FAIL_RETENTION) -> dict:
    """Agreement as a function of the band position the model actually saw.

    In crop ``i`` at shift ``k`` the band sits at ``e_ik = offset_i - dz_k * px`` relative to the
    nominal anchor (``offset_i``: the crop's median band offset at ``dz = 0``). The crop's text
    content sets the level of r, so ``r_ik = a_i + g(bin(e_ik)) + noise`` is fitted by
    alternating means (``fit_offset_curve``; bins with < ``min_pairs`` pairs are excluded).

    * Support: the contiguous run of supported bins around the best bin; every walk stops at
      the first unsupported bin.
    * Retention is relative to the best bin. The crop-paired bootstrap re-selects the best bin
      in every replicate (the peak is not fixed to the data-selected bin), so the CIs include
      the uncertainty of the peak location (``peak_bootstrap``).
    * Safe = contiguous bins around the peak with point retention >= ``safe_retention``; the
      "safe_ci" core uses the bootstrap 95 % lower bound instead. Fail = linear crossing of
      ``fail_retention``; a side without a crossing inside the support is censored at the
      support edge (``*_reached_within_data = False``).
    """
    shifts = np.asarray(shifts, float)
    ok = np.isfinite(offsets_um)
    r, off = r[ok], offsets_um[ok]
    e = off[:, None] - shifts[None, :] * pixel_size_um
    lo_edge = np.floor(np.nanmin(e) / bin_um) * bin_um
    edges = np.arange(lo_edge, np.nanmax(e) + bin_um, bin_um)
    b = np.clip(np.digitize(e, edges) - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    centers = edges[:-1] + bin_um / 2

    curve, cnt, use = fit_offset_curve(r, b, nb, min_pairs)
    if not use.any():
        raise ValueError(f"no band-offset bin has >= {min_pairs} pairs")
    peak = int(np.nanargmax(np.where(use, curve, np.nan)))
    run = support_run(use, peak)
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, nb), np.nan)
    peaks = np.full(n_boot, -1)
    for i in range(n_boot):
        pick = rng.integers(0, len(r), len(r))
        cb = fit_offset_curve(r[pick], b[pick], nb, min_pairs)[0]
        boots[i] = cb
        sub = cb[run]
        if np.isfinite(sub).any():
            peaks[i] = run[int(np.nanargmax(sub))]
    good = peaks >= 0
    ratio = np.full_like(boots, np.nan)
    ratio[good] = boots[good] / boots[good, peaks[good]][:, None]
    point = curve / curve[peak]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rlo, rhi = np.nanpercentile(ratio, 2.5, axis=0), np.nanpercentile(ratio, 97.5, axis=0)
            clo, chi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    n_valid_boot = np.isfinite(boots).sum(0)

    xs, pt, lo = centers[run], point[run], rlo[run]
    k0 = int(np.flatnonzero(run == peak)[0])

    def walk(values, level):
        out = [xs[k0], xs[k0]]
        for direction, slot in ((-1, 0), (1, 1)):
            k = k0
            while 0 <= k + direction < len(xs) and np.isfinite(values[k + direction]) \
                    and values[k + direction] >= level:
                k += direction
            out[slot] = xs[k]
        # bins are labelled by their centre; a safe range covers whole bins
        return [round(float(out[0] - bin_um / 2), 2), round(float(out[1] + bin_um / 2), 2)]

    safe_ci = walk(lo, safe_retention)
    safe_rng = walk(pt, safe_retention)
    fneg, rneg = _crossing(list(xs), pt, fail_retention, k0, -1)
    fpos, rpos = _crossing(list(xs), pt, fail_retention, k0, +1)
    if not rneg:   # no crossing inside the support: censored at the edge of the evidence
        fneg = xs[0] - bin_um / 2
    if not rpos:
        fpos = xs[-1] + bin_um / 2
    fail_rng = [round(float(min(fneg, safe_rng[0])), 2), round(float(max(fpos, safe_rng[1])), 2)]
    pk = centers[peaks[good]]
    uniq, freq = np.unique(pk, return_counts=True)
    in_run = np.zeros(nb, bool)
    in_run[run] = True
    return {
        "method": ("r_ik = a_i + g(band offset bin) fitted by alternating means (bins with < min_pairs pairs "
                   "excluded from the fit); e_ik = crop median band offset - dz*px from the nominal anchor "
                   "z_nom = 54; crop-paired bootstrap with the best bin re-selected in every replicate"),
        "bin_um": bin_um, "min_pairs": min_pairs, "n_crops": int(len(r)), "n_boot": int(n_boot),
        "support_um": [round(float(xs[0] - bin_um / 2), 2), round(float(xs[-1] + bin_um / 2), 2)],
        "bins": [{"center_um": round(float(c), 2), "n_pairs": int(n), "r": None if not np.isfinite(v) else round(float(v), 4),
                  "r_ci95": [None if not np.isfinite(a_) else round(float(a_), 4),
                             None if not np.isfinite(z_) else round(float(z_), 4)],
                  "retention": None if not np.isfinite(q) else round(float(q), 4),
                  "retention_ci95": [None if not np.isfinite(a2) else round(float(a2), 4),
                                     None if not np.isfinite(z2) else round(float(z2), 4)],
                  "boot_replicates_supported": int(nv), "used": bool(u)}
                 for c, n, v, a_, z_, q, a2, z2, nv, u in zip(centers, cnt, curve, clo, chi, point, rlo, rhi,
                                                              n_valid_boot, in_run)],
        "peak_center_um": round(float(centers[peak]), 2),
        "peak_bootstrap": {"ci95_um": np.percentile(pk, [2.5, 97.5]).round(2).tolist() if pk.size else None,
                           "share_by_bin_um": {f"{u_:.2f}": round(float(f_) / max(pk.size, 1), 4)
                                               for u_, f_ in zip(uniq, freq)},
                           "share_at_point_peak": round(float((peaks[good] == peak).mean()), 4) if pk.size else None},
        "rule": {"safe": f"contiguous supported bins around the best bin with point retention >= {safe_retention}",
                 "safe_ci": f"same, with the bootstrap 95% lower bound >= {safe_retention} (proven-safe core; "
                            "best bin re-selected per replicate)",
                 "fail": f"point retention < {fail_retention} (linear interpolation between bin centres); "
                         "without a crossing inside the support, censored at the support edge"},
        "safe_band_offset_um": safe_rng,
        "safe_ci_band_offset_um": safe_ci,
        "fail_band_offset_um": {"neg": fail_rng[0], "pos": fail_rng[1],
                                "neg_reached_within_data": bool(rneg), "pos_reached_within_data": bool(rpos)},
    }


def bin_width_sensitivity(r: np.ndarray, offsets_um: np.ndarray, shifts, pixel_size_um: float,
                          widths=BIN_WIDTHS_UM, n_boot: int = 500, seed: int = 0) -> dict:
    """Band-offset thresholds at several bin widths (same rules, fewer bootstrap draws)."""
    rows = []
    for w in widths:
        c = band_offset_curve(r, offsets_um, shifts, pixel_size_um, bin_um=w, n_boot=n_boot, seed=seed)
        rows.append({"bin_um": w, "peak_center_um": c["peak_center_um"],
                     "peak_ci95_um": c["peak_bootstrap"]["ci95_um"], "safe_band_offset_um": c["safe_band_offset_um"],
                     "safe_ci_band_offset_um": c["safe_ci_band_offset_um"], "fail_band_offset_um": c["fail_band_offset_um"],
                     "support_um": c["support_um"]})
    pos = [x["fail_band_offset_um"]["pos"] for x in rows]
    neg = [x["fail_band_offset_um"]["neg"] for x in rows]
    return {"n_boot": n_boot, "widths": rows,
            "fail_pos_range_um": [min(pos), max(pos)], "fail_neg_range_um": [min(neg), max(neg)],
            "peak_range_um": [min(x["peak_center_um"] for x in rows), max(x["peak_center_um"] for x in rows)]}


def calibration_flag_share(records: list[dict], thr, load_crop, pixel_size_um: float) -> dict:
    """Auditor class shares on the calibration crops (ink-bearing text regions).

    This is a *flag share*, not a specificity: the calibration meshes are not verified-correct,
    so the number of truly clean cells (the specificity denominator) is unknown. ``load_crop(rec)``
    must return the crop's CT (from the cache or S3), so the result never depends on what the
    cache happened to keep."""
    from .auditor import REASON_BIT, classify

    tot = np.zeros(4)
    per_crop_fail = []
    reasons = {}
    for rec in records:
        a = load_crop(rec)
        f = detect_bands(a, pixel_size_um, BandParams(cell_size=8))
        cls, reason = classify(f, thr)[:2]
        v = cls != 255
        k = np.array([(cls[v] == i).sum() for i in range(4)], float)
        tot += k
        per_crop_fail.append((k[2] + k[3]) / max(v.sum(), 1))
        for name in ("DRIFT_FAIL", "DRIFT_BEYOND_MEASURED", "NOMINAL_IN_AIR", "AIR_GAP_FAIL", "WINDOW_EMPTY",
                     "NO_BAND", "LATERAL_JUMP", "COMPETING_BANDS", "SHEET_ID_UNCERTAIN"):
            reasons[name] = reasons.get(name, 0) + int(((reason & REASON_BIT[name]) > 0)[v].sum())
    if len(per_crop_fail) != len(records):
        raise RuntimeError("calibration crops missing")
    n = max(tot.sum(), 1)
    pcf = np.array(per_crop_fail)
    return {
        "n_crops": len(per_crop_fail), "n_cells": int(tot.sum()),
        "share": {"OK": round(tot[0] / n, 4), "WARN_DRIFT": round(tot[1] / n, 4),
                  "FAIL_OFF_SHEET": round(tot[2] / n, 4), "FAIL_SHEET_SWITCH": round(tot[3] / n, 4)},
        "reason_share": {k: round(v / n, 4) for k, v in reasons.items()},
        "crops_with_fail_share_above_10pct": int((pcf > 0.10).sum()),
        "note": ("calibration flag share: auditor class shares on the ink-bearing calibration crops. Not a "
                 "specificity: the meshes are not verified-correct and the clean-cell denominator is unknown."),
    }


def summarize_metric(x: np.ndarray, shifts, idx: np.ndarray) -> list[dict]:
    rows = []
    for k, s in enumerate(shifts):
        col = x[:, k]
        good = col[np.isfinite(col)]
        lo, hi = bootstrap_mean_ci(col, idx)
        q1, med, q3 = np.percentile(good, [25, 50, 75]) if good.size else (np.nan,) * 3
        rows.append({"dz": s, "n": int(good.size), "mean": round(float(np.nanmean(col)), 4),
                     "ci95": [round(lo, 4), round(hi, 4)], "median": round(float(med), 4),
                     "iqr": [round(float(q1), 4), round(float(q3), 4)]})
    return rows


def calibrate(n_crops: int = 48, shifts=SHIFTS, out_dir: Path = OUT_DIR, seed: int = 0,
              min_sep: int = 1024, cache: BoundedCache | None = None) -> dict:
    from PIL import Image

    from .viz import plot_tolerance_curve

    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = cache or BoundedCache()
    ref_seg = SEGMENTS["PHerc0139"]
    shifts = validate_shifts(shifts)

    vol = LayersVolume(ref_seg)
    try:
        px = vol.pixel_size_um
        D, H, W = vol.shape(0)
        h8, w8 = H // 8, W // 8
        ref8 = reference_1um_on_grid(ref_seg, cache, (h8, w8))
        # valid CT at ds8 from pyramid level 5 (32 px), eroded one L5 pixel against mean-pooled padding
        l5 = cache.load_array(f"{vol.cache_tag}/L5_valid.npz")
        if l5 is None:
            print("reading level 5 for the valid mask ...", flush=True)
            l5 = (vol.array(5)[:] != 0).any(axis=0)
            cache.save_array(f"{vol.cache_tag}/L5_valid.npz", l5)
        from scipy import ndimage as ndi
        l5v = ndi.binary_erosion(l5, iterations=1)
        valid8 = np.kron(l5v, np.ones((4, 4), bool))[:h8, :w8]
        if valid8.shape != (h8, w8):
            pad = np.zeros((h8, w8), bool)
            pad[:valid8.shape[0], :valid8.shape[1]] = valid8
            valid8 = pad
        crops = select_crops(ref8, valid8, n_crops + 8, min_sep=min_sep)
        print(f"{len(crops)} candidate crops selected from the 1.129 um reference", flush=True)

        model = InkModel()
        print(f"model on {model.device_name}", flush=True)
        params = BandParams(cell_size=8)
        records = []
        r_ref, r_self, fg_ret, mean_p = [], [], [], []
        for crop in crops:
            if len(records) >= n_crops:
                break
            t0 = time.perf_counter()
            v = read_crop(vol, crop.y0, crop.x0, TILE, cache=cache)
            if not (v != 0).any(axis=0).all():
                print(f"  skip ({crop.y0}, {crop.x0}): zero-padded pixels", flush=True)
                continue
            field = detect_bands(v, px, params)
            meas = np.isfinite(field.delta_z_um)
            ref_patch = ref8[crop.y0 // 8:(crop.y0 + TILE) // 8, crop.x0 // 8:(crop.x0 + TILE) // 8] / 255.0
            preds = {s: model.predict(v[START + s:END + s]) for s in shifts}
            p0 = preds[0]
            fg0 = p0 > 0.5
            rr, rs, fr, mp = [], [], [], []
            for s in shifts:
                p = preds[s]
                rr.append(pearson(ds8(p), ref_patch))
                rs.append(pearson(p, p0))
                fr.append(float((fg0 & (p > 0.5)).sum() / fg0.sum()) if fg0.mean() >= 0.01 else np.nan)
                mp.append(float(p.mean()))
            r_ref.append(rr), r_self.append(rs), fg_ret.append(fr), mean_p.append(mp)
            rec = {
                "y0": crop.y0, "x0": crop.x0, "ref_text_score": round(crop.score, 5),
                "ref_mean": round(crop.ref_mean, 3), "ref_std": round(crop.ref_std, 3),
                "band_measured_cells": int(meas.sum()),
                "band_delta_um_median": round(float(np.median(field.delta_z_um[meas])), 2) if meas.any() else None,
                "band_abs_delta_um_median": round(float(np.median(np.abs(field.delta_z_um[meas]))), 2) if meas.any() else None,
                "nominal_in_air_share": round(float(field.nominal_in_air[field.valid].mean()), 3),
                "r_ref": [round(x, 4) for x in rr], "fg0_share": round(float(fg0.mean()), 4),
                "best_dz": shifts[int(np.nanargmax(rr))] if np.isfinite(rr).any() else None,
            }
            records.append(rec)
            print(f"  crop {len(records):2d} ({crop.y0:5d}, {crop.x0:5d}) r0={rr[shifts.index(0)]:+.3f} "
                  f"best dz={rec['best_dz']:+d} band median {rec['band_delta_um_median']} um "
                  f"[{time.perf_counter() - t0:.1f}s]", flush=True)
    finally:
        vol.close()

    r_ref, r_self, fg_ret, mean_p = map(np.asarray, (r_ref, r_self, fg_ret, mean_p))
    header = {
        "segment": {"sample": ref_seg.sample, "segment": segment_record(ref_seg)["long_id"],
                    "volume": ref_seg.volume, "pixel_size_um": px, "level0_shape_zyx": [D, H, W]},
        "reference": {"type": "published 1.129 um-scan ink map (model mrg20736) resized to the 2.4 um ds8 grid",
                      "volume": "20260413113053", "metric": "Pearson r on 32x32 ds8 cells per crop"},
        "model": {"checkpoint": CKPT_PATH.name, "recipe": "new_canon_autoresearch_recipe",
                  "window": f"[{START}+dz, {END}+dz)", "clip_scale": MAX_CLIP, "precision": "fp32",
                  "batch_size": 1, "device": model.device_name},
        "measure_runtime_s": round(time.perf_counter() - t_start, 1),
    }
    np.savez_compressed(out_dir / "ink_tolerance_raw.npz", r_ref=r_ref, r_self=r_self, fg_ret=fg_ret,
                        mean_p=mean_p, shifts=np.array(shifts))
    (out_dir / "ink_tolerance_crops.json").write_text(json.dumps({"header": header, "crops": records}, indent=1),
                                                      encoding="utf-8")
    return analyze(header, records, r_ref, r_self, fg_ret, mean_p, shifts, out_dir, seed=seed, cache=cache)


def reanalyze(out_dir: Path = OUT_DIR, seed: int = 0) -> dict:
    """Recompute statistics, thresholds and the figure from a previous run's raw arrays (no GPU)."""
    out_dir = Path(out_dir)
    crops_path = out_dir / "ink_tolerance_crops.json"
    if crops_path.exists():
        saved = json.loads(crops_path.read_text(encoding="utf-8"))
        header, records = saved["header"], saved["crops"]
    else:  # output of an older run: header and crops live in the curve JSON
        old = json.loads((out_dir / "ink_tolerance_curve.json").read_text(encoding="utf-8"))
        header = {k: old[k] for k in ("segment", "reference", "model")}
        header["measure_runtime_s"] = old.get("measure_runtime_s", old.get("runtime_s"))
        records = old["crops"]
        crops_path.write_text(json.dumps({"header": header, "crops": records}, indent=1), encoding="utf-8")
    raw = np.load(out_dir / "ink_tolerance_raw.npz")
    return analyze(header, records, raw["r_ref"], raw["r_self"], raw["fg_ret"], raw["mean_p"],
                   [int(v) for v in raw["shifts"]], out_dir, seed=seed)


def analyze(header: dict, records: list[dict], r_ref, r_self, fg_ret, mean_p, shifts, out_dir: Path,
            seed: int = 0, cache: BoundedCache | None = None) -> dict:
    from scipy.stats import spearmanr

    from .auditor import Thresholds
    from .viz import plot_tolerance_curve

    t0 = time.perf_counter()
    cache = cache or BoundedCache()
    out_dir = Path(out_dir)
    px = header["segment"]["pixel_size_um"]
    shifts = validate_shifts(shifts)
    k0 = shifts.index(0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(records), size=(4000, len(records)))
    ret = retention_stats(r_ref, shifts, seed=seed)

    # the CT crops come from the cache or, when evicted, from S3: never silently skipped
    ref_seg = SEGMENTS["PHerc0139"]
    vol = LayersVolume(ref_seg)

    def load_crop(rec):
        return read_crop(vol, rec["y0"], rec["x0"], TILE, cache=cache)

    # band offsets re-measured with the current detector (the stored ones may predate it)
    stored = np.array([np.nan if r["band_delta_um_median"] is None else r["band_delta_um_median"]
                       for r in records], float)
    offsets_all = np.full(len(records), np.nan)
    for i, rec in enumerate(records):
        f = detect_bands(load_crop(rec), px, BandParams(cell_size=8))
        m = np.isfinite(f.delta_z_um)
        if m.any():
            offsets_all[i] = round(float(np.median(f.delta_z_um[m])), 2)
        rec["band_delta_um_median"] = None if not m.any() else float(offsets_all[i])
        rec["band_measured_cells"] = int(m.sum())
    d = offsets_all - stored
    both = np.isfinite(d)
    offset_recompute = {
        "detector": "current band_detector (BandParams(cell_size=8), min_band_rule='faces_um')",
        "n_changed": int((np.abs(d[both]) > 0.005).sum()),
        "max_abs_change_um": round(float(np.abs(d[both]).max()), 2) if both.any() else None,
        "n_measured": int(np.isfinite(offsets_all).sum()), "n_stored_measured": int(np.isfinite(stored).sum()),
    }
    offsets = offsets_all[np.isfinite(offsets_all)]
    ref_offset = float(np.median(offsets))
    thr = derive_thresholds(shifts, ret["point"], ret["lo"], px, ref_offset)

    # absolute band-offset frame (what the auditor measures) replaces the mirrored window ranges
    curve = band_offset_curve(r_ref, offsets_all, shifts, px, seed=seed)
    fo = curve["fail_band_offset_um"]
    crossed = [fo["neg_reached_within_data"], fo["pos_reached_within_data"]]
    crossed_abs = [abs(v) for v, c in zip((fo["neg"], fo["pos"]), crossed) if c]
    thr["band_frame_window_mirror"] = {**thr["band_frame"], "reference_offset_um": round(ref_offset, 2)}
    thr["band_frame"] = {
        "safe_drift_um": curve["safe_band_offset_um"],
        "fail_drift_um": [fo["neg"], fo["pos"]],
        # a side without an empirical crossing is censored: drift beyond it is a warning only
        "fail_drift_crossed": crossed,
        # nominal slice in air: the nearest band face is farther than the smaller *crossed* fail distance
        "air_gap_fail_um": round(min(crossed_abs) if crossed_abs else min(abs(fo["neg"]), abs(fo["pos"])), 2),
        "switch_jump_um": SWITCH_JUMP_UM, "drift_window_cells": DRIFT_WINDOW_CELLS,
        "switch_min_cells": SWITCH_MIN_CELLS,
        "provisional": True,
        "source": "band_offset_curve (safe/fail) + fixed detector-noise rules (switch, smoothing)",
    }
    thr["status"] = "provisional"
    thr["provisional_scope"] = list(PROVISIONAL_SCOPE)
    if not all(crossed):
        side = "negative" if not crossed[0] else "positive"
        thr["censored_edge_note"] = (
            f"the {side} fail edge is censored at the support edge ({fo['neg'] if not crossed[0] else fo['pos']} um): "
            "the retention never fell below the fail level inside the data. Drift beyond it is reported as "
            "WARN_DRIFT + DRIFT_BEYOND_MEASURED, not FAIL_OFF_SHEET.")
    bw = bin_width_sensitivity(r_ref, offsets_all, shifts, px, seed=seed)
    thr["reference_offset_um"] = 0.0
    thr["frame_note"] = (
        "band_frame is measured directly as agreement vs the band position relative to the window "
        "centre (band_offset_curve), so drift_um = delta_z_um (reference offset 0). The window-shift "
        "ranges mirrored around the median crop offset are kept as band_frame_window_mirror.")

    informative = np.nanmean(r_ref, axis=1) >= 0.2
    sens = {"rule": "mean r over all shifts >= 0.2", "n_crops": int(informative.sum())}
    if informative.sum() >= 10:
        ret_i = retention_stats(r_ref[informative], shifts, seed=seed)
        sens["agreement_retention"] = [{"dz": s, "point": round(float(p), 4),
                                        "ci95": [round(float(a), 4), round(float(b), 4)]}
                                       for s, p, a, b in zip(shifts, ret_i["point"], ret_i["lo"], ret_i["hi"])]
        sens["thresholds_window_frame"] = derive_thresholds(shifts, ret_i["point"], ret_i["lo"], px,
                                                            ref_offset)["window_frame"]
        cur_i = band_offset_curve(r_ref[informative], offsets_all[informative], shifts, px, seed=seed)
        sens["band_offset_curve"] = {k: cur_i[k] for k in ("peak_center_um", "peak_bootstrap", "safe_band_offset_um",
                                                          "safe_ci_band_offset_um", "fail_band_offset_um")}
    paired = r_ref - r_ref[:, [k0]]
    paired_rows = []
    for k, s in enumerate(shifts):
        lo_, hi_ = bootstrap_mean_ci(paired[:, k], idx)
        paired_rows.append({"dz": s, "mean_delta_r": round(float(np.nanmean(paired[:, k])), 4),
                            "ci95": [round(lo_, 4), round(hi_, 4)]})

    best = np.array([r["best_dz"] for r in records], float)
    off_vx = offsets_all / px
    ok = np.isfinite(best) & np.isfinite(off_vx)
    rho = spearmanr(best[ok], off_vx[ok])
    r0 = r_ref[:, k0]

    band = thr["band_frame"]
    audit_thr = Thresholds(safe_drift_um=tuple(band["safe_drift_um"]), fail_drift_um=tuple(band["fail_drift_um"]),
                           fail_drift_crossed=tuple(band["fail_drift_crossed"]),
                           reference_offset_um=0.0, switch_jump_um=band["switch_jump_um"],
                           air_gap_fail_um=band["air_gap_fail_um"], drift_window_cells=band["drift_window_cells"],
                           switch_min_cells=band["switch_min_cells"], source="calibration", provisional=True)
    per_cell = Thresholds(safe_drift_um=audit_thr.safe_drift_um, fail_drift_um=audit_thr.fail_drift_um,
                          fail_drift_crossed=audit_thr.fail_drift_crossed,
                          switch_jump_um=audit_thr.switch_jump_um, air_gap_fail_um=audit_thr.air_gap_fail_um,
                          source="per-cell", provisional=True)
    try:
        flag_share = {"audit_rules": calibration_flag_share(records, audit_thr, load_crop, px),
                      "per_cell_rules_no_smoothing": calibration_flag_share(records, per_cell, load_crop, px)}
    finally:
        vol.close()

    result = {
        **{k: header[k] for k in ("segment", "reference", "model")},
        "n_crops": len(records), "crop_size": TILE, "shifts_vx": shifts,
        "shifts_um": [round(s * px, 2) for s in shifts],
        "agreement_vs_reference": summarize_metric(r_ref, shifts, idx),
        "agreement_retention": [{"dz": s, "point": round(float(p), 4), "ci95": [round(float(a), 4), round(float(b), 4)]}
                                for s, p, a, b in zip(shifts, ret["point"], ret["lo"], ret["hi"])],
        "paired_delta_r_vs_dz0": paired_rows,
        "band_offset_curve": curve,
        "band_offset_bin_width_sensitivity": bw,
        "band_offset_recompute": offset_recompute,
        "sensitivity_informative_crops": sens,
        "self_consistency_vs_dz0": summarize_metric(r_self, shifts, idx),
        "foreground_retention": summarize_metric(fg_ret, shifts, idx),
        "mean_probability": summarize_metric(mean_p, shifts, idx),
        "r0_distribution": {"median": round(float(np.median(r0)), 4),
                            "share_r0_above_0p3": round(float((r0 > 0.3).mean()), 3),
                            "min": round(float(r0.min()), 4), "max": round(float(r0.max()), 4)},
        "best_dz_histogram": {str(s): int((best == s).sum()) for s in shifts},
        "best_dz_vs_band_offset_spearman": {"rho": round(float(rho.statistic), 4),
                                            "p": round(float(rho.pvalue), 4), "n": int(ok.sum())},
        "band_offset_um_across_crops": {"median": round(ref_offset, 2),
                                        "p10_p90": np.percentile(offsets, [10, 90]).round(2).tolist()},
        "thresholds": thr,
        "calibration_flag_share_on_calibration_crops": flag_share,
        "crops": records,
        "measure_runtime_s": header.get("measure_runtime_s"),
        "analysis_runtime_s": round(time.perf_counter() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ink_tolerance_curve.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    plot_tolerance_curve(result, r_ref, out_dir / "ink_tolerance_curve.png")
    print(json.dumps({"thresholds": thr, "calibration_flag_share": {k: v["share"] for k, v in flag_share.items()},
                      "bin_width_sensitivity": {k: bw[k] for k in ("fail_pos_range_um", "fail_neg_range_um",
                                                                   "peak_range_um")}}, indent=1), flush=True)
    return result
