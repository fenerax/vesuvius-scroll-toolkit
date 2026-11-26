"""Per-cell placement QA classes, ranked hotspots and a training keep-mask.

Classes
-------
0   OK                 band near the nominal slice, within the calibrated safe drift range
1   WARN_DRIFT         any of: band beyond the safe range but not past an *empirical* failure
                       crossing; drift beyond a *censored* failure edge (the calibration never saw
                       agreement drop below the fail level there: ``DRIFT_BEYOND_MEASURED``);
                       nominal slice in air (``NOMINAL_IN_AIR``) closer than the air-gap failure
                       distance; nearest-band flip between two continuing sheets
                       (``COMPETING_BANDS``); band midpoint unmeasurable because it touches the
                       stack limit (``CLIPPED_BAND``)
2   FAIL_OFF_SHEET     the nominal slice misses the papyrus: drift beyond an empirically crossed
                       failure threshold, nominal slice in an air gap at least the failure
                       distance from the band, no band in the 62-layer ink window, or no band at
                       all. A blank ink map here means "surface missed", not "blank papyrus".
3   FAIL_SHEET_SWITCH  discontinuous lateral step of the nearest band (the selected band ends)
                       at or above ``switch_jump_um`` in a group of >= ``switch_min_cells`` cells
255 INVALID_PAD        zero-padded exterior (no valid surface coordinates)

Precedence (first match wins): INVALID_PAD > FAIL_OFF_SHEET > FAIL_SHEET_SWITCH > WARN_DRIFT > OK.

Reason bits (``reason``, uint16): every condition that holds is recorded, whatever the class.

Drift frame: ``drift_um = delta_z_um - reference_offset_um``. ``reference_offset_um`` is the
median band offset in the calibration crops, i.e. the placement convention the published
meshes (and therefore the ink model's training data) follow. The calibration measures ink
agreement against a *window* shift ``dz``; moving the window by ``+dz`` makes the band appear
at ``-dz`` relative to the window centre, so the band-frame ranges are the window ranges
mirrored (see ``ink_tolerance.derive_thresholds``).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .band_detector import BandField, with_steps

OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH, INVALID_PAD = 0, 1, 2, 3, 255
CLASS_NAMES = {OK: "OK", WARN_DRIFT: "WARN_DRIFT", FAIL_OFF_SHEET: "FAIL_OFF_SHEET",
               FAIL_SHEET_SWITCH: "FAIL_SHEET_SWITCH", INVALID_PAD: "INVALID_PAD"}
# status palette (dataviz reference): good / warning / critical; violet for the switch class
CLASS_COLORS = {OK: "#0ca30c", WARN_DRIFT: "#fab219", FAIL_OFF_SHEET: "#d03b3b",
                FAIL_SHEET_SWITCH: "#6250d6", INVALID_PAD: "#e1e0d9"}

REASONS = {
    1: "DRIFT_WARN", 2: "DRIFT_FAIL", 4: "NOMINAL_IN_AIR", 8: "AIR_GAP_FAIL",
    16: "WINDOW_EMPTY", 32: "NO_BAND", 64: "CLIPPED_BAND", 128: "LATERAL_JUMP",
    256: "COMPETING_BANDS", 512: "DRIFT_BEYOND_MEASURED", 1024: "SHEET_ID_UNCERTAIN",
}
REASON_BIT = {name: bit for bit, name in REASONS.items()}
REASON_DTYPE = np.uint16


@dataclass(frozen=True)
class Thresholds:
    """Classification thresholds, all in micrometres (band frame)."""

    safe_drift_um: tuple[float, float] = (-9.6, 9.6)
    fail_drift_um: tuple[float, float] = (-24.0, 24.0)
    reference_offset_um: float = 0.0
    switch_jump_um: float = 40.0
    air_gap_fail_um: float = 24.0
    drift_window_cells: int = 1     # odd edge of the nan-median window applied to drift (1 = per cell)
    switch_min_cells: int = 1       # smallest 8-connected group of jump cells that counts as a switch
    # was each fail edge an empirical crossing (True) or the censored edge of the measured data (False)?
    fail_drift_crossed: tuple[bool, bool] = (True, True)
    provisional: bool = False       # thresholds transferred beyond what the calibration measured
    source: str = "built-in defaults (uncalibrated)"

    def __post_init__(self):
        s, f = self.safe_drift_um, self.fail_drift_um
        # the calibrated safe range need not contain 0: the model's optimum sits below the window centre
        if not (f[0] <= s[0] <= s[1] <= f[1]):
            raise ValueError(f"need fail_lo <= safe_lo <= safe_hi <= fail_hi, got {s} / {f}")
        if self.switch_jump_um <= 0 or self.air_gap_fail_um <= 0:
            raise ValueError("switch_jump_um and air_gap_fail_um must be positive")
        if self.drift_window_cells < 1 or self.drift_window_cells % 2 == 0 or self.switch_min_cells < 1:
            raise ValueError("drift_window_cells must be odd and >= 1, switch_min_cells >= 1")
        object.__setattr__(self, "fail_drift_crossed", tuple(bool(v) for v in self.fail_drift_crossed))
        if len(self.fail_drift_crossed) != 2:
            raise ValueError("fail_drift_crossed needs two flags (negative edge, positive edge)")

    @classmethod
    def from_calibration(cls, path: str | Path, **overrides) -> "Thresholds":
        cal = json.loads(Path(path).read_text(encoding="utf-8"))
        band = cal["thresholds"]["band_frame"]
        crossed = band.get("fail_drift_crossed")
        if crossed is None:     # older calibration files: read the flags from the curve
            fo = cal.get("band_offset_curve", {}).get("fail_band_offset_um", {})
            crossed = (fo.get("neg_reached_within_data", True), fo.get("pos_reached_within_data", True))
        provisional = bool(band.get("provisional", cal["thresholds"].get("status") == "provisional"))
        kw = dict(
            safe_drift_um=tuple(band["safe_drift_um"]),
            fail_drift_um=tuple(band["fail_drift_um"]),
            reference_offset_um=float(cal["thresholds"]["reference_offset_um"]),
            air_gap_fail_um=float(band["air_gap_fail_um"]),
            fail_drift_crossed=tuple(bool(v) for v in crossed),
            provisional=provisional,
            source=f"calibrated{' (provisional)' if provisional else ''}: {Path(path).name}",
        )
        for k in ("switch_jump_um", "air_gap_fail_um"):
            if k in band:
                kw[k] = float(band[k])
        for k in ("drift_window_cells", "switch_min_cells"):
            if k in band:
                kw[k] = int(band[k])
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditResult:
    field: BandField
    cls: np.ndarray            # uint8 class map (gh, gw)
    reason: np.ndarray         # uint16 reason bitmask (gh, gw)
    drift_um: np.ndarray       # band-frame drift (NaN without midpoint)
    thresholds: Thresholds
    origin_l0: tuple[int, int] = (0, 0)    # level-0 (Y, X) of cell (0, 0)'s corner
    cell_l0: int = 8                       # level-0 pixels per cell edge
    meta: dict = field(default_factory=dict)
    drift_raw_um: np.ndarray | None = None  # per-cell drift before the nan-median window

    def __post_init__(self):
        if self.drift_raw_um is None:
            self.drift_raw_um = self.drift_um

    def cell_center_l0(self, iy, ix):
        return (self.origin_l0[0] + np.asarray(iy) * self.cell_l0 + self.cell_l0 // 2,
                self.origin_l0[1] + np.asarray(ix) * self.cell_l0 + self.cell_l0 // 2)

    def cell_of(self, y: int, x: int) -> tuple[int, int]:
        return (int((y - self.origin_l0[0]) // self.cell_l0), int((x - self.origin_l0[1]) // self.cell_l0))

    def class_counts(self) -> dict:
        valid = self.cls != INVALID_PAD
        n = max(int(valid.sum()), 1)
        out = {}
        for c, name in CLASS_NAMES.items():
            k = int((self.cls == c).sum())
            out[name] = {"cells": k, "share_of_valid": None if c == INVALID_PAD else round(k / n, 4)}
        return out

    def cell_report(self, y: int, x: int) -> dict:
        iy, ix = self.cell_of(y, x)
        f = self.field
        if not (0 <= iy < f.shape[0] and 0 <= ix < f.shape[1]):
            raise IndexError(f"({y}, {x}) is outside this audit")
        cy, cx = self.cell_center_l0(iy, ix)

        def num(a):
            v = float(a[iy, ix])
            return None if not np.isfinite(v) else round(v, 2)

        return {
            "query_yx_l0": [int(y), int(x)], "cell_center_yx_l0": [int(cy), int(cx)],
            "class": int(self.cls[iy, ix]), "class_name": CLASS_NAMES[int(self.cls[iy, ix])],
            "reasons": reason_names(int(self.reason[iy, ix])),
            "z_top": num(f.z_top), "z_bot": num(f.z_bot), "z_mid": num(f.z_mid),
            "delta_z_vx": num(f.delta_z_vx), "delta_z_um": num(f.delta_z_um),
            "drift_um": num(self.drift_um), "drift_raw_um": num(self.drift_raw_um),
            "switch_jump_um": num(f.switch_jump_um), "flip_jump_um": num(f.flip_jump_um),
            "uncertain_jump_um": num(f.uncertain_jump_um), "midpoint_step_um": num(f.midpoint_step_um),
            "gap_um": num(f.gap_um), "thickness_um": num(f.thickness_um),
            "n_bands": int(f.n_bands[iy, ix]), "competing_bands": int(f.competing_bands[iy, ix]),
            "nominal_in_air": bool(f.nominal_in_air[iy, ix]), "window_empty": bool(f.window_empty[iy, ix]),
        }


def reason_names(bits: int) -> list[str]:
    return [name for bit, name in REASONS.items() if bits & bit]


def reason_counts(res: "AuditResult") -> dict:
    """Valid cells carrying each reason bit, split by final class."""
    v = res.cls != INVALID_PAD
    out = {}
    for bit, name in REASONS.items():
        on = v & ((res.reason & bit) > 0)
        out[name] = {"cells": int(on.sum()),
                     "by_class": {CLASS_NAMES[c]: int((on & (res.cls == c)).sum())
                                  for c in (OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH)}}
    return out


def nanmedian_window(a: np.ndarray, size: int, min_count: int | None = None, rows: int = 128,
                     z_top: np.ndarray | None = None, z_bot: np.ndarray | None = None) -> np.ndarray:
    """NaN-aware median over a ``size`` x ``size`` window (edge cells use the cells available).

    With ``z_top``/``z_bot``, only window cells whose selected band overlaps the centre cell's
    band in ``z`` (the same sheet) take part, so a region where the nearest-band choice flips
    between two sheets is not averaged into a value neither sheet has. Cells whose window holds
    fewer than ``min_count`` usable values (default: a third of the window) keep their own value.
    NaN cells stay NaN."""
    if size <= 1:
        return a.copy()
    r = size // 2
    min_count = max(1, size * size // 3) if min_count is None else min_count
    win = np.lib.stride_tricks.sliding_window_view
    pad = lambda x: np.pad(x.astype(np.float32), r, constant_values=np.nan)  # noqa: E731
    p = pad(a)
    same = z_top is not None and z_bot is not None
    if same:
        pt, pb = pad(z_top), pad(z_bot)
    out = np.empty(a.shape, np.float32)
    for y0 in range(0, a.shape[0], rows):
        y1 = min(a.shape[0], y0 + rows)
        w = win(p[y0:y1 + 2 * r], (size, size)).reshape(y1 - y0, a.shape[1], -1)
        if same:
            wt = win(pt[y0:y1 + 2 * r], (size, size)).reshape(w.shape)
            wb = win(pb[y0:y1 + 2 * r], (size, size)).reshape(w.shape)
            ct, cb = z_top[y0:y1, :, None], z_bot[y0:y1, :, None]
            with np.errstate(invalid="ignore"):
                w = np.where(np.minimum(wb, cb) - np.maximum(wt, ct) >= 0, w, np.nan)
        n = np.isfinite(w).sum(-1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(w, axis=-1)
        out[y0:y1] = np.where(n >= min_count, med, a[y0:y1])
    return np.where(np.isfinite(a), out, np.nan).astype(np.float64)


def grouped(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """Cells of ``mask`` inside an 8-connected group of at least ``min_cells`` cells."""
    if min_cells <= 1 or not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    size = np.bincount(lab.ravel(), minlength=n + 1)
    size[0] = 0
    return size[lab] >= min_cells


def classify(f: BandField, thr: Thresholds) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cls uint8, reason uint16 bitmask, drift_um, drift_raw_um)`` for every cell of ``f``.

    Drift classes use ``drift_um``, the nan-median of the per-cell drift over
    ``drift_window_cells`` among cells on the same sheet (overlapping band); the ink tolerance was
    calibrated for a coherent window shift, and a single 8 px cell is too small to move an ink
    prediction on its own. Air-gap, empty-window and no-band tests stay per cell.

    Drift beyond a fail edge is FAIL_OFF_SHEET only where that edge was an empirical crossing
    (``fail_drift_crossed``); beyond a censored edge it is WARN_DRIFT with ``DRIFT_BEYOND_MEASURED``.
    A switch (selected band ends) and a nearest-band flip (both bands continue) count only inside an
    8-connected group of at least ``switch_min_cells`` cells; flips become ``COMPETING_BANDS``."""
    v = f.valid
    raw = np.where(v, f.delta_z_um - thr.reference_offset_um, np.nan)
    drift = nanmedian_window(raw, thr.drift_window_cells, z_top=f.z_top, z_bot=f.z_bot)
    fin = np.isfinite(drift)
    crossed_lo, crossed_hi = thr.fail_drift_crossed
    with np.errstate(invalid="ignore"):
        warn = fin & ((drift < thr.safe_drift_um[0]) | (drift > thr.safe_drift_um[1]))
        out_lo = fin & (drift < thr.fail_drift_um[0])
        out_hi = fin & (drift > thr.fail_drift_um[1])
        fail = (out_lo & crossed_lo) | (out_hi & crossed_hi)
        beyond = (out_lo & (not crossed_lo)) | (out_hi & (not crossed_hi))
        air_fail = f.nominal_in_air & (f.gap_um >= thr.air_gap_fail_um)
        jump = grouped(v & (np.nan_to_num(f.switch_jump_um, nan=0.0) >= thr.switch_jump_um), thr.switch_min_cells)
        flip = grouped(v & (np.nan_to_num(f.flip_jump_um, nan=0.0) >= thr.switch_jump_um), thr.switch_min_cells)
        uncertain = jump & (np.nan_to_num(f.uncertain_jump_um, nan=0.0) >= thr.switch_jump_um)
    clipped = f.clipped & ~fin

    reason = np.zeros(v.shape, REASON_DTYPE)
    for name, flag in (("DRIFT_WARN", warn), ("DRIFT_FAIL", fail), ("NOMINAL_IN_AIR", f.nominal_in_air),
                       ("AIR_GAP_FAIL", air_fail), ("WINDOW_EMPTY", f.window_empty), ("NO_BAND", f.no_band),
                       ("CLIPPED_BAND", clipped), ("LATERAL_JUMP", jump), ("COMPETING_BANDS", flip),
                       ("DRIFT_BEYOND_MEASURED", beyond), ("SHEET_ID_UNCERTAIN", uncertain)):
        reason |= np.where(v & flag, REASON_BIT[name], 0).astype(REASON_DTYPE)

    off = v & (fail | air_fail | f.window_empty | f.no_band)
    switch = v & jump & ~off
    warned = v & (warn | beyond | clipped | f.nominal_in_air | flip) & ~off & ~switch
    cls = np.full(v.shape, INVALID_PAD, np.uint8)
    cls[v] = OK
    cls[warned] = WARN_DRIFT
    cls[switch] = FAIL_SHEET_SWITCH
    cls[off] = FAIL_OFF_SHEET
    return cls, reason, np.where(v, drift, np.nan), raw


def audit_field(f: BandField, thr: Thresholds, origin_l0=(0, 0), cell_l0: int | None = None,
                meta: dict | None = None) -> AuditResult:
    cls, reason, drift, raw = classify(f, thr)
    return AuditResult(f, cls, reason, drift, thr, tuple(int(v) for v in origin_l0),
                       int(cell_l0 or f.cell_size), dict(meta or {}), raw)


def cell_severity(res: AuditResult) -> np.ndarray:
    """Dimensionless per-cell severity: 1 at a threshold, larger beyond it."""
    t = res.thresholds
    f = res.field
    with np.errstate(invalid="ignore"):
        d = res.drift_um
        lo, hi = t.fail_drift_um
        drift_sev = np.where(d < 0, d / lo if lo else 0, d / hi if hi else 0)
        parts = [
            np.nan_to_num(drift_sev, nan=0.0),
            np.nan_to_num(f.switch_jump_um / t.switch_jump_um, nan=0.0),
            np.where(f.nominal_in_air, np.nan_to_num(f.gap_um / t.air_gap_fail_um, nan=0.0), 0.0),
            np.where(f.window_empty, 2.0, 0.0),
            np.where(f.no_band, 1.0, 0.0),
        ]
    return np.where(f.valid, np.max(parts, axis=0), 0.0)


def find_hotspots(res: AuditResult, classes=(FAIL_OFF_SHEET, FAIL_SHEET_SWITCH), top: int | None = 200,
                  min_cells: int = 1) -> list[dict]:
    """Connected (8-neighbour) regions of each failure class, ranked by summed severity."""
    sev = cell_severity(res)
    f = res.field
    out = []
    for c in classes:
        lab, n = ndi.label(res.cls == c, structure=np.ones((3, 3), bool))
        if n == 0:
            continue
        flat = lab.ravel()
        on = flat > 0
        ids = flat[on]
        cells = np.flatnonzero(on)
        s = sev.ravel()[cells]
        count = np.bincount(ids, minlength=n + 1)
        score = np.bincount(ids, weights=s, minlength=n + 1)
        order = np.lexsort((-s, ids))
        firsts = order[np.r_[True, ids[order][1:] != ids[order][:-1]]]
        peak_cell = np.zeros(n + 1, np.int64)
        peak_cell[ids[firsts]] = cells[firsts]
        absd = np.nan_to_num(np.abs(res.drift_um.ravel()[cells]), nan=-1.0)
        maxd = np.full(n + 1, -1.0)
        np.maximum.at(maxd, ids, absd)
        jmp = np.nan_to_num(f.switch_jump_um.ravel()[cells], nan=0.0)
        maxj = np.zeros(n + 1)
        np.maximum.at(maxj, ids, jmp)
        iy_all, ix_all = np.divmod(cells, f.shape[1])
        sy = np.bincount(ids, weights=iy_all, minlength=n + 1)
        sx = np.bincount(ids, weights=ix_all, minlength=n + 1)
        for k in range(1, n + 1):
            if count[k] < min_cells:
                continue
            iy, ix = divmod(int(peak_cell[k]), f.shape[1])
            py, px = res.cell_center_l0(iy, ix)
            cy, cx = res.cell_center_l0(sy[k] / count[k], sx[k] / count[k])

            def val(a):
                x = float(a[iy, ix])
                return round(x, 2) if np.isfinite(x) else None

            out.append({
                "class": int(c), "class_name": CLASS_NAMES[c], "n_cells": int(count[k]),
                "area_mm2": round(count[k] * (res.cell_l0 * f.pixel_size_um / 1000.0) ** 2, 5),
                "score": round(float(score[k]), 3),
                "peak_yx_l0": [int(py), int(px)], "centroid_yx_l0": [int(round(cy)), int(round(cx))],
                "drift_um": val(res.drift_um), "drift_raw_um": val(res.drift_raw_um),
                "delta_z_um": val(f.delta_z_um),
                "max_abs_drift_um": round(float(maxd[k]), 2) if maxd[k] >= 0 else None,
                "jump_um": val(f.switch_jump_um), "max_jump_um": round(float(maxj[k]), 2),
                "flip_jump_um": val(f.flip_jump_um), "midpoint_step_um": val(f.midpoint_step_um),
                "gap_um": val(f.gap_um), "z_top": val(f.z_top), "z_bot": val(f.z_bot),
                "competing_bands": int(f.competing_bands[iy, ix]),
                "reasons": reason_names(int(res.reason[iy, ix])),
            })
    out.sort(key=lambda h: (-h["score"], -h["n_cells"]))
    for i, h in enumerate(out, 1):
        h["rank"] = i
    return out[:top] if top else out


def train_keep_mask(cls: np.ndarray, keep_classes=(OK, WARN_DRIFT), margin_cells: int = 1) -> np.ndarray:
    """Cells safe to keep for 3D surface training: allowed classes, minus a margin around failures."""
    keep = np.isin(cls, keep_classes)
    fail = np.isin(cls, (FAIL_OFF_SHEET, FAIL_SHEET_SWITCH))
    if margin_cells > 0 and fail.any():
        fail = ndi.binary_dilation(fail, structure=np.ones((3, 3), bool), iterations=margin_cells)
    return keep & ~fail


def restrict_valid(f: BandField, valid: np.ndarray) -> BandField:
    """Copy of ``f`` with extra cells marked invalid (e.g. exact level-0 padding), with the lateral
    step fields recomputed so that no step to a newly invalid cell survives."""
    v = f.valid & valid
    nan = lambda a: np.where(v, a, np.nan)  # noqa: E731
    off = lambda a: a & v  # noqa: E731
    g = replace(
        f, valid=v, n_bands=np.where(v, f.n_bands, 0), competing_bands=np.where(v, f.competing_bands, 0),
        z_top=nan(f.z_top), z_bot=nan(f.z_bot), z_mid=nan(f.z_mid), thickness_um=nan(f.thickness_um),
        delta_z_vx=nan(f.delta_z_vx), delta_z_um=nan(f.delta_z_um), nominal_in_air=off(f.nominal_in_air),
        gap_um=nan(f.gap_um), window_empty=off(f.window_empty), clipped=off(f.clipped),
        no_band=off(f.no_band), occupancy=np.where(v, f.occupancy, np.uint64(0)),
    )
    return with_steps(g)


def save_audit(res: AuditResult, out_dir: Path, prefix: str, hotspots: list[dict] | None = None,
               keep_kwargs: dict | None = None) -> dict:
    """Write ``<prefix>_fields.npz``, ``<prefix>_hotspots.json``, ``<prefix>_train_keep_mask.png``
    (+ ``.json`` sidecar) and ``<prefix>_qa_classes.png``. Returns the summary dict."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = res.field
    hotspots = find_hotspots(res) if hotspots is None else hotspots
    keep = train_keep_mask(res.cls, **(keep_kwargs or {}))
    np.savez_compressed(
        out_dir / f"{prefix}_fields.npz",
        cls=res.cls, reason=res.reason, drift_um=res.drift_um.astype(np.float32),
        drift_raw_um=res.drift_raw_um.astype(np.float32), midpoint_step_um=f.midpoint_step_um.astype(np.float32),
        delta_z_um=f.delta_z_um.astype(np.float32), z_top=f.z_top.astype(np.float32),
        z_bot=f.z_bot.astype(np.float32), switch_jump_um=f.switch_jump_um.astype(np.float32),
        flip_jump_um=f.flip_jump_um.astype(np.float32), uncertain_jump_um=f.uncertain_jump_um.astype(np.float32),
        gap_um=f.gap_um.astype(np.float32), n_bands=f.n_bands.astype(np.uint8),
        competing_bands=f.competing_bands.astype(np.uint8), keep=keep,
        origin_l0=np.array(res.origin_l0), cell_l0=np.array(res.cell_l0),
    )
    Image.fromarray(keep.astype(np.uint8) * 255).save(out_dir / f"{prefix}_train_keep_mask.png")
    rgb = class_rgb(res.cls)
    Image.fromarray(rgb).save(out_dir / f"{prefix}_qa_classes.png")
    summary = {
        "meta": res.meta, "thresholds": res.thresholds.to_dict(),
        "origin_l0_yx": list(res.origin_l0), "cell_l0_px": res.cell_l0, "grid_shape": list(res.cls.shape),
        "class_counts": res.class_counts(), "reason_counts": reason_counts(res), "band_summary": f.summary(),
        "train_keep_mask": {"png": f"{prefix}_train_keep_mask.png", "kept_cells": int(keep.sum()),
                            "rule": keep_kwargs or {"keep_classes": ["OK", "WARN_DRIFT"], "margin_cells": 1},
                            "pixel_to_l0": "cell (iy, ix) covers L0 Y [oy + iy*c, oy + (iy+1)*c), X likewise"},
        "n_hotspots": len(hotspots),
    }
    (out_dir / f"{prefix}_hotspots.json").write_text(
        json.dumps({"summary": summary, "hotspots": hotspots}, indent=1), encoding="utf-8")
    return summary


def class_rgb(cls: np.ndarray) -> np.ndarray:
    rgb = np.zeros(cls.shape + (3,), np.uint8)
    for c, hexcol in CLASS_COLORS.items():
        rgb[cls == c] = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
    return rgb
