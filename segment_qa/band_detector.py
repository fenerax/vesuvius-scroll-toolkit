"""Vectorized raw-CT papyrus band and air-gap detector for rendered surface volumes.

Input is a ``(D, H, W)`` slab from a ``layers-zarr`` surface volume: the layer axis ``z``
is the mesh normal. For every lateral cell (``cell_size`` x ``cell_size`` pixels) the
detector averages the CT profile along ``z``, smooths it, finds contiguous bright bands
(papyrus) separated by dark gaps (air), and picks the band nearest the nominal slice.

Nominal anchor vs. window centre
--------------------------------
Two "centres" exist and are kept apart on purpose:

* ``z_nom = D // 2`` (layer **54** of the 109-layer render) is the mesh surface itself, the
  layer the renderer places on the mesh. Every offset (``delta_z``, ``gap``, the band-frame
  thresholds) is measured from this **nominal anchor**.
* The ``ink_canonical_2um`` window reads layers ``[23, 85)``, i.e. samples 23..84. Its
  midpoint is **53.5**, half a layer below the anchor. The window is stored relative to the
  anchor as ``-31 .. +30`` layers (``ink_window_layers``) and converted to micrometres with the
  reference voxel size, so a band face lying exactly on layer 23 or layer 84 is inside the window.

The calibration (``ink_tolerance``) measures band offsets from the same anchor (54), so the
half-layer difference is part of the measured thresholds and is not corrected twice.

The band rule reproduces ``scripts/inspect_label_drift.py::measure`` (Sol 6.2's audit),
vectorized over all cells, with every *physical* length in micrometres so that a volume
rendered at a different voxel size gets the same physical thresholds. The minimum band
thickness is applied to the interpolated faces in micrometres (``min_band_rule="faces_um"``);
``"samples"`` keeps the reference loop's run-length rule (4 samples) for regression tests.
Intensity parameters (``contrast``, ``min_excess``, ``min_peak``) stay in uint8 grey levels.

Zero-padded exterior: the renderer writes all-zero ``z`` columns where the mesh has no
valid surface coordinates. A pixel is valid when ``any(slab != 0, axis=0)``; a cell is
valid when at least ``min_valid_fraction`` of its pixels are valid (default: all of
them). Invalid cells never enter band detection, so padding can never be read as an
air gap. Mean-pooled pyramid levels cannot express partial padding; see
``runs.l0_exact_validity`` for the exact level-0 check.

Sheet continuity across a lateral edge
--------------------------------------
A large step of the selected band's midpoint between two 4-neighbours is a sheet switch only
if the selected band *ends*. If the selected band of each cell overlaps (exact, continuous
``[z_top, z_bot]`` intervals) some accepted band of the other cell, both sheets continue and
only the nearest-band choice flipped (``flip_jump_um``). A uint64 occupancy bitmask
(``occ_vx`` layers per bit) is kept as a fast prefilter; continuity is decided by the exact
interval test after it. Edges where the prefilter reports continuation but the exact intervals
are disjoint (near-touching bands within one bit) count as switches and are reported as
``sheet_id_uncertain`` because the sheet identity across them cannot be resolved at this
z resolution.

Sign convention: ``z_top`` is the band face at the lower layer index and ``z_bot`` the
face at the higher index. ``delta_z = z_mid - z_nom`` is positive when the papyrus band
sits at a higher layer index than the nominal slice. Which side is the recto depends on
the render polarity and is not established by the OME-Zarr metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage as ndi

REFERENCE_PIXEL_UM = 2.399          # voxel size of the ink model's training renders
INK_WINDOW_LAYERS = (-31, 30)       # layers [23, 85) of the 109-layer render, relative to z_nom = 54
WINDOW_TOL_VX = 1e-6                # float slack so a face exactly on layer 23 / 84 is inside


@dataclass(frozen=True)
class BandParams:
    """Detector parameters. Lengths are in micrometres; intensities in uint8 grey levels."""

    cell_size: int = 8
    smooth_sigma_um: float = 2.999        # 1.25 voxels at 2.399 um
    min_band_um: float = 9.5              # interpolated face-to-face thickness (4 samples at 2.399 um)
    gap_margin_um: float = 4.798          # nominal plane >= 2 voxels outside the band at 2.399 um
    contrast: float = 0.30                # band cutoff = p15 + max(min_excess, contrast * (p90 - p15))
    peak_extra: float = 0.12              # peak floor = p15 + max(min_peak, (contrast + extra) * span)
    min_excess: float = 12.0
    min_peak: float = 22.0
    # ink_canonical_2um reads samples 23..84 of the 109-layer render: -31..+30 layers from z_nom = 54
    ink_window_um: tuple[float, float] = (INK_WINDOW_LAYERS[0] * REFERENCE_PIXEL_UM,
                                          INK_WINDOW_LAYERS[1] * REFERENCE_PIXEL_UM)
    min_valid_fraction: float = 1.0
    pad_erode_cells: int = 0              # optional extra erosion of the valid mask
    min_band_rule: str = "faces_um"       # "faces_um" (default) or "samples" (reference loop)

    def __post_init__(self):
        if self.min_band_rule not in ("faces_um", "samples"):
            raise ValueError("min_band_rule must be 'faces_um' or 'samples'")

    def samples(self, length_um: float, pixel_size_um: float) -> float:
        return length_um / pixel_size_um


GRID_FIELDS = ("valid", "n_bands", "competing_bands", "z_top", "z_bot", "z_mid", "thickness_um", "delta_z_vx",
               "delta_z_um", "nominal_in_air", "gap_um", "window_empty", "clipped", "no_band", "switch_jump_um",
               "midpoint_step_um", "flip_jump_um", "uncertain_jump_um", "occupancy", "cont_down", "cont_right",
               "pre_down", "pre_right")


@dataclass
class BandField:
    """Per-cell band measurements on a ``(gh, gw)`` grid. Lengths in voxels end in ``_vx``
    (layer index units) and in micrometres end in ``_um``."""

    pixel_size_um: float
    cell_size: int
    depth: int
    z_nom: int
    valid: np.ndarray                 # bool: fully supported (not zero-padded) cell
    n_bands: np.ndarray               # int: accepted bands anywhere in the stack
    competing_bands: np.ndarray       # int: other accepted bands that intersect the ink window
    z_top: np.ndarray                 # float vx: low-index face of the nearest band (NaN: no band)
    z_bot: np.ndarray                 # float vx: high-index face of the nearest band
    z_mid: np.ndarray                 # float vx: midpoint (NaN when no band or band clipped by the stack)
    thickness_um: np.ndarray
    delta_z_vx: np.ndarray            # z_mid - z_nom
    delta_z_um: np.ndarray
    nominal_in_air: np.ndarray        # bool: nominal slice is below cutoff and outside the band by >= gap margin
    gap_um: np.ndarray                # distance from the nominal slice to the nearest band face (0 inside)
    window_empty: np.ndarray          # bool: no accepted band intersects the 62-layer ink window
    clipped: np.ndarray               # bool: nearest band touches the first/last layer
    no_band: np.ndarray               # bool: valid cell without any accepted band
    switch_jump_um: np.ndarray        # max |delta_z| step to a 4-neighbour where the selected band ends
    midpoint_step_um: np.ndarray      # max |delta_z| step to any 4-neighbour (includes merge/split noise)
    occupancy: np.ndarray             # uint64 bitmask of z covered by any accepted band (occ_vx per bit)
    occ_vx: int                       # layers per occupancy bit
    flip_jump_um: np.ndarray = None   # max step to a 4-neighbour where both selected bands continue (flip)
    uncertain_jump_um: np.ndarray = None  # max switch step decided by the exact test against the prefilter
    cont_down: np.ndarray = None      # bool edge (iy, ix)-(iy+1, ix): both selected bands continue (exact)
    cont_right: np.ndarray = None     # bool edge (iy, ix)-(iy, ix+1)
    pre_down: np.ndarray = None       # bool edge: bitmask prefilter says both continue
    pre_right: np.ndarray = None
    band_spacing_um: np.ndarray = field(default_factory=lambda: np.zeros(0))  # consecutive band midpoints
    band_spacing_cell: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # flat cell of each

    def __post_init__(self):
        shape = self.valid.shape
        for name in ("flip_jump_um", "uncertain_jump_um"):
            if getattr(self, name) is None:
                setattr(self, name, np.where(np.isfinite(self.delta_z_um), 0.0, np.nan))
        for name in ("cont_down", "cont_right", "pre_down", "pre_right"):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(shape, bool))

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape

    @classmethod
    def empty(cls, shape: tuple[int, int], pixel_size_um: float, cell_size: int, depth: int,
              z_nom: int | None = None) -> "BandField":
        """All-invalid field (every cell zero-padded)."""
        nan = np.full(shape, np.nan)
        off = np.zeros(shape, bool)
        zero = np.zeros(shape, int)
        return cls(pixel_size_um=float(pixel_size_um), cell_size=int(cell_size), depth=int(depth),
                   z_nom=depth // 2 if z_nom is None else int(z_nom), valid=off.copy(), n_bands=zero.copy(),
                   competing_bands=zero.copy(), z_top=nan.copy(), z_bot=nan.copy(), z_mid=nan.copy(),
                   thickness_um=nan.copy(), delta_z_vx=nan.copy(), delta_z_um=nan.copy(),
                   nominal_in_air=off.copy(), gap_um=nan.copy(), window_empty=off.copy(), clipped=off.copy(),
                   no_band=off.copy(), switch_jump_um=nan.copy(), midpoint_step_um=nan.copy(),
                   occupancy=np.zeros(shape, np.uint64), occ_vx=max(1, -(-depth // 64)))

    def crop(self, ys: slice, xs: slice) -> "BandField":
        """Sub-grid view (copies). Edge flags on the crop's last row/column keep their value, i.e.
        they still describe the edge to the cell just outside the crop."""
        kw = {n: getattr(self, n)[ys, xs].copy() for n in GRID_FIELDS}
        gh, gw = self.shape
        y0, y1, _ = ys.indices(gh)
        x0, x1, _ = xs.indices(gw)
        iy, ix = np.divmod(self.band_spacing_cell, gw)
        keep = (iy >= y0) & (iy < y1) & (ix >= x0) & (ix < x1)
        cell = (iy[keep] - y0) * (x1 - x0) + (ix[keep] - x0)
        return replace(self, band_spacing_um=self.band_spacing_um[keep], band_spacing_cell=cell, **kw)

    def summary(self) -> dict:
        v = self.valid
        m = np.isfinite(self.delta_z_um)
        a = np.abs(self.delta_z_um[m])
        out = {
            "cells": int(v.size), "valid": int(v.sum()), "measured": int(m.sum()),
            "no_band": int(self.no_band.sum()), "clipped": int((self.clipped & v).sum()),
            "nominal_in_air": int(self.nominal_in_air.sum()), "window_empty": int(self.window_empty.sum()),
            "multi_band": int((self.n_bands >= 2).sum()),
        }
        if a.size:
            out["abs_delta_um_p50_p90_p95_max"] = np.percentile(a, [50, 90, 95, 100]).round(2).tolist()
            out["signed_delta_um_median"] = round(float(np.median(self.delta_z_um[m])), 2)
        if self.band_spacing_um.size:
            out["band_spacing_um_median"] = round(float(np.median(self.band_spacing_um)), 2)
        return out


def cell_profiles(slab: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``z`` profile per cell ``(D, gh, gw)`` and the per-cell valid-pixel fraction."""
    if slab.ndim != 3:
        raise ValueError(f"expected a (D, H, W) slab, got shape {slab.shape}")
    d, h, w = slab.shape
    gh, gw = h // cell, w // cell
    if gh == 0 or gw == 0:
        raise ValueError(f"slab {slab.shape} is smaller than one {cell}px cell")
    s = slab[:, : gh * cell, : gw * cell]
    support = (s != 0).any(axis=0)
    if cell == 1:
        return s.astype(np.float32), support.astype(np.float32)
    prof = s.reshape(d, gh, cell, gw, cell).mean(axis=(2, 4), dtype=np.float32)
    frac = support.reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    return prof, frac


def pool_all(mask: np.ndarray, cell: int) -> np.ndarray:
    """``all`` over ``cell`` x ``cell`` blocks; a partial trailing block counts as invalid."""
    h, w = mask.shape
    gh, gw = -(-h // cell), -(-w // cell)
    pad = np.zeros((gh * cell, gw * cell), bool)
    pad[:h, :w] = mask
    return pad.reshape(gh, cell, gw, cell).all(axis=(1, 3))


def occupancy_bits(lo_vx: np.ndarray, hi_vx: np.ndarray, occ_vx: int) -> np.ndarray:
    """uint64 mask with the bits of ``z`` intervals ``[lo, hi]`` (vx) set, ``occ_vx`` layers per bit.
    NaN intervals give 0."""
    fin = np.isfinite(lo_vx) & np.isfinite(hi_vx)
    a = np.clip(np.floor(np.where(fin, lo_vx, 0) / occ_vx), 0, 63).astype(np.uint64)
    b = np.clip(np.floor(np.where(fin, hi_vx, 0) / occ_vx), 0, 63).astype(np.uint64)
    one = np.uint64(1)
    upper = np.where(b >= 63, ~np.uint64(0), (one << (b + one)) - one)
    bits = upper & ~((one << a) - one)
    return np.where(fin, bits, np.uint64(0))


def ink_window_mask(low_vx, high_vx, z_nom: float, w_lo_vx: float, w_hi_vx: float,
                    tol: float = WINDOW_TOL_VX) -> np.ndarray:
    """True where the band ``[low, high]`` intersects the closed window ``[z_nom + w_lo, z_nom + w_hi]``
    (sample positions). A face lying exactly on the first or last window sample is inside."""
    low_vx, high_vx = np.asarray(low_vx, float), np.asarray(high_vx, float)
    return ~((high_vx < z_nom + w_lo_vx - tol) | (low_vx > z_nom + w_hi_vx + tol))


def edge_continuation(col: np.ndarray, low: np.ndarray, high: np.ndarray, sel_lo: np.ndarray,
                      sel_hi: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Exact continuity of the selected bands across every 4-neighbour edge.

    ``col``/``low``/``high``: all accepted bands (flat cell index, faces in vx); ``sel_lo``/``sel_hi``
    (flat, NaN without band): the selected band per cell. Returns ``(cont_down, cont_right)``,
    bool ``(gh, gw)``: edge to the cell below / to the right where the selected band of *each* cell
    overlaps (closed continuous intervals) some accepted band of the other cell."""
    gh, gw = shape
    n = gh * gw
    r, c = np.divmod(col, gw)
    out = []
    for off, has_prev, has_next in ((gw, r >= 1, r < gh - 1), (1, c >= 1, c < gw - 1)):
        fwd = np.zeros(n, bool)    # selected band of cell q overlaps a band of q + off
        bwd = np.zeros(n, bool)    # selected band of cell q overlaps a band of q - off
        k = np.flatnonzero(has_prev)
        src = col[k] - off
        with np.errstate(invalid="ignore"):
            ov = np.minimum(high[k], sel_hi[src]) - np.maximum(low[k], sel_lo[src]) >= 0
        fwd[src[ov]] = True
        k = np.flatnonzero(has_next)
        dst = col[k] + off
        with np.errstate(invalid="ignore"):
            ov = np.minimum(high[k], sel_hi[dst]) - np.maximum(low[k], sel_lo[dst]) >= 0
        bwd[dst[ov]] = True
        fwd, bwd = fwd.reshape(shape), bwd.reshape(shape)
        edge = np.zeros(shape, bool)
        if off == gw:
            edge[:-1] = fwd[:-1] & bwd[1:]
        else:
            edge[:, :-1] = fwd[:, :-1] & bwd[:, 1:]
        out.append(edge)
    return out[0], out[1]


def prefilter_continuation(z_top: np.ndarray, z_bot: np.ndarray, occupancy: np.ndarray,
                           occ_vx: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantized (bitmask) version of :func:`edge_continuation`: a superset of the exact result."""
    sel = occupancy_bits(z_top, z_bot, occ_vx)
    out = []
    for axis in (0, 1):
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        edge = np.zeros(z_top.shape, bool)
        edge[lo] = ((sel[lo] & occupancy[hi]) != 0) & ((sel[hi] & occupancy[lo]) != 0)
        out.append(edge)
    return out[0], out[1]


@dataclass
class LateralSteps:
    switch_um: np.ndarray       # selected bands disjoint and at least one of them ends
    flip_um: np.ndarray         # selected bands disjoint but both continue: nearest-band flip
    uncertain_um: np.ndarray    # switch steps where only the exact test (not the prefilter) says "ends"
    midpoint_um: np.ndarray     # any step


def lateral_steps(delta_um: np.ndarray, z_top: np.ndarray | None = None, z_bot: np.ndarray | None = None,
                  cont: tuple[np.ndarray, np.ndarray] | None = None,
                  pre: tuple[np.ndarray, np.ndarray] | None = None) -> LateralSteps:
    """Largest absolute steps of ``delta_um`` to 4-neighbours, split by sheet continuity.

    ``cont``: exact continuation edges ``(down, right)``; ``pre``: the bitmask prefilter edges.
    An edge continues when the prefilter passes *and* the exact interval test passes. Without
    ``z_top``/``z_bot`` every finite step is a switch candidate. NaN where the cell has no midpoint."""
    shape = delta_um.shape
    sw, fl, unc, mid = (np.zeros(shape) for _ in range(4))
    for axis in (0, 1):
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        step = np.abs(delta_um[hi] - delta_um[lo])
        fin = np.isfinite(step)
        step = np.where(fin, step, 0.0)
        disjoint = np.ones(step.shape, bool)
        if z_top is not None and z_bot is not None:
            with np.errstate(invalid="ignore"):
                disjoint = np.minimum(z_bot[lo], z_bot[hi]) - np.maximum(z_top[lo], z_top[hi]) < 0
        exact = cont[axis][lo] if cont is not None else np.zeros(step.shape, bool)
        prefilter = pre[axis][lo] if pre is not None else exact
        continues = prefilter & exact
        uncertain = prefilter & ~exact
        cand = fin & disjoint
        for out, m in ((sw, cand & ~continues), (fl, cand & continues), (unc, cand & ~continues & uncertain),
                       (mid, fin)):
            s = np.where(m, step, 0.0)
            out[lo] = np.maximum(out[lo], s)
            out[hi] = np.maximum(out[hi], s)
    nan = ~np.isfinite(delta_um)
    return LateralSteps(*(np.where(nan, np.nan, a) for a in (sw, fl, unc, mid)))


def lateral_jumps_um(delta_um: np.ndarray, z_top: np.ndarray | None = None, z_bot: np.ndarray | None = None,
                     cont: tuple[np.ndarray, np.ndarray] | None = None,
                     pre: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Switch steps only (see :func:`lateral_steps`). Without ``z_top``/``z_bot``: any finite step."""
    return lateral_steps(delta_um, z_top, z_bot, cont, pre).switch_um


def with_steps(f: BandField) -> BandField:
    """Recompute the lateral step fields of ``f`` from its own selection and edge flags."""
    s = lateral_steps(f.delta_z_um, f.z_top, f.z_bot, (f.cont_down, f.cont_right), (f.pre_down, f.pre_right))
    return replace(f, switch_jump_um=s.switch_um, flip_jump_um=s.flip_um, uncertain_jump_um=s.uncertain_um,
                   midpoint_step_um=s.midpoint_um)


def detect_bands(slab: np.ndarray, pixel_size_um: float, params: BandParams | None = None,
                 z_nom: int | None = None) -> BandField:
    """Detect papyrus bands for every cell of a ``(D, H, W)`` uint8 surface-volume slab."""
    p = params or BandParams()
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    depth = slab.shape[0]
    mid = depth // 2 if z_nom is None else int(z_nom)
    prof, frac = cell_profiles(slab, p.cell_size)
    valid = frac >= p.min_valid_fraction - 1e-9
    if p.pad_erode_cells > 0:
        valid = ndi.binary_erosion(valid, iterations=p.pad_erode_cells, border_value=1)
    gh, gw = valid.shape
    n = gh * gw

    prof = ndi.gaussian_filter1d(prof, p.smooth_sigma_um / pixel_size_um, axis=0)
    P = np.ascontiguousarray(prof.reshape(depth, n).T)          # (N, D)
    air, bright = np.percentile(P, [15, 90], axis=1)
    span = bright - air
    cutoff = air + np.maximum(p.min_excess, p.contrast * span)
    peak_floor = air + np.maximum(p.min_peak, (p.contrast + p.peak_extra) * span)
    ex = P - cutoff[:, None]
    vflat = valid.ravel()
    inside = (ex > 0) & vflat[:, None]

    # contiguous above-cutoff runs, all cells at once
    padded = np.zeros((n, depth + 2), np.int8)
    padded[:, 1:-1] = inside
    edge = np.diff(padded, axis=1)
    col, start = np.nonzero(edge == 1)
    col_e, stop = np.nonzero(edge == -1)
    assert np.array_equal(col, col_e)
    end = stop - 1
    if col.size:
        flat = np.concatenate([P.ravel(), np.zeros(1, P.dtype)])
        idx = np.empty(2 * col.size, np.int64)
        idx[0::2] = col * depth + start
        idx[1::2] = col * depth + end + 1
        peak = np.maximum.reduceat(flat, idx)[0::2]
    else:
        peak = np.zeros(0, P.dtype)

    # sub-voxel faces by linear interpolation of the cutoff crossing (all runs)
    e_s = ex[col, start]
    e_sm = ex[col, np.maximum(start - 1, 0)]
    low = np.where(start > 0, start - e_s / np.where(start > 0, e_s - e_sm, 1.0), 0.0)
    e_e = ex[col, end]
    e_ep = ex[col, np.minimum(end + 1, depth - 1)]
    high = np.where(end < depth - 1, end + e_e / np.where(end < depth - 1, e_e - e_ep, 1.0), depth - 1.0)

    if p.min_band_rule == "samples":
        thick_ok = end - start + 1 >= max(1, int(round(p.min_band_um / pixel_size_um)))
    else:
        thick_ok = (high - low) * pixel_size_um >= p.min_band_um - 1e-9
    keep = thick_ok & (peak >= peak_floor[col])
    col, start, end, low, high = col[keep], start[keep], end[keep], low[keep], high[keep]
    clipped_band = (start == 0) | (end == depth - 1)
    center = 0.5 * (low + high)

    px = pixel_size_um
    w_lo, w_hi = p.ink_window_um[0] / px, p.ink_window_um[1] / px
    in_window = ink_window_mask(low, high, mid, w_lo, w_hi)

    n_bands = np.bincount(col, minlength=n)
    n_window = np.bincount(col[in_window], minlength=n)

    # nearest band: smallest distance from the nominal slice to the band, then nearest midpoint
    dist = np.maximum(np.maximum(low - mid, mid - high), 0.0)
    order = np.lexsort((np.abs(center - mid), dist, col))
    cs = col[order]
    first = np.ones(cs.size, bool)
    first[1:] = cs[1:] != cs[:-1]
    sel = order[first]
    sc = col[sel]

    def grid(fill, dtype=float):
        return np.full(n, fill, dtype)

    z_top, z_bot, z_mid = grid(np.nan), grid(np.nan), grid(np.nan)
    gap_vx = grid(np.nan)
    clipped = grid(False, bool)
    air_flag = grid(False, bool)
    competing = np.zeros(n, int)
    z_top[sc] = low[sel]
    z_bot[sc] = high[sel]
    clipped[sc] = clipped_band[sel]
    z_mid[sc] = np.where(clipped_band[sel], np.nan, center[sel])
    signed_gap = np.maximum(low[sel] - mid, mid - high[sel])
    gap_vx[sc] = np.maximum(signed_gap, 0.0)
    air_flag[sc] = (signed_gap >= p.gap_margin_um / px) & (ex[sc, mid] <= 0)
    competing[sc] = n_window[sc] - in_window[sel].astype(int)

    # spacing between consecutive measurable band midpoints in the same cell (wrap pitch proxy)
    unclipped = ~clipped_band
    same = (col[1:] == col[:-1]) & unclipped[1:] & unclipped[:-1]
    spacing = (center[1:] - center[:-1])[same] * px
    spacing_cell = col[1:][same]

    shape = (gh, gw)
    occ_vx = max(1, -(-depth // 64))
    occ = np.zeros(n, np.uint64)
    np.bitwise_or.at(occ, col, occupancy_bits(low, high, occ_vx))
    occ = occ.reshape(shape)
    cont = edge_continuation(col, low, high, z_top, z_bot, shape)
    zt, zb = z_top.reshape(shape), z_bot.reshape(shape)
    pre = prefilter_continuation(zt, zb, occ, occ_vx)
    delta_vx = (z_mid - mid).reshape(shape)
    delta_um = delta_vx * px
    no_band = valid & (n_bands.reshape(shape) == 0)
    f = BandField(
        pixel_size_um=px, cell_size=p.cell_size, depth=depth, z_nom=mid,
        valid=valid,
        n_bands=n_bands.reshape(shape),
        competing_bands=competing.reshape(shape),
        z_top=zt, z_bot=zb, z_mid=z_mid.reshape(shape),
        thickness_um=((z_bot - z_top) * px).reshape(shape),
        delta_z_vx=delta_vx, delta_z_um=delta_um,
        nominal_in_air=air_flag.reshape(shape),
        gap_um=(gap_vx * px).reshape(shape),
        window_empty=valid & (n_window.reshape(shape) == 0),
        clipped=clipped.reshape(shape),
        no_band=no_band,
        switch_jump_um=np.zeros(shape), midpoint_step_um=np.zeros(shape),
        occupancy=occ, occ_vx=occ_vx,
        cont_down=cont[0], cont_right=cont[1], pre_down=pre[0], pre_right=pre[1],
        band_spacing_um=spacing, band_spacing_cell=spacing_cell,
    )
    return with_steps(f)
