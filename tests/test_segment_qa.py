"""Unit tests for segment_qa on synthetic surface-volume slabs (no network, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segment_qa.auditor import (FAIL_OFF_SHEET, FAIL_SHEET_SWITCH, INVALID_PAD, OK, WARN_DRIFT,  # noqa: E402
                                Thresholds, audit_field, classify, find_hotspots, reason_names,
                                restrict_valid, train_keep_mask)
from segment_qa.band_detector import BandParams, detect_bands, lateral_jumps_um  # noqa: E402
from segment_qa.ink_tolerance import derive_thresholds, retention_stats  # noqa: E402

PX = 2.399
THR = Thresholds(safe_drift_um=(-10.0, 10.0), fail_drift_um=(-30.0, 30.0), reference_offset_um=0.0,
                 switch_jump_um=40.0, air_gap_fail_um=24.0, source="test")


def render(centers_um, thickness_um=30.0, px=PX, depth=109, extra=None, air=20.0, paper=150.0, seed=0,
           noise=2.0):
    """Synthetic slab: one papyrus band per pixel centred ``centers_um`` (H, W) from the nominal slice.

    ``extra`` is an optional list of further (H, W) centre maps (additional sheets)."""
    centers_um = np.asarray(centers_um, float)
    h, w = centers_um.shape
    z_um = (np.arange(depth) - depth // 2)[:, None, None] * px
    vol = np.full((depth, h, w), air, np.float32)
    for c in [centers_um] + list(extra or []):
        inside = np.abs(z_um - c[None]) <= thickness_um / 2
        vol[inside] = paper
    vol = ndi.gaussian_filter1d(vol, 1.2 / px * 2.4, axis=0)
    vol += np.random.default_rng(seed).normal(0, noise, vol.shape)
    return np.clip(vol, 1, 255).astype(np.uint8)


def const(value, h=64, w=64):
    return np.full((h, w), float(value))


def run(slab, thr=THR, px=PX, **kw):
    field = detect_bands(slab, px, BandParams(cell_size=8, **kw))
    return field, audit_field(field, thr)


def test_clean_band_is_ok():
    field, res = run(render(const(0.0)))
    assert field.valid.all()
    assert np.all(np.abs(field.delta_z_um) < 1.5)
    assert np.all(field.thickness_um > 25) and np.all(field.thickness_um < 40)
    assert not field.nominal_in_air.any() and not field.window_empty.any()
    assert (field.competing_bands == 0).all()
    assert (res.cls == OK).all()


def test_reference_offset_defines_the_ok_frame():
    slab = render(const(-20.0))
    _, res_plain = run(slab)
    _, res_ref = run(slab, thr=Thresholds(**{**THR.to_dict(), "reference_offset_um": -20.0}))
    assert (res_plain.cls == WARN_DRIFT).all()
    assert (res_ref.cls == OK).all()
    assert np.nanmax(np.abs(res_ref.drift_um)) < 1.5


def test_mild_drift_warns_with_measured_offset():
    field, res = run(render(const(18.0)))
    assert np.allclose(field.delta_z_um, 18.0, atol=1.5)
    assert (res.cls == WARN_DRIFT).all()
    assert all("DRIFT_WARN" in reason_names(int(b)) for b in res.reason.ravel())


def test_off_sheet_air_gap_fails():
    # band centred +55 um, faces at +40 / +70 um: nominal slice sits 40 um into the air gap
    field, res = run(render(const(55.0)))
    assert field.nominal_in_air.all()
    # the 30 % contrast cutoff sits on the blurred flank, so the measured face is ~4 um closer
    assert np.allclose(field.gap_um, 36.0, atol=3.0)
    assert (res.cls == FAIL_OFF_SHEET).all()
    names = reason_names(int(res.reason[0, 0]))
    assert {"NOMINAL_IN_AIR", "AIR_GAP_FAIL", "DRIFT_FAIL"} <= set(names)


def test_band_outside_ink_window_is_window_empty():
    # +105 um centre, low face +90 um, beyond the +72 um top of the 62-layer window
    field, res = run(render(const(105.0)))
    assert field.window_empty.all()
    assert (res.cls == FAIL_OFF_SHEET).all()
    assert "WINDOW_EMPTY" in reason_names(int(res.reason[3, 3]))


def test_no_band_is_off_sheet_not_pad():
    slab = np.full((109, 64, 64), 25, np.uint8)
    field, res = run(slab)
    assert field.valid.all() and field.no_band.all()
    assert (res.cls == FAIL_OFF_SHEET).all()


def test_sharp_sheet_switch():
    c = const(-18.0)
    c[:, 32:] = 24.0          # 42 um step between columns 31 and 32 (cells 3 | 4)
    field, res = run(render(c))
    jump = field.switch_jump_um
    assert np.allclose(jump[:, 3], 42.0, atol=2.0) and np.allclose(jump[:, 4], 42.0, atol=2.0)
    assert (res.cls[:, 3:5] == FAIL_SHEET_SWITCH).all()
    assert (res.cls[:, :3] == WARN_DRIFT).all() and (res.cls[:, 5:] == WARN_DRIFT).all()
    hot = find_hotspots(res)
    assert hot[0]["class_name"] == "FAIL_SHEET_SWITCH"
    assert hot[0]["n_cells"] == 16
    assert hot[0]["max_jump_um"] == pytest.approx(42.0, abs=2.0)


def test_competing_band_counted():
    c1 = const(0.0)
    c2 = const(60.0)
    field, _ = run(render(c1, extra=[c2]))
    assert (field.n_bands == 2).all()
    assert (field.competing_bands == 1).all()
    assert np.allclose(field.delta_z_um, 0.0, atol=1.5)
    assert np.median(field.band_spacing_um) == pytest.approx(60.0, abs=2.5)


def test_zero_padded_exterior_is_never_air_gap():
    slab = render(const(0.0))
    slab[:, :, 40:] = 0            # padded exterior from column 40 (cell 5 is partially padded)
    slab[:, :20, :12] = 0          # notch that leaves cell (0..2, 1) partially padded
    field, res = run(slab)
    assert (res.cls[:, 5:] == INVALID_PAD).all()
    assert (res.cls[:2, 0] == INVALID_PAD).all() and (res.cls[:2, 1] == INVALID_PAD).all()
    assert not np.any((res.cls == FAIL_OFF_SHEET))
    assert not field.nominal_in_air[~field.valid].any()
    assert not field.window_empty[~field.valid].any()
    # interior cells next to padding keep their measurement
    assert (res.cls[4:, :5] == OK).all()


def test_pad_erosion_option():
    slab = render(const(0.0))
    slab[:, :, 40:] = 0
    field = detect_bands(slab, PX, BandParams(cell_size=8, pad_erode_cells=1))
    assert not field.valid[:, 4].any() and field.valid[:, 3].all()


@pytest.mark.parametrize("centre_um", [0.0, 18.0, -22.0, 55.0])
def test_pixel_size_scaling_invariance(centre_um):
    """Same physical geometry at 2.4 um (109 layers) and 1.2 um (217 layers) -> same um results."""
    c = const(centre_um)
    coarse = render(c, px=2.4, depth=109)
    fine = render(c, px=1.2, depth=217)
    f1, r1 = run(coarse, px=2.4)
    f2, r2 = run(fine, px=1.2)
    assert np.allclose(f1.delta_z_um, f2.delta_z_um, atol=1.5, equal_nan=True)
    assert np.allclose(f1.thickness_um, f2.thickness_um, atol=3.0)
    assert np.array_equal(r1.cls, r2.cls)
    assert np.array_equal(f1.nominal_in_air, f2.nominal_in_air)


def test_lateral_jumps_ignore_nan():
    d = np.array([[0.0, 10.0, np.nan], [0.0, 50.0, 5.0]])
    j = lateral_jumps_um(d)
    assert np.isnan(j[0, 2])
    assert j[1, 1] == pytest.approx(50.0)
    assert j[0, 1] == pytest.approx(40.0)
    assert j[0, 0] == pytest.approx(10.0)


def test_hotspot_ranking_and_coordinates():
    c = const(0.0, 128, 128)
    c[8:40, 8:40] = 80.0         # large, severe off-sheet patch (4x4 cells)
    c[96:104, 96:104] = 40.0     # single mild off-sheet cell
    field = detect_bands(render(c), PX, BandParams(cell_size=8))
    res = audit_field(field, THR, origin_l0=(1000, 2000), cell_l0=8)
    hot = find_hotspots(res, classes=(FAIL_OFF_SHEET,))
    assert [h["rank"] for h in hot] == list(range(1, len(hot) + 1))
    assert hot[0]["n_cells"] >= 16 and hot[0]["score"] > hot[-1]["score"]
    py, px_ = hot[0]["peak_yx_l0"]
    assert 1000 + 8 <= py < 1000 + 40 and 2000 + 8 <= px_ < 2000 + 40
    single = [h for h in hot if h["n_cells"] == 1]
    assert single and single[0]["peak_yx_l0"] == [1000 + 12 * 8 + 4, 2000 + 12 * 8 + 4]
    assert res.cell_report(1068, 2120)["class_name"] == "OK"


def test_train_keep_mask_margin():
    cls = np.zeros((7, 7), np.uint8)
    cls[3, 3] = FAIL_SHEET_SWITCH
    cls[0, 0] = INVALID_PAD
    cls[6, 6] = WARN_DRIFT
    keep = train_keep_mask(cls, margin_cells=1)
    assert not keep[2:5, 2:5].any()
    assert not keep[0, 0] and keep[6, 6] and keep[0, 6]
    strict = train_keep_mask(cls, keep_classes=(OK,), margin_cells=0)
    assert not strict[6, 6] and strict[2, 2]


def test_restrict_valid_recomputes_jumps():
    c = const(-18.0)
    c[:, 32:] = 24.0
    field = detect_bands(render(c), PX, BandParams(cell_size=8))
    mask = np.ones(field.shape, bool)
    mask[:, 4] = False
    f2 = restrict_valid(field, mask)
    assert np.isnan(f2.delta_z_um[:, 4]).all()
    cls, _, _, _ = classify(f2, THR)
    assert (cls[:, 4] == INVALID_PAD).all() and not (cls == FAIL_SHEET_SWITCH).any()


@pytest.mark.parametrize("rule", ["samples", "faces_um"])
def test_vectorized_detector_matches_reference_loop(rule):
    """The vectorized detector reproduces Sol's per-cell loop (scripts/inspect_label_drift.measure).
    ``samples`` is the loop's own thickness rule; ``faces_um`` (default) agrees on this slab because
    no band sits between the two rules (see test_min_band_rule_faces_um_vs_samples)."""
    pytest.importorskip("s3fs")
    sys.path.insert(0, str(ROOT / "scripts"))
    import inspect_label_drift as ref

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:256, 0:256]
    c1 = 25 * np.sin(yy / 37.0) + 15 * np.cos(xx / 23.0) - 10
    c2 = c1 + 65 + 10 * np.sin(xx / 17.0)
    slab = render(c1, extra=[c2], seed=4, noise=6.0)
    slab[:, :, 230:] = 0
    slab[rng.integers(0, 109, 400), rng.integers(0, 256, 400), rng.integers(0, 230, 400)] = 255
    want = ref.measure(slab)
    got = detect_bands(slab, PX, BandParams(cell_size=8, min_band_rule=rule))
    assert np.array_equal(want.valid, got.valid)
    assert np.array_equal(want.count, got.n_bands)
    assert np.array_equal(np.isnan(want.center), np.isnan(got.z_mid))
    assert np.nanmax(np.abs(want.center - got.z_mid)) < 0.05
    assert np.array_equal(want.nominal_gap, got.nominal_in_air)
    assert np.array_equal(want.window_empty, got.window_empty & (got.n_bands > 0))


def test_derive_thresholds_mirrors_band_frame():
    shifts = [-16, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12, 16]
    point = np.array([0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0, 0.97, 0.9, 0.75, 0.6, 0.4, 0.3])
    lo = point - 0.04
    t = derive_thresholds(shifts, point, lo, 2.0, reference_offset_um=-20.0)
    w = t["window_frame"]
    assert w["safe_dz_vx"] == [-4.0, 2.0]
    assert w["fail_dz_vx"]["neg"] == pytest.approx(-7.0)     # 0.8 -> 0.6 between -6 and -8
    assert w["fail_dz_vx"]["pos"] == pytest.approx(6.667, abs=1e-3)   # 0.75 -> 0.6 between 6 and 8
    b = t["band_frame"]
    assert b["safe_drift_um"] == [-4.0, 8.0]
    assert b["fail_drift_um"] == pytest.approx([-13.33, 14.0], abs=0.01)
    assert t["reference_offset_um"] == -20.0
    Thresholds(safe_drift_um=tuple(b["safe_drift_um"]), fail_drift_um=tuple(b["fail_drift_um"]))


def test_retention_bootstrap_brackets_point():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 0.6, size=(40, 1))
    decay = np.array([0.5, 0.8, 1.0, 0.8, 0.5])
    r = base * decay + rng.normal(0, 0.02, size=(40, 5))
    s = retention_stats(r, [-4, -2, 0, 2, 4], n_boot=500)
    assert s["point"][2] == pytest.approx(1.0)
    assert np.all(s["lo"] <= s["point"] + 1e-9) and np.all(s["hi"] >= s["point"] - 1e-9)
    assert s["point"][0] == pytest.approx(0.5, abs=0.05)


def test_threshold_validation():
    with pytest.raises(ValueError):
        Thresholds(safe_drift_um=(-10, 10), fail_drift_um=(-5, 30))


def test_nearest_band_flip_between_two_continuing_sheets_is_not_a_switch():
    """Two sheets 60 um apart with the nominal slice between them: the nearest-band choice flips
    across the tile but both sheets continue, so the papyrus did not move."""
    lower = const(-32.0)
    upper = const(32.0)
    lower[:, 32:] = -28.0     # sheets drift by 4 um: the nearer one changes at column 32
    upper[:, 32:] = 36.0
    field, res = run(render(lower, extra=[upper]))
    assert np.nanmax(field.midpoint_step_um) > 50.0       # midpoint hops from one sheet to the other
    assert np.nanmax(field.switch_jump_um) == 0.0          # but both bands continue across the edge
    assert not (res.cls == FAIL_SHEET_SWITCH).any()


def test_mesh_step_moves_all_bands_and_is_a_switch():
    lower = const(-20.0)
    upper = const(40.0)
    lower[:, 32:] = 26.0      # every band moves by +46 um: a real step in the surface
    upper[:, 32:] = 86.0
    field, res = run(render(lower, extra=[upper]))
    assert np.nanmax(field.switch_jump_um) == pytest.approx(46.0, abs=3.0)
    assert (res.cls[:, 3:5] == FAIL_SHEET_SWITCH).all()


def test_switch_min_cells_drops_isolated_jump_cells():
    c = const(0.0)
    c[24:32, 24:32] = 48.0    # one 8 px cell lifted by 48 um: 5 jump cells in a plus shape
    strict = Thresholds(**{**THR.to_dict(), "switch_min_cells": 6})
    _, res1 = run(render(c))
    _, res6 = run(render(c), thr=strict)
    assert (res1.cls == FAIL_SHEET_SWITCH).sum() >= 1
    assert not (res6.cls == FAIL_SHEET_SWITCH).any()


def test_drift_window_median_ignores_single_outlier_cell():
    tight = {**THR.to_dict(), "safe_drift_um": (-8.0, 8.0), "fail_drift_um": (-15.0, 15.0), "switch_jump_um": 100.0}
    smooth = Thresholds(**{**tight, "drift_window_cells": 5})
    raw = Thresholds(**tight)
    c = const(0.0)
    c[24:32, 24:32] = 20.0    # one cell 20 um off, still on the same sheet (bands overlap in z)
    _, r_raw = run(render(c), thr=raw)
    _, r_s = run(render(c), thr=smooth)
    assert r_raw.cls[3, 3] == FAIL_OFF_SHEET
    assert r_s.cls[3, 3] == OK
    assert r_s.drift_raw_um[3, 3] == pytest.approx(20.0, abs=1.5)
    assert abs(r_s.drift_um[3, 3]) < 1.5
    big = c.copy()
    big[8:56, 8:56] = 20.0    # a coherent 6 x 6-cell drift survives the median
    _, r_big = run(render(big), thr=smooth)
    assert r_big.cls[3, 3] == FAIL_OFF_SHEET


def test_drift_window_median_does_not_mix_sheets():
    """Where the nearest band flips between two sheets, the median only uses the cell's own sheet."""
    lower = const(-51.0)
    upper = const(71.0)
    lower[:, 32:] = -62.0
    upper[:, 32:] = 60.0      # the upper sheet becomes the nearer one right of column 32
    smooth = Thresholds(**{**THR.to_dict(), "drift_window_cells": 5, "safe_drift_um": (-60.0, 60.0),
                            "fail_drift_um": (-80.0, 80.0)})
    field, res = run(render(lower, extra=[upper]), thr=smooth)
    assert res.drift_um[:, 3] == pytest.approx(np.full(8, -51.0), abs=2.0)
    assert res.drift_um[:, 4] == pytest.approx(np.full(8, 60.0), abs=2.0)


def test_band_offset_curve_recovers_planted_optimum():
    from segment_qa.ink_tolerance import band_offset_curve

    rng = np.random.default_rng(1)
    shifts = np.array([-16, -12, -8, -4, 0, 4, 8, 12, 16])
    offsets = rng.uniform(-30, 0, 60)
    level = rng.uniform(0.2, 0.6, 60)                          # crop-specific text contrast
    e = offsets[:, None] - shifts[None, :] * PX
    r = level[:, None] * np.exp(-0.5 * ((e + 15.0) / 22.0) ** 2) + rng.normal(0, 0.01, e.shape)
    cur = band_offset_curve(r, offsets, shifts, PX, n_boot=200)
    assert -24.0 <= cur["peak_center_um"] <= -6.0
    lo, hi = cur["safe_band_offset_um"]
    assert lo < -15.0 < hi
    # point retention 0.7 at |e + 15| = 22 * sqrt(-2 ln 0.7) = 18.5 um
    assert cur["fail_band_offset_um"]["pos"] == pytest.approx(3.5, abs=5.0)
    assert cur["fail_band_offset_um"]["neg"] == pytest.approx(-33.5, abs=5.0)


# ------------------------------------------------------------------ round-2 fixes (Sol review)
from segment_qa.auditor import REASON_BIT  # noqa: E402
from segment_qa.band_detector import BandField, ink_window_mask, pool_all  # noqa: E402

SHARP = dict(smooth_sigma_um=0.01)     # no detector smoothing: faces sit exactly 0.7 vx outside the paper


def step_slab(columns, h=64, w=64, depth=109, air=20, paper=150):
    """Noise-free slab: ``columns`` is a list of ``(x0, x1, [(z0, z1), ...])`` paper layer ranges
    (inclusive). With ``SHARP`` params the cutoff (p15 + 0.3 span = 59) puts each face 91/130 = 0.7
    layers outside the paper run."""
    a = np.full((depth, h, w), air, np.uint8)
    for x0, x1, bands in columns:
        for z0, z1 in bands:
            a[z0:z1 + 1, :, x0:x1] = paper
    return a


def mean_pool(slab, f):
    d, h, w = slab.shape
    return slab.reshape(d, h // f, f, w // f, f).mean(axis=(2, 4)).round().astype(np.uint8)


UNC_THR = Thresholds(safe_drift_um=(-30.0, 30.0), fail_drift_um=(-60.0, 60.0), switch_jump_um=30.0,
                     air_gap_fail_um=24.0, source="test")


@pytest.mark.parametrize("level", [0, 3])
def test_near_touching_disjoint_bands_are_a_switch_with_uncertain_sheet_id(level):
    """Selected bands [36.3, 50.7] | [51.3, 65.7] vx: disjoint by 0.6 vx, yet in the same 2-layer
    occupancy bit (25). The prefilter says "continues"; the exact interval test says "ends"."""
    slab = step_slab([(0, 32, [(37, 50)]), (32, 64, [(52, 65)])])
    if level == 0:
        f = detect_bands(slab, PX, BandParams(cell_size=8, **SHARP))
    else:
        f = detect_bands(mean_pool(slab, 8), PX, BandParams(cell_size=1, **SHARP))
    assert f.z_bot[0, 3] == pytest.approx(50.7, abs=1e-3) and f.z_top[0, 4] == pytest.approx(51.3, abs=1e-3)
    assert int(f.z_bot[0, 3] // f.occ_vx) == int(f.z_top[0, 4] // f.occ_vx)        # same prefilter bit
    assert f.pre_right[:, 3].all() and not f.cont_right[:, 3].any()
    assert np.allclose(f.switch_jump_um[:, 3:5], 15 * PX, atol=1e-3)
    assert np.allclose(f.uncertain_jump_um[:, 3:5], 15 * PX, atol=1e-3)
    assert np.nanmax(f.flip_jump_um) == 0.0
    cls, reason, _, _ = classify(f, UNC_THR)
    assert (cls[:, 3:5] == FAIL_SHEET_SWITCH).all()
    assert ((reason[:, 3:5] & REASON_BIT["SHEET_ID_UNCERTAIN"]) > 0).all()
    assert not ((reason[:, :3] & REASON_BIT["SHEET_ID_UNCERTAIN"]) > 0).any()


@pytest.mark.parametrize("level", [0, 3])
def test_disjoint_bands_in_different_bits_are_a_certain_switch(level):
    slab = step_slab([(0, 32, [(38, 51)]), (32, 64, [(53, 66)])])     # faces 51.7 | 52.3: bits 25 | 26
    p = BandParams(cell_size=8 if level == 0 else 1, **SHARP)
    f = detect_bands(slab if level == 0 else mean_pool(slab, 8), PX, p)
    assert not f.pre_right[:, 3].any() and not f.cont_right[:, 3].any()
    cls, reason, _, _ = classify(f, UNC_THR)
    assert (cls[:, 3:5] == FAIL_SHEET_SWITCH).all()
    assert not ((reason & REASON_BIT["SHEET_ID_UNCERTAIN"]) > 0).any()


@pytest.mark.parametrize("level", [0, 3])
def test_touching_bands_continue(level):
    slab = step_slab([(0, 32, [(37, 50)]), (32, 64, [(51, 64)])])     # faces 50.7 | 50.3 overlap
    p = BandParams(cell_size=8 if level == 0 else 1, **SHARP)
    f = detect_bands(slab if level == 0 else mean_pool(slab, 8), PX, p)
    assert f.cont_right[:, 3].all()
    assert np.nanmax(f.switch_jump_um) == 0.0
    assert not (classify(f, UNC_THR)[0] == FAIL_SHEET_SWITCH).any()


def test_nominal_in_air_is_a_warning():
    """Nominal slice ~10 um into the air, below the 24 um air-gap failure distance, drift inside the
    safe range: WARN_DRIFT (NOMINAL_IN_AIR), not OK."""
    thr = Thresholds(safe_drift_um=(-30.0, 30.0), fail_drift_um=(-60.0, 60.0), air_gap_fail_um=24.0, source="test")
    field, res = run(render(const(20.0), thickness_um=16.0), thr=thr)
    assert field.nominal_in_air.all()
    assert (field.gap_um < 24.0).all()
    assert (res.cls == WARN_DRIFT).all()
    names = set(reason_names(int(res.reason[3, 3])))
    assert "NOMINAL_IN_AIR" in names and not names & {"AIR_GAP_FAIL", "DRIFT_WARN", "DRIFT_FAIL"}


def test_drift_beyond_censored_edge_warns_and_crossed_edge_fails():
    thr = Thresholds(safe_drift_um=(-10.0, 10.0), fail_drift_um=(-30.0, 30.0), air_gap_fail_um=200.0,
                     fail_drift_crossed=(False, True), source="test")
    _, neg = run(render(const(-40.0), thickness_um=90.0), thr=thr)
    _, pos = run(render(const(40.0), thickness_um=90.0), thr=thr)
    assert (neg.cls == WARN_DRIFT).all()
    n = set(reason_names(int(neg.reason[3, 3])))
    assert "DRIFT_BEYOND_MEASURED" in n and "DRIFT_FAIL" not in n
    assert (pos.cls == FAIL_OFF_SHEET).all()
    p = set(reason_names(int(pos.reason[3, 3])))
    assert "DRIFT_FAIL" in p and "DRIFT_BEYOND_MEASURED" not in p


def test_flip_between_continuing_sheets_is_competing_bands_warning():
    lower, upper = const(-30.0), const(36.0)
    lower[:, 32:] = -36.0
    upper[:, 32:] = 30.0
    thr = Thresholds(safe_drift_um=(-40.0, 40.0), fail_drift_um=(-60.0, 60.0), air_gap_fail_um=50.0, source="test")
    field, res = run(render(lower, thickness_um=40.0, extra=[upper]), thr=thr)
    assert np.nanmax(field.switch_jump_um) == 0.0
    assert np.allclose(field.flip_jump_um[:, 3:5], 60.0, atol=3.0)
    assert (res.cls[:, 3:5] == WARN_DRIFT).all()
    assert ((res.reason[:, 3:5] & REASON_BIT["COMPETING_BANDS"]) > 0).all()
    assert not ((res.reason[:, :2] & REASON_BIT["COMPETING_BANDS"]) > 0).any()


def test_pair_check_reports_flip_and_switch():
    from segment_qa.runs import pair_check

    lower, upper = const(-30.0), const(36.0)
    lower[:, 32:] = -36.0
    upper[:, 32:] = 30.0
    thr = Thresholds(safe_drift_um=(-40.0, 40.0), fail_drift_um=(-60.0, 60.0), air_gap_fail_um=50.0, source="test")
    _, res = run(render(lower, thickness_um=40.0, extra=[upper]), thr=thr)
    pc = pair_check(res, (12, 28), (12, 36), thr, "flip")
    assert pc["kind"] == "flip" and pc["bands_disjoint"] and pc["both_continue_exact"] and pc["passed"]
    slab = step_slab([(0, 32, [(38, 51)]), (32, 64, [(53, 66)])])
    f = detect_bands(slab, PX, BandParams(cell_size=8, **SHARP))
    pc2 = pair_check(audit_field(f, UNC_THR), (4, 28), (4, 36), UNC_THR, "flip")
    assert pc2["kind"] == "switch" and pc2["flagged"] and not pc2["passed"]
    with pytest.raises(ValueError):
        pair_check(res, (12, 28), (12, 44), thr)


class FakeArray:
    def __init__(self, a, chunks):
        self.a, self.chunks, self.shape = a, chunks, a.shape

    def __getitem__(self, k):
        return self.a[k]


class FakeVolume:
    """In-memory stand-in for ``LayersVolume`` with a mean-pooled pyramid."""

    def __init__(self, l0, levels=(0, 3), chunk=16, tag="fake", px=PX):
        self.arrays = {lv: l0 if lv == 0 else mean_pool(l0, 2 ** lv) for lv in levels}
        self.chunk, self.cache_tag, self.pixel_size_um = chunk, tag, px
        self.ref = None

    def array(self, level=0):
        a = self.arrays[level]
        return FakeArray(a, (a.shape[0], self.chunk, self.chunk))

    def shape(self, level=0):
        return self.arrays[level].shape

    def read(self, level, y0, x0, h, w):
        return self.arrays[level][:, y0:y0 + h, x0:x0 + w].copy()

    def lateral_um(self, level):
        return 2 ** level * 7.91


def test_empty_sweep_returns_all_invalid_field():
    from segment_qa.runs import sweep_level

    vol = FakeVolume(np.zeros((109, 64, 64), np.uint8))
    field, info = sweep_level(vol, 3, block=4)
    assert field.shape == (8, 8) and not field.valid.any() and info["empty"]
    res = audit_field(field, THR)
    assert (res.cls == INVALID_PAD).all()


def test_sweep_uses_exact_l0_validity_and_halo_seams():
    """Columns 0..43 valid at L0: cell 5 (L0 40..47) is partly padded, so the mean-pooled level 3 is
    non-zero there but the exact L0 rule marks it invalid. Cell 4 (aligned boundary at 40) is kept.
    Block seams every 4 cells must give the same fields as one block."""
    from segment_qa.runs import sweep_level

    c = const(-18.0)
    c[:, 24:] = 24.0                          # a real switch across the seam between cells 2 | 3
    l0 = render(c)
    l0[:, :, 44:] = 0
    vol = FakeVolume(l0)
    assert (vol.arrays[3][:, :, 5] != 0).any()            # pyramid alone would call cell 5 valid
    small, info = sweep_level(vol, 3, block=4)
    whole, _ = sweep_level(vol, 3, block=64)
    assert small.valid[:, :5].all() and not small.valid[:, 5:].any()
    assert info["cells_removed"] == 8
    for n in ("valid", "z_top", "z_bot", "switch_jump_um", "flip_jump_um", "cont_right", "cont_down"):
        assert np.array_equal(getattr(small, n), getattr(whole, n), equal_nan=True), n
    assert np.allclose(small.switch_jump_um[:, 2:4], 42.0, atol=2.0)
    assert np.isnan(small.switch_jump_um[:, 5:]).all()


def test_pool_all_partial_cells_are_invalid():
    m = np.ones((10, 10), bool)
    m[3, 7] = False
    out = pool_all(m, 4)
    assert out.shape == (3, 3)
    assert out[0, 0] and not out[0, 1] and not out[2, 2]


def test_crop_cache_is_keyed_by_resolved_asset(tmp_path):
    from segment_qa.data import BoundedCache, read_crop

    cache = BoundedCache(tmp_path, limit=10 ** 8)
    a = FakeVolume(np.full((109, 64, 64), 7, np.uint8), tag="PHerc0001_seg_volA_aaaaaaaaaa")
    b = FakeVolume(np.full((109, 64, 64), 9, np.uint8), tag="PHerc0001_seg_volB_bbbbbbbbbb")
    assert (read_crop(a, 0, 0, 32, cache=cache) == 7).all()
    assert (read_crop(b, 0, 0, 32, cache=cache) == 9).all()        # same coordinates, other asset
    assert (tmp_path / a.cache_tag / "L0_0_0_32.npz").exists() and (tmp_path / b.cache_tag / "L0_0_0_32.npz").exists()


def test_output_and_asset_tags_are_qualified():
    from segment_qa.data import SegmentRef, asset_tag, output_tag

    r1, r2 = SegmentRef("PHerc0139", "seg1", "vol1"), SegmentRef("PHerc0139", "seg1", "vol2")
    assert output_tag(r1) == "pherc0139_seg1_vol1" and output_tag(r1) != output_tag(r2)
    assert asset_tag(r1, "a/layers.zarr") != asset_tag(r1, "a/layers-t2.zarr")
    assert asset_tag(r1, "a/layers.zarr").startswith("PHerc0139_seg1_vol1_")


def test_cache_writes_are_atomic_and_reject_oversized_entries(tmp_path):
    from segment_qa.data import BoundedCache

    cache = BoundedCache(tmp_path, limit=4000)
    noise = np.random.default_rng(0).integers(0, 255, 20000, dtype=np.uint8)
    assert cache.save_array("x/big.npz", noise) is False
    assert not (tmp_path / "x" / "big.npz").exists()
    assert cache.save_array("x/small.npz", np.zeros(10)) is True
    assert np.array_equal(cache.load_array("x/small.npz"), np.zeros(10))
    assert not list(tmp_path.rglob("*.part"))


def test_check_crop_bounds():
    from segment_qa.data import check_crop

    check_crop((109, 512, 512), 256, 256, 256, 8)
    for args in ((-1, 0, 256), (0, 300, 256), (0, 0, 0), (0, 0, 100)):
        with pytest.raises(ValueError):
            check_crop((109, 512, 512), *args, 8)


def test_audit_crop_validates_bounds_before_reading():
    from segment_qa.runs import audit_crop

    class NoRead(FakeVolume):
        def read(self, *a):
            raise AssertionError("read before bounds check")

    vol = NoRead(np.zeros((109, 64, 64), np.uint8))
    with pytest.raises(ValueError):
        audit_crop(vol, 48, 0, THR, size=32)


def test_validate_shifts():
    from segment_qa.ink_tolerance import validate_shifts

    assert validate_shifts([-4, 0, 4]) == [-4, 0, 4]
    for bad in ([0, -4, 4], [-4, 0, 0, 4], [-4, 4], [-24, 0], [0, 25], [0, 1.5], [],
                [-4, 0, float("inf")], [-4, 0, float("-inf")], [-4, 0, float("nan")]):
        with pytest.raises(ValueError):
            validate_shifts(bad)


def test_cli_rejects_bad_shifts_before_any_read(capsys):
    from segment_qa.cli import main

    with pytest.raises(SystemExit):
        main(["calibrate", "--shifts", "4,0,-4"])
    assert "sorted and unique" in capsys.readouterr().err
    for raw in ("-4,0,inf", "-4,0,-inf", "-4,0,nan"):
        with pytest.raises(SystemExit):
            main(["calibrate", f"--shifts={raw}"])
        assert "finite integer" in capsys.readouterr().err


def test_sweep_level4_and_level5_catch_isolated_20px_padding():
    """At level 4 (16 px) or level 5 (32 px), an isolated 20 x 20 L0 padded quad can sit inside a
    valid interior without creating any fully-zero neighbor cell. The full sweep must still mark
    every affected cell INVALID_PAD, matching audit_crop."""
    from segment_qa.data import SEGMENTS
    from segment_qa.runs import audit_crop, sweep_level

    # 96 x 96 L0 -> 3 x 3 cells at level 5 (32 px) and 6 x 6 cells at level 4 (16 px)
    l0 = render(const(0.0, 96, 96))
    # 20 x 20 zero quad strictly inside center level-5 cell (32..64, 32..64): L0 38..58
    l0[:, 38:58, 38:58] = 0
    vol = FakeVolume(l0, levels=(0, 4, 5), chunk=32)
    vol.ref = SEGMENTS["PHerc0139"]

    # Level 5: all 3x3 cells are non-zero in the mean-pooled pyramid, so no cell has an invalid neighbor
    assert (vol.arrays[5] != 0).any(axis=0).all()
    f5, info5 = sweep_level(vol, 5, block=4)
    res5 = audit_field(f5, THR)
    crop5, _ = audit_crop(vol, 0, 0, THR, size=3, level=5, cell=1)
    assert res5.cls[1, 1] == INVALID_PAD
    assert np.array_equal(res5.cls, crop5.cls)
    assert info5["cells_removed"] == 1 and "all level-valid cells" in info5["candidates"]

    # Level 4: L0 38..58 straddles cells 2..3 (32..48, 48..64) in both Y and X with zero fully-empty cells
    assert (vol.arrays[4] != 0).any(axis=0).all()
    f4, info4 = sweep_level(vol, 4, block=4)
    res4 = audit_field(f4, THR)
    crop4, _ = audit_crop(vol, 0, 0, THR, size=6, level=4, cell=1)
    assert (res4.cls[2:4, 2:4] == INVALID_PAD).all()
    assert np.array_equal(res4.cls, crop4.cls)
    assert info4["cells_removed"] == 4 and "all level-valid cells" in info4["candidates"]


def test_ink_window_edges_are_inclusive():
    # window samples 23..84 around z_nom = 54
    lo, hi = INK = (-31.0, 30.0)
    assert ink_window_mask(10.0, 23.0, 54, lo, hi)
    assert not ink_window_mask(10.0, 22.99, 54, lo, hi)
    assert ink_window_mask(84.0, 95.0, 54, lo, hi)
    assert not ink_window_mask(84.01, 95.0, 54, lo, hi)
    p = BandParams()
    assert p.ink_window_um[0] / 2.399 == pytest.approx(-31.0) and p.ink_window_um[1] / 2.399 == pytest.approx(30.0)


@pytest.mark.parametrize("px", [2.4, 1.2])
def test_thin_bands_use_face_thickness_at_any_pixel_size(px):
    """Two 11 um sheets 40 um apart: both kept at 2.4 and 1.2 um, since the rule is the interpolated
    face-to-face thickness in um, not a sample count."""
    depth = 109 if px == 2.4 else 217
    f = detect_bands(render(const(-20.0), thickness_um=11.0, px=px, depth=depth, extra=[const(20.0)]), px,
                     BandParams(cell_size=8))
    assert (f.n_bands == 2).all()
    assert np.allclose(f.thickness_um, f.thickness_um[0, 0], atol=0.5)


def test_min_band_rule_faces_um_vs_samples():
    """A faint 4-sample band (grey 80 against cutoff 59) has interpolated faces only 3.7 vx = 8.9 um
    apart: the faces_um rule (>= 9.5 um) drops it; the sample-count rule (round(9.5 / 2.399) = 4
    samples, the reference loop) keeps it."""
    slab = np.full((109, 64, 64), 20, np.uint8)
    slab[20:34] = 150                     # a 14-layer band keeps p90 on paper (cutoff 59)
    slab[51:55] = 80
    faces = detect_bands(slab, PX, BandParams(cell_size=8, **SHARP))
    samples = detect_bands(slab, PX, BandParams(cell_size=8, min_band_rule="samples", **SHARP))
    assert (faces.n_bands == 1).all() and (samples.n_bands == 2).all()
    with pytest.raises(ValueError):
        BandParams(min_band_rule="pixels")


def test_band_offset_curve_bootstraps_the_peak_and_guards_sparse_bins():
    from segment_qa.ink_tolerance import band_offset_curve

    rng = np.random.default_rng(2)
    shifts = np.array([-8, -4, 0, 4, 8])
    offsets = rng.uniform(-20, 0, 50)
    level = rng.uniform(0.2, 0.6, 50)
    e = offsets[:, None] - shifts[None, :] * PX
    # flat on the negative side (never drops below 0.7: censored), falls off on the positive side
    r = level[:, None] * np.where(e < 0, 1.0, np.exp(-0.5 * (e / 10.0) ** 2)) + rng.normal(0, 0.01, e.shape)
    cur = band_offset_curve(r, offsets, shifts, PX, n_boot=300, min_pairs=20)
    pb = cur["peak_bootstrap"]
    assert sum(pb["share_by_bin_um"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0 < pb["share_at_point_peak"] < 1           # flat top: the peak location is uncertain
    used = [b for b in cur["bins"] if b["used"]]
    assert all(b["n_pairs"] >= 20 for b in used)
    assert all(b["r"] is None for b in cur["bins"] if b["n_pairs"] < 20)
    fo = cur["fail_band_offset_um"]
    assert not fo["neg_reached_within_data"] and fo["neg"] == cur["support_um"][0]
    assert fo["pos_reached_within_data"]


def test_band_offset_curve_stops_at_unsupported_bins():
    from segment_qa.band_detector import BandParams as _BP  # noqa: F401
    from segment_qa.ink_tolerance import support_run

    use = np.array([True, False, True, True, True, False, True])
    assert support_run(use, 3).tolist() == [2, 3, 4]


def test_calibration_flag_share_never_skips_crops():
    from segment_qa.ink_tolerance import calibration_flag_share

    recs = [{"y0": 0, "x0": 0}, {"y0": 0, "x0": 256}]
    out = calibration_flag_share(recs, THR, lambda rec: render(const(0.0)), PX)
    assert out["n_crops"] == 2 and out["share"]["OK"] == 1.0
    assert "specificity" not in out

    def missing(rec):
        raise FileNotFoundError("evicted and offline")

    with pytest.raises(FileNotFoundError):
        calibration_flag_share(recs, THR, missing, PX)


def test_reference_concordance_holds_out_calibration_blocks():
    from segment_qa.runs import reference_concordance

    rng = np.random.default_rng(0)
    gh = gw = 512                                    # ds8 grid: 16 x 16 tiles, 4 x 4 blocks
    cls = np.zeros((gh, gw), np.uint8)
    cls[:, 256:] = FAIL_OFF_SHEET                    # right half fails
    ref = rng.integers(20, 230, (gh, gw)).astype(np.uint8)
    noise = rng.normal(0, 60, (gh, gw))
    noise[:, :256] *= 0.2                            # left half agrees well
    ink = np.clip(ref + noise, 1, 255).astype(np.uint8)
    full = reference_concordance(cls, ink, ref, n_boot=200)
    # a 256 px L0 crop at ds8 (40, 40)..(71, 71): 2 x 2 tiles, all inside block (0, 0)
    held = reference_concordance(cls, ink, ref, exclude_l0=[(40 * 8, 40 * 8, 256)], n_boot=200)
    assert full["n_tiles"] == 256 and full["n_blocks"] == 16
    assert held["held_out"]["blocks_excluded"] == 1 and held["held_out"]["tiles_excluded_by_block"] == 16
    assert held["held_out"]["tiles_overlapping_calibration_crops"] == 4
    # a crop straddling a block corner removes all four blocks it touches
    corner = reference_concordance(cls, ink, ref, exclude_l0=[(100 * 8, 100 * 8, 256)], n_boot=50)
    assert corner["held_out"]["blocks_excluded"] == 4 and corner["n_tiles"] == 192
    assert held["n_tiles"] == 240 and held["n_blocks"] == 15
    d = held["mean_r_no_fail_minus_fail_ge_20pct"]
    assert d["mean"] > 0 and d["ci95"][0] > 0
    assert "block bootstrap" in held["ci_method"]


def test_thresholds_from_calibration_reads_censored_flags(tmp_path):
    import json

    cal = {"thresholds": {"reference_offset_um": 0.0, "status": "provisional",
                          "band_frame": {"safe_drift_um": [-38.4, 9.6], "fail_drift_um": [-76.8, 25.5],
                                         "fail_drift_crossed": [False, True], "air_gap_fail_um": 25.5,
                                         "switch_jump_um": 40.0, "drift_window_cells": 5, "switch_min_cells": 4,
                                         "provisional": True}}}
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal), encoding="utf-8")
    t = Thresholds.from_calibration(p)
    assert t.fail_drift_crossed == (False, True) and t.provisional and "provisional" in t.source
