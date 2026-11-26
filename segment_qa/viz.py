"""Figures: ink tolerance curve, full-segment QA overlay, three-scroll hotspot verification."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .auditor import CLASS_COLORS, CLASS_NAMES, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH, INVALID_PAD, OK, WARN_DRIFT

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
ZONE = {"safe": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def _shade_zones(ax, thr_win_um):
    lo_safe, hi_safe = thr_win_um["safe_dz_range_um"]
    lo_w, hi_w = thr_win_um["warn_dz_range_um"]
    xmin, xmax = ax.get_xlim()
    ax.axvspan(lo_safe, hi_safe, color=ZONE["safe"], alpha=0.10, lw=0, zorder=0)
    ax.axvspan(lo_w, lo_safe, color=ZONE["warn"], alpha=0.14, lw=0, zorder=0)
    ax.axvspan(hi_safe, hi_w, color=ZONE["warn"], alpha=0.14, lw=0, zorder=0)
    ax.axvspan(xmin, lo_w, color=ZONE["fail"], alpha=0.08, lw=0, zorder=0)
    ax.axvspan(hi_w, xmax, color=ZONE["fail"], alpha=0.08, lw=0, zorder=0)
    ax.set_xlim(xmin, xmax)


def plot_tolerance_curve(result: dict, r_ref: np.ndarray, path: Path) -> None:
    shifts_um = np.array(result["shifts_um"])
    thr = result["thresholds"]
    win = thr["window_frame"]
    agree = result["agreement_vs_reference"]
    ret = result["agreement_retention"]
    selfc = result["self_consistency_vs_dz0"]
    fg = result["foreground_retention"]
    px = result["segment"]["pixel_size_um"]

    curve = result.get("band_offset_curve")
    ncol = 3 if curve else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.6 * ncol, 5.6), layout="constrained")
    a1, a2 = axes[0], axes[1]
    pad = 0.04 * (shifts_um.max() - shifts_um.min())
    for ax in (a1, a2):
        ax.set_xlim(shifts_um.min() - pad, shifts_um.max() + pad)

    # (a) absolute agreement with the independent 1.129 um reference
    for row in r_ref:
        a1.plot(shifts_um, row, color=INK2, lw=0.5, alpha=0.18, zorder=1)
    mean = np.array([a["mean"] for a in agree])
    lo = np.array([a["ci95"][0] for a in agree])
    hi = np.array([a["ci95"][1] for a in agree])
    q1 = np.array([a["iqr"][0] for a in agree])
    q3 = np.array([a["iqr"][1] for a in agree])
    med = np.array([a["median"] for a in agree])
    a1.fill_between(shifts_um, q1, q3, color=BLUE, alpha=0.12, lw=0, zorder=2)
    a1.fill_between(shifts_um, lo, hi, color=BLUE, alpha=0.30, lw=0, zorder=3)
    a1.plot(shifts_um, mean, color=BLUE, lw=2, marker="o", ms=5, zorder=4)
    a1.plot(shifts_um, med, color=BLUE, lw=1, ls="--", zorder=4)
    _shade_zones(a1, win)
    a1.set(xlabel="Ink-window shift dz (µm); layers [23+dz, 85+dz)",
           ylabel="Pearson r vs. independent 1.129 µm ink map (ds8)",
           title=f"(a) Agreement with the independent reference, {result['n_crops']} PHerc0139 crops")
    a1.legend(handles=[
        Line2D([], [], color=BLUE, lw=2, marker="o", label="Mean across crops"),
        Patch(color=BLUE, alpha=0.30, label="95% bootstrap CI of the mean"),
        Line2D([], [], color=BLUE, lw=1, ls="--", label="Median"),
        Patch(color=BLUE, alpha=0.12, label="Interquartile range"),
        Line2D([], [], color=INK2, lw=0.6, alpha=0.4, label="Individual crops"),
    ], loc="lower center", fontsize=8.5, ncols=2)

    # (b) retention relative to the as-published placement
    p = np.array([r["point"] for r in ret])
    plo = np.array([r["ci95"][0] for r in ret])
    phi = np.array([r["ci95"][1] for r in ret])
    a2.fill_between(shifts_um, plo, phi, color=BLUE, alpha=0.25, lw=0)
    a2.plot(shifts_um, p, color=BLUE, lw=2, marker="o", ms=5, label="Reference agreement retention, r(dz) / r(0)")
    a2.plot(shifts_um, [s["mean"] for s in selfc], color=ORANGE, lw=1.6, marker="s", ms=4,
            label="Self-consistency r vs. dz = 0 prediction")
    a2.plot(shifts_um, [s["mean"] for s in fg], color=AQUA, lw=1.6, marker="^", ms=5,
            label="Foreground retention (p > 0.5 kept)")
    rule = thr["rule"]
    a2.axhline(float(rule["safe"].split(">=")[-1]), color=ZONE["safe"], lw=1, ls=":")
    a2.axhline(float(rule["fail"].split("<")[-1].split()[0]), color=ZONE["fail"], lw=1, ls=":")
    _shade_zones(a2, win)
    a2.set_ylim(min(0.0, float(np.nanmin(plo)) - 0.05), max(1.08, float(np.nanmax(phi)) + 0.03))
    a2.set(xlabel="Ink-window shift dz (µm)", ylabel="Fraction of the dz = 0 value",
           title="(b) Retention and derived thresholds")
    s_lo, s_hi = win["safe_dz_range_um"]
    f = win["fail_dz_um"]
    zones = [
        Patch(color=ZONE["safe"], alpha=0.25, label=f"Safe: {s_lo:+.1f} to {s_hi:+.1f} µm"),
        Patch(color=ZONE["warn"], alpha=0.35, label=f"Warn: to {f['neg']:+.1f} / {f['pos']:+.1f} µm"),
        Patch(color=ZONE["fail"], alpha=0.20, label="Fail: beyond"),
    ]
    h, _ = a2.get_legend_handles_labels()
    a2.legend(handles=h + zones, loc="lower center", fontsize=8.5, ncols=2)
    for ax in (a1, a2):
        sec = ax.secondary_xaxis("top", functions=(lambda u: u / px, lambda v: v * px))
        sec.set_xlabel("layers (2.399 µm voxels)", color=INK2)
        sec.tick_params(colors=INK2)
    if curve:
        _plot_band_offset(axes[2], result, curve, px)
    fig.suptitle("ink_canonical_2um tolerance to surface-window placement  ·  PHerc0139 w023, 2.399 µm",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_band_offset(ax, result: dict, curve: dict, px: float) -> None:
    """(c) agreement vs the band position each prediction saw, with the auditor's band-frame zones."""
    bins = [b for b in curve["bins"] if b["used"]]
    x = np.array([b["center_um"] for b in bins])
    r = np.array([b["r"] for b in bins])
    lo = np.array([b["r_ci95"][0] for b in bins])
    hi = np.array([b["r_ci95"][1] for b in bins])
    band = result["thresholds"]["band_frame"]
    s_lo, s_hi = band["safe_drift_um"]
    f_lo, f_hi = band["fail_drift_um"]
    pad = curve["bin_um"]
    ax.set_xlim(x.min() - pad, x.max() + pad)
    xmin, xmax = ax.get_xlim()
    ax.axvspan(s_lo, s_hi, color=ZONE["safe"], alpha=0.10, lw=0, zorder=0)
    ax.axvspan(f_lo, s_lo, color=ZONE["warn"], alpha=0.14, lw=0, zorder=0)
    ax.axvspan(s_hi, f_hi, color=ZONE["warn"], alpha=0.14, lw=0, zorder=0)
    crossed = band.get("fail_drift_crossed", [True, True])
    # a censored side has no measured failure: beyond it stays a (hatched) warning
    ax.axvspan(xmin, f_lo, color=ZONE["fail" if crossed[0] else "warn"], alpha=0.08, lw=0, zorder=0,
               hatch=None if crossed[0] else "//")
    ax.axvspan(f_hi, xmax, color=ZONE["fail" if crossed[1] else "warn"], alpha=0.08, lw=0, zorder=0,
               hatch=None if crossed[1] else "//")
    ax.set_xlim(xmin, xmax)
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.28, lw=0, zorder=2)
    ax.plot(x, r, color=BLUE, lw=2, marker="o", ms=5, zorder=3, label="Fitted r (crop effect removed)")
    for xi, n in zip(x, [b["n_pairs"] for b in bins]):
        ax.annotate(str(n), (xi, 1), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7,
                    color=INK2, xycoords=("data", "axes fraction"))
    offs = np.array([c["band_delta_um_median"] for c in result["crops"] if c["band_delta_um_median"] is not None])
    ax.plot(offs, np.full(offs.shape, 0.035), "|", color=ORANGE, ms=10, mew=1.2, transform=ax.get_xaxis_transform(),
            zorder=3, label="Crop band offsets at dz = 0")
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.set(xlabel="Band midpoint relative to the nominal anchor z = 54 (µm); − = lower layer index",
           ylabel="Pearson r vs. 1.129 µm reference",
           title="(c) Agreement vs. the band position the model saw")
    zones = [
        Patch(color=ZONE["safe"], alpha=0.25, label=f"Safe: {s_lo:+.1f} to {s_hi:+.1f} µm"),
        Patch(color=ZONE["warn"], alpha=0.35, label=f"Warn: to {f_lo:+.1f}{'' if crossed[0] else ' (censored)'}"
                                                    f" / {f_hi:+.1f}{'' if crossed[1] else ' (censored)'} µm"),
        Patch(color=ZONE["fail"], alpha=0.20, label="Fail: beyond a crossed edge"),
    ]
    if not all(crossed):
        zones.append(Patch(facecolor=ZONE["warn"], alpha=0.35, hatch="//",
                           label="Beyond censored edge: WARN (DRIFT_BEYOND_MEASURED)"))
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + [Patch(color=BLUE, alpha=0.28, label="95% crop-bootstrap CI")] + zones,
              loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=8.5, ncols=1)
    pk = curve.get("peak_bootstrap", {})
    ax.text(0.99, 0.9, f"top row: crop × shift pairs per {curve['bin_um']:g} µm bin (bins < {curve.get('min_pairs', 20)} "
            f"pairs dropped)" + chr(10) + f"bootstrapped best bin: {curve['peak_center_um']:+.1f} µm in "
            f"{100 * (pk.get('share_at_point_peak') or 0):.0f}% of replicates; thresholds provisional",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
    ax.set_ylim(min(0.0, float(lo.min()) - 0.02), float(hi.max()) + 0.08)
    sec = ax.secondary_xaxis("top", functions=(lambda u: u / px, lambda v: v * px))
    sec.set_xlabel("layers (2.399 µm voxels)", color=INK2)
    sec.tick_params(colors=INK2)


def class_cmap() -> tuple[ListedColormap, dict]:
    order = [OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH, INVALID_PAD]
    return ListedColormap([CLASS_COLORS[c] for c in order]), {c: i for i, c in enumerate(order)}


def class_index_image(cls: np.ndarray) -> np.ndarray:
    _, lut = class_cmap()
    out = np.zeros(cls.shape, np.uint8)
    for c, i in lut.items():
        out[cls == c] = i
    return out


def class_legend(counts: dict | None = None, include_pad=True):
    handles = []
    for c in (OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH, INVALID_PAD):
        if c == INVALID_PAD and not include_pad:
            continue
        label = f"{c}: {CLASS_NAMES[c]}"
        if counts and c != INVALID_PAD:
            label += f"  ({100 * counts[CLASS_NAMES[c]]['share_of_valid']:.1f}%)"
        handles.append(Patch(facecolor=CLASS_COLORS[c], edgecolor=AXIS, label=label))
    return handles


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("drift", [BLUE, "#e1e0d9", ORANGE])
    cmap.set_bad(GRID)
    return cmap


def _class_rgb(cls: np.ndarray) -> np.ndarray:
    from .auditor import class_rgb

    return class_rgb(cls)


def plot_full_overlay(res, ink: np.ndarray, hotspots: list, path: Path, marks=(), calibration: dict | None = None,
                      check: dict | None = None, zoom_half_l0: int = 1536, ref_conc: dict | None = None) -> None:
    """Full-segment QA class map, drift map and the published ds8 ink preview side by side,
    with zoom panels around the first marked point."""
    cls = res.cls
    f = res.cell_l0
    counts = res.class_counts()
    t = res.thresholds
    fig = plt.figure(figsize=(18, 14.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], left=0.04, right=0.985, top=0.9, bottom=0.135,
                          hspace=0.28, wspace=0.06)
    extent = (0, cls.shape[1] * f, cls.shape[0] * f, 0)
    ax_c, ax_d, ax_i = (fig.add_subplot(gs[0, k]) for k in range(3))
    ax_c.imshow(_class_rgb(cls), extent=extent, interpolation="antialiased")
    ax_c.set_title(f"(a) QA class per {f}×{f} px cell ({f * res.field.pixel_size_um:.0f} µm)", loc="left")
    lo, hi = t.fail_drift_um
    lim = max(abs(lo), abs(hi))
    im = ax_d.imshow(np.ma.masked_invalid(res.drift_um), extent=extent, cmap=diverging_cmap(), vmin=-lim, vmax=lim,
                     interpolation="antialiased")
    ax_d.set_title("(b) Band offset from the nominal anchor z = 54 (µm)", loc="left")
    cax = ax_d.inset_axes([0.08, -0.13, 0.84, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"drift_um: same-sheet {t.drift_window_cells}×{t.drift_window_cells}-cell median; − = band at lower "
                 "layer index", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    for v in (*t.safe_drift_um, *t.fail_drift_um):
        cb.ax.axvline(v, color=INK, lw=0.8)
    ax_i.imshow(ink, cmap="gray", extent=(0, ink.shape[1] * 8, ink.shape[0] * 8, 0), interpolation="antialiased")
    ax_i.set_title("(c) Published ink_canonical_2um prediction (ds8)", loc="left")
    cal_matches = bool(
        calibration
        and calibration.get("segment", {}).get("sample") == res.meta.get("sample")
        and calibration.get("segment", {}).get("volume") == res.meta.get("volume")
    )
    if cal_matches:
        for cr in calibration.get("crops", []):
            ax_i.add_patch(plt.Rectangle((cr["x0"], cr["y0"]), 256, 256, fill=False, ec=BLUE, lw=0.9))
    for ax in (ax_c, ax_d, ax_i):
        ax.grid(False)
        ax.set_xlabel("X (level-0 px)", labelpad=1)
        if ax is not ax_c:
            ax.tick_params(labelleft=False)
        for y, x, label in marks:
            ax.add_patch(plt.Circle((x, y), 420, fill=False, ec=INK, lw=1.4))
            ax.annotate(label, (x, y), xytext=(x + 700, y - 900), fontsize=9, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax_c.set_ylabel("Y (level-0 px)")
    handles = class_legend(counts)
    if cal_matches:
        handles.append(Patch(facecolor="none", edgecolor=BLUE, label=f"Calibration crops (n={len(calibration['crops'])})"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.945), ncols=6, fontsize=9.5)

    if marks:
        y, x, label = marks[0]
        y0, y1 = max(0, y - zoom_half_l0), y + zoom_half_l0
        x0, x1 = max(0, x - zoom_half_l0), x + zoom_half_l0
        sl = (slice(y0 // f, y1 // f), slice(x0 // f, x1 // f))
        ext = (x0, x0 + (sl[1].stop - sl[1].start) * f, y0 + (sl[0].stop - sl[0].start) * f, y0)
        z_c, z_d, z_i = (fig.add_subplot(gs[1, k]) for k in range(3))
        z_c.imshow(_class_rgb(cls[sl]), extent=ext, interpolation="nearest")
        z_d.imshow(np.ma.masked_invalid(res.drift_um[sl]), extent=ext, cmap=diverging_cmap(), vmin=-lim, vmax=lim,
                   interpolation="nearest")
        z_i.imshow(ink[y0 // 8:y1 // 8, x0 // 8:x1 // 8], cmap="gray", extent=ext, interpolation="nearest")
        for ax, name in ((z_c, "classes"), (z_d, "drift"), (z_i, "ink preview")):
            ax.grid(False)
            ax.plot([x], [y], marker="+", ms=16, mew=1.6, color=ORANGE if ax is z_i else INK)
            ax.set_title(f"Zoom on {label}: {name}", loc="left", fontsize=10)
            ax.set_xlabel("X (level-0 px)")
        for ax in (ax_c, ax_d, ax_i):
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=0.8, ls="--"))
    share = {k: v["share_of_valid"] for k, v in counts.items() if v["share_of_valid"] is not None}
    txt = (f"Thresholds ({t.source}): safe drift {t.safe_drift_um[0]:+.1f} to {t.safe_drift_um[1]:+.1f} µm, "
           f"fail beyond {t.fail_drift_um[0]:+.1f}{'' if t.fail_drift_crossed[0] else ' (censored: WARN only)'} / "
           f"{t.fail_drift_um[1]:+.1f}{'' if t.fail_drift_crossed[1] else ' (censored: WARN only)'} µm"
           f"{' [provisional outside PHerc0139]' if t.provisional else ''}, reference offset "
           f"{t.reference_offset_um:+.1f} µm, sheet switch >= {t.switch_jump_um:.0f} µm, air gap >= "
           f"{t.air_gap_fail_um:.1f} µm. Valid cells: "
           f"{sum(v['cells'] for k, v in counts.items() if k != 'INVALID_PAD'):,}; "
           f"FAIL share {100 * (share['FAIL_OFF_SHEET'] + share['FAIL_SHEET_SWITCH']):.1f}%.")
    if check and "exact_agreement" in check:
        kci = check.get("cohen_kappa_ci95_tile_bootstrap")
        txt += (f"\nL0 cross-check, {check['n_tiles']} interior tiles (halo-trimmed, {check['cells']:,} cells, 8×8 px "
                f"cells vs. this map): exact agreement {100 * check['exact_agreement']:.1f}%, Cohen κ = "
                f"{check['cohen_kappa']:.3f}" + (f" [{kci[0]:.3f}, {kci[1]:.3f}] tile bootstrap" if kci else "") +
                f", fail/non-fail agreement {100 * check['fail_vs_nonfail_agreement']:.1f}%.")
        pb = check.get("padding_boundary")
        if pb and pb.get("cells"):
            txt += (f" Padding boundary ({pb['n_tiles']} tiles): valid-mask agreement "
                    f"{100 * pb['valid_mask_agreement']:.2f}%.")
    if ref_conc and "by_fail_share" in ref_conc:
        parts = ", ".join(f"{b['bin']}: {b['mean_r']:.3f} (n={b['n_tiles']})" for b in ref_conc["by_fail_share"])
        sp = ref_conc["spearman_r_vs_fail_share"]
        spci = sp.get("ci95_block_bootstrap")
        txt += (f"\nPublished 2.4 µm vs. independent 1.129 µm ink map, mean Pearson r per 256 px tile by FAIL share "
                f"(blocks touching calibration crops held out): {parts}; Spearman ρ = {sp['rho'] if sp['rho'] is None else format(sp['rho'], '+.3f')}"
                + (f" [{spci[0]:+.3f}, {spci[1]:+.3f}] block bootstrap." if spci else "."))
    import textwrap
    txt = "\n".join(textwrap.fill(para, 205) for para in txt.split("\n"))
    fig.text(0.04, 0.006, txt, fontsize=9, color=INK2, va="bottom")
    m = res.meta
    fig.suptitle(f"{m['sample']} {m['segment']}: raw-CT placement QA over the whole segment "
                 f"(level-{m['level']} sweep, {m['pixel_size_um']} µm layers)", x=0.04, y=0.985, ha="left",
                 fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _section(ax, res, slab, axis: str, line_px: int, focus_px: int | None, title: str):
    """Raw-CT cut along the crop with band faces, midpoints coloured by class and a class strip."""
    f = res.field
    D = slab.shape[0]
    ci = line_px // f.cell_size
    if axis == "x":
        plane = slab[:, line_px, :]
        top, bot, mid, cls = f.z_top[ci], f.z_bot[ci], f.z_mid[ci], res.cls[ci]
        origin = res.origin_l0[1]
        moving, fixed_name = "X", "Y"
        fixed = res.origin_l0[0] + line_px
    else:
        plane = slab[:, :, line_px]
        top, bot, mid, cls = f.z_top[:, ci], f.z_bot[:, ci], f.z_mid[:, ci], res.cls[:, ci]
        origin = res.origin_l0[0]
        moving, fixed_name = "Y", "X"
        fixed = res.origin_l0[1] + line_px
    vals = plane[plane > 0]
    lo, hi = np.percentile(vals, [3, 99]) if vals.size else (0, 255)
    n = plane.shape[1]
    ax.imshow(plane, cmap="gray", vmin=lo, vmax=hi, aspect="auto", origin="lower",
              extent=(origin - 0.5, origin + n - 0.5, -0.5, D - 0.5), interpolation="nearest")
    ax.axhspan(23 - 0.5, 85 - 0.5, color=BLUE, alpha=0.08, lw=0)
    ax.axhline(D // 2, color="#d03b3b", ls="--", lw=1.1)
    xs = origin + np.arange(top.size) * f.cell_size + f.cell_size / 2
    ax.scatter(xs, top, color="#71e2fa", s=8, marker="_", lw=0.9)
    ax.scatter(xs, bot, color="#71e2fa", s=8, marker="_", lw=0.9)
    for c in np.unique(cls):
        sel = (cls == c) & np.isfinite(mid)
        ax.scatter(xs[sel], mid[sel], color=CLASS_COLORS[int(c)], s=18, edgecolors=INK, linewidths=0.4, zorder=5)
    ax.imshow(_class_rgb(cls[None, :]), extent=(origin - 0.5, origin + n - 0.5, D + 1, D + 7), aspect="auto",
              interpolation="nearest")
    ax.set_ylim(-0.5, D + 7)
    ax.set_xlim(origin - 0.5, origin + n - 0.5)
    if focus_px is not None:
        ax.axvline(origin + focus_px, color="white", lw=0.9, ls=":")
    ax.set_xlabel(f"{moving} (level-0 px) at {fixed_name} = {fixed}")
    ax.set_ylabel("layer z")
    ax.set_yticks([0, 23, 54, 84, 108])
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=9.2)


def _map(ax, res, slab, mark_yx, title: str):
    D = slab.shape[0]
    nom = slab[D // 2].astype(float)
    vals = nom[nom > 0]
    lo, hi = np.percentile(vals, [3, 99]) if vals.size else (0, 255)
    y0, x0 = res.origin_l0
    ax.imshow(nom, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest",
              extent=(x0 - 0.5, x0 + slab.shape[2] - 0.5, y0 + slab.shape[1] - 0.5, y0 - 0.5))
    c = res.cell_l0
    rgba = np.dstack([_class_rgb(res.cls), np.full(res.cls.shape, 125, np.uint8)])
    ax.imshow(rgba, extent=(x0, x0 + res.cls.shape[1] * c, y0 + res.cls.shape[0] * c, y0), interpolation="nearest")
    if mark_yx is not None:
        ax.plot([mark_yx[1]], [mark_yx[0]], marker="s", ms=9, mfc="none", mec="white", mew=1.8)
    ax.set_title(title, loc="left", fontsize=9.2)
    ax.set_xticks([x0, x0 + 128, x0 + 256])
    ax.set_yticks([y0, y0 + 128, y0 + 256])
    ax.tick_params(labelsize=7.5)
    ax.grid(False)


def plot_hotspot_verification(panels, thr, path: Path) -> None:
    import textwrap

    rows = len(panels)
    height = 4.6 * rows + 1.9
    fig = plt.figure(figsize=(21, height))
    gs = fig.add_gridspec(rows, 4, width_ratios=[1, 2.1, 1, 2.1], left=0.045, right=0.99, top=1 - 1.75 / height,
                          bottom=0.55 / height + 0.03, hspace=0.62, wspace=0.2)

    def fmt(v):
        return "n/a" if v is None else f"{v:+.1f} µm"

    for r, (case, hot, hot_slab, ctrl, ctrl_slab, entry) in enumerate(panels):
        cell = entry["hotspot_cell"]
        hy, hx = case.hotspot_yx
        oy, ox = hot.origin_l0
        line = (hy - oy) if case.section_axis == "x" else (hx - ox)
        focus = (hx - ox) if case.section_axis == "x" else (hy - oy)
        cc = entry["hotspot_crop_class_counts"]
        _map(fig.add_subplot(gs[r, 0]), hot, hot_slab, case.hotspot_yx,
             f"{case.sample} Row {case.row} hotspot crop\nFAIL {100 * entry['hotspot_crop_fail_share']:.1f}%, "
             f"OK {100 * cc['OK']['share_of_valid']:.1f}%")
        in_window = cell["competing_bands"] + (0 if cell["window_empty"] else 1)
        reasons = ", ".join(r_ for r_ in cell["reasons"] if r_ != "DRIFT_WARN" or "DRIFT_FAIL" not in cell["reasons"])
        _section(fig.add_subplot(gs[r, 1]), hot, hot_slab, case.section_axis, line, focus,
                 f"{case.sample} {entry['segment']}  ·  cell ({hy}, {hx}) → {cell['class_name']}\n"
                 f"reasons: {reasons or '-'}\n"
                 f"drift {fmt(cell['drift_um'])} (cell {fmt(cell['drift_raw_um'])}), gap {fmt(cell['gap_um'])}, "
                 f"bands in ink window {in_window}" + chr(10) +
                 f"switch jump {fmt(cell['switch_jump_um'])} (raw midpoint step {fmt(cell['midpoint_step_um'])})")
        ctrl_y, ctrl_x = entry["control_origin_l0"]
        ccc = entry["control_class_counts"]
        _map(fig.add_subplot(gs[r, 2]), ctrl, ctrl_slab, None,
             f"Control crop ({ctrl_y}, {ctrl_x})\nFAIL {100 * entry['control_fail_share']:.1f}%, "
             f"OK {100 * ccc['OK']['share_of_valid']:.1f}%")
        cs = entry["control_selection"]
        crit = textwrap.fill(f"Control (candidate #{cs.get('rank', 1)}, {'clean' if cs.get('clean', True) else 'NOT clean'}"
                             f"): first a-priori candidate whose L0 CT audit has FAIL < 2%", 88)
        pc = entry.get("pair_check")
        if pc:
            crit += (f"\nHotspot pair {tuple(pc['a_l0_yx'])}→{tuple(pc['b_l0_yx'])}: {pc['kind']}, step "
                     f"{pc['step_um']} µm, {'flagged' if pc['flagged'] else 'NOT flagged'}")
        _section(fig.add_subplot(gs[r, 3]), ctrl, ctrl_slab, "x", 132, None,
                 f"{crit}\nsection at crop row 132 · WARN {100 * ccc['WARN_DRIFT']['share_of_valid']:.1f}%, OFF_SHEET "
                 f"{100 * ccc['FAIL_OFF_SHEET']['share_of_valid']:.1f}%, SWITCH "
                 f"{100 * ccc['FAIL_SHEET_SWITCH']['share_of_valid']:.1f}%")
    handles = class_legend(include_pad=True) + [
        Line2D([], [], color="#d03b3b", ls="--", label="Nominal z = 54"),
        Line2D([], [], color="#71e2fa", marker="_", lw=0, ms=10, label="Band faces"),
        Patch(color=BLUE, alpha=0.2, label="Ink window [23, 85)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 1 - 0.62 / height), ncols=8, fontsize=9.5)
    fig.suptitle("Three-scroll hotspot verification: auditor class at Sol's label-drift hotspots vs. controls",
                 x=0.045, y=1 - 0.12 / height, ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.045, 0.12 / height, f"Thresholds ({thr.source}): safe {thr.safe_drift_um[0]:+.1f} to "
             f"{thr.safe_drift_um[1]:+.1f} µm, fail beyond {thr.fail_drift_um[0]:+.1f}"
             f"{'' if thr.fail_drift_crossed[0] else ' (censored: WARN + DRIFT_BEYOND_MEASURED)'} / "
             f"{thr.fail_drift_um[1]:+.1f}{'' if thr.fail_drift_crossed[1] else ' (censored)'} µm "
             f"(band offset from the nominal anchor z = 54), drift = same-sheet median of {thr.drift_window_cells}×"
             f"{thr.drift_window_cells} cells, sheet switch >= {thr.switch_jump_um:.0f} µm in >= {thr.switch_min_cells} "
             f"cells, air gap >= {thr.air_gap_fail_um:.1f} µm.\nMaps: nominal slice z = 54 with a class tint per 8×8 px "
             "cell; white square = hotspot cell. Sections: raw CT through the hotspot (dotted line), band midpoints "
             "coloured by class, class strip along the top.", fontsize=9, color=INK2)
    fig.savefig(path, dpi=130)
    plt.close(fig)
