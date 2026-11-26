"""Read a bounded set of full-depth Vesuvius surface-volume crops from S3."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr
from matplotlib.lines import Line2D
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import rankdata, spearmanr
from zarr.core.sync import sync
from zarr.storage import FsspecStore


ROOT = Path(__file__).resolve().parents[1]
BUCKET = "vesuvius-challenge-open-data"
SIZE = 256
STRIDE = 8
INK_HALF_WINDOW = 31


@dataclass(frozen=True)
class Segment:
    sample: str
    key: str
    volume: str


@dataclass(frozen=True)
class Crop:
    name: str
    segment: Segment
    y: int
    x: int
    context: str


W023 = Segment("PHerc0139", "20260311000000", "20260102150214")
W029 = Segment("PHerc1667", "20251212185248", "20251217075048")
W053 = Segment("PHercParis4", "20260623144957", "20260411134726")

CROPS = (
    Crop("A", W023, 5120, 16128, "Upper interior, high ink contrast"),
    Crop("B", W023, 12544, 15616, "Middle interior, high ink contrast"),
    Crop("C", W023, 20096, 19456, "Lower interior, high ink contrast"),
    Crop("D", W023, 20608, 19328, "Lower interior, preview-dark"),
    Crop("E", W023, 7680, 21632, "Right mesh edge"),
    Crop("F", W023, 16000, 1280, "Left mesh edge"),
    Crop("G", W029, 18944, 15616, "Interior"),
    Crop("H", W029, 18944, 2048, "Left side"),
    Crop("I", W053, 24320, 93312, "Interior"),
    Crop("J", W053, 24320, 46464, "Interior, west"),
)


def record(catalog, spec):
    return catalog[spec.sample]["segments"][spec.key]


def asset(catalog, spec, kind):
    return next(
        item["origins"][0]["path"].strip("/")
        for item in record(catalog, spec)["data"]
        if item["type"] == kind and item["parameters"]["target_volume"] == spec.volume
    )


def open_layers(fs, catalog, spec):
    key = asset(catalog, spec, "layers-zarr")
    store = FsspecStore(fs, path=f"{BUCKET}/{key}", read_only=True)
    group = zarr.open_group(store, mode="r")
    metadata = group.attrs["multiscales"][0]
    transforms = metadata["datasets"][0]["coordinateTransformations"]
    spacing = next(item["scale"][0] for item in transforms if item["type"] == "scale")
    return group["0"], spacing


def read_crop(array, crop):
    depth, height, width = array.shape
    if not (0 <= crop.y <= height - SIZE and 0 <= crop.x <= width - SIZE):
        raise ValueError(f"Crop {crop.name} is outside {array.shape}")
    cz, cy, cx = array.chunks
    if cz != depth or crop.y % cy or crop.x % cx or SIZE % cy or SIZE % cx:
        raise ValueError(f"Crop {crop.name} does not align with chunks {array.chunks}")
    return array[:, crop.y:crop.y + SIZE, crop.x:crop.x + SIZE]


@dataclass
class Measurement:
    bounds: np.ndarray
    count: np.ndarray
    valid: np.ndarray
    nominal_gap: np.ndarray
    window_empty: np.ndarray

    @property
    def center(self):
        return self.bounds.mean(axis=-1)


def measure(volume, contrast=0.30):
    """Find resolvable CT bands on 8x8 XY cells, with no preferred z direction."""
    depth = volume.shape[0]
    grid = SIZE // STRIDE
    profiles = volume.reshape(depth, grid, STRIDE, grid, STRIDE).mean(axis=(2, 4))
    profiles = ndi.gaussian_filter1d(profiles, 1.25, axis=0)
    air, bright = np.percentile(profiles, [15, 90], axis=0)
    span = bright - air
    cutoff = air + np.maximum(12, contrast * span)
    peak_floor = air + np.maximum(22, (contrast + 0.12) * span)
    # Zarr renders invalid surface coordinates as all-zero columns.
    supported = (volume != 0).any(axis=0)
    valid = supported.reshape(grid, STRIDE, grid, STRIDE).all(axis=(1, 3))
    bounds = np.full((grid, grid, 2), np.nan)
    count = np.zeros((grid, grid), dtype=int)
    nominal_gap = np.zeros((grid, grid), dtype=bool)
    window_empty = np.zeros((grid, grid), dtype=bool)
    mid = depth // 2

    for iy, ix in zip(*np.nonzero(valid)):
        profile = profiles[:, iy, ix]
        excess = profile - cutoff[iy, ix]
        inside = excess > 0
        starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
        ends = np.flatnonzero(inside & ~np.r_[inside[1:], False])
        bands = []
        for start, end in zip(starts, ends):
            if end - start + 1 < 4 or profile[start:end + 1].max() < peak_floor[iy, ix]:
                continue
            # A band touching the stack limit contributes to ambiguity/gap checks,
            # but its midpoint and clipped boundary cannot be measured.
            low = start - excess[start] / (excess[start] - excess[start - 1]) if start else 0
            high = end + excess[end] / (excess[end] - excess[end + 1]) if end < depth - 1 else depth - 1
            bands.append((low, high, start == 0 or end == depth - 1))
        count[iy, ix] = len(bands)
        if not bands:
            continue
        low, high, clipped = min(
            bands,
            key=lambda band: (max(band[0] - mid, mid - band[1], 0),
                              abs((band[0] + band[1]) / 2 - mid)),
        )
        nominal_gap[iy, ix] = max(low - mid, mid - high) >= 2 and excess[mid] <= 0
        window_empty[iy, ix] = all(
            upper < mid - INK_HALF_WINDOW or lower > mid + INK_HALF_WINDOW - 1
            for lower, upper, _ in bands
        )
        if not clipped:
            bounds[iy, ix] = (low, high)
    return Measurement(bounds, count, valid, nominal_gap, window_empty)


def stats(values, spacing):
    if not len(values):
        return None
    summary = np.percentile(values, [50, 90, 95, 100])
    return {"vx": summary.round(2).tolist(), "um": (summary * spacing).round(2).tolist()}


def strongest_jump(center):
    """Find the largest jump between adjacent cells."""
    jumps = []
    for axis in (0, 1):
        d = np.abs(np.diff(center, axis=axis))
        if np.isfinite(d).any():
            index = np.unravel_index(np.nanargmax(d), d.shape)
            other = list(index)
            other[axis] += 1
            jumps.append((float(d[index]), index, tuple(other)))
    return max(jumps) if jumps else None


def yx(crop, index):
    iy, ix = index
    return [crop.y + int(iy) * STRIDE + STRIDE // 2,
            crop.x + int(ix) * STRIDE + STRIDE // 2]


def summarize(crop, result, mid, spacing):
    center = result.center
    measured = np.isfinite(center)
    delta = center[measured] - mid
    drift = np.abs(delta)
    low, high = (result.bounds[measured] - mid).T
    report = {
        "valid": int(result.valid.sum()),
        "measured": int(measured.sum()),
        "no_band": int((result.valid & (result.count == 0)).sum()),
        "clipped_nearest": int((result.valid & ~measured & (result.count > 0)).sum()),
        "multiple_bands": int((result.count >= 2).sum()),
        "z54_gap": int(result.nominal_gap.sum()),
        "no_band_in_62_slice_window": int(result.window_empty.sum()),
        "delta_median": float(np.median(delta).round(2)) if len(delta) else None,
        "low_median": float(np.median(low).round(2)) if len(delta) else None,
        "high_median": float(np.median(high).round(2)) if len(delta) else None,
        "abs_delta": stats(drift, spacing),
        "abs_low": stats(np.abs(low), spacing),
        "abs_high": stats(np.abs(high), spacing),
        "centered": int((drift <= 3).sum()),
        "moderate": int(((drift >= 8) & (drift <= 25)).sum()),
        "severe": int((drift > 25).sum()),
    }
    if measured.any():
        index = np.unravel_index(np.nanargmax(np.abs(center - mid)), center.shape)
        report["max_yx"] = yx(crop, index)
        report["max_borders"] = result.bounds[index].round(2).tolist()
    jump = strongest_jump(center)
    if jump is not None:
        distance, first, second = jump
        report["jump"] = [round(distance, 2), yx(crop, first), yx(crop, second),
                          round(float(center[first]), 2), round(float(center[second]), 2)]
    return report


def read_ink(catalog, shape):
    key = asset(catalog, W023, "ink-detection-downsampled")
    with Image.open(ROOT / "cache" / "stage1" / key) as image:
        ink = np.asarray(image.convert("L"))
    if tuple(STRIDE * dim for dim in ink.shape) != shape[1:]:
        raise ValueError(f"ds8 ink shape {ink.shape} does not match level 0 {shape}")
    return ink


def rank_center(values):
    return rankdata(values) / len(values) - (len(values) + 1) / (2 * len(values))


def ink_audit(captures, ink):
    if ink is None:
        return
    paired = []
    for crop, _, vol, result in captures:
        if crop.segment != W023:
            continue
        y, x = crop.y // STRIDE, crop.x // STRIDE
        side = SIZE // STRIDE
        pixels = ink[y:y + side, x:x + side]
        drift = np.abs(result.center - vol.shape[0] // 2)
        keep = np.isfinite(drift)
        paired.append((crop.name, drift[keep], pixels[keep]))
    groups = [
        ("<=3", lambda d: d <= 3),
        (">3 to <8", lambda d: (d > 3) & (d < 8)),
        ("8 to 25", lambda d: (d >= 8) & (d <= 25)),
        (">25", lambda d: d > 25),
    ]
    all_d = np.concatenate([d for _, d, _ in paired])
    all_p = np.concatenate([p for _, _, p in paired])
    fig, (scatter, summary) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name, d, p in paired:
        rho = spearmanr(d, p).statistic
        print(f"Ink {name} n={len(d)} rho={rho:.3f} mean={p.mean():.1f}", flush=True)
        scatter.scatter(d, p, s=8, alpha=0.28, label=name, edgecolors="none")
    pooled = spearmanr(all_d, all_p).statistic
    centered_d = np.concatenate([rank_center(d) for _, d, _ in paired])
    centered_p = np.concatenate([rank_center(p) for _, _, p in paired])
    within = np.corrcoef(centered_d, centered_p)[0, 1]
    print(f"Ink pooled n={len(all_d)} rho={pooled:.3f}; within-crop rank r={within:.3f}", flush=True)
    interior = [(d, p) for name, d, p in paired if name in "ABCD"]
    if interior:
        ds, ps = zip(*interior)
        d, p = np.concatenate(ds), np.concatenate(ps)
        print(f"Ink interior n={len(d)} rho={spearmanr(d, p).statistic:.3f}", flush=True)
    for at, (label, selector) in enumerate(groups):
        p = all_p[selector(all_d)]
        if len(p):
            q1, med, q3 = np.percentile(p, [25, 50, 75])
            print(f"Ink bin {label} n={len(p)} mean={p.mean():.1f} median={med:.1f} "
                  f"fraction>=128={(p >= 128).mean():.3f}", flush=True)
            summary.errorbar(at, med, yerr=[[med - q1], [q3 - med]],
                             fmt="o", color="#19587C", capsize=5, lw=2)
        summary.text(at, 12, f"n={len(p):,}", ha="center", fontsize=8)
    scatter.legend(title="Crop", ncols=3, loc="upper right")
    scatter.set(xlabel="Absolute CT band-center offset (voxels)",
                ylabel="Published ds8 JPEG intensity",
                title=f"Pooled rank correlation ρ = {pooled:.2f}", ylim=(0, 255))
    summary.set(xlabel="Absolute offset (voxels)", ylabel="JPEG median and middle 50%",
                xticks=range(len(groups)), xticklabels=[label for label, _ in groups],
                ylim=(0, 255), title="Preview brightness by offset")
    for ax in (scatter, summary):
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("PHerc0139 w023 · Published ink prediction versus CT band offset")
    fig.savefig(ROOT / "outputs" / "label_drift_ink.png", dpi=155)
    plt.close(fig)


def pooled_stats(captures):
    for segment in (None, W023, W029, W053):
        selected = [entry for entry in captures if segment is None or entry[0].segment == segment]
        if not selected:
            continue
        offsets = [result.center - vol.shape[0] // 2 for _, _, vol, result in selected]
        voxels = np.concatenate([d[np.isfinite(d)] for d in offsets])
        microns = np.concatenate([d[np.isfinite(d)] * entry[1]
                                  for d, entry in zip(offsets, selected)])
        voxel_stats = stats(np.abs(voxels), 1)
        micron_stats = stats(np.abs(microns), 1)
        data = {
            "valid": sum(int(result.valid.sum()) for _, _, _, result in selected),
            "measured": len(voxels),
            "centered": int((np.abs(voxels) <= 3).sum()),
            "moderate": int(((np.abs(voxels) >= 8) & (np.abs(voxels) <= 25)).sum()),
            "severe": int((np.abs(voxels) > 25).sum()),
            "z54_gap": sum(int(result.nominal_gap.sum()) for _, _, _, result in selected),
            "no_band_in_62_slice_window": sum(int(result.window_empty.sum())
                                             for _, _, _, result in selected),
            "abs_delta_vx": voxel_stats["vx"] if voxel_stats else None,
            "abs_delta_um": micron_stats["vx"] if micron_stats else None,
        }
        print(f"Pooled {segment.sample if segment else 'all'} {json.dumps(data)}", flush=True)


def focal_cell(crop, result, mid):
    positions = {
        "B": (12620, 15708),
        "E": (7804, 21700),
        "G": (18980, 15740),
        "I": (24484, 93396),
    }
    if crop.name in positions:
        y, x = positions[crop.name]
        return (y - crop.y) // STRIDE, (x - crop.x) // STRIDE
    delta = result.center - mid
    if crop.name == "D":
        candidates = (np.abs(delta) <= 3) & (result.count == 1)
    else:
        candidates = (delta >= -16) & (delta <= -10) & ~result.nominal_gap
    iy, ix = np.indices(delta.shape)
    k = delta.shape[0] // 2
    distance = np.where(candidates, (iy - k) ** 2 + (ix - k) ** 2, np.inf)
    return np.unravel_index(distance.argmin(), delta.shape)


def draw_section(ax, crop, volume, result, anchor, along_x, lo, hi):
    y, x = yx(crop, anchor)
    iy, ix = anchor
    if along_x:
        section = volume[:, y - crop.y, :]
        bounds = result.bounds[iy, :]
        origin, focus, fixed = crop.x, x, y
        moving, constant = "X", "Y"
    else:
        section = volume[:, :, x - crop.x]
        bounds = result.bounds[:, ix]
        origin, focus, fixed = crop.y, y, x
        moving, constant = "Y", "X"
    mid = volume.shape[0] // 2
    ax.imshow(section, cmap="gray", vmin=lo, vmax=hi, aspect="auto", origin="lower",
              interpolation="nearest", extent=(origin - 0.5, origin + SIZE - 0.5,
                                               -0.5, volume.shape[0] - 0.5))
    ax.axhline(mid, color="#ff5b52", ls="--", lw=1.15)
    locations = origin + np.arange(bounds.shape[0]) * STRIDE + STRIDE // 2
    ax.scatter(locations, bounds[:, 0], color="#71e2fa", s=6, marker="_", linewidths=0.8)
    ax.scatter(locations, bounds[:, 1], color="#71e2fa", s=6, marker="_", linewidths=0.8)
    ax.scatter(locations, bounds.mean(axis=-1), color="#ffdf53", s=8, linewidths=0)
    ax.axvline(focus, color="white", lw=0.75, ls=":", alpha=0.8)
    ax.set(xlabel=f"{moving} (level-0 voxel) · {constant} = {fixed}",
           ylabel="Z layer", yticks=[0, 23, 54, 84, 108])
    ax.set_xticks(origin + np.arange(4) * 64)


def gallery(captures, catalog):
    captions = {
        "D": "Centered band in preview-dark interior",
        "C": "Offset band with z=54 near one boundary",
        "E": "Rendered edge and zero padding",
        "B": "Local handoff between bands",
        "G": "Competing bands around the nominal slice",
        "I": "Wide gap and steeply folded bands",
    }
    by_name = {item[0].name: item for item in captures}
    selected = [by_name[name] for name in captions if name in by_name]
    if not selected:
        return
    fig = plt.figure(figsize=(13, 2.37 * len(selected) + 0.9))
    layout = fig.add_gridspec(len(selected), 2, left=0.075, right=0.977,
                             bottom=0.067, top=0.913, hspace=0.68, wspace=0.20)
    fig.suptitle("Where published surface volumes place the papyrus",
                 x=0.075, y=0.981, ha="left", fontsize=19, fontweight="bold")
    handles = [
        Line2D([], [], color="#d72b2b", ls="--", label="Nominal z = 54"),
        Line2D([], [], color="#8e6500", marker="o", lw=0, label="CT band midpoint"),
        Line2D([], [], color="#16768c", marker="_", lw=0, label="CT band boundaries"),
        Line2D([], [], color="#666666", ls=":", label="Focal cell"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.961),
               frameon=False, ncols=4, fontsize=10)
    for row, (crop, spacing, volume, result) in enumerate(selected):
        mid = volume.shape[0] // 2
        anchor = focal_cell(crop, result, mid)
        y, x = yx(crop, anchor)
        center = result.center[anchor]
        borders = result.bounds[anchor]
        values = volume[volume > 0]
        lo, hi = np.percentile(values, [5, 98]) if values.size else (0, 255)
        long_id = record(catalog, crop.segment)["long_id"]
        print(f"Gallery {crop.name} [{y}, {x}] z={center:.2f} "
              f"borders={borders.round(2).tolist()} gap={result.nominal_gap[anchor]} "
              f"bands={result.count[anchor]} empty_window={result.window_empty[anchor]}", flush=True)
        ax_x = fig.add_subplot(layout[row, 0])
        ax_y = fig.add_subplot(layout[row, 1])
        draw_section(ax_x, crop, volume, result, anchor, True, lo, hi)
        draw_section(ax_y, crop, volume, result, anchor, False, lo, hi)
        label = (f"{crop.name}  {captions[crop.name]}   ·   {crop.segment.sample} / {long_id}\n"
                 f"Focal (Y, X) = ({y}, {x})  ·  "
                 f"z = {center:.1f} [{borders[0]:.1f}, {borders[1]:.1f}]  ·  "
                 f"Δz = {center - mid:+.1f} vx ({(center - mid) * spacing:+.1f} µm)")
        ax_x.set_title(label, loc="left", fontsize=10.5, pad=8, linespacing=1.35)
    fig.savefig(ROOT / "outputs" / "label_drift_gallery.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", default="".join(crop.name for crop in CROPS))
    args = parser.parse_args()
    if not args.crops or set(args.crops) - {crop.name for crop in CROPS}:
        parser.error("--crops must contain crop IDs A through J")
    catalog = json.loads((ROOT / "data" / "s3_metadata.min.json").read_text(encoding="utf-8"))["samples"]
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    sessions = {}
    captures = []
    ink = None
    try:
        for crop in CROPS:
            if crop.name not in args.crops:
                continue
            if crop.segment not in sessions:
                sessions[crop.segment] = open_layers(fs, catalog, crop.segment)
            array, spacing = sessions[crop.segment]
            print(f"{crop.name} {crop.segment.sample} {array.shape}, "
                  f"{spacing} um, {crop.context}. Reading four chunks.", flush=True)
            if crop.segment == W023 and ink is None:
                ink = read_ink(catalog, array.shape)
            volume = read_crop(array, crop)
            result = measure(volume)
            mid = volume.shape[0] // 2
            print(crop.name, json.dumps(summarize(crop, result, mid, spacing)), flush=True)
            for threshold in (0.25, 0.40):
                check = measure(volume, contrast=threshold)
                drift = np.abs(check.center - mid)
                print(f"{crop.name} sensitivity {threshold:.2f} "
                      f"n={np.isfinite(drift).sum()} p95={np.nanpercentile(drift, 95):.2f} "
                      f"severe={(drift > 25).sum()} gap={check.nominal_gap.sum()}", flush=True)
            captures.append((crop, spacing, volume, result))
    finally:
        if fs._s3 is not None:
            sync(fs._s3creator.__aexit__(None, None, None))
    pooled_stats(captures)
    gallery(captures, catalog)
    ink_audit(captures, ink)


if __name__ == "__main__":
    main()
