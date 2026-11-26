"""
Stage 1 — Vesuvius Challenge Lightweight Streaming Sandbox (Option B: zarr + s3fs)

Streams targeted pyramid levels and chunk-aligned crops from AWS Open Data
(s3://vesuvius-challenge-open-data) with ZERO multi-TB downloads.

Verifies:
  1. OME-Zarr multiscale pyramid metadata (shapes, chunks, codecs, virtual GB)
  2. Cold vs. warm S3 chunk read throughput + strict local cache cap
  3. Coordinate alignment between:
       - PHerc. 139 CT volume (20260102150214, 2.399 um, 78 keV)
       - Scroll umbilicus centerline JSON
       - Precomputed neural surface-prediction Zarr
       - Transformed tifxyz papyrus mesh (segment 20260311000000-w023_202603112281)
       - Published downsampled ink-detection preview JPEG
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import tifffile
import zarr
from PIL import Image

BUCKET = "vesuvius-challenge-open-data"
S3_HTTP = f"https://{BUCKET}.s3.us-east-1.amazonaws.com"

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "s3_metadata.min.json"
CACHE_DIR = ROOT / "cache" / "stage1"
OUT_DIR = ROOT / "outputs"
MAX_CACHE_BYTES = 2 * 1024**3  # 2 GB hard cap for Stage 1 sandbox

TARGET_SAMPLE = "PHerc0139"
TARGET_VOLUME_ID = "20260102150214"
TARGET_SEGMENT_ID = "20260311000000"


def cache_usage_bytes(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def download_small_cached(s3_key: str, max_single_mb: float = 50.0) -> Path:
    """Download a small metadata/mesh/preview file into CACHE_DIR with a hard cap."""
    local = CACHE_DIR / s3_key
    if local.exists():
        return local

    used = cache_usage_bytes(CACHE_DIR)
    if used >= MAX_CACHE_BYTES:
        raise RuntimeError(
            f"Local cache cap exceeded ({used / 1e9:.2f} GB >= {MAX_CACHE_BYTES / 1e9:.2f} GB). "
            f"Prune {CACHE_DIR} before downloading more."
        )

    url = f"{S3_HTTP}/{s3_key}"
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "vesuvius-sandbox"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        size = int(resp.headers.get("Content-Length", "0"))
    if size > max_single_mb * 1024 * 1024:
        raise RuntimeError(f"Refusing to download {s3_key} ({size / 1e6:.1f} MB > {max_single_mb} MB limit)")

    local.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    urllib.request.urlretrieve(url, local)
    dt = time.perf_counter() - t0
    print(f"  [cached] {s3_key.split('/')[-1]} ({size / 1e6:.2f} MB in {dt:.2f}s -> {size / max(dt, 1e-6) / 1e6:.2f} MB/s)")
    return local


def open_s3_zarr(s3_key: str):
    """Open an anonymous S3 OME-Zarr group cleanly on both zarr v3 and zarr v2."""
    clean_path = f"{BUCKET}/{s3_key.strip('/')}"
    zarr_major = int(zarr.__version__.split(".")[0])
    if zarr_major >= 3:
        from zarr.storage import FsspecStore
        fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
        store = FsspecStore(fs, path=clean_path, read_only=True)
        return zarr.open_group(store=store, mode="r")
    else:
        import fsspec
        mapper = fsspec.get_mapper(f"s3://{clean_path}", anon=True)
        return zarr.open_group(mapper, mode="r")


def inspect_zarr_pyramid(label: str, group) -> dict[str, dict]:
    """Print OME-Zarr multiscale metadata without downloading any voxel chunks."""
    print(f"\n=== OME-Zarr Pyramid: {label} ===")
    attrs = dict(group.attrs)
    multiscales = attrs.get("multiscales", [])
    if multiscales:
        axes = [a.get("name", "?") for a in multiscales[0].get("axes", [])]
        print(f"  Axes: {axes}")
    levels = {}
    keys = sorted([k for k in group.keys() if k.isdigit()], key=int)
    for k in keys:
        arr = group[k]
        shape = tuple(arr.shape)
        chunks = tuple(arr.chunks)
        dtype = arr.dtype
        virtual_gb = (int(np.prod(shape)) * dtype.itemsize) / (1024**3)
        chunk_mib = (int(np.prod(chunks)) * dtype.itemsize) / (1024**2)
        compressor = getattr(arr, "compressors", None) or getattr(arr, "compressor", None)
        levels[k] = {
            "shape": shape,
            "chunks": chunks,
            "dtype": str(dtype),
            "virtual_gb": virtual_gb,
            "chunk_mib": chunk_mib,
            "compressor": str(compressor),
        }
        print(
            f"  Level {k}: shape={shape}  chunks={chunks} ({chunk_mib:.2f} MiB/chunk)  "
            f"dtype={dtype}  virtual={virtual_gb:.2f} GiB  compressor={compressor}"
        )
    return levels


def lookup_targets() -> dict[str, str]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sample = meta["samples"][TARGET_SAMPLE]
    vol = sample["volumes"][TARGET_VOLUME_ID]
    seg = sample["segments"][TARGET_SEGMENT_ID]

    paths: dict[str, str] = {}
    for item in vol["data"]:
        t = item["type"]
        p = item["origins"][0]["path"]
        if t == "ome-zarr":
            paths["ct_zarr"] = p
        elif t == "surface-prediction-zarr":
            paths["surface_zarr"] = p
        elif t == "umbilicus":
            paths["umbilicus_json"] = p

    for item in seg["data"]:
        t = item["type"]
        params = item.get("parameters") or {}
        p = item["origins"][0]["path"]
        if t == "tifxyz-transformed" and params.get("target_volume") == TARGET_VOLUME_ID:
            paths["tifxyz_dir"] = p
        elif t == "ink-detection-downsampled" and params.get("target_volume") == TARGET_VOLUME_ID:
            paths["ink_jpg"] = p

    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"zarr={zarr.__version__}  s3fs={s3fs.__version__}  numpy={np.__version__}")
    paths = lookup_targets()
    print("\n=== Target Paths (from local catalog) ===")
    for k, v in paths.items():
        print(f"  {k:<15}: {v}")

    # 1. Inspect CT and surface-prediction Zarr pyramids (metadata only — kilobytes)
    ct_grp = open_s3_zarr(paths["ct_zarr"])
    ct_levels = inspect_zarr_pyramid("PHerc0139 CT (2.399 um)", ct_grp)

    surf_grp = open_s3_zarr(paths["surface_zarr"])
    surf_levels = inspect_zarr_pyramid("PHerc0139 Surface Prediction", surf_grp)

    # 2. Fetch small assets (umbilicus JSON, ink preview JPEG, tifxyz coords ~20 MB total)
    print("\n=== Downloading Small Assets (cached, hard-capped) ===")
    umb_path = download_small_cached(paths["umbilicus_json"], max_single_mb=5)
    ink_path = download_small_cached(paths["ink_jpg"], max_single_mb=20)
    tifxyz_base = paths["tifxyz_dir"].rstrip("/")
    tif_meta_path = download_small_cached(f"{tifxyz_base}/meta.json", max_single_mb=1)
    x_path = download_small_cached(f"{tifxyz_base}/x.tif", max_single_mb=20)
    y_path = download_small_cached(f"{tifxyz_base}/y.tif", max_single_mb=20)
    z_path = download_small_cached(f"{tifxyz_base}/z.tif", max_single_mb=20)

    with open(umb_path, "r", encoding="utf-8") as f:
        umb_data = json.load(f)
    with open(tif_meta_path, "r", encoding="utf-8") as f:
        tif_meta = json.load(f)
    print(f"  tifxyz meta.json: {tif_meta}")

    xs = tifffile.imread(x_path).astype(np.float32)
    ys = tifffile.imread(y_path).astype(np.float32)
    zs = tifffile.imread(z_path).astype(np.float32)
    valid = (xs > 0) & (ys > 0) & (zs > 0) & np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    print(f"  tifxyz grid shape={xs.shape}, valid points={valid.sum():,} ({100 * valid.mean():.1f}%)")
    print(
        f"  tifxyz bounds (Level-0 voxels): "
        f"Z=[{zs[valid].min():.0f}, {zs[valid].max():.0f}], "
        f"Y=[{ys[valid].min():.0f}, {ys[valid].max():.0f}], "
        f"X=[{xs[valid].min():.0f}, {xs[valid].max():.0f}]"
    )

    # Pick a clean center point on the valid papyrus mesh (near the middle of the sheet)
    valid_rc = np.argwhere(valid)
    mid_r, mid_c = valid_rc[len(valid_rc) // 2]
    cz, cy, cx = float(zs[mid_r, mid_c]), float(ys[mid_r, mid_c]), float(xs[mid_r, mid_c])
    print(f"  Selected mesh anchor (Level-0 Z,Y,X): ({cz:.1f}, {cy:.1f}, {cx:.1f})")

    # 3. Stream a coarse overview slice at Level 4 (1/16 resolution -> ~4x faster than Level 3!)
    coarse_lvl = "4" if "4" in ct_levels else max(ct_levels.keys(), key=int)
    scale = 2 ** int(coarse_lvl)
    ct_coarse = ct_grp[coarse_lvl]
    z_coarse = int(round(cz / scale))
    z_coarse = max(0, min(z_coarse, ct_coarse.shape[0] - 1))

    print(f"\n=== Streaming Coarse Overview (Level {coarse_lvl}, scale 1/{scale}, Z={z_coarse}) ===")
    t0 = time.perf_counter()
    coarse_slice = np.asarray(ct_coarse[z_coarse, :, :])
    dt_coarse = time.perf_counter() - t0
    print(f"  Fetched coarse XY slice {coarse_slice.shape} in {dt_coarse:.2f}s")

    # Find mesh points near this coarse Z plane
    z_band = valid & (np.abs(zs / scale - z_coarse) <= 0.75)
    mx_c = xs[z_band] / scale
    my_c = ys[z_band] / scale

    # Parse umbilicus centerline point nearest cz
    umb_xy = None
    if isinstance(umb_data, dict):
        pts = (
            umb_data.get("control_points")
            or umb_data.get("points")
            or umb_data.get("centerline")
            or umb_data.get("coords")
        )
    elif isinstance(umb_data, list):
        pts = umb_data
    else:
        pts = None
    if pts and isinstance(pts, list) and len(pts) > 0:
        try:
            arr_u = np.array(
                [[p["x"], p["y"], p["z"]] if isinstance(p, dict) else p for p in pts],
                dtype=np.float32,
            )
            if arr_u.ndim == 2 and arr_u.shape[1] >= 3:
                l0_z, l0_y, l0_x = ct_levels["0"]["shape"]
                if arr_u[:, 2].max() <= l0_z * 1.05 and arr_u[:, 0].max() <= l0_x * 1.05:
                    ux, uy, uz = arr_u[:, 0], arr_u[:, 1], arr_u[:, 2]
                else:
                    uz, uy, ux = arr_u[:, 0], arr_u[:, 1], arr_u[:, 2]
                idx_u = int(np.argmin(np.abs(uz - cz)))
                umb_xy = (float(ux[idx_u] / scale), float(uy[idx_u] / scale))
                print(f"  Umbilicus at Z~{cz:.0f}: Level-0 ({ux[idx_u]:.0f}, {uy[idx_u]:.0f}) -> coarse {umb_xy}")
        except Exception as e:
            print(f"  (Could not parse umbilicus format: {e})")

    # 4. Stream a CHUNK-ALIGNED native Level-0 256^3 crop centered around (cz, cy, cx) + benchmark
    ch_z, ch_y, ch_x = ct_levels["0"]["chunks"]
    crop_size = 256  # exactly 2x2x2 chunks of 128^3 = 8 chunks = 16 MiB
    z0 = (int(cz - crop_size // 2 + ch_z // 2) // ch_z) * ch_z
    y0 = (int(cy - crop_size // 2 + ch_y // 2) // ch_y) * ch_y
    x0 = (int(cx - crop_size // 2 + ch_x // 2) // ch_x) * ch_x
    z0 = max(0, min(z0, ct_levels["0"]["shape"][0] - crop_size))
    y0 = max(0, min(y0, ct_levels["0"]["shape"][1] - crop_size))
    x0 = max(0, min(x0, ct_levels["0"]["shape"][2] - crop_size))
    z1, y1, x1 = z0 + crop_size, y0 + crop_size, x0 + crop_size

    ct_l0 = ct_grp["0"]
    print(f"\n=== Benchmark: Chunk-Aligned Native Level-0 Crop [{z0}:{z1}, {y0}:{y1}, {x0}:{x1}] ===")
    # Single chunk benchmark first (128^3 = 2 MiB)
    t0 = time.perf_counter()
    single = np.asarray(ct_l0[z0 : z0 + ch_z, y0 : y0 + ch_y, x0 : x0 + ch_x])
    dt_single = time.perf_counter() - t0
    single_mb = single.nbytes / (1024**2)
    print(f"  1 chunk  ({ch_z}^3 = {single_mb:.2f} MiB): {dt_single:.2f}s -> {single_mb / dt_single:.2f} MiB/s")

    # Full 256^3 aligned block (8 chunks = 16 MiB, fetched concurrently by zarr/s3fs)
    t0 = time.perf_counter()
    ct_crop = np.asarray(ct_l0[z0:z1, y0:y1, x0:x1])
    dt_block = time.perf_counter() - t0
    block_mb = ct_crop.nbytes / (1024**2)
    print(f"  8 chunks ({crop_size}^3 = {block_mb:.2f} MiB): {dt_block:.2f}s -> {block_mb / dt_block:.2f} MiB/s")

    # Stream matching surface-prediction crop (same coordinates if L0 matches!)
    surf_crop = None
    if "0" in surf_levels and surf_levels["0"]["shape"] == ct_levels["0"]["shape"]:
        t0 = time.perf_counter()
        surf_crop = np.asarray(surf_grp["0"][z0:z1, y0:y1, x0:x1])
        dt_surf = time.perf_counter() - t0
        print(
            f"  Surface prediction crop {surf_crop.shape}: {dt_surf:.2f}s "
            f"(min={surf_crop.min()}, max={surf_crop.max()}, nonzero={(surf_crop > 0).mean()*100:.2f}%)"
        )

    # Filter tifxyz points inside this 256^3 crop (widen slice band to +/-10 voxels since tifxyz grid scale=0.05 -> 20 vx spacing!)
    in_box = valid & (zs >= z0) & (zs < z1) & (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
    lz = zs[in_box] - z0
    ly = ys[in_box] - y0
    lx = xs[in_box] - x0
    print(f"  tifxyz grid vertices inside 256^3 crop: {len(lz):,} (grid step ~20 native voxels)")

    # Pick slice indices near the median of mesh points inside the crop
    sz = int(np.clip(np.median(lz), 0, crop_size - 1)) if len(lz) else crop_size // 2
    sy = int(np.clip(np.median(ly), 0, crop_size - 1)) if len(ly) else crop_size // 2
    sx = int(np.clip(np.median(lx), 0, crop_size - 1)) if len(lx) else crop_size // 2

    # 5. Figure 1: Coarse Overview + Mesh Footprint + Ink Preview
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=140)
    axes[0].imshow(coarse_slice, cmap="gray", origin="upper")
    if len(mx_c):
        axes[0].scatter(mx_c, my_c, s=2.5, c="#00e5ff", alpha=0.9, label="tifxyz segment")
    if umb_xy:
        axes[0].scatter([umb_xy[0]], [umb_xy[1]], s=90, c="#ffea00", marker="+", linewidths=2.5, label="umbilicus")
    rect = plt.Rectangle(
        (x0 / scale, y0 / scale),
        max(crop_size / scale, 8),
        max(crop_size / scale, 8),
        fill=False,
        edgecolor="#ff3d00",
        linewidth=2,
        label="256^3 native crop",
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"PHerc. 139 Level-{coarse_lvl} Overview (Z={z_coarse})\n+ tifxyz Surface, Umbilicus & Crop Box")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.75)
    axes[0].axis("off")

    ink_img = Image.open(ink_path)
    axes[1].imshow(ink_img, cmap="gray")
    axes[1].set_title(f"Published Ink Preview (ds8)\nSegment {TARGET_SEGMENT_ID}")
    axes[1].axis("off")

    fig.tight_layout()
    overview_out = OUT_DIR / "stage1_overview.png"
    fig.savefig(overview_out, bbox_inches="tight")
    plt.close(fig)

    # 6. Figure 2: Native Level-0 Orthogonal Slices + tifxyz Overlay + Surface Prediction
    band = 10.5  # half of the ~20-voxel tifxyz grid spacing so nearby vertices always appear cleanly
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=140)
    planes = [
        ("XY (Z=" + str(z0 + sz) + ")", ct_crop[sz, :, :], surf_crop[sz, :, :] if surf_crop is not None else None,
         lx[np.abs(lz - sz) <= band], ly[np.abs(lz - sz) <= band]),
        ("XZ (Y=" + str(y0 + sy) + ")", ct_crop[:, sy, :], surf_crop[:, sy, :] if surf_crop is not None else None,
         lx[np.abs(ly - sy) <= band], lz[np.abs(ly - sy) <= band]),
        ("YZ (X=" + str(x0 + sx) + ")", ct_crop[:, :, sx], surf_crop[:, :, sx] if surf_crop is not None else None,
         ly[np.abs(lx - sx) <= band], lz[np.abs(lx - sx) <= band]),
    ]

    for col, (title, ct_2d, sp_2d, px_pts, py_pts) in enumerate(planes):
        axes[0, col].imshow(ct_2d, cmap="gray", origin="upper")
        if len(px_pts):
            axes[0, col].scatter(px_pts, py_pts, s=16, c="#00e5ff", edgecolors="black", linewidths=0.4, alpha=0.95)
        axes[0, col].set_title(f"CT Native 2.4um — {title}\n(cyan = tifxyz vertices)")
        axes[0, col].axis("off")

        if sp_2d is not None:
            axes[1, col].imshow(ct_2d, cmap="gray", origin="upper")
            axes[1, col].imshow(sp_2d, cmap="magma", alpha=0.45, origin="upper")
            if len(px_pts):
                axes[1, col].scatter(px_pts, py_pts, s=12, c="#00e5ff", edgecolors="black", linewidths=0.3, alpha=0.85)
            axes[1, col].set_title(f"CT + S3 Surface Prediction — {title}")
        else:
            axes[1, col].text(0.5, 0.5, "No matching L0 surface pred", ha="center", va="center")
        axes[1, col].axis("off")

    fig.tight_layout()
    crop_out = OUT_DIR / "stage1_native_crop.png"
    fig.savefig(crop_out, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== Done ===")
    print(f"  Total local cache usage: {cache_usage_bytes(CACHE_DIR) / 1e6:.2f} MB (cap: {MAX_CACHE_BYTES / 1e9:.1f} GB)")
    print(f"  Saved: {overview_out}")
    print(f"  Saved: {crop_out}")


if __name__ == "__main__":
    main()
