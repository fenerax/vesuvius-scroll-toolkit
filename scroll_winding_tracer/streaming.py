"""Lazy anonymous S3 & local OME-Zarr 3D Z-slab streaming with hard cache caps."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import s3fs
import zarr

BUCKET = "vesuvius-challenge-open-data"
S3_HTTP = f"https://{BUCKET}.s3.us-east-1.amazonaws.com"


def open_zarr_array(path_or_s3_key: str, level: str = "0"):
    """Open an anonymous S3 or local OME-Zarr array cleanly on both zarr v2 and v3."""
    p = Path(path_or_s3_key)
    if p.exists():
        grp = zarr.open_group(str(p), mode="r")
        return grp[level] if level in grp else zarr.open_array(str(p), mode="r")

    clean = path_or_s3_key.replace("s3://", "").strip("/")
    if not clean.startswith(f"{BUCKET}/"):
        clean = f"{BUCKET}/{clean}"

    zarr_major = int(zarr.__version__.split(".")[0])
    if zarr_major >= 3:
        from zarr.storage import FsspecStore
        fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
        store = FsspecStore(fs, path=clean, read_only=True)
        grp = zarr.open_group(store=store, mode="r")
    else:
        import fsspec
        mapper = fsspec.get_mapper(f"s3://{clean}", anon=True)
        grp = zarr.open_group(mapper, mode="r")
    return grp[level]


def resolve_scroll_assets(scroll_id: str, catalog_path: Path, volume_id: str | None = None) -> dict[str, str]:
    """Resolve surface-prediction Zarr & umbilicus JSON S3 paths from local catalog metadata."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if scroll_id not in meta["samples"]:
        raise KeyError(f"Scroll '{scroll_id}' not found in catalog {catalog_path}")

    vols = meta["samples"][scroll_id].get("volumes", {})
    candidates = [volume_id] if volume_id else list(vols.keys())

    # Prefer volumes that have BOTH a surface-m7 (9um/L2) prediction AND an umbilicus JSON
    best: dict[str, str] | None = None
    for vid in candidates:
        v = vols.get(vid)
        if not v:
            continue
        surf_m7 = None
        surf_any = None
        umb = None
        for d in v.get("data", []):
            t = d["type"]
            path = d["origins"][0]["path"]
            if t == "umbilicus":
                umb = path
            elif t == "surface-prediction-zarr":
                if "surface-m7" in path:
                    surf_m7 = path
                elif surf_any is None:
                    surf_any = path
        if umb and (surf_m7 or surf_any):
            best = {
                "scroll_id": scroll_id,
                "volume_id": vid,
                "pixel_size_um": str(v.get("properties", {}).get("pixel_size_um", "")),
                "surface_zarr": surf_m7 or surf_any,
                "umbilicus_json": umb,
            }
            if surf_m7:
                break

    if best is None:
        raise ValueError(f"No volume with both surface-prediction-zarr and umbilicus found for {scroll_id}")
    return best


def load_umbilicus_points(umb_path_or_s3_key: str, cache_dir: Path | None = None) -> np.ndarray:
    """Load umbilicus control points as (N, 3) float64 XYZ array."""
    p = Path(umb_path_or_s3_key)
    if not p.exists():
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            local = cache_dir / Path(umb_path_or_s3_key).name
            if not local.exists():
                urllib.request.urlretrieve(f"{S3_HTTP}/{umb_path_or_s3_key.lstrip('/')}", local)
            p = local
        else:
            raw = urllib.request.urlopen(f"{S3_HTTP}/{umb_path_or_s3_key.lstrip('/')}", timeout=20).read()
            data = json.loads(raw.decode("utf-8"))
            return np.array([[q["x"], q["y"], q["z"]] for q in data["control_points"]], dtype=np.float64)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array([[q["x"], q["y"], q["z"]] for q in data["control_points"]], dtype=np.float64)


def interpolate_umbilicus_xy(umb_pts_xyz: np.ndarray, z: float) -> tuple[float, float]:
    """Linearly interpolate scroll umbilicus (x, y) at slice z."""
    order = np.argsort(umb_pts_xyz[:, 2])
    pts = umb_pts_xyz[order]
    ux = float(np.interp(z, pts[:, 2], pts[:, 0]))
    uy = float(np.interp(z, pts[:, 2], pts[:, 1]))
    return ux, uy


def stream_z_slab(
    arr,
    z_center: int,
    half_z: int = 8,
    y_slice: tuple[int, int] | None = None,
    x_slice: tuple[int, int] | None = None,
    cache_file: Path | None = None,
) -> tuple[np.ndarray, int, int, int]:
    """
    Stream a compact binary 3D Z-slab [z_center-half_z : z_center+half_z+1] from S3/local Zarr.
    Optionally caches bit-packed compressed `.npz` (~10-20 MB/slab) to avoid repeat transfers.
    Returns: (slab_bool_zyx, y0, x0, z0)
    """
    if cache_file is not None and cache_file.exists():
        npz = np.load(cache_file)
        shape = tuple(int(v) for v in npz["shape"])
        slab = np.unpackbits(npz["packed"], axis=-1)[..., : shape[-1]].astype(np.bool_)
        return slab, int(npz["y0"]), int(npz["x0"]), int(npz["z0"])

    z0 = max(0, z_center - half_z)
    z1 = min(int(arr.shape[0]), z_center + half_z + 1)
    y0, y1 = y_slice if y_slice else (0, int(arr.shape[1]))
    x0, x1 = x_slice if x_slice else (0, int(arr.shape[2]))

    slab = np.asarray(arr[z0:z1, y0:y1, x0:x1]) > 0
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        packed = np.packbits(slab, axis=-1)
        np.savez_compressed(cache_file, packed=packed, shape=np.array(slab.shape), y0=y0, x0=x0, z0=z0, z=z_center)
    return slab, y0, x0, z0
