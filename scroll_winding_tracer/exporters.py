"""Dual official exporters: VC3D Point Collections JSON & WindingInferenceStore v1."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def _canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def export_vc3d_json(items: list, out_path: Path, *, is_same_winding: bool = False) -> Path:
    """Export relative-winding chains or same-winding tracks as VC3D point collections JSON v1."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    collections = {}
    now_ms = int(time.time() * 1000)
    rng = np.random.default_rng(77 if not is_same_winding else 99)

    for idx, item in enumerate(items, 1):
        pts = item if isinstance(item, np.ndarray) else item["points_xyz"]
        pts_dict = {}
        for k, (x, y, z) in enumerate(pts):
            pts_dict[str(k + 1)] = {
                "creation_time": now_ms,
                "p": [float(x), float(y), float(z)],
                "wind_a": 0.0 if is_same_winding else float(k),
            }
        meta = {"winding_is_absolute": False, "generator": "scroll_winding_tracer_v0.3"}
        if not is_same_winding and isinstance(item, dict):
            meta["mean_confidence"] = round(float(item.get("mean_conf", 0.0)), 4)
            meta["min_confidence"] = round(float(item.get("min_conf", 0.0)), 4)
        collections[str(idx)] = {
            "autoFillConstant": 0.0,
            "autoFillMode": 1,
            "color": rng.uniform(0.25, 0.95, size=3).tolist(),
            "metadata": meta,
            "name": f"{'auto_sw' if is_same_winding else 'auto_rw'}_z{int(pts[0, 2])}_{idx:04d}",
            "points": pts_dict,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"vc_pointcollections_json_version": 1, "collections": collections}, f, indent=2)
    return out_path


def _split_collinear(pts_xyz: np.ndarray, max_err_px: float) -> list[np.ndarray]:
    subrays = []
    n = len(pts_xyz)
    i = 0
    while i < n - 1:
        j = i + 2
        while j < n:
            seg = pts_xyz[i : j + 1]
            v = seg[-1] - seg[0]
            L = float(np.linalg.norm(v))
            if L < 1e-6:
                break
            u = v / L
            proj = ((seg - seg[0]) @ u)[:, None] * u[None, :]
            if float(np.linalg.norm((seg - seg[0]) - proj, axis=1).max()) > max_err_px:
                break
            j += 1
        subrays.append(pts_xyz[i:j])
        i = max(i + 1, j - 1)
    return subrays


def export_winding_inference_store_v1(
    chains: list[dict],
    out_dir: Path,
    *,
    max_collinear_err_px: float = 2.0,
    verify: bool = True,
) -> dict:
    """
    Export chains as a checksum-verified `WindingInferenceStore` v1 directory directly
    consumable by `ScrollPrize/villa/spiral-fitting/winding_supervision.py`.
    """
    out_dir = Path(out_dir)
    shard_dir = out_dir / "shard_0"
    shard_dir.mkdir(parents=True, exist_ok=True)

    origins, steps, seed_w, cross_t, cross_lvl = [], [], [], [], []
    offsets = [0]

    for ch in chains:
        for sub in _split_collinear(ch["points_xyz"], max_collinear_err_px):
            if len(sub) < 2:
                continue
            zyx = sub[:, [2, 1, 0]].astype(np.float64)
            origin = zyx[0]
            vec = zyx[-1] - zyx[0]
            total = float(np.linalg.norm(vec))
            if total < 1e-3:
                continue
            step_vec = vec / total
            ts = ((zyx - origin) @ step_vec).astype(np.float32)
            lvls = np.arange(len(sub), dtype=np.int16)

            origins.append(origin.astype(np.float32))
            steps.append(step_vec.astype(np.float32))
            seed_w.append(0)
            cross_t.append(ts)
            cross_lvl.append(lvls)
            offsets.append(offsets[-1] + len(sub))

    arrs = {
        "ray_origin_zyx": np.asarray(origins, dtype="<f4") if origins else np.empty((0, 3), dtype="<f4"),
        "ray_step_zyx": np.asarray(steps, dtype="<f4") if steps else np.empty((0, 3), dtype="<f4"),
        "seed_winding": np.asarray(seed_w, dtype="<i2"),
        "crossing_t": np.concatenate(cross_t).astype("<f4") if cross_t else np.empty((0,), dtype="<f4"),
        "crossing_level": np.concatenate(cross_lvl).astype("<i2") if cross_lvl else np.empty((0,), dtype="<i2"),
        "crossing_offsets": np.asarray(offsets, dtype="<i8"),
    }

    arr_meta = {}
    for name, a in arrs.items():
        fpath = shard_dir / f"{name}.npy"
        np.save(fpath, a)
        arr_meta[name] = {
            "file": f"{name}.npy",
            "shape": list(a.shape),
            "dtype": str(a.dtype.str),
            "bytes": int(fpath.stat().st_size),
            "sha256": _sha256_file(fpath),
        }

    manifest = {
        "artifact_type": "winding_inference_crossings",
        "format_version": 1,
        "coordinate_order": "zyx",
        "coordinate_space": "reference zarr scale-0 voxels",
        "generator": "scroll_winding_tracer_v0.3",
        "num_rays": int(len(arrs["ray_origin_zyx"])),
        "num_crossings": int(len(arrs["crossing_t"])),
        "shards": [{"name": "shard_0", "arrays": arr_meta}],
    }
    manifest["fingerprint"] = _canonical_digest(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verify:
        loaded = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        view = copy.deepcopy(loaded)
        fp = view.pop("fingerprint")
        if fp != _canonical_digest(view):
            raise RuntimeError("Manifest canonical SHA-256 fingerprint mismatch")
        if int(arrs["crossing_offsets"][-1]) != len(arrs["crossing_t"]):
            raise RuntimeError("Ragged crossing offsets mismatch")
    return manifest
