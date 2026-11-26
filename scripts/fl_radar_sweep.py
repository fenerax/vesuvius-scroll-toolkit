"""First-Letters 9 um Radar Sweeper (Tranche 0 local + cloud-ready).

Fast, chunk-cached, auto-upright, 3-model consensus sweeper for Vesuvius Challenge
8.64 / 9.36 um First Letters scrolls:
  - Shared thread-safe LRU S3 chunk cache with in-flight deduplication (zero duplicate chunk reads)
  - Automatic upright UV orientation (scroll Z-axis vertical, papyrus fibers horizontal; PR #2664 fix)
  - 3 independent 9 um models: reader-v3 (384px) + argus-v3 (256px) + scrollprize/ink_9um (128px)
  - Forward + reversed layer orders, directional consensus, inter-model spatial correlation,
    and 1D vertical Greek line-spacing periodicity score.

Works out-of-the-box on local GTX 1070 Ti (FP32, batch 2-4, <3 GB VRAM) and scales cleanly
to AWS us-east-1 Spot L4/A10G or RTX 4090 with `--workers 24 --batch-size 8 --fp16`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import tifffile
import torch
import zarr
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from zarr.storage import FsspecStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "cache" / "villa_src" / "vesuvius" / "src"))
import stroke_scorer as ss

# PyTorch 2.6 shim for villa's torch.compiler.disable(reason=...)
_orig_disable = torch.compiler.disable
torch.compiler.disable = lambda fn=None, recursive=True, **kw: _orig_disable(fn, recursive=recursive)

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.checkpoint import load_checkpoint, select_inference_weights
from vesuvius.ink_detection.models.model import make_model

SCROLL_VOLUMES = {
    "PHerc0846B": ("vesuvius-challenge-open-data/PHerc0846B/volumes/20250804142305-9.362um-1.2m-113keV-masked.zarr", 9.362),
    "PHerc0211": ("vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr", 9.362),
    "PHerc0343": ("vesuvius-challenge-open-data/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr", 8.640),
    "PHerc0306B": ("vesuvius-challenge-open-data/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr", 8.640),
    "PHerc0125": ("vesuvius-challenge-open-data/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr", 9.362),
    "PHerc0257": ("vesuvius-challenge-open-data/PHerc0257/volumes/20250821151750-9.362um-1.2m-113keV-masked.zarr", 9.362),
    "PHerc0268": ("vesuvius-challenge-open-data/PHerc0268/volumes/20251110183117-8.640um-1.2m-116keV-masked.zarr", 8.640),
    "PHerc0358": ("vesuvius-challenge-open-data/PHerc0358/volumes/20250821151737-9.362um-1.2m-113keV-masked.zarr", 9.362),
    "PHerc0800": ("vesuvius-challenge-open-data/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr", 8.640),
    "PHerc0813": ("vesuvius-challenge-open-data/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr", 9.362),
    "PHerc0826": ("vesuvius-challenge-open-data/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr", 9.362),
}

MODEL_SPECS = [
    ("reader_v3", ROOT / "cache" / "models" / "reader-v3.pth"),
    ("argus_v3", ROOT / "cache" / "models" / "argus-v3.pth"),
    ("ink_9um", ROOT / "cache" / "models" / "ink_9um_seed42_step075000.pth"),
]


class ChunkCachedVolume:
    """Thread-safe LRU chunk cache with in-flight deduplication over a 3D S3 zarr array."""

    def __init__(self, zarr_arr, max_chunks: int = 768):
        self.arr = zarr_arr
        self.shape = tuple(int(v) for v in zarr_arr.shape)
        self.chunks = tuple(int(v) for v in zarr_arr.chunks)
        self.dtype = zarr_arr.dtype
        self.max_chunks = max_chunks
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._inflight: dict[tuple[int, int, int], Future] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _get_chunk(self, key: tuple[int, int, int]) -> np.ndarray:
        with self._lock:
            val = self._cache.get(key)
            if val is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return val
            fut = self._inflight.get(key)
            owner = fut is None
            if owner:
                fut = Future()
                self._inflight[key] = fut
                self.misses += 1
            else:
                self.hits += 1
        if not owner:
            return fut.result()
        cz, cy, cx = key
        sz, sy, sx = self.chunks
        z0, y0, x0 = cz * sz, cy * sy, cx * sx
        z1, y1, x1 = min(self.shape[0], z0 + sz), min(self.shape[1], y0 + sy), min(self.shape[2], x0 + sx)
        try:
            for attempt in range(4):
                try:
                    data = np.asarray(self.arr[z0:z1, y0:y1, x0:x1])
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(0.5 * (attempt + 1))
            with self._lock:
                self._cache[key] = data
                self._cache.move_to_end(key)
                while len(self._cache) > self.max_chunks:
                    self._cache.popitem(last=False)
                self._inflight.pop(key, None)
            fut.set_result(data)
            return data
        except Exception as e:
            with self._lock:
                self._inflight.pop(key, None)
            fut.set_exception(e)
            raise

    def prefetch_bbox(self, lo: np.ndarray, hi: np.ndarray, pool: ThreadPoolExecutor) -> None:
        sz, sy, sx = self.chunks
        z0, y0, x0 = max(0, int(lo[0])), max(0, int(lo[1])), max(0, int(lo[2]))
        z1, y1, x1 = min(self.shape[0], int(hi[0])), min(self.shape[1], int(hi[1])), min(self.shape[2], int(hi[2]))
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return
        keys = [
            (cz, cy, cx)
            for cz in range(z0 // sz, (z1 - 1) // sz + 1)
            for cy in range(y0 // sy, (y1 - 1) // sy + 1)
            for cx in range(x0 // sx, (x1 - 1) // sx + 1)
        ]
        futs = [pool.submit(self._get_chunk, k) for k in keys]
        for f in futs:
            f.result()

    def read_bbox_padded(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        z0, y0, x0 = (int(v) for v in lo)
        z1, y1, x1 = (int(v) for v in hi)
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=self.dtype)
        cz0, cz1 = max(0, z0), min(self.shape[0], z1)
        cy0, cy1 = max(0, y0), min(self.shape[1], y1)
        cx0, cx1 = max(0, x0), min(self.shape[2], x1)
        if cz1 <= cz0 or cy1 <= cy0 or cx1 <= cx0:
            return out
        sz, sy, sx = self.chunks
        for kz in range(cz0 // sz, (cz1 - 1) // sz + 1):
            for ky in range(cy0 // sy, (cy1 - 1) // sy + 1):
                for kx in range(cx0 // sx, (cx1 - 1) // sx + 1):
                    chk = self._get_chunk((kz, ky, kx))
                    bz0, by0, bx0 = kz * sz, ky * sy, kx * sx
                    bz1, by1, bx1 = bz0 + chk.shape[0], by0 + chk.shape[1], bx0 + chk.shape[2]
                    iz0, iz1 = max(cz0, bz0), min(cz1, bz1)
                    iy0, iy1 = max(cy0, by0), min(cy1, by1)
                    ix0, ix1 = max(cx0, bx0), min(cx1, bx1)
                    out[iz0 - z0 : iz1 - z0, iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = chk[
                        iz0 - bz0 : iz1 - bz0, iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0
                    ]
        return out


class UprightTifXYZSurface:
    """tifxyz reader with exact -1 sentinel masking and continuous Z-vertical alignment.

    Handles both pre-aligned meshes (0/90/180 deg) AND arbitrarily tilted vc_grow_seg_from_seed
    meshes (e.g. 45 deg diagonal auto_grown patches) by computing the exact continuous scroll-Z
    gradient direction and sampling directly on the upright (Z-vertical) orthonormal grid with
    zero double interpolation.
    """

    def __init__(self, seg_dir: Path, auto_upright: bool = True):
        self.meta = json.loads((seg_dir / "meta.json").read_text(encoding="utf-8"))
        x = tifffile.imread(seg_dir / "x.tif").astype(np.float64)
        y = tifffile.imread(seg_dir / "y.tif").astype(np.float64)
        z = tifffile.imread(seg_dir / "z.tif").astype(np.float64)
        sy, sx = float(self.meta["scale"][1]), float(self.meta["scale"][0])
        valid = (x >= 0) & (y >= 0) & (z >= 0) & ((x > 0) | (y > 0) | (z > 0))

        self.x, self.y, self.z = x, y, z
        self.valid_grid = valid
        self.scale_y, self.scale_x = sy, sx
        self.shape = x.shape

        # Default identity transform: v_hat=(1,0) along +row, u_hat=(0,1) along +col
        self.v_hat = np.array([1.0, 0.0], dtype=np.float64)
        self.u_hat = np.array([0.0, 1.0], dtype=np.float64)
        self.origin_rc = np.array([0.0, 0.0], dtype=np.float64)
        self.rot_deg = 0.0
        self.transposed = False
        self.flipped_v = False

        full_h0 = int(round(self.shape[0] / sy))
        full_w0 = int(round(self.shape[1] / sx))

        if auto_upright and valid.sum() > 20:
            # Compute signed Z-gradient per full-resolution pixel along grid rows (r) and cols (c)
            m_r = valid[1:, :] & valid[:-1, :]
            m_c = valid[:, 1:] & valid[:, :-1]
            dz_dr = float(np.median((z[1:, :] - z[:-1, :])[m_r])) * sy if m_r.any() else 0.0
            dz_dc = float(np.median((z[:, 1:] - z[:, :-1])[m_c])) * sx if m_c.any() else 0.0
            mag = math.hypot(dz_dr, dz_dc)
            if mag > 1e-4:
                theta = math.atan2(dz_dc, dz_dr)  # 0 = +Z already down rows
                deg = math.degrees(theta)
                self.rot_deg = round(deg, 1)
                if abs(deg) > 3.0:
                    # Snap near-90/180 angles cleanly; rotate arbitrarily for diagonal meshes!
                    if abs(abs(deg) - 90.0) <= 3.0:
                        theta = math.copysign(math.pi / 2.0, theta)
                        self.transposed = True
                    elif abs(abs(deg) - 180.0) <= 3.0:
                        theta = math.pi
                        self.flipped_v = True
                    self.v_hat = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
                    self.u_hat = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
                    # Tight bounding box of valid pixels in the new upright (v, u) frame
                    vr, vc = np.where(valid)
                    fr = vr.astype(np.float64) / sy
                    fc = vc.astype(np.float64) / sx
                    proj_v = fr * self.v_hat[0] + fc * self.v_hat[1]
                    proj_u = fr * self.u_hat[0] + fc * self.u_hat[1]
                    vmin, vmax = float(proj_v.min()), float(proj_v.max())
                    umin, umax = float(proj_u.min()), float(proj_u.max())
                    self.origin_rc = vmin * self.v_hat + umin * self.u_hat
                    full_h0 = max(1, int(math.ceil(vmax - vmin)) + 1)
                    full_w0 = max(1, int(math.ceil(umax - umin)) + 1)

        self.full_resolution_shape = (full_h0, full_w0)

    def tile_geometry(self, bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r0, r1, c0, c1 = bounds
        rr = np.arange(r0, r1, dtype=np.float64)
        cc = np.arange(c0, c1, dtype=np.float64)
        # Map upright (rr, cc) -> original full-res (orig_r, orig_c) -> control-grid (gr, gc)
        orig_r = self.origin_rc[0] + rr[:, None] * self.v_hat[0] + cc[None, :] * self.u_hat[0]
        orig_c = self.origin_rc[1] + rr[:, None] * self.v_hat[1] + cc[None, :] * self.u_hat[1]
        gr = orig_r * self.scale_y
        gc = orig_c * self.scale_x

        in_bounds = (gr >= 0.0) & (gr <= self.shape[0] - 1.0) & (gc >= 0.0) & (gc <= self.shape[1] - 1.0)
        ir0 = np.clip(np.floor(gr).astype(int), 0, self.shape[0] - 1)
        ic0 = np.clip(np.floor(gc).astype(int), 0, self.shape[1] - 1)
        ir1 = np.clip(ir0 + 1, 0, self.shape[0] - 1)
        ic1 = np.clip(ic0 + 1, 0, self.shape[1] - 1)
        valid = (
            in_bounds
            & self.valid_grid[ir0, ic0] & self.valid_grid[ir0, ic1]
            & self.valid_grid[ir1, ic0] & self.valid_grid[ir1, ic1]
        )
        gr_c = np.clip(gr, 0.0, self.shape[0] - 1.0)
        gc_c = np.clip(gc, 0.0, self.shape[1] - 1.0)
        coords = np.stack([gr_c, gc_c], axis=0)
        xyz = np.stack([
            map_coordinates(self.x, coords, order=1, mode="nearest"),
            map_coordinates(self.y, coords, order=1, mode="nearest"),
            map_coordinates(self.z, coords, order=1, mode="nearest"),
        ], axis=-1)

        def sample_at(du_step, dv_step):
            dr = (dv_step * self.v_hat[0] + du_step * self.u_hat[0]) * self.scale_y
            dc = (dv_step * self.v_hat[1] + du_step * self.u_hat[1]) * self.scale_x
            c = np.stack([
                np.clip(gr_c + dr, 0.0, self.shape[0] - 1.0),
                np.clip(gc_c + dc, 0.0, self.shape[1] - 1.0),
            ], axis=0)
            return np.stack([
                map_coordinates(self.x, c, order=1, mode="nearest"),
                map_coordinates(self.y, c, order=1, mode="nearest"),
                map_coordinates(self.z, c, order=1, mode="nearest"),
            ], axis=-1)

        du = sample_at(1.0, 0.0) - sample_at(-1.0, 0.0)
        dv = sample_at(0.0, 1.0) - sample_at(0.0, -1.0)
        nrm = np.cross(du, dv)
        norm = np.linalg.norm(nrm, axis=-1, keepdims=True)
        valid = valid & (norm[..., 0] > 1e-6)
        nrm = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 1e-6)
        return xyz, nrm, valid


def normalize_robust(image: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32).copy()
    if arr.size == 0:
        return arr
    std_pre = float(np.std(arr))
    lo, hi = float(np.percentile(arr, p_lo)), float(np.percentile(arr, p_hi))
    np.clip(arr, lo, hi, out=arr)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        span = hi - lo
        scale = abs(std_pre) if (np.isfinite(std_pre) and abs(std_pre) >= 1e-6) else (
            abs(span) / 2.0 if (np.isfinite(span) and abs(span) >= 2e-6) else 1.0
        )
    arr -= med
    arr /= scale
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def normalize_robust_gpu(batch: torch.Tensor, p_lo: float = 0.01, p_hi: float = 0.99) -> torch.Tensor:
    """Exact GPU-batched equivalent of normalize_robust on shape (B, D, H, W)."""
    x = batch.float()
    B = x.shape[0]
    flat = x.reshape(B, -1)
    std_pre = flat.std(dim=1, unbiased=False, keepdim=True)
    # torch.quantile supports up to 2^24 elements; for 384x384x17 = 2.5M it works per-sample
    q_pts = torch.tensor([p_lo, p_hi], device=x.device, dtype=torch.float32)
    # Process per-sample if flat.numel() > 16M to stay safely within torch.quantile element cap
    if flat.numel() <= 16_000_000:
        q = torch.quantile(flat, q_pts, dim=1)
        lo, hi = q[0].unsqueeze(1), q[1].unsqueeze(1)
    else:
        lo_list, hi_list = [], []
        for i in range(B):
            qi = torch.quantile(flat[i], q_pts)
            lo_list.append(qi[0])
            hi_list.append(qi[1])
        lo = torch.stack(lo_list).unsqueeze(1)
        hi = torch.stack(hi_list).unsqueeze(1)
    xc = flat.clamp(min=lo, max=hi)
    med = xc.median(dim=1, keepdim=True).values
    mad = (xc - med).abs().median(dim=1, keepdim=True).values
    scale = 1.4826 * mad
    span = hi - lo
    bad = (~torch.isfinite(scale)) | (scale < 1e-6)
    fallback = torch.where(
        torch.isfinite(std_pre) & (std_pre.abs() >= 1e-6),
        std_pre.abs(),
        torch.where(torch.isfinite(span) & (span.abs() >= 2e-6), span.abs() / 2.0, torch.ones_like(scale)),
    )
    scale = torch.where(bad, fallback, scale)
    out = (xc - med) / scale
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape_as(x)


class FlatModelRunner:
    def __init__(self, name: str, ckpt_path: Path, fp16: bool = False, bf16: bool = False):
        ckpt = load_checkpoint(ckpt_path)
        cfg = InkConfig.from_mapping(ckpt["config"])
        model = make_model(cfg)
        _, weights = select_inference_weights(ckpt)
        model.load_state_dict(weights, strict=True)
        self.name = name
        self.patch = int(ckpt["config"]["patch_size"][1])
        self.depth = int(ckpt["config"]["patch_size"][0])
        self.dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        self.model = model.to("cuda").eval()
        if self.dtype != torch.float32:
            self.model = self.model.to(self.dtype)

    def infer(self, layers_zyx: np.ndarray, mask_yx: np.ndarray,
              overlap: float = 0.5, reverse: bool = False, batch_size: int = 4) -> np.ndarray:
        vol = np.ascontiguousarray(layers_zyx[::-1]) if reverse else layers_zyx
        D, H, W = vol.shape
        p_sz = self.patch
        stride = max(1, int(round(p_sz * (1.0 - overlap))))
        hw = torch.outer(
            torch.hann_window(p_sz, periodic=False, dtype=torch.float32),
            torch.hann_window(p_sz, periodic=False, dtype=torch.float32),
        )
        hw = torch.clamp(hw / hw.max().clamp(min=1e-8), min=0.001).numpy()

        def starts(n):
            if n <= p_sz:
                return [0]
            s = list(range(0, n - p_sz, stride))
            if not s or s[-1] != n - p_sz:
                s.append(n - p_sz)
            return s

        coords = []
        for y0 in starts(H):
            for x0 in starts(W):
                y1, x1 = min(y0 + p_sz, H), min(x0 + p_sz, W)
                if mask_yx[y0:y1, x0:x1].any():
                    coords.append((y0, y1, x0, x1))

        prob_sum = np.zeros((H, W), dtype=np.float32)
        wt_sum = np.zeros((H, W), dtype=np.float32)
        dtype = self.dtype

        with torch.inference_mode():
            batch_arrs, batch_meta = [], []
            def flush():
                if not batch_arrs:
                    return
                raw = torch.from_numpy(np.stack(batch_arrs, axis=0)).to("cuda", non_blocking=True)
                normed = normalize_robust_gpu(raw)[:, None].to(dtype)
                out = self.model(normed)
                logits = out["ink"] if isinstance(out, dict) else out
                probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
                for pr, (y0, y1, x0, x1) in zip(probs, batch_meta):
                    vh, vw = y1 - y0, x1 - x0
                    w = hw[:vh, :vw]
                    prob_sum[y0:y1, x0:x1] += pr[:vh, :vw] * w
                    wt_sum[y0:y1, x0:x1] += w
                batch_arrs.clear()
                batch_meta.clear()

            for y0, y1, x0, x1 in coords:
                crop = vol[:, y0:y1, x0:x1]
                if not crop.any():
                    continue
                vh, vw = y1 - y0, x1 - x0
                if vh < p_sz or vw < p_sz:
                    pad = np.zeros((D, p_sz, p_sz), dtype=crop.dtype)
                    pad[:, :vh, :vw] = crop
                    crop = pad
                batch_arrs.append(crop)
                batch_meta.append((y0, y1, x0, x1))
                if len(batch_arrs) >= batch_size:
                    flush()
            flush()
        return np.divide(prob_sum, wt_sum, out=np.zeros_like(prob_sum), where=wt_sum > 0)


def render_surface_cached(surf: UprightTifXYZSurface, cvol: ChunkCachedVolume,
                          bounds: tuple[int, int, int, int] | None = None,
                          layers: int = 17, workers: int = 16) -> tuple[np.ndarray, np.ndarray, float]:
    full_h, full_w = surf.full_resolution_shape
    if bounds is None:
        r0, r1, c0, c1 = 0, full_h, 0, full_w
    else:
        r0, r1, c0, c1 = bounds
    H, W = r1 - r0, c1 - c0

    # Determine inward normal sign on coarse sample
    xyz_c, nrm_c, val_c = surf.tile_geometry((r0, min(r1, r0 + min(H, 1024)), c0, min(c1, c0 + min(W, 1024))))
    umb = np.asarray([cvol.shape[2] / 2.0, cvol.shape[1] / 2.0], dtype=np.float64)
    if val_c.any():
        towards = ((umb - xyz_c[val_c, :2]) * nrm_c[val_c, :2]).sum(1) > 0
        sign = 1.0 if towards.mean() >= 0.5 else -1.0
    else:
        sign = 1.0

    offsets = np.arange(layers, dtype=np.float64) - layers // 2
    out = np.zeros((layers, H, W), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    TILE = 256
    tiles = [(rr, min(rr + TILE, r1), cc, min(cc + TILE, c1))
             for rr in range(r0, r1, TILE) for cc in range(c0, c1, TILE)]

    t0 = time.perf_counter()
    chunk_pool = ThreadPoolExecutor(max_workers=workers)
    cpu_pool = ThreadPoolExecutor(max_workers=6)
    try:
        sz, sy, sx = cvol.chunks
        Z, Y, X = cvol.shape
        # Process tiles in batches so each batch's unique S3 chunks ALWAYS fit in cvol.max_chunks (zero LRU thrashing!)
        TILE_BATCH = 48

        def interp_one(item):
            b, zyx, lo, val, hi = item
            if zyx is None:
                return b, None, val
            blk = cvol.read_bbox_padded(lo, hi)
            s = map_coordinates(blk, (zyx - lo).T, order=1, mode="constant", cval=0.0)
            return b, np.clip(np.rint(s), 0, 255).astype(np.uint8).reshape(layers, -1), val

        for b0 in range(0, len(tiles), TILE_BATCH):
            batch_tiles = tiles[b0 : b0 + TILE_BATCH]
            geoms = []
            batch_chunk_keys = set()
            for b in batch_tiles:
                xyz, nrm, val = surf.tile_geometry(b)
                if not val.any():
                    geoms.append((b, None, None, val, None))
                    continue
                pts = xyz[val][None] + sign * offsets[:, None, None] * nrm[val][None]
                zyx = pts[..., ::-1].reshape(-1, 3)
                lo = np.floor(zyx.min(0)).astype(int) - 1
                hi = np.ceil(zyx.max(0)).astype(int) + 2
                cz0, cz1 = max(0, int(lo[0])), min(Z, int(hi[0]))
                cy0, cy1 = max(0, int(lo[1])), min(Y, int(hi[1]))
                cx0, cx1 = max(0, int(lo[2])), min(X, int(hi[2]))
                if cz1 > cz0 and cy1 > cy0 and cx1 > cx0:
                    for kz in range(cz0 // sz, (cz1 - 1) // sz + 1):
                        for ky in range(cy0 // sy, (cy1 - 1) // sy + 1):
                            for kx in range(cx0 // sx, (cx1 - 1) // sx + 1):
                                batch_chunk_keys.add((kz, ky, kx))
                geoms.append((b, zyx, lo, val, hi))

            if batch_chunk_keys:
                futs = [chunk_pool.submit(cvol._get_chunk, k) for k in sorted(batch_chunk_keys)]
                for f in futs:
                    f.result()

            for b, sampled, val in cpu_pool.map(interp_one, geoms):
                tr0, tr1, tc0, tc1 = b
                lr0, lr1, lc0, lc1 = tr0 - r0, tr1 - r0, tc0 - c0, tc1 - c0
                mask[lr0:lr1, lc0:lc1] = val
                if sampled is not None:
                    out[:, lr0:lr1, lc0:lc1][:, val] = sampled
    finally:
        cpu_pool.shutdown(wait=True)
        chunk_pool.shutdown(wait=True)
    return out, mask, time.perf_counter() - t0


def greek_line_periodicity(img: np.ndarray, mask: np.ndarray, voxel_um: float) -> float:
    """1D vertical FFT bandpower ratio in Greek line-spacing band (3.0..8.5 mm period).

    Masks out invalid padding pixels per row and extracts the longest contiguous run of
    rows with >=50% valid pixels, then linearly detrends before FFT so black padding
    edges never create artificial step-edge spikes.
    """
    H, W = img.shape
    if H < 256:
        return 0.0
    row_valid_frac = mask.mean(axis=1)
    good = row_valid_frac >= 0.5
    if int(good.sum()) < 256:
        return 0.0
    # Longest contiguous run of valid rows
    best_s, best_e, cur_s = 0, 0, None
    for i, g in enumerate(good):
        if g and cur_s is None:
            cur_s = i
        elif not g and cur_s is not None:
            if i - cur_s > best_e - best_s:
                best_s, best_e = cur_s, i
            cur_s = None
    if cur_s is not None and H - cur_s > best_e - best_s:
        best_s, best_e = cur_s, H
    L = best_e - best_s
    if L < 256:
        return 0.0
    sub_img = img[best_s:best_e]
    sub_msk = mask[best_s:best_e]
    denom = np.maximum(sub_msk.sum(axis=1), 1)
    prof = (sub_img * sub_msk).sum(axis=1, dtype=np.float64) / denom
    # Linear detrend to remove slow illumination/edge ramps
    t = np.arange(L, dtype=np.float64)
    slope, intercept = np.polyfit(t, prof, 1)
    prof = prof - (slope * t + intercept)
    if prof.std() < 1e-5:
        return 0.0
    win = np.hanning(L)
    spec = np.abs(np.fft.rfft(prof * win)) ** 2
    freqs = np.fft.rfftfreq(L, d=voxel_um / 1000.0)  # cycles per mm
    # Greek line spacing: ~3.0 to 8.5 mm -> 0.118 to 0.333 cycles/mm
    band = (freqs >= 1.0 / 8.5) & (freqs <= 1.0 / 3.0)
    bg = (freqs >= 0.04) & (freqs <= 1.0) & ~band
    if not band.any() or not bg.any():
        return 0.0
    return float(spec[band].max() / (np.median(spec[bg]) + 1e-9))


def score_predictions(preds_fwd: dict[str, np.ndarray], preds_rev: dict[str, np.ndarray],
                      mask: np.ndarray, voxel_um: float, win_px: int = 1024) -> dict:
    """Compute 3-model consensus, directional consensus, inter-model correlation & best window."""
    names = list(preds_fwd.keys())
    def gmean(d):
        prod = np.ones_like(next(iter(d.values())), dtype=np.float32)
        for v in d.values():
            prod *= np.clip(v, 0.0, 1.0)
        return prod ** (1.0 / len(d))

    cf = gmean(preds_fwd)
    cr = gmean(preds_rev)
    df = np.clip(cf - 0.5 * cr, 0.0, 1.0)
    dr = np.clip(cr - 0.5 * cf, 0.0, 1.0)

    H, W = mask.shape
    w_h, w_w = min(win_px, H), min(win_px, W)
    step = max(256, w_h // 2)

    best = None
    for side, cons, dir_map, pdict in (("fwd", cf, df, preds_fwd), ("rev", cr, dr, preds_rev)):
        for y0 in range(0, max(1, H - w_h + 1), step):
            for x0 in range(0, max(1, W - w_w + 1), step):
                m = mask[y0 : y0 + w_h, x0 : x0 + w_w]
                if m.mean() < 0.6:
                    continue
                cm = cons[y0 : y0 + w_h, x0 : x0 + w_w][m]
                dm = dir_map[y0 : y0 + w_h, x0 : x0 + w_w][m]
                cons_p99 = float(np.percentile(cm, 99))
                dir_p99 = float(np.percentile(dm, 99))
                # Pairwise correlation between reader_v3 and argus_v3
                p1 = pdict[names[0]][y0 : y0 + w_h, x0 : x0 + w_w][m]
                p2 = pdict[names[1]][y0 : y0 + w_h, x0 : x0 + w_w][m] if len(names) > 1 else p1
                r12 = float(np.corrcoef(p1, p2)[0, 1]) if (p1.std() > 1e-5 and p2.std() > 1e-5) else 0.0
                # Also correlation with 3rd model if present
                if len(names) > 2:
                    p3 = pdict[names[2]][y0 : y0 + w_h, x0 : x0 + w_w][m]
                    r13 = float(np.corrcoef(p1, p3)[0, 1]) if (p1.std() > 1e-5 and p3.std() > 1e-5) else 0.0
                    agree = 0.5 * (r12 + r13)
                else:
                    r13, agree = r12, r12
                per = greek_line_periodicity(cons[y0 : y0 + w_h, x0 : x0 + w_w], m, voxel_um)
                # Combined radar score: strong directional 3-model consensus * spatial agreement * modest periodicity bonus
                radar = dir_p99 * max(0.0, agree) * (1.0 + 0.05 * min(per, 20.0))
                cand = {
                    "side": side,
                    "win_yx": [int(y0), int(x0), int(w_h), int(w_w)],
                    "radar_score": round(radar, 4),
                    "dir_p99": round(dir_p99, 4),
                    "cons_p99": round(cons_p99, 4),
                    "agree_r_mean": round(agree, 4),
                    "r_reader3_argus3": round(r12, 4),
                    "r_reader3_ink9um": round(r13, 4),
                    "line_periodicity": round(per, 2),
                }
                if best is None or cand["radar_score"] > best["radar_score"]:
                    best = cand
    return {
        "best_window": best,
        "cons_fwd": cf,
        "cons_rev": cr,
        "dir_fwd": df,
        "dir_rev": dr,
    }


def save_panel(out_path: Path, title: str, ct_mid: np.ndarray, preds_best: dict[str, np.ndarray],
               cons_best: np.ndarray, dir_best: np.ndarray, win_yx: list[int], metrics: dict) -> None:
    y0, x0, wh, ww = win_yx
    sl = (slice(y0, y0 + wh), slice(x0, x0 + ww))
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    ct_c = ct_mid[sl]
    lo, hi = np.percentile(ct_c[ct_c > 0], [2, 98]) if (ct_c > 0).any() else (0, 255)
    axes[0].imshow(ct_c, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Upright Surface CT")
    names = list(preds_best.keys())
    for idx, nm in enumerate(names[:2], 1):
        p = preds_best[nm][sl]
        axes[idx].imshow(p, cmap="gray", vmin=0, vmax=max(0.4, float(np.percentile(p, 99.5))))
        axes[idx].set_title(f"{nm} ({metrics['side']})")
    c_img = cons_best[sl]
    axes[3].imshow(c_img, cmap="gray", vmin=0, vmax=max(0.4, float(np.percentile(c_img, 99.5))))
    axes[3].set_title(f"3-Model Cons p99={metrics['cons_p99']:.2f}")
    d_img = dir_best[sl]
    axes[4].imshow(d_img, cmap="gray", vmin=0, vmax=max(0.35, float(np.percentile(d_img, 99.5))))
    axes[4].set_title(f"Directional Score={metrics['radar_score']:.3f} r={metrics['agree_r_mean']:.2f}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_tranche0_jobs() -> list[dict]:
    jobs = [
        {
            "id": "CTRL_PHerc0846B_50k_winner",
            "scroll": "PHerc0846B",
            "mesh_dir": ROOT / "cache" / "killtest1_846b" / "PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750",
            "bounds": (1792, 3328, 1792, 3328),  # known $50k letter crop
            "note": "Positive control ($50k winner)",
            "auto_upright": False,  # keep published orientation
        }
    ]
    idx_path = ROOT / "cache" / "eligible_meshes" / "data" / "index.csv"
    with idx_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Top 5 brand-new Nov 17 PHerc0125 windings by on_sheet_2snaps (never inspected!)
    new125 = [r for r in rows if r["scroll"] == "PHerc0125" and r.get("on_sheet_2snaps")]
    new125.sort(key=lambda r: float(r["on_sheet_2snaps"]), reverse=True)
    # Pick diverse z windows among top scorers
    seen_z = set()
    picked125 = []
    for r in new125:
        if r["window_z"] in seen_z:
            continue
        seen_z.add(r["window_z"])
        picked125.append(r)
        if len(picked125) >= 5:
            break
    for r in picked125:
        jobs.append({
            "id": f"PHerc0125_z{r['window_z']}_{r['wrap']}",
            "scroll": "PHerc0125",
            "mesh_dir": ROOT / "cache" / "eligible_meshes" / r["path"],
            "bounds": None,
            "note": f"Nov17 uninspected, on_sheet_2snaps={r['on_sheet_2snaps']}",
            "auto_upright": True,
        })

    # Plus 1 uninspected w020 mesh each from PHerc0268 (116keV), PHerc0358, PHerc0257, PHerc0826
    for s in ("PHerc0268", "PHerc0358", "PHerc0257", "PHerc0826"):
        cands = [r for r in rows if r["scroll"] == s and r["wrap"] == "w020"]
        if not cands:
            cands = [r for r in rows if r["scroll"] == s]
        if cands:
            r = cands[0]
            jobs.append({
                "id": f"{s}_z{r['window_z']}_{r['wrap']}",
                "scroll": s,
                "mesh_dir": ROOT / "cache" / "eligible_meshes" / r["path"],
                "bounds": None,
                "note": f"Uninspected w020 (area={r.get('area_cm2','?')} cm2)",
                "auto_upright": True,
            })
    return jobs


def recenter_slab_ct_peak(
    layers_wide: np.ndarray, mask: np.ndarray, out_layers: int = 17, max_shift: int = 6, block: int = 256
) -> tuple[np.ndarray, dict]:
    """Smoothly recenter a wide slab (e.g., 29 layers) onto the local CT papyrus peak within +-max_shift.

    Bounded to +-max_shift (default +-6 vox) and smoothed across 256px blocks so it corrects
    unsnapped spiral-fit drift without ever jumping across an air gap onto an adjacent winding.
    """
    D, H, W = layers_wide.shape
    c0 = D // 2
    half = out_layers // 2
    max_s = min(max_shift, c0 - half)
    if max_s <= 0:
        return layers_wide[c0 - half : c0 + half + 1], {"dz_med": 0.0, "dz_p90": 0.0}

    ny = max(1, (H + block - 1) // block)
    nx = max(1, (W + block - 1) // block)
    dz_grid = np.zeros((ny, nx), dtype=np.float32)
    valid_grid = np.zeros((ny, nx), dtype=bool)
    lo_z, hi_z = c0 - max_s, c0 + max_s + 1

    for by in range(ny):
        y0, y1 = by * block, min(H, (by + 1) * block)
        for bx in range(nx):
            x0, x1 = bx * block, min(W, (bx + 1) * block)
            m = mask[y0:y1, x0:x1]
            if float(m.mean()) < 0.20:
                continue
            prof = layers_wide[lo_z:hi_z, y0:y1, x0:x1][:, m].mean(axis=1)
            # Smooth 3-tap profile to resist single-voxel noise
            if len(prof) >= 3:
                prof_s = np.convolve(prof, [0.25, 0.5, 0.25], mode="same")
                prof_s[0], prof_s[-1] = prof[0], prof[-1]
            else:
                prof_s = prof
            dz_grid[by, bx] = float(np.argmax(prof_s) - max_s)
            valid_grid[by, bx] = True

    if not valid_grid.any():
        return layers_wide[c0 - half : c0 + half + 1], {"dz_med": 0.0, "dz_p90": 0.0}

    med_dz = float(np.median(dz_grid[valid_grid]))
    p90_abs = float(np.percentile(np.abs(dz_grid[valid_grid]), 90))
    # Fill invalid blocks with median, then smooth spatially so shift varies gently across the sheet
    dz_fill = np.where(valid_grid, dz_grid, med_dz)
    dz_smooth = np.clip(gaussian_filter(dz_fill, sigma=0.8), -max_s, max_s)
    # Nearest/linear upsample to integer per-pixel shift
    dz_full = np.rint(zoom(dz_smooth, (H / ny, W / nx), order=1)).astype(np.int32)
    dz_full = np.clip(dz_full[:H, :W], -max_s, max_s)
    if dz_full.shape != (H, W):
        pad = np.zeros((H, W), dtype=np.int32)
        pad[: dz_full.shape[0], : dz_full.shape[1]] = dz_full
        dz_full = pad

    out = np.empty((out_layers, H, W), dtype=layers_wide.dtype)
    yy, xx = np.ogrid[:H, :W]
    for k in range(out_layers):
        zi = np.clip((c0 - half + k) + dz_full, 0, D - 1)
        out[k] = layers_wide[zi, yy, xx]
    return out, {"dz_med": round(med_dz, 2), "dz_p90_abs": round(p90_abs, 2)}


def build_tranche1_jobs(
    include_ctrl: bool = False,
    scrolls_filter: list[str] | None = None,
    include_pherc0211_only: bool = False,
    min_on_sheet: float = 0.0,
    only_nov17: bool = False,
) -> list[dict]:
    """Build First Letters meshes from eligible_meshes, sorted for max S3 chunk reuse."""
    jobs = []
    if include_ctrl:
        ctrl_dir = ROOT / "cache" / "killtest1_846b" / "PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750"
        if ctrl_dir.exists():
            jobs.append({
                "id": "CTRL_PHerc0846B_50k_winner",
                "scroll": "PHerc0846B",
                "mesh_dir": ctrl_dir,
                "bounds": None,  # UNBOUNDED full mesh!
                "note": "Positive control ($50k winner, full unbounded)",
                "auto_upright": False,
            })
    idx_path = ROOT / "cache" / "eligible_meshes" / "data" / "index.csv"
    with idx_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if include_pherc0211_only:
        rows = [r for r in rows if r["scroll"] == "PHerc0211"]
    else:
        rows = [r for r in rows if r["scroll"] in SCROLL_VOLUMES and r["scroll"] != "PHerc0211"]
        if scrolls_filter:
            allowed = set(scrolls_filter)
            rows = [r for r in rows if r["scroll"] in allowed]
    if only_nov17:
        rows = [r for r in rows if r.get("on_sheet_2snaps", "").strip() != ""]
    if min_on_sheet > 0.0:
        kept = []
        for r in rows:
            os2 = r.get("on_sheet_2snaps", "").strip()
            if os2:
                try:
                    if float(os2) < min_on_sheet:
                        continue
                except ValueError:
                    pass
            kept.append(r)
        rows = kept

    def sort_key(r):
        try:
            wz = int(r.get("window_z") or 0)
        except ValueError:
            wz = 0
        return (r["scroll"], wz, r.get("wrap", ""), r.get("path", ""))
    rows.sort(key=sort_key)
    for r in rows:
        mesh_dir = ROOT / "cache" / "eligible_meshes" / r["path"]
        if not (mesh_dir / "meta.json").exists():
            continue
        jobs.append({
            "id": f"{r['scroll']}_z{r['window_z']}_{r['wrap']}",
            "scroll": r["scroll"],
            "mesh_dir": mesh_dir,
            "bounds": None,
            "note": f"area={r.get('area_cm2','?')}cm2 on_sheet={r.get('on_sheet_2snaps','')}",
            "auto_upright": True,
            "mesh_path": r["path"],
        })
    return jobs


def save_tile_montages(out_prefix: Path, top_tiles: list[dict], tile_crops: dict, title_prefix: str, per_page: int = 5):
    """Save clean 5-row visual montages of the global top-ranked tiles."""
    show = [t for t in top_tiles if (t["mesh_id"], tuple(t["win_yx"]), t["side"]) in tile_crops]
    for page in range(0, len(show), per_page):
        chunk = show[page : page + per_page]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(13.5, 3.2 * len(chunk)))
        if len(chunk) == 1:
            axes = axes[None, :]
        for r, t in enumerate(chunk):
            ct_c, raw_c, dir_c = tile_crops[(t["mesh_id"], tuple(t["win_yx"]), t["side"])]
            y0, x0, wh, ww = t["win_yx"]
            lo, hi = np.percentile(ct_c[ct_c > 0], [2, 98]) if (ct_c > 0).any() else (0, 255)
            axes[r, 0].imshow(ct_c, cmap="gray", vmin=lo, vmax=hi)
            axes[r, 0].set_title(f"#{page+r+1} {t['mesh_id']}\n({t['side']}) y={y0} x={x0}", fontsize=8.5)
            axes[r, 1].imshow(raw_c, cmap="gray", vmin=0, vmax=max(0.45, float(np.percentile(raw_c, 99.5))))
            axes[r, 1].set_title(f"reader-v3 raw={t['raw_score']:.3f} p99={t['p99']:.2f}", fontsize=8.5)
            axes[r, 2].imshow(dir_c, cmap="gray", vmin=0, vmax=max(0.40, float(np.percentile(dir_c, 99.5))))
            extra_r = f" r_a={t['r_reader3_argus_v3']:.2f}" if "r_reader3_argus_v3" in t else ""
            axes[r, 2].set_title(
                f"Stroke={t['stroke_score']:.3f} n={t['n_strokes']} c={t['cells']} g={t['giant']:.2f}{extra_r}",
                fontsize=8.5,
            )
            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        fig.suptitle(f"{title_prefix}: Top {page+1}-{page+len(chunk)} Tiles (100% Full-Mesh Stroke Ranking)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_part{page // per_page + 1}.png", dpi=125)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("tranche0", "tranche1", "pherc0211"), default="tranche1")
    ap.add_argument("--scrolls", nargs="*", default=None, help="Optional subset of scrolls for tranche1")
    ap.add_argument("--only-nov17", action="store_true", help="Restrict to the 278 fresh Nov 17 PHerc0125 meshes")
    ap.add_argument("--min-on-sheet", type=float, default=0.0, help="Filter meshes with on_sheet_2snaps < threshold")
    ap.add_argument("--fast-reader-only", action="store_true", help="Primary screen with reader-v3 only (3x faster)")
    ap.add_argument("--confirm-thresh", type=float, default=0.10, help="Auto-confirm tiles >= thresh with argus-v3 + ink_9um")
    ap.add_argument("--aranke-thresh", type=float, default=0.0, help="If >0, run Aranke studio-softloss on tiles >= thresh")
    ap.add_argument("--aranke-disagree-samples", type=int, default=0, help="Run Aranke on N matched reader-v3 negative tiles")
    ap.add_argument("--recenter-ct", action="store_true", help="Smoothly recenter slab on local CT peak (+-6 vox)")
    ap.add_argument("--save-full-maps", action="store_true", help="Save 2x-downsampled calibrated full-mesh PNGs")
    ap.add_argument("--win-px", type=int, default=1024, help="Tile window size in full-res pixels")
    ap.add_argument("--stride-px", type=int, default=512, help="Tile stride in full-res pixels")
    ap.add_argument("--workers", type=int, default=16, help="S3 chunk prefetch threads")
    ap.add_argument("--batch-size", type=int, default=4, help="GPU batch size")
    ap.add_argument("--max-chunks", type=int, default=768, help="LRU S3 chunk cache capacity per scroll")
    ap.add_argument("--fp16", action="store_true", help="Use FP16 (only on Tensor-Core GPUs, NOT Pascal)")
    ap.add_argument("--bf16", action="store_true", help="Use BF16 (recommended on Ampere/Ada/Blackwell)")
    ap.add_argument("--max-dim", type=int, default=0, help="Optional center-crop limit (0 = FULL mesh, default)")
    ap.add_argument("--top-k-montage", type=int, default=20, help="Number of global top tiles to render in montages")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "fl_sweep"))
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out)
    maps_dir = out_dir / "full_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    lb_path = out_dir / f"{args.mode}_leaderboard.json"
    tiles_path = out_dir / f"{args.mode}_top_tiles.json"

    specs_to_load = [MODEL_SPECS[0]] if args.fast_reader_only else MODEL_SPECS
    print(f"Loading {len(specs_to_load)} model(s) on {torch.cuda.get_device_name(0)}...", flush=True)
    runners = [FlatModelRunner(name, path, fp16=args.fp16, bf16=args.bf16) for name, path in specs_to_load]
    confirm_runners: list[FlatModelRunner] | None = None if args.fast_reader_only else runners[1:]
    print("  loaded:", [(r.name, r.patch, str(r.dtype)) for r in runners], f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    def get_confirm_runners() -> list[FlatModelRunner]:
        nonlocal confirm_runners
        if confirm_runners is None:
            print("  [CONFIRM] Lazy-loading argus_v3 + ink_9um for hit confirmation...", flush=True)
            confirm_runners = [FlatModelRunner(name, path, fp16=args.fp16, bf16=args.bf16) for name, path in MODEL_SPECS[1:]]
        return confirm_runners

    aranke_runner = None
    def get_aranke():
        nonlocal aranke_runner
        if aranke_runner is None:
            import aranke_runner as ar
            print("  [ARANKE] Lazy-loading studio-softloss-20261023 (223M 3D ResUNet)...", flush=True)
            aranke_runner = ar.ArankeStudioRunner(bf16=args.bf16)
        return aranke_runner

    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    vol_cache: dict[str, tuple[ChunkCachedVolume, float]] = {}

    def get_vol(scroll: str) -> tuple[ChunkCachedVolume, float]:
        if scroll not in vol_cache:
            if args.mode in ("tranche1", "pherc0211") and vol_cache:
                vol_cache.clear()
            vpath, vum = SCROLL_VOLUMES[scroll]
            arr = zarr.open_group(FsspecStore(fs, path=vpath, read_only=True), mode="r")["0"]
            vol_cache[scroll] = (ChunkCachedVolume(arr, max_chunks=args.max_chunks), vum)
        return vol_cache[scroll]

    if args.mode == "tranche0":
        jobs = build_tranche0_jobs()
    elif args.mode == "pherc0211":
        jobs = build_tranche1_jobs(include_ctrl=False, include_pherc0211_only=True, min_on_sheet=args.min_on_sheet)
    else:
        jobs = build_tranche1_jobs(
            include_ctrl=False, scrolls_filter=args.scrolls, min_on_sheet=args.min_on_sheet, only_nov17=args.only_nov17
        )

    print(
        f"\nRunning {args.mode}: {len(jobs)} jobs (fast_reader_only={args.fast_reader_only}, recenter_ct={args.recenter_ct}, "
        f"aranke_thresh={args.aranke_thresh}, disagree_n={args.aranke_disagree_samples}, save_full_maps={args.save_full_maps}, "
        f"win={args.win_px}, stride={args.stride_px}, bs={args.batch_size})",
        flush=True,
    )
    results = []
    global_tiles: list[dict] = []
    aranke_hits: list[dict] = []
    disagree_records: list[dict] = []
    disagree_stride = max(1, len(jobs) // max(1, args.aranke_disagree_samples)) if args.aranke_disagree_samples > 0 else 999999
    tile_crops: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    t_all = time.perf_counter()

    def flush_leaderboard():
        ranked_meshes = sorted(
            results, key=lambda r: (r["best_tile"]["stroke_score"] if r.get("best_tile") else -1.0), reverse=True
        )
        ranked_tiles = sorted(global_tiles, key=lambda t: t["stroke_score"], reverse=True)
        tot_valid = round(sum(r.get("valid_area_cm2", 0.0) for r in results), 2)
        tot_bbox = round(sum(r.get("bbox_area_cm2", 0.0) for r in results), 2)
        lb_path.write_text(json.dumps({
            "mode": args.mode,
            "completed": len(results),
            "total_jobs": len(jobs),
            "total_valid_cm2": tot_valid,
            "total_bbox_cm2": tot_bbox,
            "total_tiles_evaluated": sum(r.get("n_tiles_evaluated", 0) for r in results),
            "total_seconds": round(time.perf_counter() - t_all, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "aranke_hits_count": len(aranke_hits),
            "disagree_samples_count": len(disagree_records),
            "leaderboard": ranked_meshes,
        }, indent=2), encoding="utf-8")
        tiles_path.write_text(json.dumps(ranked_tiles[:150], indent=2), encoding="utf-8")
        if aranke_hits:
            (out_dir / f"{args.mode}_aranke_hits.json").write_text(
                json.dumps(sorted(aranke_hits, key=lambda x: x["studio_score"], reverse=True), indent=2), encoding="utf-8"
            )
        if disagree_records:
            (out_dir / f"{args.mode}_disagree_pilot.json").write_text(
                json.dumps(sorted(disagree_records, key=lambda x: x["studio_score"] - x["reader_score"], reverse=True), indent=2),
                encoding="utf-8",
            )

    for ji, job in enumerate(jobs, 1):
        print(f"\n[{ji}/{len(jobs)}] {job['id']} ({job['note']})", flush=True)
        try:
            cvol, vum = get_vol(job["scroll"])
            surf = UprightTifXYZSurface(job["mesh_dir"], auto_upright=job["auto_upright"])
            fh, fw = surf.full_resolution_shape
            bounds = job["bounds"]
            if bounds is None and args.max_dim > 0:
                ch, cw = min(fh, args.max_dim), min(fw, args.max_dim)
                r0, c0 = max(0, (fh - ch) // 2), max(0, (fw - cw) // 2)
                bounds = (r0, r0 + ch, c0, c0 + cw)
            H = (bounds[1] - bounds[0]) if bounds else fh
            W = (bounds[3] - bounds[2]) if bounds else fw

            hits_before, misses_before = cvol.hits, cvol.misses
            n_layers = 29 if args.recenter_ct else 17
            layers_raw, mask, dt_ren = render_surface_cached(surf, cvol, bounds=bounds, layers=n_layers, workers=args.workers)
            if args.recenter_ct:
                layers, rec_info = recenter_slab_ct_peak(layers_raw, mask, out_layers=17, max_shift=6, block=256)
            else:
                layers, rec_info = layers_raw, {"dz_med": 0.0, "dz_p90_abs": 0.0}
            new_misses = cvol.misses - misses_before
            new_hits = cvol.hits - hits_before
            valid_frac = float(mask.mean())
            mid_ct = layers[8]
            ct_std = float(mid_ct[mask].std()) if mask.any() else 0.0

            t_gpu = time.perf_counter()
            r0 = runners[0]
            bs0 = args.batch_size if r0.patch <= 256 else max(1, args.batch_size // 2)
            pf0 = r0.infer(layers, mask, overlap=0.5, reverse=False, batch_size=bs0)
            pr0 = r0.infer(layers, mask, overlap=0.5, reverse=True, batch_size=bs0)

            extra_fwd, extra_rev = {}, {}
            if not args.fast_reader_only:
                for rx in runners[1:]:
                    bsx = args.batch_size if rx.patch <= 256 else max(1, args.batch_size // 2)
                    extra_fwd[rx.name] = rx.infer(layers, mask, overlap=0.5, reverse=False, batch_size=bsx)
                    extra_rev[rx.name] = rx.infer(layers, mask, overlap=0.5, reverse=True, batch_size=bsx)
            torch.cuda.synchronize()

            scored = ss.score_full_mesh_tiles(
                pf0, pr0, mask, vum,
                win_px=args.win_px, stride_px=args.stride_px, top_k=8, nms_iou=0.35,
                extra_fwd=extra_fwd or None, extra_rev=extra_rev or None,
            )

            # Auto-confirm strong hits with argus_v3 + ink_9um in fast-reader-only mode!
            bt = scored["best_tile"]
            if args.fast_reader_only and bt is not None and bt["stroke_score"] >= args.confirm_thresh:
                for rx in get_confirm_runners():
                    bsx = args.batch_size if rx.patch <= 256 else max(1, args.batch_size // 2)
                    extra_fwd[rx.name] = rx.infer(layers, mask, overlap=0.5, reverse=False, batch_size=bsx)
                    extra_rev[rx.name] = rx.infer(layers, mask, overlap=0.5, reverse=True, batch_size=bsx)
                torch.cuda.synchronize()
                scored = ss.score_full_mesh_tiles(
                    pf0, pr0, mask, vum,
                    win_px=args.win_px, stride_px=args.stride_px, top_k=8, nms_iou=0.35,
                    extra_fwd=extra_fwd, extra_rev=extra_rev,
                )
                bt = scored["best_tile"]

            # Aranke studio-softloss super-resolution confirmation on flagged tiles (>= aranke_thresh)
            if args.aranke_thresh > 0.0 and scored["top_tiles"]:
                import aranke_runner as ar
                for t in scored["top_tiles"][:3]:
                    if t["stroke_score"] >= args.aranke_thresh:
                        y0, x0, wh, ww = t["win_yx"]
                        ps = pf0 if t["side"] == "fwd" else pr0
                        s_info, s_best, s_dir = get_aranke().score_tile(
                            layers[:, y0:y0+wh, x0:x0+ww], mask[y0:y0+wh, x0:x0+ww],
                            ps[y0:y0+wh, x0:x0+ww], vum, batch_size=max(8, args.batch_size // 2), stride=64,
                        )
                        t.update(s_info)
                        hit_rec = {"mesh_id": job["id"], "scroll": job["scroll"], "win_yx": t["win_yx"],
                                   "reader_score": t["stroke_score"], **s_info}
                        aranke_hits.append(hit_rec)
                        ar.save_aranke_confirm_panel(
                            out_dir / "aranke_hits" / f"{job['id']}_y{y0}_x{x0}.png",
                            f"HIT {job['id']} ({t['side']}) y={y0} x={x0}",
                            mid_ct[y0:y0+wh, x0:x0+ww], ps[y0:y0+wh, x0:x0+ww], s_best, s_dir, t["stroke_score"], s_info,
                        )
                        print(f"    [ARANKE HIT] y={y0} x={x0}: reader={t['stroke_score']:.3f} -> studio={s_info['studio_score']:.3f} r={s_info['r_reader_studio']:.2f}", flush=True)

            # Disagreement mining pilot: sample N matched reader-v3-negative clean tiles for Aranke!
            if (
                args.aranke_disagree_samples > 0
                and len(disagree_records) < args.aranke_disagree_samples
                and (ji - 1) % disagree_stride == 0
                and bt is not None
                and bt["stroke_score"] < 0.04
                and bt["valid_frac"] >= 0.70
            ):
                import aranke_runner as ar
                y0, x0, wh, ww = bt["win_yx"]
                ps = pf0 if bt["side"] == "fwd" else pr0
                s_info, s_best, s_dir = get_aranke().score_tile(
                    layers[:, y0:y0+wh, x0:x0+ww], mask[y0:y0+wh, x0:x0+ww],
                    ps[y0:y0+wh, x0:x0+ww], vum, batch_size=max(8, args.batch_size // 2), stride=64,
                )
                d_rec = {"mesh_id": job["id"], "scroll": job["scroll"], "win_yx": bt["win_yx"],
                         "reader_score": bt["stroke_score"], **s_info}
                disagree_records.append(d_rec)
                if s_info["studio_score"] >= 0.12:
                    ar.save_aranke_confirm_panel(
                        out_dir / "disagree_hits" / f"{job['id']}_y{y0}_x{x0}.png",
                        f"DISAGREE {job['id']} y={y0} x={x0}",
                        mid_ct[y0:y0+wh, x0:x0+ww], ps[y0:y0+wh, x0:x0+ww], s_best, s_dir, bt["stroke_score"], s_info,
                    )
                print(f"    [DISAGREE #{len(disagree_records)}] reader={bt['stroke_score']:.3f} vs studio={s_info['studio_score']:.3f} r={s_info['r_reader_studio']:.2f}", flush=True)

            dt_gpu = time.perf_counter() - t_gpu

            if args.save_full_maps:
                ss.save_full_mesh_maps(maps_dir, job["id"], mid_ct, pf0, pr0, mask)

            cm2_hr = scored["valid_area_cm2"] / max(1e-6, (dt_ren + dt_gpu) / 3600.0)
            rec = {
                "id": job["id"],
                "scroll": job["scroll"],
                "mesh_path": job.get("mesh_path", ""),
                "note": job["note"],
                "rendered_hw": [int(H), int(W)],
                "valid_area_cm2": scored["valid_area_cm2"],
                "bbox_area_cm2": scored["bbox_area_cm2"],
                "valid_frac": round(valid_frac, 3),
                "ct_std": round(ct_std, 1),
                "recenter": rec_info,
                "n_tiles_evaluated": scored["n_tiles_evaluated"],
                "render_seconds": round(dt_ren, 1),
                "gpu_seconds": round(dt_gpu, 1),
                "s3_chunks_fetched": int(new_misses),
                "s3_cache_hits": int(new_hits),
                "valid_cm2_hr": round(cm2_hr, 1),
                "best_tile": bt,
                "top_tiles": scored["top_tiles"][:5],
            }
            results.append(rec)

            # Merge all top-K NMS tiles from this mesh into the global tile leaderboard
            for t in scored["top_tiles"]:
                gt = {"mesh_id": job["id"], "scroll": job["scroll"], **t}
                global_tiles.append(gt)

            # Retain visual crops only for current global top-(top_k_montage + 10) tiles (bounded RAM!)
            global_tiles.sort(key=lambda t: t["stroke_score"], reverse=True)
            keep_n = max(30, args.top_k_montage + 10)
            keep_keys = {(t["mesh_id"], tuple(t["win_yx"]), t["side"]) for t in global_tiles[:keep_n]}
            for t in scored["top_tiles"]:
                k = (job["id"], tuple(t["win_yx"]), t["side"])
                if k in keep_keys:
                    y0, x0, wh, ww = t["win_yx"]
                    ps = pf0 if t["side"] == "fwd" else pr0
                    po = pr0 if t["side"] == "fwd" else pf0
                    tile_crops[k] = (
                        mid_ct[y0:y0+wh, x0:x0+ww].copy(),
                        ps[y0:y0+wh, x0:x0+ww].copy(),
                        np.clip(ps[y0:y0+wh, x0:x0+ww] - 0.5 * po[y0:y0+wh, x0:x0+ww], 0.0, 1.0),
                    )
            for old_k in list(tile_crops.keys()):
                if old_k not in keep_keys:
                    tile_crops.pop(old_k, None)

            if bt is not None:
                print(
                    f"  -> ren={dt_ren:.1f}s gpu={dt_gpu:.1f}s | valid={scored['valid_area_cm2']:.2f}cm2 "
                    f"tiles={scored['n_tiles_evaluated']} | top1=({bt['side']}) stroke={bt['stroke_score']:.4f} "
                    f"n={bt['n_strokes']} c={bt['cells']} g={bt['giant']:.2f} p99={bt['p99']:.2f} dz={rec_info['dz_med']:+.1f}",
                    flush=True,
                )
        except Exception as e:
            print(f"  !! ERROR on {job['id']}: {e}", flush=True)
            results.append({"id": job["id"], "scroll": job["scroll"], "error": str(e), "best_tile": None})
        if ji % 10 == 0 or ji == len(jobs):
            flush_leaderboard()

    flush_leaderboard()
    global_tiles.sort(key=lambda t: t["stroke_score"], reverse=True)
    print(f"\n=== {args.mode.upper()} GLOBAL TOP 25 TILES (across {len(results)} meshes) ===", flush=True)
    for rank, t in enumerate(global_tiles[:25], 1):
        y0, x0, wh, ww = t["win_yx"]
        print(
            f"  #{rank:2d} {t['mesh_id']:30s} ({t['side']}) y={y0:4d} x={x0:4d} | "
            f"stroke={t['stroke_score']:.4f} n={t['n_strokes']:2d} c={t['cells']:2d} g={t['giant']:.2f} p99={t['p99']:.2f}",
            flush=True,
        )

    save_tile_montages(
        out_dir / f"{args.mode}_top_tiles",
        global_tiles[: args.top_k_montage],
        tile_crops,
        args.mode.upper(),
        per_page=5,
    )
    print(f"\nDone in {time.perf_counter()-t_all:.1f}s -> {lb_path}", flush=True)


if __name__ == "__main__":
    main()
