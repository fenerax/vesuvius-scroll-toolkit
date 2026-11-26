"""Kill Test 1A: Reproduce today's $50k PHerc0846B First Letters win with reader-v3 on GTX 1070 Ti.

Streams only the 1536x1536 letter region of PHerc0846B-s20mm-b03-g08-t2-01 from S3 (9.362 um),
runs reader-v3.pth (FP32, 384x384, 50% overlap, Hann blend, robust_mad 1-99%), and compares
pixel-for-pixel against brkzak's published _reader-v3_raw.tif.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import tifffile
import torch
import zarr
from scipy.ndimage import map_coordinates
from zarr.storage import FsspecStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cache" / "villa_src" / "vesuvius" / "src"))

# PyTorch 2.6 shim for torch.compiler.disable(reason=...)
_orig_disable = torch.compiler.disable
torch.compiler.disable = lambda fn=None, recursive=True, **kw: _orig_disable(fn, recursive=recursive)

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.checkpoint import load_checkpoint, select_inference_weights
from vesuvius.ink_detection.models.model import make_model


class TifXYZSurface:
    """Pure NumPy/SciPy reader for Vesuvius tifxyz surfaces (no vc_reader C++ dependency)."""

    def __init__(self, seg_dir: Path):
        self.meta = json.loads((seg_dir / "meta.json").read_text(encoding="utf-8"))
        self.x = tifffile.imread(seg_dir / "x.tif").astype(np.float64)
        self.y = tifffile.imread(seg_dir / "y.tif").astype(np.float64)
        self.z = tifffile.imread(seg_dir / "z.tif").astype(np.float64)
        self.scale_y, self.scale_x = float(self.meta["scale"][1]), float(self.meta["scale"][0])
        self.shape = self.x.shape
        self.full_resolution_shape = (
            int(round(self.shape[0] / self.scale_y)),
            int(round(self.shape[1] / self.scale_x)),
        )
        self.valid_grid = (self.x > 0) | (self.y > 0) | (self.z > 0)

    def tile_geometry(self, bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r0, r1, c0, c1 = bounds
        rr = np.arange(r0, r1, dtype=np.float64) * self.scale_y
        cc = np.arange(c0, c1, dtype=np.float64) * self.scale_x
        gr, gc = np.meshgrid(rr, cc, indexing="ij")
        h, w = gr.shape

        # Valid iff all 4 surrounding grid corners are valid
        ir0 = np.clip(np.floor(gr).astype(int), 0, self.shape[0] - 1)
        ic0 = np.clip(np.floor(gc).astype(int), 0, self.shape[1] - 1)
        ir1 = np.clip(ir0 + 1, 0, self.shape[0] - 1)
        ic1 = np.clip(ic0 + 1, 0, self.shape[1] - 1)
        valid = (
            (gr >= 0) & (gr <= self.shape[0] - 1) & (gc >= 0) & (gc <= self.shape[1] - 1)
            & self.valid_grid[ir0, ic0] & self.valid_grid[ir0, ic1]
            & self.valid_grid[ir1, ic0] & self.valid_grid[ir1, ic1]
        )
        coords = np.stack([gr, gc], axis=0)
        px = map_coordinates(self.x, coords, order=1, mode="nearest")
        py = map_coordinates(self.y, coords, order=1, mode="nearest")
        pz = map_coordinates(self.z, coords, order=1, mode="nearest")
        xyz = np.stack([px, py, pz], axis=-1)

        # Tangent vectors via central finite differences at 1-px step in full-res space
        eps_r, eps_c = self.scale_y, self.scale_x
        def sample_at(dr, dc):
            c = np.stack([np.clip(gr + dr, 0, self.shape[0] - 1), np.clip(gc + dc, 0, self.shape[1] - 1)], axis=0)
            return np.stack([
                map_coordinates(self.x, c, order=1, mode="nearest"),
                map_coordinates(self.y, c, order=1, mode="nearest"),
                map_coordinates(self.z, c, order=1, mode="nearest"),
            ], axis=-1)

        du = sample_at(0.0, eps_c) - sample_at(0.0, -eps_c)
        dv = sample_at(eps_r, 0.0) - sample_at(-eps_r, 0.0)
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


def load_reader_v3(ckpt_path: Path = ROOT / "cache" / "models" / "reader-v3.pth"):
    ckpt = load_checkpoint(ckpt_path)
    cfg = InkConfig.from_mapping(ckpt["config"])
    model = make_model(cfg)
    _, weights = select_inference_weights(ckpt)
    model.load_state_dict(weights, strict=True)
    return model.to("cuda").eval()


def infer_reader_v3(model, layers_zyx: np.ndarray, mask_yx: np.ndarray | None = None,
                    patch: int = 384, overlap: float = 0.5, reverse: bool = False) -> tuple[np.ndarray, float]:
    """Run reader-v3 on (17, H, W) uint8 surface volume. Returns (prob_float32, gpu_seconds)."""
    vol = layers_zyx[::-1] if reverse else layers_zyx
    D, H, W = vol.shape
    assert D == 17, f"expected 17 layers, got {D}"
    stride = max(1, int(round(patch * (1.0 - overlap))))
    hw = torch.outer(
        torch.hann_window(patch, periodic=False, dtype=torch.float32),
        torch.hann_window(patch, periodic=False, dtype=torch.float32),
    )
    hw = torch.clamp(hw / hw.max().clamp(min=1e-8), min=0.001).numpy()

    def starts(n):
        if n <= patch:
            return [0]
        s = list(range(0, n - patch, stride))
        if not s or s[-1] != n - patch:
            s.append(n - patch)
        return s

    prob_sum = np.zeros((H, W), dtype=np.float32)
    wt_sum = np.zeros((H, W), dtype=np.float32)
    t0 = time.perf_counter()
    n_tiles = 0
    with torch.inference_mode():
        for y0 in starts(H):
            for x0 in starts(W):
                y1, x1 = min(y0 + patch, H), min(x0 + patch, W)
                vh, vw = y1 - y0, x1 - x0
                if mask_yx is not None and not mask_yx[y0:y1, x0:x1].any():
                    continue
                crop = vol[:, y0:y1, x0:x1]
                if not crop.any():
                    continue
                if vh < patch or vw < patch:
                    pad = np.zeros((D, patch, patch), dtype=crop.dtype)
                    pad[:, :vh, :vw] = crop
                    crop = pad
                norm = normalize_robust(crop)
                t = torch.from_numpy(np.ascontiguousarray(norm))[None, None].to("cuda", torch.float32)
                out = model(t)
                logits = out["ink"] if isinstance(out, dict) else out
                p = torch.sigmoid(logits)[0, 0, :vh, :vw].cpu().numpy()
                w = hw[:vh, :vw]
                prob_sum[y0:y1, x0:x1] += p * w
                wt_sum[y0:y1, x0:x1] += w
                n_tiles += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    prob = np.divide(prob_sum, wt_sum, out=np.zeros_like(prob_sum), where=wt_sum > 0)
    print(f"  reader-v3 ({'rev' if reverse else 'fwd'}): {n_tiles} tiles in {dt:.2f}s ({dt/max(1,n_tiles)*1000:.1f} ms/tile)", flush=True)
    return prob, dt


def read_bbox_padded(arr, lo, hi):
    z0, y0, x0 = (int(v) for v in lo)
    z1, y1, x1 = (int(v) for v in hi)
    Z, Y, X = arr.shape
    cz0, cz1 = max(0, z0), min(Z, z1)
    cy0, cy1 = max(0, y0), min(Y, y1)
    cx0, cx1 = max(0, x0), min(X, x1)
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=arr.dtype)
    if cz1 > cz0 and cy1 > cy0 and cx1 > cx0:
        out[cz0 - z0 : cz1 - z0, cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = np.asarray(
            arr[cz0:cz1, cy0:cy1, cx0:cx1]
        )
    return out


def main():
    t_all = time.perf_counter()
    base = ROOT / "cache" / "killtest1_846b"
    seg_name = "PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750"
    seg_dir = base / seg_name
    ref_raw = tifffile.imread(base / f"{seg_name}_reader-v3_raw.tif")
    surf = TifXYZSurface(seg_dir)
    full_h, full_w = surf.full_resolution_shape
    print(f"846B surface full shape: {(full_h, full_w)}, ref_raw: {ref_raw.shape}", flush=True)

    # Open PHerc0846B 9.362um L0 volume on S3
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    vol_path = "vesuvius-challenge-open-data/PHerc0846B/volumes/20250804142305-9.362um-1.2m-113keV-masked.zarr"
    vol = zarr.open_group(FsspecStore(fs, path=vol_path, read_only=True), mode="r")["0"]
    umbilicus_xy = (vol.shape[2] / 2.0, vol.shape[1] / 2.0)

    # Normal sign: point towards scroll axis on >50% of the segment
    xyz_c, nrm_c, val_c = surf.tile_geometry((0, full_h, 0, full_w)) if full_h <= 1024 else surf.tile_geometry((1792, 3328, 1792, 3328))
    towards_axis = ((np.asarray(umbilicus_xy, np.float64) - xyz_c[val_c, :2]) * nrm_c[val_c, :2]).sum(1) > 0
    sign = 1.0 if towards_axis.mean() >= 0.5 else -1.0
    print(f"  normal sign={sign:+.0f} (towards axis {max(towards_axis.mean(), 1-towards_axis.mean()):.0%})", flush=True)

    # Crop 1536x1536 around brightest letter region (2048, 2048)
    R0, C0, SIZE = 1792, 1792, 1536
    R1, C1 = R0 + SIZE, C0 + SIZE
    LAYERS = 17
    offsets = np.arange(LAYERS, dtype=np.float64) - LAYERS // 2
    layers = np.zeros((LAYERS, SIZE, SIZE), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    TILE = 256
    tiles = [(r, min(r + TILE, R1), c, min(c + TILE, C1))
             for r in range(R0, R1, TILE) for c in range(C0, C1, TILE)]

    t_ren = time.perf_counter()
    def render_tile(b):
        r0, r1, c0, c1 = b
        xyz, nrm, val = surf.tile_geometry(b)
        if not val.any():
            return b, None, val
        pts = xyz[val][None] + sign * offsets[:, None, None] * nrm[val][None]
        zyx = pts[..., ::-1].reshape(-1, 3)
        lo = np.floor(zyx.min(0)).astype(int) - 1
        hi = np.ceil(zyx.max(0)).astype(int) + 2
        blk = read_bbox_padded(vol, lo, hi)
        sampled = map_coordinates(blk, (zyx - lo).T, order=1, mode="constant", cval=0.0)
        sampled = np.clip(np.rint(sampled), 0, 255).astype(np.uint8).reshape(LAYERS, -1)
        return b, sampled, val

    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, (b, sampled, val) in enumerate(ex.map(render_tile, tiles), 1):
            r0, r1, c0, c1 = b
            lr0, lr1, lc0, lc1 = r0 - R0, r1 - R0, c0 - C0, c1 - C0
            mask[lr0:lr1, lc0:lc1] = val
            if sampled is not None:
                layers[:, lr0:lr1, lc0:lc1][:, val] = sampled
            if i % 12 == 0 or i == len(tiles):
                print(f"  rendered {i}/{len(tiles)} tiles ({time.perf_counter()-t_ren:.1f}s)", flush=True)
    dt_ren = time.perf_counter() - t_ren

    torch.cuda.reset_peak_memory_stats()
    model = load_reader_v3()
    prob_fwd, dt_gpu = infer_reader_v3(model, layers, mask, reverse=False)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    ours_u8 = np.clip(np.rint(prob_fwd * 255.0), 0, 255).astype(np.uint8)
    ref_crop = ref_raw[R0:R1, C0:C1]
    # Trim 192px border where overlap with outside-crop tiles differs slightly
    sl = (slice(192, SIZE - 192), slice(192, SIZE - 192))
    m_in = mask[sl]
    r_full = float(np.corrcoef(ours_u8[mask].ravel(), ref_crop[mask].ravel())[0, 1])
    r_core = float(np.corrcoef(ours_u8[sl][m_in].ravel(), ref_crop[sl][m_in].ravel())[0, 1])
    mae_core = float(np.abs(ours_u8[sl][m_in].astype(float) - ref_crop[sl][m_in].astype(float)).mean())
    area_cm2 = (SIZE * 9.362 / 1e4) ** 2
    cm2_per_hr_gpu = area_cm2 / (dt_gpu / 3600.0)
    cm2_per_hr_total = area_cm2 / ((dt_ren + dt_gpu) / 3600.0)

    summary = {
        "segment": seg_name,
        "crop_rc_size": [R0, C0, SIZE],
        "area_cm2": round(area_cm2, 3),
        "render_seconds": round(dt_ren, 2),
        "gpu_seconds": round(dt_gpu, 2),
        "peak_vram_gb": round(peak_gb, 3),
        "throughput_cm2_per_hr_gpu": round(cm2_per_hr_gpu, 1),
        "throughput_cm2_per_hr_end_to_end": round(cm2_per_hr_total, 1),
        "pearson_r_full_crop": round(r_full, 4),
        "pearson_r_interior_core": round(r_core, 4),
        "mae_u8_interior_core": round(mae_core, 2),
        "pass_bar_r_ge_0.9": r_core >= 0.9,
        "pass_bar_cm2_hr_ge_20": cm2_per_hr_total >= 20.0,
    }
    print("\n=== KILL TEST 1A SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    out_dir = ROOT / "outputs" / "killtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "killtest1a_846b_repro.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Side-by-side visual verification
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(layers[LAYERS // 2][sl], cmap="gray")
    axes[0].set_title("PHerc0846B Surface CT (mid-layer, 9.36 um)")
    axes[1].imshow(ref_crop[sl], cmap="gray", vmin=0, vmax=230)
    axes[1].set_title("Published $50k brkzak reader-v3_raw")
    axes[2].imshow(ours_u8[sl], cmap="gray", vmin=0, vmax=230)
    axes[2].set_title(f"Ours (GTX 1070 Ti FP32): r={r_core:.4f}, {cm2_per_hr_total:.0f} cm2/hr")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Kill Test 1A: PHerc0846B First Letters Reproduction (r={r_core:.4f}, MAE={mae_core:.1f}/255, VRAM={peak_gb:.2f} GB)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "killtest1a_846b_repro.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'killtest1a_846b_repro.png'} in {time.perf_counter()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
