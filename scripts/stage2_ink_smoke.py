"""
Stage 2 — Vesuvius Challenge Path 2 Ink-Detection Smoke Test (GTX 1070 Ti / Pascal)

Reproduces the official `scrollprize/ink_canonical_2um` (ResNet3D-152 + 3D Decoder,
`new_canon_autoresearch_recipe`) on a chunk-aligned 768x768 surface-volume crop
streamed directly from AWS S3 (PHerc0139 segment 20260311000000), and compares
our local prediction side-by-side against the official published ink preview.

Official preprocessing verified against ScrollPrize/villa/ink-detection/optimized_inference:
  - 62 depth layers centered on the 109-layer surface volume: layers [23, 85)
  - Clip uint8 to [0, 200] and scale by / 200.0 (float32)
  - Input tensor: (1, 1, 62, 256, 256)
  - Sliding window: tile_size=256, stride=128, Hann 2D blending
  - Output: sigmoid(logits) -> bilinear upsample 64x64 -> 256x256
  - FP32 on Pascal GTX 1070 Ti (no Tensor Cores / slow FP16)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import torch
import torch.nn.functional as F
import zarr
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "vendor"))

from model_resnet3d_3d_decoder import load_model  # noqa: E402

BUCKET = "vesuvius-challenge-open-data"
CATALOG_PATH = ROOT / "data" / "s3_metadata.min.json"
CKPT_PATH = ROOT / "cache" / "models" / "r152_3ddec_v2_l5_epoch13.ckpt"
CACHE_STAGE1 = ROOT / "cache" / "stage1"
OUT_DIR = ROOT / "outputs"

TARGET_SAMPLE = "PHerc0139"
TARGET_VOLUME_ID = "20260102150214"
TARGET_SEGMENT_ID = "20260311000000"

START_LAYER = 23
END_LAYER = 85  # 62 channels centered on 109-layer volume
MAX_CLIP = 200.0
TILE_SIZE = 256
STRIDE = 128
CROP_SIZE = 768  # 6x6 chunks of 128px (~61 MiB streamed) -> 5x5 = 25 tiles


def hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    win = np.outer(wy, wx)
    return np.clip(win, 1e-3, None)


def verify_device() -> torch.device:
    print(f"PyTorch: {torch.__version__}  CUDA built: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  CUDA not available -> falling back to CPU")
        return torch.device("cpu")

    arches = torch.cuda.get_arch_list()
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"  GPU: {name} (compute capability {cc[0]}.{cc[1]})")
    print(f"  Built arches: {arches}")
    has_sm = f"sm_{cc[0]}{cc[1]}" in arches
    print(f"  Matches sm_{cc[0]}{cc[1]}: {has_sm}")

    try:
        x = torch.randn(1, 1, 8, 32, 32, device="cuda", dtype=torch.float32)
        conv = torch.nn.Conv3d(1, 4, 3, padding=1).cuda()
        with torch.inference_mode():
            y = conv(x)
        torch.cuda.synchronize()
        print(f"  Real Conv3d CUDA forward pass OK: {tuple(y.shape)}")
        return torch.device("cuda")
    except Exception as exc:
        print(f"  CUDA forward pass FAILED ({exc}) -> falling back to CPU")
        return torch.device("cpu")


def lookup_paths() -> tuple[str, Path]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    seg = meta["samples"][TARGET_SAMPLE]["segments"][TARGET_SEGMENT_ID]
    layers_zarr = None
    ink_jpg_key = None
    for item in seg["data"]:
        params = item.get("parameters") or {}
        if params.get("target_volume") != TARGET_VOLUME_ID:
            continue
        if item["type"] == "layers-zarr":
            layers_zarr = item["origins"][0]["path"]
        elif item["type"] == "ink-detection-downsampled":
            ink_jpg_key = item["origins"][0]["path"]
    assert layers_zarr and ink_jpg_key, "Missing layers-zarr or ink-detection-downsampled in catalog"
    return layers_zarr, CACHE_STAGE1 / ink_jpg_key


def pick_letter_rich_roi(ink_ds8: np.ndarray, full_h: int, full_w: int, crop_px: int) -> tuple[int, int]:
    """Pick a chunk-aligned (128px) ROI with strong ink contrast (mid-density strokes, not empty/saturated)."""
    ds = full_h / ink_ds8.shape[0]
    win = int(round(crop_px / ds))
    best_score = -1.0
    best_yx = (full_h // 2, full_w // 2)

    # Search interior region (avoid noisy scroll edges)
    h8, w8 = ink_ds8.shape
    step8 = max(16, int(round(128 / ds)))
    arr = ink_ds8.astype(np.float32) / 255.0
    for r in range(int(h8 * 0.15), int(h8 * 0.85) - win, step8):
        for c in range(int(w8 * 0.15), int(w8 * 0.85) - win, step8):
            patch = arr[r : r + win, c : c + win]
            mean = float(patch.mean())
            std = float(patch.std())
            # Prefer clear stroke/background contrast: ~20-55% bright ink with high variance & sharp gradients
            if 0.18 <= mean <= 0.55:
                gx = np.abs(np.diff(patch, axis=1)).mean()
                gy = np.abs(np.diff(patch, axis=0)).mean()
                score = std * float(gx + gy)
                if score > best_score:
                    best_score = score
                    best_yx = (int(round(r * ds)), int(round(c * ds)))

    # Snap cleanly to 128-px Zarr chunk boundaries
    y0 = max(0, min((best_yx[0] // 128) * 128, full_h - crop_px))
    x0 = max(0, min((best_yx[1] // 128) * 128, full_w - crop_px))
    return y0, x0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

    device = verify_device()
    layers_key, ink_jpg_path = lookup_paths()
    ink_ds8 = np.array(Image.open(ink_jpg_path).convert("L"))
    print(f"\nLoaded published ds8 ink preview: {ink_ds8.shape}")

    # Open surface-volume Zarr level 0 (anonymous S3)
    from zarr.storage import FsspecStore
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    store = FsspecStore(fs, path=f"{BUCKET}/{layers_key.strip('/')}", read_only=True)
    grp = zarr.open_group(store=store, mode="r")
    arr_l0 = grp["0"]
    depth, full_h, full_w = arr_l0.shape
    print(f"Opened surface-volume layers-zarr L0: shape={arr_l0.shape} chunks={arr_l0.chunks}")

    # Pick a high-contrast Greek-letter ROI aligned to 128px chunks
    y0, x0 = pick_letter_rich_roi(ink_ds8, full_h, full_w, CROP_SIZE)
    y1, x1 = y0 + CROP_SIZE, x0 + CROP_SIZE
    print(f"Selected chunk-aligned ROI: Y=[{y0}:{y1}], X=[{x0}:{x1}], Z=[{START_LAYER}:{END_LAYER}) (62 layers)")

    # Stream ONLY the requested 62x768x768 surface-volume crop (~61 MiB on wire)
    t0 = time.perf_counter()
    vol_crop = np.asarray(arr_l0[START_LAYER:END_LAYER, y0:y1, x0:x1])  # (62, 768, 768) uint8
    dt_stream = time.perf_counter() - t0
    print(
        f"Streamed surface-volume crop {vol_crop.shape} ({vol_crop.nbytes / 1e6:.1f} MB) "
        f"in {dt_stream:.2f}s ({vol_crop.nbytes / max(dt_stream, 1e-6) / 1e6:.2f} MB/s)"
    )

    # Load official ResNet3D-152 + 3D decoder checkpoint (FP32 for Pascal)
    t0 = time.perf_counter()
    wrapper = load_model(str(CKPT_PATH), device=device, num_frames=END_LAYER - START_LAYER)
    wrapper.eval()
    dt_load = time.perf_counter() - t0
    print(f"Loaded ResNet3D-152 3D-decoder model onto {device} in {dt_load:.2f}s")

    # Sliding-window inference: tile_size=256, stride=128, batch_size=1, FP32
    ys = list(range(0, CROP_SIZE - TILE_SIZE + 1, STRIDE))
    xs = list(range(0, CROP_SIZE - TILE_SIZE + 1, STRIDE))
    coords = [(yy, xx) for yy in ys for xx in xs]
    win = hann2d(TILE_SIZE, TILE_SIZE)

    pred_acc = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    weight_acc = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for idx, (yy, xx) in enumerate(coords, 1):
            tile = vol_crop[:, yy : yy + TILE_SIZE, xx : xx + TILE_SIZE]  # (62, 256, 256)
            tile_f = np.clip(tile, 0, MAX_CLIP).astype(np.float32) / MAX_CLIP
            tens = torch.from_numpy(tile_f)[None, None].to(device)  # (1, 1, 62, 256, 256)

            # FP32 forward pass (no autocast on Pascal sm_61)
            logits = wrapper.forward(tens)  # (1, 1, 64, 64)
            probs = torch.sigmoid(logits)
            probs_up = F.interpolate(
                probs.float(), size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()

            pred_acc[yy : yy + TILE_SIZE, xx : xx + TILE_SIZE] += probs_up * win
            weight_acc[yy : yy + TILE_SIZE, xx : xx + TILE_SIZE] += win
            if idx == 1 or idx % 5 == 0 or idx == len(coords):
                print(f"  tile {idx:2d}/{len(coords)} done ({time.perf_counter() - t0:.1f}s elapsed)")

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_vram_gb = 0.0
    dt_inf = time.perf_counter() - t0
    pred_map = pred_acc / np.clip(weight_acc, 1e-6, None)

    print(
        f"\nInference finished: {len(coords)} tiles in {dt_inf:.2f}s "
        f"({dt_inf / len(coords):.2f}s/tile, peak VRAM={peak_vram_gb:.2f} GiB)"
    )

    # Compare against corresponding crop of published ds8 ink preview
    ds = full_h / ink_ds8.shape[0]
    r0, c0 = int(round(y0 / ds)), int(round(x0 / ds))
    r1, c1 = int(round(y1 / ds)), int(round(x1 / ds))
    pub_crop_ds8 = ink_ds8[r0:r1, c0:c1]

    # Downsample our prediction to match pub_crop_ds8 for Pearson correlation
    our_ds8 = np.array(
        Image.fromarray((np.clip(pred_map, 0, 1) * 255).astype(np.uint8)).resize(
            (pub_crop_ds8.shape[1], pub_crop_ds8.shape[0]), Image.BILINEAR
        )
    )
    corr = float(np.corrcoef(our_ds8.ravel(), pub_crop_ds8.ravel())[0, 1])
    print(f"Pearson correlation vs. official published ds8 ink prediction: r = {corr:.4f}")

    # Plot 4-panel verification figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=140)

    axes[0].imshow(ink_ds8, cmap="gray")
    axes[0].add_patch(
        plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="#ff3d00", linewidth=2.5)
    )
    axes[0].set_title(f"Full Published Ink (ds8)\nROI = 768x768 native")
    axes[0].axis("off")

    mid_ct = vol_crop[vol_crop.shape[0] // 2]  # middle layer = surface center (z=54)
    axes[1].imshow(mid_ct, cmap="gray")
    axes[1].set_title(f"Streamed Surface Volume\nMiddle CT Slice (layer 54)")
    axes[1].axis("off")

    axes[2].imshow(pub_crop_ds8, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Official Published Ink\n(ds8 crop from S3)")
    axes[2].axis("off")

    axes[3].imshow(pred_map, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(
        f"Local GTX 1070 Ti Reproduction\n(FP32, r={corr:.3f}, {peak_vram_gb:.2f} GiB VRAM)"
    )
    axes[3].axis("off")

    fig.tight_layout()
    out_png = OUT_DIR / "stage2_ink_smoke.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
