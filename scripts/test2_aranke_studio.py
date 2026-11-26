"""Test 2 ($0 local): Priyank Aranke's $50k-winning studio-softloss-20261023 on GTX 1070 Ti.

Checks:
  1. Does the 223M-param DeepResidualUNet run in local 8 GB VRAM (FP32)?
  2. How does it score on all 4 $50k winners (113 keV + 116 keV) vs Tranche 1's 3 worst
     false positives AND competitor held-back PHerc0358_z11280_w020?
  3. How much does it agree vs disagree with reader-v3?
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import torch
import zarr
from safetensors.torch import load_file
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "cache" / "models" / "aranke_studio"))
import fl_radar_sweep as fl
import stroke_scorer as ss
from model import DeepResidualUNet


def hann96() -> np.ndarray:
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96)
    return (w[:, None] * w[None, :]).astype(np.float32)


def crop128(arr: np.ndarray, y: int, x: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    tile = np.asarray(arr[:, max(0, y) : min(h, y + 128), max(0, x) : min(w, x + 128)])
    pad = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
    return np.pad(tile, pad, mode="reflect") if any(a or b for a, b in pad) else tile


@torch.inference_mode()
def infer_aranke(model, vol16: np.ndarray, mask: np.ndarray, batch_size: int = 4, stride: int = 64) -> np.ndarray:
    h, w = vol16.shape[-2:]
    res = np.zeros((h, w), dtype=np.float32)
    wt = np.zeros((h, w), dtype=np.float32)
    kern = hann96()
    ys = sorted(set(list(range(0, h, stride)) + [max(0, h - 96)]))
    xs = sorted(set(list(range(0, w, stride)) + [max(0, w - 96)]))
    coords = [(y, x) for y in ys for x in xs if mask[y : y + 96, x : x + 96].any()]
    for i in range(0, len(coords), batch_size):
        batch = coords[i : i + batch_size]
        raw = np.stack([crop128(vol16, y - 16, x - 16) for y, x in batch], axis=0)
        x = torch.from_numpy(raw[:, None]).to("cuda", dtype=torch.float32)
        x.clamp_(0, 200).div_(255.0)
        probs = model(x)[:, 0, 16:112, 16:112].sigmoid().cpu().numpy()
        for (y, x0), p in zip(batch, probs):
            hh, ww = min(96, h - y), min(96, w - x0)
            res[y : y + hh, x0 : x0 + ww] += p[:hh, :ww] * kern[:hh, :ww]
            wt[y : y + hh, x0 : x0 + ww] += kern[:hh, :ww]
    out = np.divide(res, wt, out=np.zeros_like(res), where=wt > 1e-12)
    out[~mask] = 0.0
    return out


def main():
    out_dir = ROOT / "outputs" / "council_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ROOT / "cache/models/aranke_studio/studio-softloss-20261023-ema.safetensors"
    assert ckpt_path.exists(), f"Missing {ckpt_path}"

    torch.cuda.reset_peak_memory_stats()
    print("Loading Aranke studio-softloss-20261023 (223M 3D ResUNet, FP32) on", torch.cuda.get_device_name(0), "...", flush=True)
    studio = DeepResidualUNet(deeper=False)
    studio.load_state_dict(load_file(str(ckpt_path), device="cpu"), strict=True)
    studio = studio.eval().to("cuda")
    print(f"  Studio VRAM loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    reader = fl.FlatModelRunner("reader_v3", ROOT / "cache/models/reader-v3.pth", fp16=False, bf16=False)
    print(f"  Both models VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    cases = [
        ("WIN_PHerc0846B_113", "PHerc0846B", ROOT / "cache/killtest1_846b/PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750", False, (3584, 4608, 3072, 4096)),
        ("WIN_PHerc0211_113",  "PHerc0211",  ROOT / "cache/winners/PHerc0211", True, (696, 1720, 3584, 4608)),
        ("WIN_PHerc0343_116",  "PHerc0343",  ROOT / "cache/winners/PHerc0343", True, (1024, 2048, 1536, 2560)),
        ("WIN_PHerc0306B_116", "PHerc0306B", ROOT / "cache/winners/PHerc0306B", True, (1536, 2560, 2048, 3072)),
        ("FP_0125_z6944_w060", "PHerc0125",  ROOT / "cache/eligible_meshes/meshes/PHerc0125/z6944_w060", True, (316, 1340, 7168, 8192)),
        ("FP_0813_z12496_w060","PHerc0813",  ROOT / "cache/eligible_meshes/meshes/PHerc0813/z12496_w060", True, (0, 1024, 6656, 7680)),
        ("ATLAS_0813_z13088",  "PHerc0813",  ROOT / "cache/eligible_meshes/meshes/PHerc0813/z13088_w040", True, None),
    ]

    results = []
    fig, axes = plt.subplots(len(cases), 4, figsize=(15, 3.3 * len(cases)))

    for row_i, (name, scroll, seg, auto_up, bounds) in enumerate(cases):
        vpath, vum = fl.SCROLL_VOLUMES[scroll]
        arr = zarr.open_group(FsspecStore(fs, path=vpath, read_only=True), mode="r")["0"]
        cvol = fl.ChunkCachedVolume(arr, max_chunks=256)
        surf = fl.UprightTifXYZSurface(seg, auto_upright=auto_up)
        H_f, W_f = surf.full_resolution_shape
        if bounds is None:
            # Render 29 layers + CT-recenter full mesh to pick the exact best tile first
            layers29, mask_full, dt_ren = fl.render_surface_cached(surf, cvol, bounds=None, layers=29, workers=16)
            slab_full, _ = fl.recenter_slab_ct_peak(layers29, mask_full, out_layers=17, max_shift=6, block=256)
            rf_tmp = reader.infer(slab_full, mask_full, overlap=0.5, reverse=False, batch_size=4)
            rr_tmp = reader.infer(slab_full, mask_full, overlap=0.5, reverse=True, batch_size=4)
            bt_tmp = ss.score_full_mesh_tiles(rf_tmp, rr_tmp, mask_full, vum, win_px=1024, stride_px=512, top_k=1)["best_tile"]
            y0, x0, wh, ww = bt_tmp["win_yx"]
            slab = slab_full[:, y0:y0+wh, x0:x0+ww]
            mask = mask_full[y0:y0+wh, x0:x0+ww]
        else:
            r0, r1, c0, c1 = bounds[0], min(bounds[1], H_f), bounds[2], min(bounds[3], W_f)
            slab, mask, dt_ren = fl.render_surface_cached(surf, cvol, bounds=(r0, r1, c0, c1), layers=17, workers=16)

        t0 = time.perf_counter()
        rf = reader.infer(slab, mask, overlap=0.5, reverse=False, batch_size=4)
        rr = reader.infer(slab, mask, overlap=0.5, reverse=True, batch_size=4)
        torch.cuda.synchronize()
        dt_r = time.perf_counter() - t0

        t1 = time.perf_counter()
        sf = infer_aranke(studio, slab[:16], mask, batch_size=2, stride=64)
        sr = infer_aranke(studio, np.ascontiguousarray(slab[1:][::-1]), mask, batch_size=2, stride=64)
        torch.cuda.synchronize()
        dt_s = time.perf_counter() - t1

        sc_r = ss.score_full_mesh_tiles(rf, rr, mask, vum, win_px=1024, stride_px=512, top_k=1)["best_tile"]
        sc_s = ss.score_full_mesh_tiles(sf, sr, mask, vum, win_px=1024, stride_px=512, top_k=1)["best_tile"]

        r_best_map = rf if (sc_r and sc_r["side"] == "fwd") else rr
        s_best_map = sf if (sc_s and sc_s["side"] == "fwd") else sr
        # Check max correlation across both pairings (studio depth order may be opposite reader-v3)
        r_ff = float(np.corrcoef(r_best_map[mask], sf[mask])[0, 1]) if mask.any() else 0.0
        r_fr = float(np.corrcoef(r_best_map[mask], sr[mask])[0, 1]) if mask.any() else 0.0
        corr_best = max(r_ff, r_fr)
        s_aligned = sf if r_ff >= r_fr else sr

        rec = {
            "name": name,
            "scroll": scroll,
            "reader_s": round(dt_r, 1),
            "studio_s": round(dt_s, 1),
            "reader_score": sc_r["stroke_score"] if sc_r else 0.0,
            "studio_score": sc_s["stroke_score"] if sc_s else 0.0,
            "reader_side": sc_r["side"] if sc_r else "?",
            "studio_side": sc_s["side"] if sc_s else "?",
            "corr_reader_studio": round(corr_best, 4),
            "studio_p99": sc_s["p99"] if sc_s else 0.0,
        }
        results.append(rec)
        print(
            f"  {name:20s} | reader={rec['reader_score']:.4f} ({dt_r:.1f}s) | "
            f"studio={rec['studio_score']:.4f} ({dt_s:.1f}s, p99={rec['studio_p99']:.2f}) | r_agree={corr_best:.3f}",
            flush=True,
        )

        ct = slab[8]
        lo, hi = np.percentile(ct[mask & (ct > 0)], [2, 98]) if (mask & (ct > 0)).any() else (0, 255)
        axes[row_i, 0].imshow(ct, cmap="gray", vmin=lo, vmax=hi)
        axes[row_i, 0].set_title(f"{name}\nCT", fontsize=8.5)
        axes[row_i, 1].imshow(r_best_map, cmap="gray", vmin=0, vmax=max(0.45, float(np.percentile(r_best_map, 99.5))))
        axes[row_i, 1].set_title(f"reader-v3\nsc={rec['reader_score']:.3f}", fontsize=8.5)
        axes[row_i, 2].imshow(s_best_map, cmap="gray", vmin=0, vmax=max(0.45, float(np.percentile(s_best_map, 99.5))))
        axes[row_i, 2].set_title(f"Aranke studio-softloss\nsc={rec['studio_score']:.3f} r={corr_best:.2f}", fontsize=8.5)
        diff = np.clip(s_aligned - r_best_map, 0.0, 1.0)
        axes[row_i, 3].imshow(diff, cmap="magma", vmin=0, vmax=0.6)
        axes[row_i, 3].set_title("Disagreement\n(studio - reader3)+", fontsize=8.5)
        for c in range(4):
            axes[row_i, c].set_xticks([]); axes[row_i, c].set_yticks([])

    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"\nPeak VRAM (both models on GTX 1070 Ti FP32): {peak_gb} GB", flush=True)
    fig.suptitle(f"Test 2 ($0 Local): Aranke studio-softloss-20261023 vs reader-v3 (Peak VRAM {peak_gb} GB)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "test2_aranke_vs_reader3.png", dpi=125)
    plt.close(fig)
    (out_dir / "test2_aranke_vs_reader3.json").write_text(json.dumps({"peak_vram_gb": peak_gb, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'test2_aranke_vs_reader3.png'}", flush=True)


if __name__ == "__main__":
    main()
