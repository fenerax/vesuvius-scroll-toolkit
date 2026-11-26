"""Tranche 1B sweeper: downloads & 3-model-sweeps existing S3 tifxyz segments (e.g. PHerc1203/segments/raw).

Reuses the exact GPU-batched 3-model consensus engine from fl_radar_sweep.py.
"""
from __future__ import annotations

import argparse
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
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fl_radar_sweep as fl

S3_SEGMENT_SOURCES = {
    "PHerc1203": {
        "volume": ("vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr", 9.362),
        "segments_prefix": "vesuvius-challenge-open-data/PHerc1203/segments/raw",
    },
}


def download_s3_tifxyz(fs: s3fs.S3FileSystem, s3_dir: str, dst_dir: pathlib.Path) -> pathlib.Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("meta.json", "x.tif", "y.tif", "z.tif"):
        local = dst_dir / fname
        if not local.exists():
            local.write_bytes(fs.cat(f"{s3_dir}/{fname}"))
    return dst_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scroll", default="PHerc1203", choices=sorted(S3_SEGMENT_SOURCES.keys()))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-chunks", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "fl_sweep_1b"))
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    out_dir = pathlib.Path(args.out)
    panel_dir = out_dir / "panels"
    cache_dir = ROOT / "cache" / "s3_segments" / args.scroll
    panel_dir.mkdir(parents=True, exist_ok=True)

    src = S3_SEGMENT_SOURCES[args.scroll]
    vpath, vum = src["volume"]
    fs_sync = s3fs.S3FileSystem(anon=True)
    fs_async = s3fs.S3FileSystem(anon=True, asynchronous=True)

    s3_dirs = sorted(fs_sync.ls(src["segments_prefix"], detail=False))
    print(f"Found {len(s3_dirs)} S3 tifxyz segments for {args.scroll}", flush=True)
    print("Loading 3 independent 9 um models on", torch.cuda.get_device_name(0), "...", flush=True)
    runners = [fl.FlatModelRunner(name, path, fp16=args.fp16, bf16=args.bf16) for name, path in fl.MODEL_SPECS]
    arr = zarr.open_group(FsspecStore(fs_async, path=vpath, read_only=True), mode="r")["0"]
    cvol = fl.ChunkCachedVolume(arr, max_chunks=args.max_chunks)

    results = []
    top_vis = []
    t_all = time.perf_counter()

    for idx, s3_dir in enumerate(s3_dirs, 1):
        seg_name = s3_dir.split("/")[-1]
        jid = f"{args.scroll}_{seg_name}"
        print(f"\n[{idx}/{len(s3_dirs)}] {jid}", flush=True)
        try:
            local_seg = download_s3_tifxyz(fs_sync, s3_dir, cache_dir / seg_name)
            surf = fl.UprightTifXYZSurface(local_seg, auto_upright=True)
            H, W = surf.full_resolution_shape
            area_cm2 = (H * W) * ((vum / 1e4) ** 2)
            print(f"  full={H}x{W} ({area_cm2:.2f} cm2, T={surf.transposed}, FV={surf.flipped_v})", flush=True)

            hits_b, misses_b = cvol.hits, cvol.misses
            layers, mask, dt_ren = fl.render_surface_cached(surf, cvol, bounds=None, layers=17, workers=args.workers)
            new_m, new_h = cvol.misses - misses_b, cvol.hits - hits_b
            valid_frac = float(mask.mean())
            mid_ct = layers[8]
            ct_std = float(mid_ct[mask].std()) if mask.any() else 0.0
            print(f"  rendered in {dt_ren:.1f}s (S3: {new_m} fetched, {new_h} hits; valid={valid_frac:.2f}, std={ct_std:.1f})", flush=True)

            t_gpu = time.perf_counter()
            preds_fwd, preds_rev = {}, {}
            for r in runners:
                bs = args.batch_size if r.patch <= 256 else max(1, args.batch_size // 2)
                preds_fwd[r.name] = r.infer(layers, mask, overlap=0.5, reverse=False, batch_size=bs)
                preds_rev[r.name] = r.infer(layers, mask, overlap=0.5, reverse=True, batch_size=bs)
            torch.cuda.synchronize()
            dt_gpu = time.perf_counter() - t_gpu

            scored = fl.score_predictions(preds_fwd, preds_rev, mask, vum, win_px=min(1024, H, W))
            bw = scored["best_window"]
            cm2_hr = area_cm2 / max(1e-6, (dt_ren + dt_gpu) / 3600.0)
            results.append({
                "id": jid,
                "scroll": args.scroll,
                "s3_dir": s3_dir,
                "rendered_hw": [int(H), int(W)],
                "area_cm2": round(area_cm2, 3),
                "valid_frac": round(valid_frac, 3),
                "ct_std": round(ct_std, 1),
                "render_seconds": round(dt_ren, 1),
                "gpu_6pass_3model_seconds": round(dt_gpu, 1),
                "throughput_cm2_hr": round(cm2_hr, 1),
                "best_window": bw,
            })
            if bw is not None:
                print(f"  -> GPU={dt_gpu:.1f}s ({cm2_hr:.0f} cm2/hr) | side={bw['side']} radar={bw['radar_score']:.4f} "
                      f"dir_p99={bw['dir_p99']:.3f} agree_r={bw['agree_r_mean']:.3f} per={bw['line_periodicity']:.1f}", flush=True)
                side = bw["side"]
                pbest = preds_fwd if side == "fwd" else preds_rev
                cbest = scored["cons_fwd"] if side == "fwd" else scored["cons_rev"]
                dbest = scored["dir_fwd"] if side == "fwd" else scored["dir_rev"]
                fl.save_panel(panel_dir / f"{jid}.png", jid, mid_ct, pbest, cbest, dbest, bw["win_yx"], bw)
                y0, x0, wh, ww = bw["win_yx"]
                top_vis.append((bw["radar_score"], jid, mid_ct[y0:y0+wh, x0:x0+ww].copy(),
                                cbest[y0:y0+wh, x0:x0+ww].copy(), dbest[y0:y0+wh, x0:x0+ww].copy(), bw))
        except Exception as e:
            print(f"  !! ERROR on {jid}: {e}", flush=True)
            results.append({"id": jid, "scroll": args.scroll, "error": str(e), "best_window": None})

    results.sort(key=lambda r: (r["best_window"]["radar_score"] if r.get("best_window") else -1.0), reverse=True)
    lb_path = out_dir / f"tranche1b_{args.scroll}_leaderboard.json"
    lb_path.write_text(json.dumps({
        "scroll": args.scroll,
        "completed": len(results),
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "leaderboard": results,
    }, indent=2), encoding="utf-8")

    top_vis.sort(key=lambda x: x[0], reverse=True)
    show = top_vis[:12]
    if show:
        fig, axes = plt.subplots(len(show), 3, figsize=(13, 3.6 * len(show)))
        if len(show) == 1:
            axes = axes[None, :]
        for row, (sc_val, jid, ct_c, c_c, d_c, bw) in enumerate(show):
            lo, hi = np.percentile(ct_c[ct_c > 0], [2, 98]) if (ct_c > 0).any() else (0, 255)
            axes[row, 0].imshow(ct_c, cmap="gray", vmin=lo, vmax=hi)
            axes[row, 0].set_title(f"#{row+1} {jid}\nUpright CT (side={bw['side']})", fontsize=9)
            axes[row, 1].imshow(c_c, cmap="gray", vmin=0, vmax=max(0.4, float(np.percentile(c_c, 99.5))))
            axes[row, 1].set_title(f"3-Model Consensus p99={bw['cons_p99']:.3f} r={bw['agree_r_mean']:.3f}", fontsize=9)
            axes[row, 2].imshow(d_c, cmap="gray", vmin=0, vmax=max(0.35, float(np.percentile(d_c, 99.5))))
            axes[row, 2].set_title(f"Directional Score={sc_val:.4f} (per={bw['line_periodicity']:.1f})", fontsize=9)
            for col in range(3):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        fig.suptitle(f"Tranche 1B ({args.scroll}): Top {len(show)} S3 Segments", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / f"tranche1b_{args.scroll}_top_candidates.png", dpi=135)
        plt.close(fig)
    print("=== TRANCHE_1B_COMPLETE_OK ===", flush=True)


if __name__ == "__main__":
    main()
