"""Step 0 ($0 local): Blind discovery benchmark on the FULL UNBOUNDED 6320x6320 PHerc0846B mesh.

Part A (fast, ~3s): Score the published 6320x6320 reader-v3_raw.tif across 100% of tiles,
visualize the full 35 cm2 sheet with top-10 blind tiles overlaid, and check stroke-width DoG tuning.

Part B (full live end-to-end): Run our upgraded reader-v3 pipeline (NO hardcoded bounds!) on
the FULL 6320x6320 PHerc0846B mesh (fwd + rev) PLUS 3 real distractor meshes from eligible_meshes
(PHerc0125, PHerc0800, PHerc0813 — including previously high-radar delamination junk!), and verify
that the real $50k PHerc0846B Greek letters surface cleanly in the top 1% / top 10 tiles blindly!
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
import matplotlib.patches as patches
import numpy as np
import s3fs
import tifffile
import torch
import zarr
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fl_radar_sweep as fl
import stroke_scorer as ss


def run_part_a_published():
    out_dir = ROOT / "outputs" / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = ROOT / "cache" / "killtest1_846b" / "PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750"
    surf = fl.UprightTifXYZSurface(seg, auto_upright=False)
    raw = tifffile.imread(
        ROOT / "cache" / "killtest1_846b" / "PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750_reader-v3_raw.tif"
    ).astype(np.float32) / 255.0
    _, _, valid = surf.tile_geometry((0, raw.shape[0], 0, raw.shape[1]))

    t0 = time.perf_counter()
    res = ss.score_full_mesh_tiles(
        raw, None, valid, 9.362, win_px=1024, stride_px=512, top_k=12, nms_iou=0.35, return_all_tiles=True
    )
    print(f"[Part A] Scored {res['n_tiles_evaluated']} tiles (1024px, stride 512) across 6320x6320 "
          f"({res['valid_area_cm2']} cm2 valid / {res['bbox_area_cm2']} cm2 bbox) in {time.perf_counter()-t0:.2f}s")
    for i, t in enumerate(res["top_tiles"], 1):
        y0, x0, h, w = t["win_yx"]
        in_box = (y0 + h > 1792 and y0 < 3328 and x0 + w > 1792 and x0 < 3328)
        print(f"  #{i:2d} y={y0:4d}..{y0+h:4d} x={x0:4d}..{x0+w:4d} | score={t['stroke_score']:.4f} "
              f"n={t['n_strokes']:2d} c={t['cells']:2d} dog99={t['dog99']:.3f} giant={t['giant']:.3f} "
              f"p99={t['p99']:.3f} | in_50k_box={in_box}")

    # Save full 6320x6320 overview (4x downsampled) with blind top-10 boxes + top-10 tile strip!
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.4, 1.0])
    ax_full = fig.add_subplot(gs[0, :3])
    ax_full.imshow(raw[::4, ::4], cmap="gray", vmin=0, vmax=0.75)
    ax_full.set_title("Full Unbounded PHerc0846B (6320x6320, 19.5 cm2 valid) — Blind Top-10 Tiles", fontsize=11)
    # Green dashed = known $50k letter box
    ax_full.add_patch(patches.Rectangle((1792/4, 1792/4), 1536/4, 1536/4, lw=2, ec="#22c55e", fc="none", ls="--"))
    for i, t in enumerate(res["top_tiles"][:10], 1):
        y0, x0, h, w = t["win_yx"]
        col = "#ef4444" if i <= 3 else "#f59e0b"
        ax_full.add_patch(patches.Rectangle((x0/4, y0/4), w/4, h/4, lw=1.8, ec=col, fc="none"))
        ax_full.text(x0/4 + 8, y0/4 + 35, f"#{i} ({t['stroke_score']:.2f})", color=col, fontsize=8, fontweight="bold",
                     bbox=dict(fc="black", alpha=0.55, pad=1, ec="none"))

    # Show top 7 tiles zoomed
    for idx, pos in enumerate([(0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]):
        if idx >= len(res["top_tiles"]):
            break
        t = res["top_tiles"][idx]
        y0, x0, h, w = t["win_yx"]
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.imshow(raw[y0:y0+h, x0:x0+w], cmap="gray", vmin=0, vmax=0.75)
        ax.set_title(f"#{idx+1} sc={t['stroke_score']:.3f} n={t['n_strokes']} c={t['cells']}\n({y0},{x0})", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    out_p = out_dir / "partA_846b_full_blind.png"
    fig.savefig(out_p, dpi=130)
    plt.close(fig)
    print(f"Saved {out_p}")


def run_part_b_live():
    """Live end-to-end GPU blind discovery: FULL 6320x6320 PHerc0846B + 3 real distractor meshes."""
    out_dir = ROOT / "outputs" / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick 3 tough real adversarial distractor meshes from Tranche 1 (old radar_score #1, #3, #6!)
    distractors = [
        ("PHerc0125_z6944_w060", "PHerc0125", ROOT / "cache/eligible_meshes/meshes/PHerc0125/z6944_w060", True),
        ("PHerc0813_z12496_w060", "PHerc0813", ROOT / "cache/eligible_meshes/meshes/PHerc0813/z12496_w060", True),
        ("PHerc0125_z14736_w010", "PHerc0125", ROOT / "cache/eligible_meshes/meshes/PHerc0125/z14736_w010", True),
    ]
    jobs = [
        (
            "PHerc0846B_FULL_UNBOUNDED",
            "PHerc0846B",
            ROOT / "cache/killtest1_846b/PHerc0846B-s20mm-b03-g08-t2-01-grow20261020153902-grow20261021205750",
            False,
        ),
    ] + [d for d in distractors if (d[2] / "meta.json").exists()]

    print(f"\n[Part B] Live reader-v3-only blind discovery across {len(jobs)} FULL meshes...", flush=True)
    runner = fl.FlatModelRunner("reader_v3", ROOT / "cache/models/reader-v3.pth", fp16=False, bf16=False)
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)

    global_tiles = []
    mesh_summaries = []
    saved_crops = {}  # key -> (ct, prob_side, prob_dir)

    for jid, scroll, mdir, auto_up in jobs:
        t0 = time.perf_counter()
        vpath, vum = fl.SCROLL_VOLUMES[scroll]
        arr = zarr.open_group(FsspecStore(fs, path=vpath, read_only=True), mode="r")["0"]
        cvol = fl.ChunkCachedVolume(arr, max_chunks=768)
        surf = fl.UprightTifXYZSurface(mdir, auto_upright=auto_up)
        H, W = surf.full_resolution_shape
        print(f"\n  Running {jid} ({H}x{W})...", flush=True)
        layers, mask, dt_ren = fl.render_surface_cached(surf, cvol, bounds=None, layers=17, workers=16)
        print(f"    rendered in {dt_ren:.1f}s (chunks fetched={cvol.misses}, hits={cvol.hits}, valid={mask.mean():.2f}), running reader-v3...", flush=True)
        t_gpu = time.perf_counter()
        pf = runner.infer(layers, mask, overlap=0.5, reverse=False, batch_size=4)
        pr = runner.infer(layers, mask, overlap=0.5, reverse=True, batch_size=4)
        torch.cuda.synchronize()
        dt_gpu = time.perf_counter() - t_gpu

        scored = ss.score_full_mesh_tiles(
            pf, pr, mask, vum, win_px=1024, stride_px=512, top_k=10, nms_iou=0.35, return_all_tiles=True
        )
        ss.save_full_mesh_maps(out_dir / "full_maps", jid, layers[8], pf, pr, mask)

        print(f"    ren={dt_ren:.1f}s gpu={dt_gpu:.1f}s | tiles={scored['n_tiles_evaluated']} "
              f"valid={scored['valid_area_cm2']:.2f} cm2 | top1={scored['best_tile']}", flush=True)
        mesh_summaries.append({
            "id": jid,
            "scroll": scroll,
            "hw": [H, W],
            "valid_area_cm2": scored["valid_area_cm2"],
            "bbox_area_cm2": scored["bbox_area_cm2"],
            "n_tiles_evaluated": scored["n_tiles_evaluated"],
            "render_s": round(dt_ren, 1),
            "gpu_s": round(dt_gpu, 1),
            "best_tile": scored["best_tile"],
        })
        for t in scored["all_tiles"]:
            rec = {"mesh_id": jid, "scroll": scroll, **t}
            global_tiles.append(rec)
        # Cache crops for top-10 NMS tiles of each mesh for global visual montage
        for t in scored["top_tiles"]:
            y0, x0, h, w = t["win_yx"]
            key = (jid, y0, x0, t["side"])
            p_s = pf if t["side"] == "fwd" else pr
            p_o = pr if t["side"] == "fwd" else pf
            saved_crops[key] = (
                layers[8, y0:y0+h, x0:x0+w].copy(),
                p_s[y0:y0+h, x0:x0+w].copy(),
                np.clip(p_s[y0:y0+h, x0:x0+w] - 0.5 * p_o[y0:y0+h, x0:x0+w], 0.0, 1.0),
            )

    # Global ranking across ALL tiles from ALL meshes (NMS per mesh already in top_tiles; also compute exact percentile over ALL tiles!)
    global_tiles.sort(key=lambda t: t["stroke_score"], reverse=True)
    # Build NMS-deduplicated global top 15
    global_nms = []
    for t in global_tiles:
        if any(t["mesh_id"] == g["mesh_id"] and ss._box_iou(t["win_yx"], g["win_yx"]) > 0.35 for g in global_nms):
            continue
        global_nms.append(t)
        if len(global_nms) >= 15:
            break

    n_all = len(global_tiles)
    box_ranks = [
        i + 1 for i, t in enumerate(global_tiles)
        if t["mesh_id"] == "PHerc0846B_FULL_UNBOUNDED"
        and t["win_yx"][0] + t["win_yx"][2] > 1792 and t["win_yx"][0] < 3328
        and t["win_yx"][1] + t["win_yx"][3] > 1792 and t["win_yx"][1] < 3328
    ]
    print(f"\n=== PART B GLOBAL BLIND DISCOVERY ({n_all} total tiles across {len(jobs)} full meshes) ===")
    print(f"PHerc0846B $50k-box tile ranks in ALL {n_all} tiles: {box_ranks[:10]} (top-1% cutoff = #{max(1, int(0.01*n_all))})")
    for rank, t in enumerate(global_nms[:12], 1):
        y0, x0, h, w = t["win_yx"]
        print(f"  #{rank:2d} {t['mesh_id']:26s} ({t['side']}) y={y0:4d} x={x0:4d} | "
              f"score={t['stroke_score']:.4f} (dir={t['dir_score']:.4f} raw={t['raw_score']:.4f}) "
              f"n={t['n_strokes']:2d} c={t['cells']:2d} giant={t['giant']:.3f}")

    (out_dir / "partB_blind_discovery.json").write_text(json.dumps({
        "total_tiles": n_all,
        "box_ranks_first10": box_ranks[:10],
        "meshes": mesh_summaries,
        "global_top15_nms": global_nms,
    }, indent=2), encoding="utf-8")

    # Save visual montage of Global Blind Top-10 NMS tiles!
    show = [t for t in global_nms[:10] if (t["mesh_id"], t["win_yx"][0], t["win_yx"][1], t["side"]) in saved_crops]
    if show:
        fig, axes = plt.subplots(len(show), 3, figsize=(13, 3.2 * len(show)))
        if len(show) == 1:
            axes = axes[None, :]
        for r, t in enumerate(show):
            y0, x0, h, w = t["win_yx"]
            ct_c, p_c, d_c = saved_crops[(t["mesh_id"], y0, x0, t["side"])]
            lo, hi = np.percentile(ct_c[ct_c > 0], [2, 98]) if (ct_c > 0).any() else (0, 255)
            axes[r, 0].imshow(ct_c, cmap="gray", vmin=lo, vmax=hi)
            axes[r, 0].set_title(f"#{r+1} {t['mesh_id']}\n({t['side']}) y={y0} x={x0}", fontsize=8.5)
            axes[r, 1].imshow(p_c, cmap="gray", vmin=0, vmax=max(0.45, float(np.percentile(p_c, 99.5))))
            axes[r, 1].set_title(f"reader-v3 raw={t['raw_score']:.3f} p99={t['p99']:.2f}", fontsize=8.5)
            axes[r, 2].imshow(d_c, cmap="gray", vmin=0, vmax=max(0.4, float(np.percentile(d_c, 99.5))))
            axes[r, 2].set_title(f"Dir score={t['stroke_score']:.3f} n={t['n_strokes']} c={t['cells']} g={t['giant']:.2f}", fontsize=8.5)
            for c in range(3):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        fig.suptitle(f"Step 0 Blind Discovery: Global Top {len(show)} Tiles Across {len(jobs)} Unbounded Meshes ({n_all} Tiles)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "partB_blind_top10.png", dpi=125)
        plt.close(fig)
        print(f"Saved {out_dir / 'partB_blind_top10.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("a", "b", "both"), default="a")
    args = ap.parse_args()
    if args.part in ("a", "both"):
        run_part_a_published()
    if args.part in ("b", "both"):
        run_part_b_live()
