"""Step 0B ($0 local): Blind recall verification on the 3 additional $50k First Letters winners!

Tests our exact fast reader-v3 + 100%-tile raw-float thin-stroke DoG scorer on:
  - PHerc0211 (9.362 um / 113 keV, Aranke winner)
  - PHerc0343 (8.640 um / 116 keV, Nieuwlaar winner)
  - PHerc0306B (8.640 um / 116 keV, SirProdigle winner)

Also measures CT-peak offset on each winner and compares raw vs --recenter-ct so we know
with 100% empirical certainty how reader-v3 + stroke_scorer performs on BOTH 113 keV and 116 keV!
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
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fl_radar_sweep as fl
import stroke_scorer as ss


def main():
    out_dir = ROOT / "outputs" / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = fl.FlatModelRunner("reader_v3", ROOT / "cache/models/reader-v3.pth", fp16=False, bf16=False)
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)

    winners = [
        ("WIN_PHerc0211_113keV", "PHerc0211", ROOT / "cache/winners/PHerc0211"),
        ("WIN_PHerc0343_116keV", "PHerc0343", ROOT / "cache/winners/PHerc0343"),
        ("WIN_PHerc0306B_116keV", "PHerc0306B", ROOT / "cache/winners/PHerc0306B"),
    ]

    summary = []
    vis_rows = []
    for jid, scroll, mdir in winners:
        vpath, vum = fl.SCROLL_VOLUMES[scroll]
        arr = zarr.open_group(FsspecStore(fs, path=vpath, read_only=True), mode="r")["0"]
        cvol = fl.ChunkCachedVolume(arr, max_chunks=768)
        surf = fl.UprightTifXYZSurface(mdir, auto_upright=True)
        H, W = surf.full_resolution_shape
        print(f"\nRunning {jid} ({scroll}, {vum} um): full={H}x{W} (rot={surf.rot_deg} deg)...", flush=True)
        # Render 29 layers so we can inspect BOTH raw dz=0 AND smooth CT-recentered!
        layers29, mask, dt_ren = fl.render_surface_cached(surf, cvol, bounds=None, layers=29, workers=16)
        layers_dz0 = layers29[6:23]  # exact [-8..+8]
        layers_rec, rec_info = fl.recenter_slab_ct_peak(layers29, mask, out_layers=17, max_shift=6, block=256)

        t_gpu = time.perf_counter()
        pf = runner.infer(layers_dz0, mask, overlap=0.5, reverse=False, batch_size=4)
        pr = runner.infer(layers_dz0, mask, overlap=0.5, reverse=True, batch_size=4)
        torch.cuda.synchronize()
        dt_gpu = time.perf_counter() - t_gpu

        scored = ss.score_full_mesh_tiles(pf, pr, mask, vum, win_px=1024, stride_px=512, top_k=5, nms_iou=0.35)
        ss.save_full_mesh_maps(out_dir / "full_maps", jid, layers_dz0[8], pf, pr, mask)

        # Also check recentered score if median |dz| >= 2
        pf_r = runner.infer(layers_rec, mask, overlap=0.5, reverse=False, batch_size=4)
        pr_r = runner.infer(layers_rec, mask, overlap=0.5, reverse=True, batch_size=4)
        scored_r = ss.score_full_mesh_tiles(pf_r, pr_r, mask, vum, win_px=1024, stride_px=512, top_k=5, nms_iou=0.35)

        bt = scored["best_tile"]
        btr = scored_r["best_tile"]
        print(f"  ren={dt_ren:.1f}s gpu={dt_gpu:.1f}s valid={scored['valid_area_cm2']:.2f}cm2 tiles={scored['n_tiles_evaluated']} ct_dz={rec_info}", flush=True)
        print(f"  dz=0     top1: {bt}", flush=True)
        print(f"  recenter top1: {btr}", flush=True)
        summary.append({
            "id": jid,
            "scroll": scroll,
            "vum": vum,
            "hw": [H, W],
            "valid_area_cm2": scored["valid_area_cm2"],
            "n_tiles": scored["n_tiles_evaluated"],
            "ct_recenter_info": rec_info,
            "dz0_best": bt,
            "dz0_top3": scored["top_tiles"][:3],
            "recenter_best": btr,
        })
        if bt is not None:
            y0, x0, h, w = bt["win_yx"]
            ps = pf if bt["side"] == "fwd" else pr
            po = pr if bt["side"] == "fwd" else pf
            vis_rows.append((
                f"{jid} (dz=0)",
                layers_dz0[8, y0:y0+h, x0:x0+w],
                ps[y0:y0+h, x0:x0+w],
                np.clip(ps[y0:y0+h, x0:x0+w] - 0.5 * po[y0:y0+h, x0:x0+w], 0.0, 1.0),
                bt,
            ))
        if btr is not None:
            y0, x0, h, w = btr["win_yx"]
            ps = pf_r if btr["side"] == "fwd" else pr_r
            po = pr_r if btr["side"] == "fwd" else pf_r
            vis_rows.append((
                f"{jid} (CT-recentered dz={rec_info['dz_med']:+.1f})",
                layers_rec[8, y0:y0+h, x0:x0+w],
                ps[y0:y0+h, x0:x0+w],
                np.clip(ps[y0:y0+h, x0:x0+w] - 0.5 * po[y0:y0+h, x0:x0+w], 0.0, 1.0),
                btr,
            ))

    (out_dir / "step0b_3winners.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if vis_rows:
        fig, axes = plt.subplots(len(vis_rows), 3, figsize=(13.5, 3.2 * len(vis_rows)))
        for r, (lbl, ct_c, p_c, d_c, bt) in enumerate(vis_rows):
            lo, hi = np.percentile(ct_c[ct_c > 0], [2, 98]) if (ct_c > 0).any() else (0, 255)
            axes[r, 0].imshow(ct_c, cmap="gray", vmin=lo, vmax=hi)
            axes[r, 0].set_title(f"{lbl}\n({bt['side']}) y={bt['win_yx'][0]} x={bt['win_yx'][1]}", fontsize=8.5)
            axes[r, 1].imshow(p_c, cmap="gray", vmin=0, vmax=max(0.45, float(np.percentile(p_c, 99.5))))
            axes[r, 1].set_title(f"reader-v3 raw={bt['raw_score']:.3f} p99={bt['p99']:.2f}", fontsize=8.5)
            axes[r, 2].imshow(d_c, cmap="gray", vmin=0, vmax=max(0.40, float(np.percentile(d_c, 99.5))))
            axes[r, 2].set_title(f"Stroke={bt['stroke_score']:.3f} n={bt['n_strokes']} c={bt['cells']} g={bt['giant']:.2f}", fontsize=8.5)
            for c in range(3):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        fig.suptitle("Step 0B ($0 Local): Blind Recall on 3 Additional $50k Winners (PHerc0211, PHerc0343 8.64um, PHerc0306B 8.64um)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "step0b_3winners.png", dpi=125)
        plt.close(fig)
        print(f"\nSaved {out_dir / 'step0b_3winners.png'}", flush=True)


if __name__ == "__main__":
    main()
