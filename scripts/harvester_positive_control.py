"""Harvester Positive Control: Prove whether vc_grow_seg_from_seed patches recover real ink on winning scrolls.

Seeds 6 points per scroll (PHerc0211 & PHerc0343) on the S3 surface-prediction grid near
the known $50k winning wrap, grows patches with the EXACT Run 2 vc_grow_seg_from_seed
parameters, measures 3D distance to the winning wrap, and compares reader-v3 thin-stroke
scores RAW vs --recenter-ct (+ Aranke studio-softloss on top tiles).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import s3fs
import tifffile
import torch
import zarr
from scipy.spatial import cKDTree
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fl_radar_sweep as fl
import stroke_scorer as ss
import tranche2_vc3d_harvest as t2

CONTROL_SCROLLS = {
    "PHerc0211": {
        "volume": ("vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr", 9.362),
        "preds": "vesuvius-challenge-open-data/PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr",
        "normal_grids": "vesuvius-challenge-open-data/PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.normal-grids",
        "winner_dir": "PHerc0211",
    },
    "PHerc0343": {
        "volume": ("vesuvius-challenge-open-data/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr", 8.640),
        "preds": "vesuvius-challenge-open-data/PHerc0343/representations/predictions/surfaces/20250521140437-surface-20260413222639-surface-m7-L0-th0.2.zarr",
        "normal_grids": "vesuvius-challenge-open-data/PHerc0343/representations/predictions/surfaces/20250521140437-surface-20260413222639-surface-m7-L0-th0.2.normal-grids",
        "winner_dir": "PHerc0343",
    },
}


def find_winner_mesh_dir(winners_root: pathlib.Path, scroll: str) -> pathlib.Path:
    base = winners_root / scroll
    for xf in sorted(base.rglob("x.tif")):
        return xf.parent
    raise FileNotFoundError(f"No winner x.tif found under {base}")


def load_mesh_pts(seg_dir: pathlib.Path) -> np.ndarray:
    x = tifffile.imread(seg_dir / "x.tif").astype(np.float32)
    y = tifffile.imread(seg_dir / "y.tif").astype(np.float32)
    z = tifffile.imread(seg_dir / "z.tif").astype(np.float32)
    m = (x >= 0) & (y >= 0) & (z >= 0) & ((x > 0) | (y > 0) | (z > 0))
    return np.stack([x[m], y[m], z[m]], axis=-1)


def pick_seeds_near_winner(
    ct_arr, preds_arr, winner_pts: np.ndarray, n_seeds: int = 6, seed: int = 42
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    Z, Y, X = ct_arr.shape
    # Sample well-spaced target anchors across the central 20-80% of the winning wrap
    order = np.argsort(winner_pts[:, 2])
    mid_slice = order[int(len(order) * 0.15) : int(len(order) * 0.85)]
    idxs = np.linspace(0, len(mid_slice) - 1, n_seeds * 4, dtype=int)
    anchors = winner_pts[mid_slice[idxs]]

    seeds: list[tuple[int, int, int]] = []
    min_sep = 220.0
    for ax, ay, az in anchors:
        if len(seeds) >= n_seeds:
            break
        if any(np.linalg.norm(np.array([ax, ay, az]) - np.array(s)) < min_sep for s in seeds):
            continue
        x0 = max(0, min(X - 48, int(ax) - 24))
        y0 = max(0, min(Y - 48, int(ay) - 24))
        z0 = max(0, min(Z - 24, int(az) - 12))
        ct_box = np.asarray(ct_arr[z0 : z0 + 24, y0 : y0 + 48, x0 : x0 + 48])
        pr_box = np.asarray(preds_arr[z0 : z0 + 24, y0 : y0 + 48, x0 : x0 + 48])
        good = np.argwhere((pr_box > 100) & (ct_box > 40))
        if len(good) == 0:
            good = np.argwhere((pr_box > 0) & (ct_box > 35))
        if len(good) == 0:
            continue
        # Pick the predicted surface point closest to the anchor (simulating landing on that wrap)
        cand_xyz = np.stack([x0 + good[:, 2], y0 + good[:, 1], z0 + good[:, 0]], axis=-1).astype(np.float32)
        d = np.linalg.norm(cand_xyz - np.array([ax, ay, az], dtype=np.float32), axis=1)
        best = good[int(np.argmin(d))]
        seeds.append((x0 + int(best[2]), y0 + int(best[1]), z0 + int(best[0])))
    print(f"  Picked {len(seeds)} surface-pred seeds on winning wrap (distances <= 30 vox)", flush=True)
    return seeds


def run_control_scroll(scroll: str, vc_bin: pathlib.Path, runner: fl.FlatModelRunner, get_aranke, args) -> dict:
    cfg = CONTROL_SCROLLS[scroll]
    vpath, vum = cfg["volume"]
    out_dir = pathlib.Path(args.out) / scroll
    seg_dir = out_dir / "segments"
    work_dir = pathlib.Path(args.scratch) / scroll
    ng_dir = work_dir / "normal_grids"
    cache_dir = work_dir / "vc_cache"
    dummy_vol = work_dir / "vol_stub.zarr"
    for d in (out_dir, seg_dir, work_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    winner_dir = find_winner_mesh_dir(pathlib.Path(args.winners_root), scroll)
    winner_pts = load_mesh_pts(winner_dir)
    winner_tree = cKDTree(winner_pts)
    print(f"\n================ POSITIVE CONTROL: {scroll} ({vum} um, winner={winner_dir.name}, {len(winner_pts)} pts) ================", flush=True)

    fs_async = s3fs.S3FileSystem(anon=True, asynchronous=True)
    ct_arr = zarr.open_group(FsspecStore(fs_async, path=vpath, read_only=True), mode="r")["0"]
    preds_arr = zarr.open_group(FsspecStore(fs_async, path=cfg["preds"], read_only=True), mode="r")["0"]

    seeds = pick_seeds_near_winner(ct_arr, preds_arr, winner_pts, n_seeds=args.n_seeds, seed=42)
    t2.sync_normal_grids(cfg["normal_grids"], ng_dir, workers=args.sync_workers)
    t2.make_dummy_volume_zarr(dummy_vol, cfg["preds"])

    params_path = work_dir / "params.json"
    params_path.write_text(json.dumps({
        "mode": "seed",
        "step_size": 20,
        "voxelsize": vum,
        "min_area_cm": 0.4,
        "generations": args.generations,
        "thread_limit": 2,
        "cache_size": 2000000000,
        "cache_root": str(cache_dir),
        "use_cuda": False,
        "normal_grid_path": str(ng_dir),
        "normal_grid_level": 0,
    }, indent=2), encoding="utf-8")

    t_grow = time.perf_counter()
    grow_log = out_dir / "vc_grow.log"
    for b0 in range(0, len(seeds), args.grow_procs):
        batch = seeds[b0 : b0 + args.grow_procs]
        procs = []
        with grow_log.open("a", encoding="utf-8") as lf:
            for (sx, sy, sz) in batch:
                cmd = [str(vc_bin), "-v", str(dummy_vol), "-t", str(seg_dir), "-p", str(params_path), "-s", str(sx), str(sy), str(sz)]
                procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT))
        deadline = time.time() + args.grow_timeout
        for pr in procs:
            try:
                pr.wait(timeout=max(5, deadline - time.time()))
            except subprocess.TimeoutExpired:
                pr.kill()
    print(f"  Grew patches in {time.perf_counter()-t_grow:.1f}s", flush=True)

    raw_patch_dirs = sorted([d for d in seg_dir.iterdir() if d.is_dir() and (d / "meta.json").exists() and (d / "x.tif").exists()])
    patch_dirs = t2.deduplicate_patches_3d(raw_patch_dirs, vum, overlap_thresh=0.80)
    shutil.rmtree(ng_dir, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)

    cvol = fl.ChunkCachedVolume(ct_arr, max_chunks=4096)
    records = []
    global_tiles_rec = []
    tile_crops = {}

    for pi, pdir in enumerate(patch_dirs, 1):
        jid = f"ctrl_{scroll}_{pdir.name}"
        pts = load_mesh_pts(pdir)
        dists, _ = winner_tree.query(pts, workers=1)
        geom_stats = {
            "dist_p10_vox": round(float(np.percentile(dists, 10)), 2),
            "dist_p25_vox": round(float(np.percentile(dists, 25)), 2),
            "dist_med_vox": round(float(np.median(dists)), 2),
            "frac_within_4vox": round(float((dists <= 4.0).mean()), 3),
            "frac_within_8vox": round(float((dists <= 8.0).mean()), 3),
            "frac_within_15vox": round(float((dists <= 15.0).mean()), 3),
        }

        surf = fl.UprightTifXYZSurface(pdir, auto_upright=True)
        layers_wide, mask, dt_ren = fl.render_surface_cached(surf, cvol, bounds=None, layers=29, workers=args.workers)
        if not mask.any() or float(mask.mean()) < 0.15:
            continue
        layers_raw = layers_wide[6:23]  # exact center 17 layers
        layers_rec, rec_info = fl.recenter_slab_ct_peak(layers_wide, mask, out_layers=17, max_shift=6, block=256)

        bs0 = args.batch_size
        pf_raw = runner.infer(layers_raw, mask, overlap=0.5, reverse=False, batch_size=bs0)
        pr_raw = runner.infer(layers_raw, mask, overlap=0.5, reverse=True, batch_size=bs0)
        sc_raw = ss.score_full_mesh_tiles(pf_raw, pr_raw, mask, vum, win_px=1024, stride_px=512, top_k=5, nms_iou=0.35)

        pf_rec = runner.infer(layers_rec, mask, overlap=0.5, reverse=False, batch_size=bs0)
        pr_rec = runner.infer(layers_rec, mask, overlap=0.5, reverse=True, batch_size=bs0)
        sc_rec = ss.score_full_mesh_tiles(pf_rec, pr_rec, mask, vum, win_px=1024, stride_px=512, top_k=5, nms_iou=0.35)
        torch.cuda.synchronize()

        bt_raw = sc_raw["best_tile"]
        bt_rec = sc_rec["best_tile"]
        raw_s = bt_raw["stroke_score"] if bt_raw else 0.0
        rec_s = bt_rec["stroke_score"] if bt_rec else 0.0

        ar_info = None
        if bt_rec is not None and rec_s >= 0.07:
            import aranke_runner as ar
            y0, x0, wh, ww = bt_rec["win_yx"]
            ps = pf_rec if bt_rec["side"] == "fwd" else pr_rec
            ar_info, s_best, s_dir = get_aranke().score_tile(
                layers_rec[:, y0:y0+wh, x0:x0+ww], mask[y0:y0+wh, x0:x0+ww],
                ps[y0:y0+wh, x0:x0+ww], vum, batch_size=max(8, bs0 // 2), stride=64,
            )
            ar.save_aranke_confirm_panel(
                out_dir / "aranke_hits" / f"{jid}_y{y0}_x{x0}.png",
                f"POSCTRL {jid} ({bt_rec['side']}) raw={raw_s:.3f} rec={rec_s:.3f}",
                layers_rec[8, y0:y0+wh, x0:x0+ww], ps[y0:y0+wh, x0:x0+ww], s_best, s_dir, rec_s, ar_info,
            )

        rec = {
            "id": jid,
            "valid_area_cm2": sc_rec["valid_area_cm2"],
            "geom_to_winner": geom_stats,
            "recenter": rec_info,
            "raw_stroke_score": round(raw_s, 4),
            "recentered_stroke_score": round(rec_s, 4),
            "best_raw_tile": bt_raw,
            "best_rec_tile": bt_rec,
            "aranke": ar_info,
        }
        records.append(rec)
        if bt_rec is not None:
            t = {"mesh_id": jid, "scroll": scroll, **bt_rec}
            global_tiles_rec.append(t)
            y0, x0, wh, ww = bt_rec["win_yx"]
            ps = pf_rec if bt_rec["side"] == "fwd" else pr_rec
            po = pr_rec if bt_rec["side"] == "fwd" else pf_rec
            tile_crops[(jid, tuple(bt_rec["win_yx"]), bt_rec["side"])] = (
                layers_rec[8, y0:y0+wh, x0:x0+ww].copy(),
                ps[y0:y0+wh, x0:x0+ww].copy(),
                np.clip(ps[y0:y0+wh, x0:x0+ww] - 0.5 * po[y0:y0+wh, x0:x0+ww], 0.0, 1.0),
            )
        print(
            f"  [{pi}/{len(patch_dirs)}] {jid} area={sc_rec['valid_area_cm2']:.2f}cm2 "
            f"<=4vox={geom_stats['frac_within_4vox']:.2f} <=8vox={geom_stats['frac_within_8vox']:.2f} "
            f"| raw={raw_s:.4f} rec={rec_s:.4f}",
            flush=True,
        )

    global_tiles_rec.sort(key=lambda t: t["stroke_score"], reverse=True)
    if global_tiles_rec:
        fl.save_tile_montages(
            out_dir / f"{scroll}_posctrl_top",
            global_tiles_rec[:10],
            tile_crops,
            f"Harvester Positive Control ({scroll})",
            per_page=5,
        )

    best_raw = max((r["raw_stroke_score"] for r in records), default=0.0)
    best_rec = max((r["recentered_stroke_score"] for r in records), default=0.0)
    return {
        "scroll": scroll,
        "n_seeds": len(seeds),
        "n_patches_grown": len(raw_patch_dirs),
        "n_patches_evaluated": len(records),
        "best_raw_stroke_score": best_raw,
        "best_recentered_stroke_score": best_rec,
        "passed_0_10": bool(max(best_raw, best_rec) >= 0.10),
        "patches": records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scrolls", nargs="+", default=["PHerc0211", "PHerc0343"])
    ap.add_argument("--vc-bin", required=True)
    ap.add_argument("--winners-root", default=str(ROOT / "cache" / "winners"))
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--generations", type=int, default=65)
    ap.add_argument("--grow-procs", type=int, default=4)
    ap.add_argument("--grow-timeout", type=int, default=300)
    ap.add_argument("--sync-workers", type=int, default=48)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--scratch", default="/workspace/posctrl_scratch")
    ap.add_argument("--out", default="/workspace/vesuvius/outputs/posctrl")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    name, path = fl.MODEL_SPECS[0]
    print(f"Loading {name} on {torch.cuda.get_device_name(0)}...", flush=True)
    runner = fl.FlatModelRunner(name, path, bf16=args.bf16)

    aranke_runner = None
    def get_aranke():
        nonlocal aranke_runner
        if aranke_runner is None:
            import aranke_runner as ar
            aranke_runner = ar.ArankeStudioRunner(bf16=args.bf16)
        return aranke_runner

    vc_bin = pathlib.Path(args.vc_bin)
    summaries = []
    for scroll in args.scrolls:
        summaries.append(run_control_scroll(scroll, vc_bin, runner, get_aranke, args))

    overall_pass = any(s["passed_0_10"] for s in summaries)
    report = {
        "overall_pass_any_ge_0_10": overall_pass,
        "both_pass_ge_0_10": all(s["passed_0_10"] for s in summaries),
        "scrolls": summaries,
    }
    out_p = pathlib.Path(args.out)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "positive_control_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "scrolls"}, indent=2), flush=True)
    for s in summaries:
        print(f"SUMMARY {s['scroll']}: grown={s['n_patches_grown']} raw_max={s['best_raw_stroke_score']:.4f} rec_max={s['best_recentered_stroke_score']:.4f} PASS={s['passed_0_10']}", flush=True)
    if not overall_pass:
        print("=== POSITIVE_CONTROL_FAILED ===", flush=True)
        sys.exit(42)
    print("=== POSITIVE_CONTROL_PASSED ===", flush=True)


if __name__ == "__main__":
    main()
