"""Tranche 2 (Upgraded): Automated VC3D L0 patch harvesting + full-mesh raw-float thin-stroke sweep.

Upgrades over v1 (fixing all 6-seat Fancy Council blind spots):
  1. Covers all untouched 116 keV scrolls (PHerc0483A/B, 0490A/B, 1218) AND lightly-covered
     113 keV scrolls (PHerc0257, 0358, 0826) plus original 4 virgin scrolls.
  2. Real 3D cKDTree spatial deduplication of grown patches (largest-first; skips >65% overlap).
  3. Archives all kept grown tifxyz meshes to <scroll>_tifxyz.tar.gz BEFORE Pod deletion!
  4. Fast reader-v3-only primary screening (3x faster) with 100%-tile raw-float thin-stroke
     DoG scoring (trailing edges included, valid_frac >= 0.30), lazy argus-v3 + ink_9um
     confirmation on hits, calibrated 2x full-mesh PNG maps, and honest valid_area_cm2 reporting.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

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


def _scroll_entry(scroll: str, prefix: str, vum: float, kev: int) -> dict:
    v_str = f"{vum:.3f}um"
    return {
        "volume": (f"vesuvius-challenge-open-data/{scroll}/volumes/{prefix}-{v_str}-1.2m-{kev}keV-masked.zarr", vum),
        "preds": f"vesuvius-challenge-open-data/{scroll}/representations/predictions/surfaces/{prefix}-surface-20260413222639-surface-m7-L0-th0.2.zarr",
        "normal_grids": f"vesuvius-challenge-open-data/{scroll}/representations/predictions/surfaces/{prefix}-surface-20260413222639-surface-m7-L0-th0.2.normal-grids",
    }


TRANCHE2_SCROLLS = {
    # Lightly-covered 9.362 um / 113 keV scrolls (exact winner protocol!)
    "PHerc0257": _scroll_entry("PHerc0257", "20250821151750", 9.362, 113),
    "PHerc0358": _scroll_entry("PHerc0358", "20250821151737", 9.362, 113),
    "PHerc0826": _scroll_entry("PHerc0826", "20250821151701", 9.362, 113),
    "PHerc0191": _scroll_entry("PHerc0191", "20250821151635", 9.362, 113),
    "PHerc1545": _scroll_entry("PHerc1545", "20250821151648", 9.362, 113),
    # Completely untouched 8.640 um / 116 keV scrolls (proven readable on PHerc0343 & PHerc0306B!)
    "PHerc0483A": _scroll_entry("PHerc0483A", "20250521140913", 8.640, 116),
    "PHerc0483B": _scroll_entry("PHerc0483B", "20251124083638", 8.640, 116),
    "PHerc0490A": _scroll_entry("PHerc0490A", "20250521151210", 8.640, 116),
    "PHerc0490B": _scroll_entry("PHerc0490B", "20250521151215", 8.640, 116),
    "PHerc1218": _scroll_entry("PHerc1218", "20250521120456", 8.640, 116),
    "PHerc0175A": _scroll_entry("PHerc0175A", "20250521115057", 8.640, 116),
    "PHerc0175B": _scroll_entry("PHerc0175B", "20250521125822", 8.640, 116),
}


def sync_normal_grids(s3_ng: str, local_ng: pathlib.Path, workers: int = 48):
    fs = s3fs.S3FileSystem(anon=True)
    local_ng.mkdir(parents=True, exist_ok=True)
    meta_dst = local_ng / "metadata.json"
    if not meta_dst.exists():
        meta_dst.write_bytes(fs.cat(f"{s3_ng}/metadata.json"))

    to_fetch: list[tuple[str, pathlib.Path]] = []
    for sub in ("xy", "xz", "yz"):
        sub_dir = local_ng / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
        for item in fs.ls(f"{s3_ng}/{sub}", detail=False):
            dst = sub_dir / item.split("/")[-1]
            if not dst.exists():
                to_fetch.append((item, dst))

    print(f"  Syncing {len(to_fetch)} normal-grid files to {local_ng} ({workers} threads)...", flush=True)
    t0 = time.perf_counter()
    done = [0]

    def fetch_one(pair):
        src, dst = pair
        for attempt in range(4):
            try:
                dst.write_bytes(fs.cat(src))
                done[0] += 1
                if done[0] % 4000 == 0:
                    print(f"    synced {done[0]}/{len(to_fetch)} ({time.perf_counter()-t0:.1f}s)", flush=True)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(0.5 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch_one, to_fetch))
    print(f"  Synced {len(to_fetch)} files in {time.perf_counter()-t0:.1f}s", flush=True)


def make_dummy_volume_zarr(local_vol: pathlib.Path, s3_preds_path: str):
    if local_vol.exists():
        shutil.rmtree(local_vol)
    fs = s3fs.S3FileSystem(anon=True)
    local_vol.mkdir(parents=True)
    for fn in (".zattrs", ".zgroup"):
        if fs.exists(f"{s3_preds_path}/{fn}"):
            (local_vol / fn).write_bytes(fs.cat(f"{s3_preds_path}/{fn}"))
    for lvl in ("0", "1", "2", "3", "4", "5"):
        if fs.exists(f"{s3_preds_path}/{lvl}/.zarray"):
            (local_vol / lvl).mkdir()
            (local_vol / lvl / ".zarray").write_bytes(fs.cat(f"{s3_preds_path}/{lvl}/.zarray"))


def sample_interior_seeds(ct_arr, preds_arr, n_seeds: int = 36, seed: int = 42) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    Z, Y, X = ct_arr.shape
    z_levels = np.linspace(int(Z * 0.18), int(Z * 0.82), 9, dtype=int)
    seeds: list[tuple[int, int, int]] = []
    min_sep = 450.0

    mid_ct = np.asarray(ct_arr[int(Z * 0.5), ::8, ::8])
    ys, xs = np.where(mid_ct > 35)
    if len(ys) == 0:
        cy, cx, r_max = Y / 2.0, X / 2.0, min(Y, X) * 0.35
    else:
        cy, cx = float(ys.mean() * 8), float(xs.mean() * 8)
        r_max = float(np.percentile(np.sqrt((ys * 8 - cy) ** 2 + (xs * 8 - cx) ** 2), 95))
    print(f"  Scroll geometry: center=({cx:.0f},{cy:.0f}), r95={r_max:.0f} vox, shape={ct_arr.shape}", flush=True)

    attempts = 0
    while len(seeds) < n_seeds and attempts < n_seeds * 14:
        attempts += 1
        z0 = max(256, min(Z - 256, int(rng.choice(z_levels)) + rng.randint(-128, 128)))
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.20 * r_max, 0.84 * r_max)
        x0 = int(np.clip(cx + rad * np.cos(ang), 128, X - 129))
        y0 = int(np.clip(cy + rad * np.sin(ang), 128, Y - 129))
        if any(np.linalg.norm(np.array([x0, y0, z0]) - np.array(s)) < min_sep for s in seeds):
            continue
        ct_box = np.asarray(ct_arr[z0 : z0 + 16, y0 : y0 + 64, x0 : x0 + 64])
        if float((ct_box > 40).mean()) < 0.15:
            continue
        pr_box = np.asarray(preds_arr[z0 : z0 + 16, y0 : y0 + 64, x0 : x0 + 64])
        good = np.argwhere((pr_box > 100) & (ct_box > 40))
        if len(good) == 0:
            good = np.argwhere((pr_box > 0) & (ct_box > 35))
        if len(good) == 0:
            continue
        dz, dy, dx = good[rng.randrange(len(good))]
        seeds.append((x0 + int(dx), y0 + int(dy), z0 + int(dz)))

    print(f"  Sampled {len(seeds)} CT-supported interior seeds in {attempts} probes", flush=True)
    return seeds


def deduplicate_patches_3d(patch_dirs: list[pathlib.Path], vum: float, overlap_thresh: float = 0.65, dist_vox: float = 25.0) -> list[pathlib.Path]:
    """Keep largest-first non-duplicate grown tifxyz patches using 3D cKDTree overlap check."""
    items = []
    for pdir in patch_dirs:
        try:
            meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
            x = tifffile.imread(pdir / "x.tif").astype(np.float32)
            y = tifffile.imread(pdir / "y.tif").astype(np.float32)
            z = tifffile.imread(pdir / "z.tif").astype(np.float32)
            valid = (x >= 0) & (y >= 0) & (z >= 0) & ((x > 0) | (y > 0) | (z > 0))
            n_val = int(valid.sum())
            if n_val < 30:
                continue
            sy, sx = float(meta["scale"][1]), float(meta["scale"][0])
            valid_cm2 = (n_val / (sy * sx)) * ((vum / 1e4) ** 2)
            if valid_cm2 < 0.6:
                continue
            pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
            if len(pts) > 600:
                idx = np.linspace(0, len(pts) - 1, 600, dtype=int)
                pts = pts[idx]
            items.append((valid_cm2, pdir, pts))
        except Exception:
            continue

    items.sort(key=lambda t: t[0], reverse=True)
    kept_dirs: list[pathlib.Path] = []
    kept_trees: list[cKDTree] = []
    for area_cm2, pdir, pts in items:
        dup = False
        for tree in kept_trees:
            dists, _ = tree.query(pts, workers=1)
            if float((dists <= dist_vox).mean()) >= overlap_thresh:
                dup = True
                break
        if not dup:
            kept_dirs.append(pdir)
            kept_trees.append(cKDTree(pts))
    print(f"  3D dedup: {len(patch_dirs)} grown -> {len(kept_dirs)} unique patches (>= 0.6 cm2 valid)", flush=True)
    return kept_dirs


def run_scroll(scroll: str, vc_bin: pathlib.Path, runners: list[fl.FlatModelRunner], get_confirm_runners, get_aranke, args) -> dict:
    cfg = TRANCHE2_SCROLLS[scroll]
    vpath, vum = cfg["volume"]
    out_dir = pathlib.Path(args.out) / scroll
    seg_dir = out_dir / "segments"
    maps_dir = out_dir / "full_maps"
    work_dir = pathlib.Path(args.scratch) / scroll
    ng_dir = work_dir / "normal_grids"
    cache_dir = work_dir / "vc_cache"
    dummy_vol = work_dir / "vol_stub.zarr"
    for d in (out_dir, seg_dir, work_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    fs_async = s3fs.S3FileSystem(anon=True, asynchronous=True)
    ct_arr = zarr.open_group(FsspecStore(fs_async, path=vpath, read_only=True), mode="r")["0"]
    preds_arr = zarr.open_group(FsspecStore(fs_async, path=cfg["preds"], read_only=True), mode="r")["0"]

    print(f"\n================ TRANCHE 2: {scroll} ({vum} um) ================", flush=True)
    t_scroll = time.perf_counter()

    seeds = sample_interior_seeds(ct_arr, preds_arr, n_seeds=args.n_seeds, seed=42)
    if not seeds:
        return {"scroll": scroll, "error": "no valid seeds"}

    sync_normal_grids(cfg["normal_grids"], ng_dir, workers=args.sync_workers)
    make_dummy_volume_zarr(dummy_vol, cfg["preds"])

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

    print(f"  Growing patches from {len(seeds)} seeds (parallel={args.grow_procs}, gen={args.generations})...", flush=True)
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
        grown_now = [d for d in seg_dir.iterdir() if d.is_dir() and (d / "x.tif").exists()]
        print(f"    grow batch {b0 + len(batch)}/{len(seeds)} -> {len(grown_now)} patches ({time.perf_counter()-t_grow:.1f}s)", flush=True)

    raw_patch_dirs = sorted([d for d in seg_dir.iterdir() if d.is_dir() and (d / "meta.json").exists() and (d / "x.tif").exists()])
    patch_dirs = deduplicate_patches_3d(raw_patch_dirs, vum)

    # Archive kept grown tifxyz meshes to compact tar.gz so they persist after Pod deletion!
    tar_path = out_dir / f"{scroll}_tifxyz.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for pdir in patch_dirs:
            tf.add(pdir, arcname=pdir.name)
    print(f"  Archived {len(patch_dirs)} tifxyz meshes -> {tar_path.name} ({tar_path.stat().st_size/1e6:.2f} MB)", flush=True)

    shutil.rmtree(ng_dir, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)

    cvol = fl.ChunkCachedVolume(ct_arr, max_chunks=args.max_chunks)
    results = []
    global_tiles: list[dict] = []
    tile_crops: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for pi, pdir in enumerate(patch_dirs, 1):
        jid = f"{scroll}_{pdir.name}"
        try:
            surf = fl.UprightTifXYZSurface(pdir, auto_upright=True)
            H, W = surf.full_resolution_shape
            n_layers = 29 if args.recenter_ct else 17
            layers_raw, mask, dt_ren = fl.render_surface_cached(surf, cvol, bounds=None, layers=n_layers, workers=args.workers)
            if args.recenter_ct:
                layers, rec_info = fl.recenter_slab_ct_peak(layers_raw, mask, out_layers=17, max_shift=6, block=256)
            else:
                layers, rec_info = layers_raw, {"dz_med": 0.0, "dz_p90_abs": 0.0}
            valid_frac = float(mask.mean())
            if valid_frac < 0.20 or not mask.any():
                continue
            mid_ct = layers[8]
            ct_std = float(mid_ct[mask].std())
            if ct_std < 8.0:
                continue

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
                        ar.save_aranke_confirm_panel(
                            out_dir / "aranke_hits" / f"{jid}_y{y0}_x{x0}.png",
                            f"HIT {jid} ({t['side']}) y={y0} x={x0}",
                            mid_ct[y0:y0+wh, x0:x0+ww], ps[y0:y0+wh, x0:x0+ww], s_best, s_dir, t["stroke_score"], s_info,
                        )
                        print(f"    [ARANKE HIT] {jid} y={y0} x={x0}: reader={t['stroke_score']:.3f} -> studio={s_info['studio_score']:.3f} r={s_info['r_reader_studio']:.2f}", flush=True)
            dt_gpu = time.perf_counter() - t_gpu

            if args.save_full_maps:
                ss.save_full_mesh_maps(maps_dir, jid, mid_ct, pf0, pr0, mask)

            cm2_hr = scored["valid_area_cm2"] / max(1e-6, (dt_ren + dt_gpu) / 3600.0)
            results.append({
                "id": jid,
                "scroll": scroll,
                "seg_name": pdir.name,
                "rendered_hw": [int(H), int(W)],
                "valid_area_cm2": scored["valid_area_cm2"],
                "bbox_area_cm2": scored["bbox_area_cm2"],
                "valid_frac": round(valid_frac, 3),
                "ct_std": round(ct_std, 1),
                "n_tiles_evaluated": scored["n_tiles_evaluated"],
                "render_seconds": round(dt_ren, 1),
                "gpu_seconds": round(dt_gpu, 1),
                "valid_cm2_hr": round(cm2_hr, 1),
                "best_tile": bt,
                "top_tiles": scored["top_tiles"][:5],
            })
            for t in scored["top_tiles"]:
                global_tiles.append({"mesh_id": jid, "scroll": scroll, **t})
            global_tiles.sort(key=lambda t: t["stroke_score"], reverse=True)
            keep_keys = {(t["mesh_id"], tuple(t["win_yx"]), t["side"]) for t in global_tiles[:25]}
            for t in scored["top_tiles"]:
                k = (jid, tuple(t["win_yx"]), t["side"])
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
                    f"  [{pi}/{len(patch_dirs)}] {jid} valid={scored['valid_area_cm2']:.2f}cm2 tiles={scored['n_tiles_evaluated']} "
                    f"gpu={dt_gpu:.1f}s | top1=({bt['side']}) stroke={bt['stroke_score']:.4f} n={bt['n_strokes']} c={bt['cells']}",
                    flush=True,
                )
        except Exception as e:
            print(f"    !! ERROR on {jid}: {e}", flush=True)

    results.sort(key=lambda r: (r["best_tile"]["stroke_score"] if r.get("best_tile") else -1.0), reverse=True)
    global_tiles.sort(key=lambda t: t["stroke_score"], reverse=True)
    tot_valid = round(sum(r.get("valid_area_cm2", 0.0) for r in results), 2)
    tot_bbox = round(sum(r.get("bbox_area_cm2", 0.0) for r in results), 2)
    summary = {
        "scroll": scroll,
        "grown_raw": len(raw_patch_dirs),
        "unique_patches": len(patch_dirs),
        "swept_patches": len(results),
        "total_valid_cm2": tot_valid,
        "total_bbox_cm2": tot_bbox,
        "total_tiles_evaluated": sum(r.get("n_tiles_evaluated", 0) for r in results),
        "elapsed_seconds": round(time.perf_counter() - t_scroll, 1),
        "leaderboard": results,
    }
    (out_dir / f"{scroll}_leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{scroll}_top_tiles.json").write_text(json.dumps(global_tiles[:100], indent=2), encoding="utf-8")

    fl.save_tile_montages(
        out_dir / f"{scroll}_stroke_top15",
        global_tiles[:15],
        tile_crops,
        f"Tranche 2 ({scroll}, {tot_valid} cm2 valid)",
        per_page=5,
    )
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scrolls", nargs="+", default=["PHerc0257", "PHerc0358", "PHerc0826", "PHerc0483A", "PHerc0483B", "PHerc0490A", "PHerc0490B", "PHerc1218"])
    ap.add_argument("--vc-bin", required=True, help="Path to vc_grow_seg_from_seed wrapper")
    ap.add_argument("--n-seeds", type=int, default=32)
    ap.add_argument("--generations", type=int, default=65)
    ap.add_argument("--grow-procs", type=int, default=4)
    ap.add_argument("--grow-timeout", type=int, default=360)
    ap.add_argument("--sync-workers", type=int, default=48)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-chunks", type=int, default=4096)
    ap.add_argument("--win-px", type=int, default=1024)
    ap.add_argument("--stride-px", type=int, default=512)
    ap.add_argument("--fast-reader-only", action="store_true")
    ap.add_argument("--confirm-thresh", type=float, default=0.10)
    ap.add_argument("--aranke-thresh", type=float, default=0.0)
    ap.add_argument("--recenter-ct", action="store_true")
    ap.add_argument("--save-full-maps", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--scratch", default="/workspace/t2_scratch")
    ap.add_argument("--out", default="/workspace/vesuvius/outputs/tranche2")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    specs = [fl.MODEL_SPECS[0]] if args.fast_reader_only else fl.MODEL_SPECS
    print(f"Loading {len(specs)} model(s) on {torch.cuda.get_device_name(0)}...", flush=True)
    runners = [fl.FlatModelRunner(name, path, bf16=args.bf16) for name, path in specs]
    confirm_runners = None if args.fast_reader_only else runners[1:]

    def get_confirm_runners():
        nonlocal confirm_runners
        if confirm_runners is None:
            print("  [CONFIRM] Lazy-loading argus_v3 + ink_9um...", flush=True)
            confirm_runners = [fl.FlatModelRunner(name, path, bf16=args.bf16) for name, path in fl.MODEL_SPECS[1:]]
        return confirm_runners

    aranke_runner = None
    def get_aranke():
        nonlocal aranke_runner
        if aranke_runner is None:
            import aranke_runner as ar
            print("  [ARANKE] Lazy-loading studio-softloss-20261023...", flush=True)
            aranke_runner = ar.ArankeStudioRunner(bf16=args.bf16)
        return aranke_runner

    vc_bin = pathlib.Path(args.vc_bin)
    assert vc_bin.exists(), f"Missing vc_grow_seg_from_seed at {vc_bin}"

    all_summaries = []
    for scroll in args.scrolls:
        s = run_scroll(scroll, vc_bin, runners, get_confirm_runners, get_aranke, args)
        all_summaries.append({k: v for k, v in s.items() if k != "leaderboard"})
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(args.out) / "tranche2_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    print("=== TRANCHE_2_COMPLETE_OK ===", flush=True)


if __name__ == "__main__":
    main()
