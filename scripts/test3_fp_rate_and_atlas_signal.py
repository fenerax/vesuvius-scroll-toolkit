"""Test 3 ($0 local): Random-Papyrus False-Positive Rate Audit + Competitor Held-Back Signal Check.

Measures our raw-float thin-stroke DoG scorer's true false-positive rate across ~200+ ordinary
papyrus tiles from 12 diverse eligible_meshes across 7 scrolls, PLUS inspects the 4 remaining
held-back meshes from rodriguescarson/eligible-scroll-atlas (PHerc0358 z11280_w020, PHerc0813
z5888_w020, z7696_w020, z13088_w040) for $0 locally!
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

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
    out_dir = ROOT / "outputs" / "council_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group A: 4 held-back meshes from rodriguescarson/eligible-scroll-atlas (reading the competitor's signal!)
    atlas_held = [
        ("ATLAS_PHerc0358_z11280_w020", "PHerc0358", ROOT / "cache/eligible_meshes/meshes/PHerc0358/z11280_w020"),
        ("ATLAS_PHerc0813_z5888_w020",  "PHerc0813", ROOT / "cache/eligible_meshes/meshes/PHerc0813/z5888_w020"),
        ("ATLAS_PHerc0813_z7696_w020",  "PHerc0813", ROOT / "cache/eligible_meshes/meshes/PHerc0813/z7696_w020"),
        ("ATLAS_PHerc0813_z13088_w040", "PHerc0813", ROOT / "cache/eligible_meshes/meshes/PHerc0813/z13088_w040"),
    ]
    # Group B: 10 diverse ordinary random papyrus meshes across 6 scrolls (fibers, folds, cracks, varied wraps)
    random_papyrus = [
        ("RAND_PHerc0125_z9344_w020",   "PHerc0125", ROOT / "cache/eligible_meshes/meshes/PHerc0125/z9344_w020"),
        ("RAND_PHerc0125_z11136_w040",  "PHerc0125", ROOT / "cache/eligible_meshes/meshes/PHerc0125/z11136_w040"),
        ("RAND_PHerc0125_z12944_w055",  "PHerc0125", ROOT / "cache/eligible_meshes/meshes/PHerc0125/z12944_w055"),
        ("RAND_PHerc0257_z8528_w020",   "PHerc0257", ROOT / "cache/eligible_meshes/meshes/PHerc0257/z8528_w020"),
        ("RAND_PHerc0257_z11536_w040",  "PHerc0257", ROOT / "cache/eligible_meshes/meshes/PHerc0257/z11536_w040"),
        ("RAND_PHerc0268_z8528_w020",   "PHerc0268", ROOT / "cache/eligible_meshes/meshes/PHerc0268/z8528_w020"),
        ("RAND_PHerc0800_z9872_w020",   "PHerc0800", ROOT / "cache/eligible_meshes/meshes/PHerc0800/z9872_w020"),
        ("RAND_PHerc0800_z14672_w040",  "PHerc0800", ROOT / "cache/eligible_meshes/meshes/PHerc0800/z14672_w040"),
        ("RAND_PHerc0813_z8896_w020",   "PHerc0813", ROOT / "cache/eligible_meshes/meshes/PHerc0813/z8896_w020"),
        ("RAND_PHerc0826_z10336_w020",  "PHerc0826", ROOT / "cache/eligible_meshes/meshes/PHerc0826/z10336_w020"),
    ]
    # Fallback resolver if a specific random mesh name doesn't exist
    jobs = []
    for grp, lst in (("atlas_held", atlas_held), ("random_papyrus", random_papyrus)):
        for jid, scroll, mdir in lst:
            if (mdir / "meta.json").exists():
                jobs.append((grp, jid, scroll, mdir))
            else:
                # pick first available mesh of that scroll not already picked
                used = {j[3] for j in jobs}
                cands = sorted(p for p in (ROOT / "cache/eligible_meshes/meshes" / scroll).iterdir()
                               if p.is_dir() and (p / "meta.json").exists() and p not in used)
                if cands:
                    pick = cands[len(cands) // 2]
                    jobs.append((grp, f"{grp.upper()[:4]}_{scroll}_{pick.name}", scroll, pick))

    print(f"Loading reader-v3 on {torch.cuda.get_device_name(0)} for {len(jobs)} meshes...", flush=True)
    runner = fl.FlatModelRunner("reader_v3", ROOT / "cache/models/reader-v3.pth", fp16=False, bf16=False)
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    vol_cache = {}

    def get_vol(scroll):
        if scroll not in vol_cache:
            vpath, vum = fl.SCROLL_VOLUMES[scroll]
            arr = zarr.open_group(FsspecStore(fs, path=vpath, read_only=True), mode="r")["0"]
            vol_cache[scroll] = (fl.ChunkCachedVolume(arr, max_chunks=768), vum)
        return vol_cache[scroll]

    all_rand_scores = []
    atlas_results = []
    mesh_records = []
    tile_crops = {}
    all_top_tiles = []

    for idx, (grp, jid, scroll, mdir) in enumerate(jobs, 1):
        cvol, vum = get_vol(scroll)
        surf = fl.UprightTifXYZSurface(mdir, auto_upright=True)
        H, W = surf.full_resolution_shape
        layers29, mask, dt_ren = fl.render_surface_cached(surf, cvol, bounds=None, layers=29, workers=16)
        layers, rec_info = fl.recenter_slab_ct_peak(layers29, mask, out_layers=17, max_shift=6, block=256)
        t_gpu = time.perf_counter()
        pf = runner.infer(layers, mask, overlap=0.5, reverse=False, batch_size=4)
        pr = runner.infer(layers, mask, overlap=0.5, reverse=True, batch_size=4)
        torch.cuda.synchronize()
        dt_gpu = time.perf_counter() - t_gpu

        scored = ss.score_full_mesh_tiles(
            pf, pr, mask, vum, win_px=1024, stride_px=512, top_k=5, nms_iou=0.35, return_all_tiles=True
        )
        bt = scored["best_tile"]
        print(
            f"[{idx:2d}/{len(jobs)}] {jid:32s} ({H}x{W}, {scored['valid_area_cm2']:4.2f}cm2, {scored['n_tiles_evaluated']:2d} tiles) "
            f"dz={rec_info['dz_med']:+.1f} | top1={bt['stroke_score'] if bt else 0:.4f} (n={bt['n_strokes'] if bt else 0}, c={bt['cells'] if bt else 0})",
            flush=True,
        )
        rec = {
            "group": grp,
            "id": jid,
            "scroll": scroll,
            "hw": [H, W],
            "valid_area_cm2": scored["valid_area_cm2"],
            "n_tiles": scored["n_tiles_evaluated"],
            "recenter": rec_info,
            "best_tile": bt,
        }
        mesh_records.append(rec)
        if grp == "atlas_held":
            atlas_results.append(rec)
        else:
            all_rand_scores.extend(t["stroke_score"] for t in scored["all_tiles"])

        for t in scored["top_tiles"][:2]:
            gt = {"mesh_id": jid, "group": grp, "scroll": scroll, **t}
            all_top_tiles.append(gt)
            y0, x0, h, w = t["win_yx"]
            ps = pf if t["side"] == "fwd" else pr
            po = pr if t["side"] == "fwd" else pf
            tile_crops[(jid, tuple(t["win_yx"]), t["side"])] = (
                layers[8, y0:y0+h, x0:x0+w].copy(),
                ps[y0:y0+h, x0:x0+w].copy(),
                np.clip(ps[y0:y0+h, x0:x0+w] - 0.5 * po[y0:y0+h, x0:x0+w], 0.0, 1.0),
            )

    arr_s = np.array(all_rand_scores, dtype=np.float32)
    fp_stats = {
        "n_random_tiles": int(len(arr_s)),
        "median": round(float(np.median(arr_s)), 5) if len(arr_s) else 0.0,
        "p90": round(float(np.percentile(arr_s, 90)), 5) if len(arr_s) else 0.0,
        "p95": round(float(np.percentile(arr_s, 95)), 5) if len(arr_s) else 0.0,
        "p99": round(float(np.percentile(arr_s, 99)), 5) if len(arr_s) else 0.0,
        "max": round(float(arr_s.max()), 5) if len(arr_s) else 0.0,
        "frac_ge_006": round(float((arr_s >= 0.06).mean()), 4) if len(arr_s) else 0.0,
        "frac_ge_008": round(float((arr_s >= 0.08).mean()), 4) if len(arr_s) else 0.0,
        "frac_ge_010": round(float((arr_s >= 0.10).mean()), 4) if len(arr_s) else 0.0,
        "frac_ge_012": round(float((arr_s >= 0.12).mean()), 4) if len(arr_s) else 0.0,
        "count_ge_010": int((arr_s >= 0.10).sum()) if len(arr_s) else 0,
    }
    print("\n=== TEST 3 RANDOM-PAPYRUS FALSE-POSITIVE AUDIT ===")
    print(json.dumps(fp_stats, indent=2))
    print("\n=== COMPETITOR HELD-BACK MESHES (eligible-scroll-atlas) ===")
    for r in atlas_results:
        bt = r["best_tile"] or {}
        print(f"  {r['id']:30s}: top1={bt.get('stroke_score',0):.4f} n={bt.get('n_strokes',0)} c={bt.get('cells',0)} dz={r['recenter']['dz_med']:+.1f}")

    (out_dir / "test3_fp_rate_audit.json").write_text(json.dumps({
        "fp_stats": fp_stats,
        "atlas_held": atlas_results,
        "meshes": mesh_records,
    }, indent=2), encoding="utf-8")

    # Save visual montage of top-10 tiles (including all 4 atlas held-back meshes!)
    all_top_tiles.sort(key=lambda t: t["stroke_score"], reverse=True)
    fl.save_tile_montages(out_dir / "test3_top_tiles", all_top_tiles[:10], tile_crops, "Test 3 FP Audit + Atlas Held-Back", per_page=5)
    print(f"Saved {out_dir / 'test3_fp_rate_audit.json'} & montages", flush=True)


if __name__ == "__main__":
    main()
