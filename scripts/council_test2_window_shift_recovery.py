"""Council Test 2: Causal whole-tile window-shift ink recovery on PHerc1667 & PHercParis4 (GTX 1070 Ti).

Tests the Opus 5.5 + Fable 5.2 make-or-break hypothesis:
  On unseen scrolls (PHerc1667 and PHercParis4 20230702185753), if we take 256x256 L0 tiles where
  raw-CT band detection says the papyrus drifted >+20 um above the nominal anchor z=54 (plus matched
  clean centered controls and negative-drift tiles) and shift the WHOLE ink-model window coherently
  (NO per-pixel z-warping, preserving 3D fiber weave!), does ink accuracy recover against:
    (a) the independent 1.129 um ink scan (ds8 Pearson r), AND
    (b) human ink-labels (L0 256x256 Dice, F0.5, recall, Pearson r)?

Conditions evaluated per tile:
  - nominal:        dz = 0 (standard layers [23, 85))
  - center:         dz = clip(round(delta_z_vx), -23, 24)        (band placed at z = 54)
  - calibrated_opt: dz = clip(round(delta_z_vx + 6), -23, 24)    (band placed at -14.4 um PHerc0139 peak)
  - wrong_sign:     dz = clip(-round(delta_z_vx), -23, 24)       (causal negative control: opposite shift)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import s3fs
import zarr
from PIL import Image
from zarr.storage import FsspecStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segment_qa.auditor import INVALID_PAD
from segment_qa.band_detector import BandParams, detect_bands
from segment_qa.data import BUCKET, OUT_DIR, BoundedCache, LayersVolume, asset_key, fetch_object, read_crop, resolve_segment
from segment_qa.ink_tolerance import END, START, InkModel

Image.MAX_IMAGE_PIXELS = None

SCROLLS = [
    {
        "name": "PHerc1667",
        "sample": "PHerc1667",
        "segment": "20251212185248",
        "volume": "20251217075048",
        "ref_vol": "20260323082859",
        "tag": "pherc1667_20251212185248_20251217075048_full",
        "label_prefix": "PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/ink-labels/2.399um-volume-20251217075048/20260918",
    },
    {
        "name": "PHercParis4",
        "sample": "PHercParis4",
        "segment": "20230702185753",
        "volume": "20260411134726",
        "ref_vol": "20260608103018",
        "tag": "phercparis4_20230702185753_20260411134726_full",
        "label_prefix": "PHercParis4/segments/20230702185753/ink-labels/2.4um-volume-20260411134726/20260918",
    },
]


def open_zarr_level(fs, prefix: str, name: str, level: str):
    store = FsspecStore(fs, path=f"{BUCKET}/{prefix}/{name}", read_only=True)
    g = zarr.open(store, mode="r")
    return g[level] if hasattr(g, "keys") else g


def select_spaced(cands: list[dict], n: int, min_sep_l0: int = 512, key=lambda c: -c["score"]) -> list[dict]:
    chosen = []
    for c in sorted(cands, key=key):
        if all(max(abs(c["y0"] - o["y0"]), abs(c["x0"] - o["x0"])) >= min_sep_l0 for o in chosen):
            chosen.append(c)
            if len(chosen) >= n:
                break
    return chosen


def match_controls(pos_tiles: list[dict], ctrl_pool: list[dict], min_sep_l0: int = 512) -> list[dict]:
    """Match each positive-drift tile to a spatially separated clean control with similar ref_std."""
    used = []
    out = []
    for pt in pos_tiles:
        target = pt["ref_std"]
        best = None
        best_diff = 1e9
        for c in ctrl_pool:
            if any(max(abs(c["y0"] - o["y0"]), abs(c["x0"] - o["x0"])) < min_sep_l0 for o in used + pos_tiles):
                continue
            d = abs(c["ref_std"] - target)
            if d < best_diff:
                best_diff = d
                best = c
        if best is not None:
            used.append(best)
            out.append(best)
    return out


def score_tile(pred_l0: np.ndarray, ref_ds8: np.ndarray, lbl_l0: np.ndarray, sup_l0: np.ndarray) -> dict:
    # ds8 Pearson r vs 1.129 um reference
    p8 = pred_l0.reshape(32, 8, 32, 8).mean(axis=(1, 3))
    r_ref = float(np.corrcoef(p8.ravel(), ref_ds8.ravel())[0, 1]) if (p8.std() > 1e-6 and ref_ds8.std() > 1e-6) else float("nan")
    # L0 metrics vs human ink-labels inside supervision
    m = sup_l0 >= 128
    if m.sum() == 0:
        return {"r_ref": r_ref, "dice_lbl": float("nan"), "f05_lbl": float("nan"), "rec_lbl": float("nan"), "r_lbl": float("nan")}
    y = lbl_l0[m] >= 128
    p = pred_l0[m]
    pb = p >= 0.5
    tp = int((pb & y).sum())
    fp = int((pb & ~y).sum())
    fn = int((~pb & y).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    f05 = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp) if (1.25 * tp + 0.25 * fn + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    r_lbl = float(np.corrcoef(p, y.astype(np.float32))[0, 1]) if (p.std() > 1e-6 and y.std() > 1e-6) else float("nan")
    return {"r_ref": r_ref, "dice_lbl": dice, "f05_lbl": f05, "rec_lbl": rec, "r_lbl": r_lbl, "pred_fg": float(pb.mean())}


def paired_summary(vals_a: list[float], vals_b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return {"n": 0}
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bd = d[idx].mean(axis=1)
    ci = np.percentile(bd, [2.5, 97.5]).round(4).tolist()
    return {
        "n": int(len(d)),
        "mean_nom": round(float(a.mean()), 4),
        "mean_shifted": round(float(b.mean()), 4),
        "delta_mean": round(float(d.mean()), 4),
        "delta_ci95": ci,
        "delta_median": round(float(np.median(d)), 4),
        "frac_improved": round(float((d > 0).mean()), 4),
    }


def main():
    t_start = time.perf_counter()
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    cache = BoundedCache()
    model = InkModel()
    results = {}
    example_panels = []

    for sc in SCROLLS:
        print(f"\n=== {sc['name']} ({sc['segment']}) ===", flush=True)
        ref = resolve_segment(sc["sample"], sc["segment"], sc["volume"])
        f = np.load(OUT_DIR / f"{sc['tag']}_fields.npz")
        cls, dz_map = f["cls"], f["delta_z_um"]
        gh, gw = cls.shape

        ref_key = asset_key(ref, "ink-detection-downsampled", sc["ref_vol"])
        ref_path = fetch_object(ref_key, cache, f"{ref.tag}/{ref_key.rsplit('/', 1)[-1]}")
        with Image.open(ref_path) as im:
            ref8 = np.asarray(im.convert("L").resize((gw, gh), Image.BILINEAR), dtype=np.float32)

        sup3 = np.asarray(open_zarr_level(fs, sc["label_prefix"], "supervision.zarr", "3"))
        lbl3 = np.asarray(open_zarr_level(fs, sc["label_prefix"], "inklabels.zarr", "3"))
        lbl0_arr = open_zarr_level(fs, sc["label_prefix"], "inklabels.zarr", "0")
        sup0_arr = open_zarr_level(fs, sc["label_prefix"], "supervision.zarr", "0")

        pos_pool, ctrl_pool, neg_pool = [], [], []
        for y in range(0, gh - 32 + 1, 32):
            for x in range(0, gw - 32 + 1, 32):
                c = cls[y : y + 32, x : x + 32]
                if (c == INVALID_PAD).any():
                    continue
                rpatch = ref8[y : y + 32, x : x + 32]
                has_ref = bool((rpatch > 0).all() and rpatch.std() >= 8.0)
                has_lbl = bool((sup3[y : y + 32, x : x + 32] >= 128).mean() >= 0.8 and (lbl3[y : y + 32, x : x + 32] >= 128).mean() >= 0.005)
                if not (has_ref or has_lbl):
                    continue
                d = dz_map[y : y + 32, x : x + 32]
                if np.isfinite(d).mean() < 0.8:
                    continue
                med_dz = float(np.nanmedian(d))
                rec = {
                    "y8": y, "x8": x, "y0": y * 8, "x0": x * 8,
                    "med_dz_um": med_dz,
                    "ref_std": float(rpatch.std()) if has_ref else 0.0,
                    "has_ref": has_ref,
                    "has_lbl": has_lbl,
                    "score": (1000.0 if (has_ref and has_lbl) else (500.0 if has_ref else 0.0)) + med_dz,
                }
                if med_dz >= 15.0:
                    pos_pool.append(rec)
                elif abs(med_dz) <= 8.0 and has_ref:
                    ctrl_pool.append(rec)
                elif med_dz <= -30.0:
                    neg_pool.append({**rec, "score": (1000.0 if (has_ref and has_lbl) else (500.0 if has_ref else 0.0)) - med_dz})

        pos_tiles = select_spaced(pos_pool, 25, min_sep_l0=512)
        ctrl_tiles = match_controls(pos_tiles, ctrl_pool, min_sep_l0=512)
        neg_tiles = select_spaced(neg_pool, 15, min_sep_l0=512)
        print(f"  selected tiles: pos={len(pos_tiles)} (pool {len(pos_pool)}, ref={sum(t['has_ref'] for t in pos_tiles)}, lbl={sum(t['has_lbl'] for t in pos_tiles)}, min_dz={min((t['med_dz_um'] for t in pos_tiles), default=0):.1f}um), "
              f"ctrl={len(ctrl_tiles)}, neg={len(neg_tiles)} (pool {len(neg_pool)})", flush=True)

        vol = LayersVolume(ref)
        px = vol.pixel_size_um
        groups_out = {}
        try:
            for gname, tiles in [("pos_drift", pos_tiles), ("clean_ctrl", ctrl_tiles), ("neg_drift", neg_tiles)]:
                tile_records = []
                for idx, tile in enumerate(tiles, 1):
                    y0, x0, y8, x8 = tile["y0"], tile["x0"], tile["y8"], tile["x8"]
                    slab = read_crop(vol, y0, x0, 256, level=0, cache=cache)
                    fld = detect_bands(slab, px, BandParams(cell_size=8))
                    l0_med_um = float(np.nanmedian(fld.delta_z_um)) if np.isfinite(fld.delta_z_um).any() else tile["med_dz_um"]
                    dz_center = int(np.clip(round(l0_med_um / px), -23, 24))
                    dz_cal = int(np.clip(round((l0_med_um + 14.4) / px), -23, 24))
                    dz_wrong = int(np.clip(-round(l0_med_um / px), -23, 24))
                    # For clean controls with dz_center==0, test a fixed +10-layer (+24um) shift as placebo? No, keep exact formula!

                    ref_patch = ref8[y8 : y8 + 32, x8 : x8 + 32]
                    lbl_l0 = np.asarray(lbl0_arr[y0 : y0 + 256, x0 : x0 + 256])
                    sup_l0 = np.asarray(sup0_arr[y0 : y0 + 256, x0 : x0 + 256])

                    shifts = {"nominal": 0, "center": dz_center, "calibrated_opt": dz_cal, "wrong_sign": dz_wrong}
                    preds = {}
                    scores = {}
                    for cname, dz in shifts.items():
                        if dz not in preds:
                            preds[dz] = model.predict(slab[START + dz : END + dz])
                        scores[cname] = score_tile(preds[dz], ref_patch if tile["has_ref"] else np.zeros((32, 32), np.float32), lbl_l0, sup_l0)

                    rec = {
                        "y0": y0, "x0": x0,
                        "l3_med_dz_um": round(tile["med_dz_um"], 2),
                        "l0_med_dz_um": round(l0_med_um, 2),
                        "shifts_vx": shifts,
                        "scores": {k: {mk: (round(mv, 4) if np.isfinite(mv) else None) for mk, mv in v.items()} for k, v in scores.items()},
                    }
                    tile_records.append(rec)

                    if gname == "pos_drift" and len(example_panels) < 6 and idx <= 3:
                        example_panels.append({
                            "title": f"{sc['name']} ({y0},{x0}) dz={l0_med_um:+.1f}um",
                            "ref": ref_patch,
                            "lbl": lbl_l0,
                            "nom": preds[0],
                            "center": preds[dz_center],
                            "cal": preds[dz_cal],
                            "shifts": shifts,
                            "scores": scores,
                        })

                # Summarize paired improvements
                agg = {"n_tiles": len(tile_records), "mean_l0_dz_um": round(float(np.mean([r["l0_med_dz_um"] for r in tile_records])), 2)}
                for cond in ("center", "calibrated_opt", "wrong_sign"):
                    agg[f"{cond}_vs_nominal"] = {}
                    for m in ("r_ref", "dice_lbl", "f05_lbl", "rec_lbl", "r_lbl"):
                        va = [r["scores"]["nominal"][m] if r["scores"]["nominal"][m] is not None else np.nan for r in tile_records]
                        vb = [r["scores"][cond][m] if r["scores"][cond][m] is not None else np.nan for r in tile_records]
                        agg[f"{cond}_vs_nominal"][m] = paired_summary(va, vb)
                # Also report strict >+25um subset for pos_drift
                if gname == "pos_drift":
                    sub = [r for r in tile_records if r["l0_med_dz_um"] >= 25.0]
                    if sub:
                        agg["strict_gt25um"] = {"n_tiles": len(sub)}
                        for cond in ("center", "calibrated_opt", "wrong_sign"):
                            agg["strict_gt25um"][f"{cond}_vs_nominal"] = {
                                m: paired_summary(
                                    [r["scores"]["nominal"][m] if r["scores"]["nominal"][m] is not None else np.nan for r in sub],
                                    [r["scores"][cond][m] if r["scores"][cond][m] is not None else np.nan for r in sub],
                                )
                                for m in ("r_ref", "dice_lbl", "rec_lbl")
                            }
                groups_out[gname] = {"summary": agg, "tiles": tile_records}
                s_c = agg["center_vs_nominal"]
                s_k = agg["calibrated_opt_vs_nominal"]
                s_w = agg["wrong_sign_vs_nominal"]
                print(f"  [{gname:10s} n={len(tile_records):2d} mean_dz={agg['mean_l0_dz_um']:+5.1f}um]", flush=True)
                print(f"    center vs nom : r_ref {s_c['r_ref']['mean_nom']}->{s_c['r_ref']['mean_shifted']} (d={s_c['r_ref']['delta_mean']:+.4f} {s_c['r_ref']['delta_ci95']}) | "
                      f"Dice {s_c['dice_lbl']['mean_nom']}->{s_c['dice_lbl']['mean_shifted']} (d={s_c['dice_lbl']['delta_mean']:+.4f} {s_c['dice_lbl']['delta_ci95']})", flush=True)
                print(f"    cal_opt vs nom: r_ref {s_k['r_ref']['mean_nom']}->{s_k['r_ref']['mean_shifted']} (d={s_k['r_ref']['delta_mean']:+.4f} {s_k['r_ref']['delta_ci95']}) | "
                      f"Dice {s_k['dice_lbl']['mean_nom']}->{s_k['dice_lbl']['mean_shifted']} (d={s_k['dice_lbl']['delta_mean']:+.4f} {s_k['dice_lbl']['delta_ci95']})", flush=True)
                print(f"    wrong  vs nom : r_ref {s_w['r_ref']['mean_nom']}->{s_w['r_ref']['mean_shifted']} (d={s_w['r_ref']['delta_mean']:+.4f} {s_w['r_ref']['delta_ci95']}) | "
                      f"Dice {s_w['dice_lbl']['mean_nom']}->{s_w['dice_lbl']['mean_shifted']} (d={s_w['dice_lbl']['delta_mean']:+.4f} {s_w['dice_lbl']['delta_ci95']})", flush=True)
        finally:
            vol.close()
        results[sc["name"]] = groups_out

    results["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    out_json = OUT_DIR / "council_test2_window_shift_recovery.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out_json} in {results['runtime_seconds']}s", flush=True)

    # Save visual montage of up to 6 positive-drift examples
    if example_panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(example_panels)
        fig, axes = plt.subplots(n, 5, figsize=(15, 3.0 * n))
        if n == 1:
            axes = axes[None, :]
        cols = ["1.129 um Ref", "Human Label", "Nominal (dz=0)", "Shifted -> z=54", "Shifted -> -14.4 um"]
        for c, title in enumerate(cols):
            axes[0, c].set_title(title, fontsize=10, fontweight="bold")
        for r, ex in enumerate(example_panels):
            imgs = [ex["ref"], ex["lbl"], ex["nom"], ex["center"], ex["cal"]]
            subs = [
                ex["title"],
                "20260918 label",
                f"r={ex['scores']['nominal']['r_ref']:.2f} D={ex['scores']['nominal']['dice_lbl']:.2f}",
                f"dz={ex['shifts']['center']:+d} r={ex['scores']['center']['r_ref']:.2f} D={ex['scores']['center']['dice_lbl']:.2f}",
                f"dz={ex['shifts']['calibrated_opt']:+d} r={ex['scores']['calibrated_opt']['r_ref']:.2f} D={ex['scores']['calibrated_opt']['dice_lbl']:.2f}",
            ]
            for c in range(5):
                axes[r, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=255 if c < 2 else 1)
                axes[r, c].set_xlabel(subs[c], fontsize=8)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        fig.suptitle("Causal Window-Shift Intervention on Positive-Drift Tiles (No Pixel Warping)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "council_test2_recovery_examples.png", dpi=150)
        plt.close(fig)
        print(f"Saved {OUT_DIR / 'council_test2_recovery_examples.png'}", flush=True)


if __name__ == "__main__":
    main()
