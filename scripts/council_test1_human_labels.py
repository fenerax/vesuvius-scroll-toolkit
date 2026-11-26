"""Council Test 1: Validate segment_qa directly against human ink-labels (20260918) on PHerc1667 & PHercParis4.

Read-only against S3 (streams level-3 inklabels.zarr / supervision.zarr / validation.zarr) and the saved
level-3 QA fields + cached 2.4 um ink previews. Evaluates:
  1. Cell-level accuracy (Dice, F0.5, IoU, Pearson r, recall, precision) by QA class and failure reason,
     plus label-hygiene audit (human ink=1 lost vs ink=0 off-sheet pollution) for Old Keep-Mask vs Hard-Fail Mask.
  2. Tile-level (256 L0 px) triage benchmark with NO std<5 filter (plus with-filter for comparison),
     comparing QA FAIL, Hard-Fail (WINDOW_EMPTY|NO_BAND|AIR_GAP_FAIL), DRIFT_FAIL, signed drift,
     dist_to_pad, and ink contrast via block-bootstrap 95% CIs (1024 L0 px blocks).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import s3fs
import zarr
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from zarr.storage import FsspecStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segment_qa.auditor import (
    CLASS_NAMES,
    FAIL_OFF_SHEET,
    FAIL_SHEET_SWITCH,
    INVALID_PAD,
    OK,
    REASON_BIT,
    WARN_DRIFT,
)
from segment_qa.data import BUCKET, OUT_DIR, BoundedCache, ink_preview_ds8, resolve_segment

TARGETS = [
    {
        "name": "PHerc1667",
        "sample": "PHerc1667",
        "segment": "20251212185248",
        "volume": "20251217075048",
        "tag": "pherc1667_20251212185248_20251217075048_full",
        "label_prefix": "PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/ink-labels/2.399um-volume-20251217075048/20260918",
        "has_validation": True,
    },
    {
        "name": "PHercParis4",
        "sample": "PHercParis4",
        "segment": "20230702185753",
        "volume": "20260411134726",
        "tag": "phercparis4_20230702185753_20260411134726_full",
        "label_prefix": "PHercParis4/segments/20230702185753/ink-labels/2.4um-volume-20260411134726/20260918",
        "has_validation": False,
    },
]

R_DRIFT_FAIL = REASON_BIT["DRIFT_FAIL"]
R_AIR_GAP = REASON_BIT["AIR_GAP_FAIL"]
R_WIN_EMPTY = REASON_BIT["WINDOW_EMPTY"]
R_NO_BAND = REASON_BIT["NO_BAND"]
R_IN_AIR = REASON_BIT["NOMINAL_IN_AIR"]
R_HARD_FAIL = R_WIN_EMPTY | R_NO_BAND | R_AIR_GAP


def load_l3_mask(fs, prefix: str, name: str) -> np.ndarray:
    store = FsspecStore(fs, path=f"{BUCKET}/{prefix}/{name}", read_only=True)
    g = zarr.open(store, mode="r")
    arr = g["3"] if hasattr(g, "keys") else g
    return np.asarray(arr)


def cell_metrics(pred_u8: np.ndarray, lbl_bin: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n_cells": 0}
    p = pred_u8[mask].astype(np.float32) / 255.0
    y = lbl_bin[mask]
    pb = p >= 0.5
    tp = int((pb & y).sum())
    fp = int((pb & ~y).sum())
    fn = int((~pb & y).sum())
    ink_n = int(y.sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    f05 = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp) if (1.25 * tp + 0.25 * fn + fp) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    r = float(np.corrcoef(p, y.astype(np.float32))[0, 1]) if (p.std() > 1e-6 and y.std() > 1e-6) else float("nan")
    return {
        "n_cells": n,
        "ink_prevalence": round(ink_n / n, 4),
        "pred_mean": round(float(p.mean()), 4),
        "precision": round(prec, 4) if np.isfinite(prec) else None,
        "recall": round(rec, 4) if np.isfinite(rec) else None,
        "dice": round(dice, 4) if np.isfinite(dice) else None,
        "f05": round(f05, 4) if np.isfinite(f05) else None,
        "iou": round(iou, 4) if np.isfinite(iou) else None,
        "pearson_r": round(r, 4) if np.isfinite(r) else None,
    }


def partial_spearman(x: np.ndarray, y: np.ndarray, covs: np.ndarray) -> float:
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rc = np.column_stack([np.ones(len(x))] + [rankdata(covs[:, j]) for j in range(covs.shape[1])])
    bx, *_ = np.linalg.lstsq(rc, rx, rcond=None)
    by, *_ = np.linalg.lstsq(rc, ry, rcond=None)
    ex = rx - rc @ bx
    ey = ry - rc @ by
    if ex.std() < 1e-9 or ey.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def tile_benchmark(
    cls: np.ndarray,
    reasons: np.ndarray,
    drift: np.ndarray,
    ink: np.ndarray,
    lbl: np.ndarray,
    sup: np.ndarray,
    dist_pad: np.ndarray,
    tile: int = 32,
    block_tiles: int = 4,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    gh, gw = cls.shape
    bpx = tile * block_tiles
    rows = []
    blocks = []
    for y in range(0, gh - tile + 1, tile):
        for x in range(0, gw - tile + 1, tile):
            c = cls[y : y + tile, x : x + tile]
            s = sup[y : y + tile, x : x + tile]
            if (c == INVALID_PAD).any() or s.mean() < 0.8:
                continue
            y_bin = lbl[y : y + tile, x : x + tile][s]
            p_u8 = ink[y : y + tile, x : x + tile][s]
            ink_prev = float(y_bin.mean())
            if ink_prev < 0.005:
                continue  # need actual supervised human ink to score Dice / recall / r
            p = p_u8.astype(np.float32) / 255.0
            pb = p >= 0.5
            tp = int((pb & y_bin).sum())
            fp = int((pb & ~y_bin).sum())
            fn = int((~pb & y_bin).sum())
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f05 = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp) if (1.25 * tp + 0.25 * fn + fp) else 0.0
            r = float(np.corrcoef(p, y_bin.astype(np.float32))[0, 1]) if (p.std() > 1e-6 and y_bin.std() > 1e-6) else 0.0
            rs = reasons[y : y + tile, x : x + tile]
            dr = drift[y : y + tile, x : x + tile]
            fail_sh = float(((c == FAIL_OFF_SHEET) | (c == FAIL_SHEET_SWITCH)).mean())
            hard_sh = float(((rs & R_HARD_FAIL) != 0).mean())
            win_empty_sh = float(((rs & (R_WIN_EMPTY | R_NO_BAND)) != 0).mean())
            air_gap_sh = float(((rs & R_AIR_GAP) != 0).mean())
            drift_fail_sh = float(((rs & R_DRIFT_FAIL) != 0).mean())
            in_air_sh = float(((rs & R_IN_AIR) != 0).mean())
            sw_sh = float((c == FAIL_SHEET_SWITCH).mean())
            mean_dr = float(np.nanmean(dr)) if np.isfinite(dr).any() else 0.0
            dp = float(dist_pad[y : y + tile, x : x + tile].mean())
            rows.append([
                dice, rec, f05, r,
                fail_sh, hard_sh, win_empty_sh, air_gap_sh, drift_fail_sh, in_air_sh, sw_sh, mean_dr,
                dp, ink_prev, float(p.std()),
            ])
            blocks.append((y // bpx) * (gw // bpx + 1) + x // bpx)
    R = np.asarray(rows, dtype=np.float64)
    blk = np.asarray(blocks, dtype=np.int64)
    ub, inv = np.unique(blk, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ub))]
    rng = np.random.default_rng(seed)
    boot_idx = [np.concatenate([members[k] for k in rng.integers(0, len(ub), len(ub))]) for _ in range(n_boot)]

    def ci(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        return np.percentile(v, [2.5, 97.5]).round(4).tolist() if v.size else None

    feats = [
        ("fail_share", 4),
        ("hard_fail_share", 5),
        ("window_empty_or_noband_share", 6),
        ("air_gap_fail_share", 7),
        ("drift_fail_share", 8),
        ("nominal_in_air_share", 9),
        ("switch_share", 10),
        ("signed_drift_mean", 11),
        ("neg_dist_to_pad", 12),  # flipped below so higher = closer to edge
        ("neg_ink_pred_std", 14), # flipped below so higher = lower pred contrast
    ]

    covs = R[:, [13, 12]]  # ink_prev, dist_to_pad
    scores_out = {}
    for metric_name, mcol in [("dice", 0), ("recall", 1), ("f05", 2), ("pearson_r", 3)]:
        y = R[:, mcol]
        res = {}
        for fname, fcol in feats:
            x = -R[:, fcol] if fname.startswith("neg_") else R[:, fcol]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = float(spearmanr(x, y).statistic)
                b_rhos = [float(spearmanr(x[i], y[i]).statistic) for i in boot_idx]
            prho = partial_spearman(x, y, covs[:, :1] if "dist_to_pad" in fname else covs)
            # Equal-budget top-10% flagged tiles: excess error (mean_all - mean_top10)
            k = max(1, int(round(0.10 * len(x))))
            order = np.argsort(-x, kind="mergesort")
            top_val = float(y[order[:k]].mean())
            rest_val = float(y[order[k:]].mean())
            # Bootstrap CI on (rest - top) = how much WORSE the flagged top-10% is
            b_lift = []
            for i in boot_idx:
                xi, yi = x[i], y[i]
                ki = max(1, int(round(0.10 * len(xi))))
                oi = np.argsort(-xi, kind="mergesort")
                b_lift.append(float(yi[oi[ki:]].mean() - yi[oi[:ki]].mean()))
            res[fname] = {
                "spearman_rho": round(rho, 4) if np.isfinite(rho) else None,
                "ci95": ci(b_rhos),
                "partial_rho_ctrl_prev_dist": round(prho, 4) if np.isfinite(prho) else None,
                "top10pct_mean": round(top_val, 4),
                "rest90pct_mean": round(rest_val, 4),
                "rest_minus_top10": round(rest_val - top_val, 4),
                "rest_minus_top10_ci95": ci(b_lift),
            }
        scores_out[metric_name] = res

    # Also report Dice by FAIL-share bins (same bins as concordance)
    bins = [(0.0, 1e-9, "no FAIL"), (1e-9, 0.05, "FAIL < 5%"), (0.05, 0.20, "FAIL 5-20%"), (0.20, 1.01, "FAIL >= 20%")]
    by_fail = []
    for lo, hi, label in bins:
        m = (R[:, 4] >= lo) & (R[:, 4] < hi)
        if not m.any():
            continue
        by_fail.append({
            "bin": label,
            "n_tiles": int(m.sum()),
            "mean_dice": round(float(R[m, 0].mean()), 4),
            "mean_recall": round(float(R[m, 1].mean()), 4),
            "mean_f05": round(float(R[m, 2].mean()), 4),
            "mean_r": round(float(R[m, 3].mean()), 4),
            "mean_ink_prev": round(float(R[m, 13].mean()), 4),
        })

    # And by Hard-Fail bins
    by_hard = []
    for lo, hi, label in [(0.0, 1e-9, "no HardFail"), (1e-9, 0.05, "HardFail < 5%"),
                          (0.05, 0.20, "HardFail 5-20%"), (0.20, 1.01, "HardFail >= 20%")]:
        m = (R[:, 5] >= lo) & (R[:, 5] < hi)
        if not m.any():
            continue
        by_hard.append({
            "bin": label,
            "n_tiles": int(m.sum()),
            "mean_dice": round(float(R[m, 0].mean()), 4),
            "mean_recall": round(float(R[m, 1].mean()), 4),
            "mean_f05": round(float(R[m, 2].mean()), 4),
            "mean_r": round(float(R[m, 3].mean()), 4),
        })

    return {
        "n_tiles": int(len(R)),
        "n_blocks": int(len(ub)),
        "overall_tile_means": {
            "dice": round(float(R[:, 0].mean()), 4),
            "recall": round(float(R[:, 1].mean()), 4),
            "f05": round(float(R[:, 2].mean()), 4),
            "pearson_r": round(float(R[:, 3].mean()), 4),
        },
        "by_fail_share": by_fail,
        "by_hard_fail_share": by_hard,
        "feature_triage": scores_out,
    }


def main():
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    cache = BoundedCache()
    report = {}

    for t in TARGETS:
        print(f"\n=== {t['name']} ({t['segment']}) ===", flush=True)
        ref = resolve_segment(t["sample"], t["segment"], t["volume"])
        f = np.load(OUT_DIR / f"{t['tag']}_fields.npz")
        cls = f["cls"]
        reasons = f["reason"] if "reason" in f else f["reasons"]
        drift = f["drift_um"]
        ink = ink_preview_ds8(ref, cache)
        lbl_raw = load_l3_mask(fs, t["label_prefix"], "inklabels.zarr")
        sup_raw = load_l3_mask(fs, t["label_prefix"], "supervision.zarr")
        val_raw = load_l3_mask(fs, t["label_prefix"], "validation.zarr") if t["has_validation"] else None

        print(f"  shapes: cls={cls.shape} ink={ink.shape} lbl={lbl_raw.shape} sup={sup_raw.shape}", flush=True)
        print(f"  unique lbl={np.unique(lbl_raw)[:10]} sup={np.unique(sup_raw)[:10]}", flush=True)
        # Binarize L3 mean-pooled labels (>127 for inklabels, >127 for supervision)
        lbl = lbl_raw >= 128
        sup = (sup_raw >= 128) & (cls != INVALID_PAD)
        val = ((val_raw >= 128) & sup) if val_raw is not None else None
        dist_pad = ndi.distance_transform_edt(cls != INVALID_PAD).astype(np.float32)

        def eval_split(split_mask: np.ndarray, split_name: str) -> dict:
            out = {"n_supervised_valid_cells": int(split_mask.sum())}
            out["all_supervised"] = cell_metrics(ink, lbl, split_mask)
            by_cls = {}
            for code in (OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH):
                by_cls[CLASS_NAMES[code]] = cell_metrics(ink, lbl, split_mask & (cls == code))
            out["by_class"] = by_cls

            hard = (reasons & R_HARD_FAIL) != 0
            win_e = (reasons & (R_WIN_EMPTY | R_NO_BAND)) != 0
            air_g = (reasons & R_AIR_GAP) != 0
            dr_only = ((reasons & R_DRIFT_FAIL) != 0) & ~hard
            in_air = (reasons & R_IN_AIR) != 0

            out["by_reason"] = {
                "WINDOW_EMPTY_OR_NO_BAND": cell_metrics(ink, lbl, split_mask & win_e),
                "AIR_GAP_FAIL": cell_metrics(ink, lbl, split_mask & air_g),
                "HARD_FAIL_ALL": cell_metrics(ink, lbl, split_mask & hard),
                "DRIFT_FAIL_ONLY": cell_metrics(ink, lbl, split_mask & dr_only),
                "NOMINAL_IN_AIR": cell_metrics(ink, lbl, split_mask & in_air),
            }

            # Label hygiene: old keep mask vs Hard-Fail-only keep mask
            fail_all = (cls == FAIL_OFF_SHEET) | (cls == FAIL_SHEET_SWITCH)
            dil_all = ndi.binary_dilation(fail_all, structure=np.ones((3, 3), bool))
            dil_hard = ndi.binary_dilation(hard, structure=np.ones((3, 3), bool))
            sup_ink = split_mask & lbl
            sup_bg = split_mask & ~lbl
            out["mask_hygiene"] = {
                "old_keep_mask": {
                    "cells_kept_frac": round(float((split_mask & ~dil_all).sum() / max(1, split_mask.sum())), 4),
                    "human_ink_dropped_frac": round(float((sup_ink & dil_all).sum() / max(1, sup_ink.sum())), 4),
                    "bg_dropped_frac": round(float((sup_bg & dil_all).sum() / max(1, sup_bg.sum())), 4),
                    "kept_metrics": cell_metrics(ink, lbl, split_mask & ~dil_all),
                    "dropped_metrics": cell_metrics(ink, lbl, split_mask & dil_all),
                },
                "hard_fail_keep_mask": {
                    "cells_kept_frac": round(float((split_mask & ~dil_hard).sum() / max(1, split_mask.sum())), 4),
                    "human_ink_dropped_frac": round(float((sup_ink & dil_hard).sum() / max(1, sup_ink.sum())), 4),
                    "bg_dropped_frac": round(float((sup_bg & dil_hard).sum() / max(1, sup_bg.sum())), 4),
                    "kept_metrics": cell_metrics(ink, lbl, split_mask & ~dil_hard),
                    "dropped_metrics": cell_metrics(ink, lbl, split_mask & dil_hard),
                },
                "bg_on_window_empty_or_noband_frac": round(float((sup_bg & win_e).sum() / max(1, sup_bg.sum())), 4),
                "bg_on_hard_fail_frac": round(float((sup_bg & hard).sum() / max(1, sup_bg.sum())), 4),
                "ink_on_window_empty_or_noband_frac": round(float((sup_ink & win_e).sum() / max(1, sup_ink.sum())), 4),
                "ink_on_hard_fail_frac": round(float((sup_ink & hard).sum() / max(1, sup_ink.sum())), 4),
            }
            print(f"  [{split_name}] all Dice={out['all_supervised']['dice']} | "
                  f"OK={by_cls['OK']['dice']} WARN={by_cls['WARN_DRIFT']['dice']} "
                  f"OFF={by_cls['FAIL_OFF_SHEET']['dice']} SW={by_cls['FAIL_SHEET_SWITCH']['dice']}", flush=True)
            print(f"  [{split_name}] WIN_EMPTY={out['by_reason']['WINDOW_EMPTY_OR_NO_BAND']['dice']} "
                  f"AIR_GAP={out['by_reason']['AIR_GAP_FAIL']['dice']} "
                  f"DRIFT_ONLY={out['by_reason']['DRIFT_FAIL_ONLY']['dice']}", flush=True)
            print(f"  [{split_name}] old_keep drops {100*out['mask_hygiene']['old_keep_mask']['human_ink_dropped_frac']:.1f}% human ink -> Dice {out['mask_hygiene']['old_keep_mask']['kept_metrics']['dice']} | "
                  f"hard_keep drops {100*out['mask_hygiene']['hard_fail_keep_mask']['human_ink_dropped_frac']:.1f}% human ink -> Dice {out['mask_hygiene']['hard_fail_keep_mask']['kept_metrics']['dice']}", flush=True)
            return out

        entry = {"all_supervised": eval_split(sup, "all_supervised")}
        if val is not None and val.any():
            entry["official_validation_split"] = eval_split(val, "validation")
        entry["tile_benchmark"] = tile_benchmark(cls, reasons, drift, ink, lbl, sup, dist_pad)
        tb = entry["tile_benchmark"]
        print(f"  [tiles n={tb['n_tiles']}, blocks={tb['n_blocks']}] Dice vs features:", flush=True)
        for fn, st in tb["feature_triage"]["dice"].items():
            print(f"    {fn:30s} rho={st['spearman_rho']} {st['ci95']} partial={st['partial_rho_ctrl_prev_dist']} "
                  f"rest-top10={st['rest_minus_top10']} {st['rest_minus_top10_ci95']}", flush=True)
        report[t["name"]] = entry

    out_path = OUT_DIR / "council_test1_human_labels.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
