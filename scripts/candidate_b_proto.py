"""
Candidate B prototype: raw-CT recto-edge snapping of a surface volume + ink_canonical_2um A/B test.

What it does
------------
1. Streams PHerc0139 segment 20260311000000 (volume 20260102150214, 2.399 um) layers-zarr from S3.
   Pyramid level 5 (109 x 843 x 775, ~70 MB, YX-only downsampling) gives a segment-wide coarse
   offset map used to pick ROIs. Only targeted 109 x 512 x 512 level-0 crops are streamed after that.
2. Snapper (`snap_offsets`): in the rendered surface volume the layer axis is the mesh normal, and the
   nominal surface (layer 54) sits on the papyrus -> air-gap edge (papyrus at lower layer index, gap at
   higher). Per lateral block we score every candidate layer z in [54-R, 54+R] with a polarity-aware
   step filter  S(z) = mean(I[z-w:z]) - mean(I[z:z+w])  (positive only for papyrus->gap descents, so the
   rising gap->next-wrap edge can never win), then regularise the 2D label field with semi-global
   matching (8 paths, P1 for 1-layer steps, P2 for larger jumps), sub-layer parabola refinement and
   confidence-weighted smoothing. R is kept below half the observed wrap pitch, so a snap cannot reach
   the neighbouring wrap's matching edge. Offsets are measured relative to the segment-wide median edge
   layer (z_ref), so the correction preserves the placement convention the ink model was trained on
   and only removes local deviations.
3. Re-slices [23 + dz, 85 + dz) with linear interpolation along the layer axis and runs the local
   ink_canonical_2um (ResNet3D-152 + 3D decoder) on baseline vs snapped inputs, on a drifted ROI and on
   an already-centred control ROI. Writes outputs/candidate_b_proto.png, outputs/candidate_b_overview.png
   and prints a JSON metrics block.

Usage:  .venv\\Scripts\\python.exe scripts\\candidate_b_proto.py [--no-ink] [--crop 512]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, zoom
from zarr.storage import FsspecStore

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "vendor"))

BUCKET = "vesuvius-challenge-open-data"
LAYERS_KEY = (
    "PHerc0139/segments/20260311000000-w023_202603112281/surface-volumes/"
    "2.399um-0.22m-78keV-volume-20260102150214.zarr"
)
INK_DS8 = (
    ROOT / "cache/stage1/PHerc0139/segments/20260311000000-w023_202603112281/ink-detection/downsampled/"
    "PHerc0139-20260311000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-"
    "new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg"
)
CKPT_PATH = ROOT / "cache" / "models" / "r152_3ddec_v2_l5_epoch13.ckpt"
OUT_DIR = ROOT / "outputs"
CACHE = Path(tempfile.gettempdir()) / "candidate_b_cache"  # outside the repo on purpose

NOMINAL = 54          # centre layer of the 109-layer render = mesh surface
START, END = 23, 85   # ink_canonical_2um 62-layer window
MAX_CLIP = 200.0
TILE, STRIDE = 256, 128

# snapper parameters (layer units = 2.399 um voxels)
SEARCH_R = 12         # candidate edge layers 54 +- 12 (observed wrap pitch here is ~25-30 layers)
STEP_W = 4            # half-width of the step filter
SGM_P1, SGM_P2 = 0.15, 1.2   # in units of normalised step score
MAX_DZ = 15.0


# --------------------------------------------------------------------------------------------- data

def open_layers():
    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    store = FsspecStore(fs, path=f"{BUCKET}/{LAYERS_KEY}", read_only=True)
    return zarr.open_group(store=store, mode="r")


def load_level5(grp) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "L5.npy"
    if p.exists():
        return np.load(p)
    a = np.asarray(grp["5"][:])
    np.save(p, a)
    return a


def load_crop(grp, y0: int, x0: int, size: int) -> np.ndarray:
    p = CACHE / f"L0_{y0}_{x0}_{size}.npy"
    if p.exists():
        return np.load(p)
    t0 = time.perf_counter()
    a = np.asarray(grp["0"][:, y0:y0 + size, x0:x0 + size])
    print(f"  streamed L0 crop {a.shape} ({a.nbytes / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f}s")
    np.save(p, a)
    return a


# ------------------------------------------------------------------------------------------ snapper

def step_scores(vol: np.ndarray, lateral_sigma: float, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Polarity-aware papyrus->gap step score for candidate layers NOMINAL-R..NOMINAL+R.

    Returns (scores[L, h, w] normalised to the block's own max, raw_strength[h, w])."""
    v = gaussian_filter(vol.astype(np.float32), (0.75, lateral_sigma, lateral_sigma))
    if block > 1:
        d, H, W = v.shape
        h, w = H // block, W // block
        v = v[:, : h * block, : w * block].reshape(d, h, block, w, block).mean((2, 4))
    cs = np.concatenate([np.zeros((1,) + v.shape[1:], np.float32), np.cumsum(v, 0)], 0)
    zs = np.arange(NOMINAL - SEARCH_R, NOMINAL + SEARCH_R + 1)
    above = (cs[zs] - cs[zs - STEP_W]) / STEP_W           # mean I[z-w:z]
    below = (cs[zs + STEP_W] - cs[zs]) / STEP_W           # mean I[z:z+w]
    s = np.maximum(above - below, 0.0)
    strength = s.max(0)
    return s / (strength[None] + 1e-3), strength


def sgm(cost: np.ndarray, p1: float, p2: float) -> np.ndarray:
    """Semi-global matching aggregation over 8 directions. cost: (L, H, W)."""
    L, H, W = cost.shape
    total = np.zeros_like(cost)
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for dy, dx in dirs:
        agg = np.zeros_like(cost)
        # traverse along the dominant axis; diagonal paths use the previous row/col shifted
        if dy == 0:
            rng = range(W) if dx > 0 else range(W - 1, -1, -1)
            prev = None
            for x in rng:
                c = cost[:, :, x]
                if prev is None:
                    a = c.copy()
                else:
                    a = c + _sgm_step(prev, p1, p2)
                agg[:, :, x] = a
                prev = a
        else:
            rng = range(H) if dy > 0 else range(H - 1, -1, -1)
            prev = None
            for y in rng:
                c = cost[:, y, :]
                if prev is None:
                    a = c.copy()
                else:
                    pv = prev
                    if dx != 0:
                        pv = np.roll(prev, dx, axis=1)
                        edge = 0 if dx > 0 else W - 1
                        pv[:, edge] = prev[:, edge]
                    a = c + _sgm_step(pv, p1, p2)
                agg[:, y, :] = a
                prev = a
        total += agg
    return total


def _sgm_step(prev: np.ndarray, p1: float, p2: float) -> np.ndarray:
    m = prev.min(0, keepdims=True)
    up = np.full_like(prev, np.inf)
    dn = np.full_like(prev, np.inf)
    up[1:] = prev[:-1] + p1
    dn[:-1] = prev[1:] + p1
    best = np.minimum(np.minimum(prev, up), np.minimum(dn, m + p2))
    return best - m


def snap_edges(vol: np.ndarray, block: int, lateral_sigma: float) -> dict:
    """Edge layer field (in absolute layer units) on the block grid + confidence."""
    s, strength = step_scores(vol, lateral_sigma, block)
    cost = 1.0 - s
    agg = sgm(cost, SGM_P1, SGM_P2)
    k = agg.argmin(0)
    L = agg.shape[0]
    # sub-layer parabola on the aggregated cost
    km, kp = np.clip(k - 1, 0, L - 1), np.clip(k + 1, 0, L - 1)
    c0 = np.take_along_axis(agg, k[None], 0)[0]
    cm = np.take_along_axis(agg, km[None], 0)[0]
    cp = np.take_along_axis(agg, kp[None], 0)[0]
    den = cm - 2 * c0 + cp
    frac = np.where((den > 1e-6) & (k > 0) & (k < L - 1), 0.5 * (cm - cp) / np.maximum(den, 1e-6), 0.0)
    edge = NOMINAL - SEARCH_R + k + np.clip(frac, -0.5, 0.5)
    # confidence: raw step strength at the chosen layer (grey levels per layer of descent), saturating
    s_at = np.take_along_axis(s, k[None], 0)[0] * strength
    at_bound = (k == 0) | (k == L - 1)
    return {"edge": edge, "strength": s_at, "at_bound": at_bound, "raw_argmax": NOMINAL - SEARCH_R + s.argmax(0)}


def smooth_field(field: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    num = gaussian_filter(field * weight, sigma)
    den = gaussian_filter(weight, sigma)
    return num / np.maximum(den, 1e-6)


def dz_field_for_crop(vol: np.ndarray, z_ref: float, block: int = 8,
                      sigmas: dict[str, float] | None = None) -> dict:
    """dz fields for several lateral smoothing scales (in blocks): 'fine' follows ~12 px structure
    (includes part of the fibre weave), 'drift' keeps only >= ~40 px drift."""
    sigmas = sigmas or {"fine": 1.5, "drift": 5.0}
    r = snap_edges(vol, block=block, lateral_sigma=2.0)
    w = np.clip(r["strength"] / 15.0, 0.05, 1.0) * (~r["at_bound"])
    w = np.maximum(w, 1e-3)
    edge_m = median_filter(r["edge"], 3)
    H, W = vol.shape[1:]
    out = {"edge_blk": r["edge"], "conf_blk": w, "at_bound_blk": r["at_bound"]}
    for name, sg in sigmas.items():
        dz_blk = np.clip(smooth_field(edge_m, w, sigma=sg) - z_ref, -MAX_DZ, MAX_DZ)
        out[name] = zoom(dz_blk, (H / dz_blk.shape[0], W / dz_blk.shape[1]), order=1).astype(np.float32)
    return out


def reslice(vol: np.ndarray, dz: np.ndarray, start: int = 0, end: int | None = None) -> np.ndarray:
    """out[k, y, x] = vol[start + k + dz(y, x), y, x] with linear interpolation along layers."""
    d = vol.shape[0]
    end = d if end is None else end
    v = vol.astype(np.float32)
    out = np.empty((end - start,) + vol.shape[1:], np.float32)
    yy, xx = np.indices(dz.shape)
    for i, k in enumerate(range(start, end)):
        z = np.clip(k + dz, 0, d - 1)
        z0 = np.floor(z).astype(np.int32)
        z1 = np.minimum(z0 + 1, d - 1)
        f = z - z0
        out[i] = v[z0, yy, xx] * (1 - f) + v[z1, yy, xx] * f
    return out


def measure_residual(vol: np.ndarray, z_ref: float) -> np.ndarray:
    """Per-block edge offset from z_ref, measured with the same detector (used before and after)."""
    r = snap_edges(vol, block=8, lateral_sigma=2.0)
    return r["edge"] - z_ref, r["at_bound"]


# --------------------------------------------------------------------------------------- ROI choice

def scan_level5(a5: np.ndarray) -> dict:
    valid = a5.max(0) > 0
    r = snap_edges(a5, block=1, lateral_sigma=0.7)
    edge = np.where(valid, r["edge"], np.nan)
    z_ref = float(np.nanmedian(edge[valid & ~r["at_bound"]]))
    edge_m = median_filter(np.nan_to_num(edge, nan=z_ref), 5)
    dz5 = np.where(valid, edge_m - z_ref, np.nan)
    return {"valid": valid, "z_ref": z_ref, "dz5": dz5, "at_bound": r["at_bound"]}


def pick_rois(scan: dict, ink5: np.ndarray, crop: int, n_drift: int = 2,
              cover5: np.ndarray | None = None) -> dict:
    """Drift ROIs: coherent offset (|median| >= 4 layers, std < 3, < 5% search-bound hits) in a
    text-bearing area (official ink ds8 std > 35). Control: text-bearing, smallest |median| + std."""
    blk = crop // 32  # L5 pixel = 32 L0 pixels
    dz5, valid, ab = scan["dz5"], scan["valid"], scan["at_bound"]
    H, W = dz5.shape
    cands, ctrl = [], None
    for r in range(0, H - blk, 4):
        for c in range(0, W - blk, 4):
            if valid[r:r + blk, c:c + blk].mean() < 0.999 or ink5[r:r + blk, c:c + blk].std() < 35:
                continue
            if cover5 is not None and cover5[r:r + blk, c:c + blk].mean() < 0.999:
                continue  # need the independent 1.1 um reference over the whole ROI
            d = dz5[r:r + blk, c:c + blk]
            med, std, bnd = float(np.median(d)), float(d.std()), float(ab[r:r + blk, c:c + blk].mean())
            if abs(med) >= 4 and std < 3 and bnd < 0.05:
                cands.append((abs(med) - std, r, c, med, std))
            sc = abs(med) + std
            if ctrl is None or sc < ctrl[0]:
                ctrl = (sc, r, c, med, std)
    cands.sort(reverse=True)
    chosen = []
    for cnd in cands:
        if all(abs(cnd[1] - o[1]) > 2 * blk or abs(cnd[2] - o[2]) > 2 * blk for o in chosen):
            chosen.append(cnd)
        if len(chosen) == n_drift:
            break
    out = {}
    for i, b in enumerate(chosen):
        out[f"drift_{chr(97 + i)}"] = b
    out["control"] = ctrl
    rois = {}
    for name, (_, r, c, med, std) in out.items():
        rois[name] = {"y0": (r * 32 // 128) * 128, "x0": (c * 32 // 128) * 128,
                      "l5_median_dz": round(med, 2), "l5_std_dz": round(std, 2)}
    return rois


# ------------------------------------------------------------------------------------------- ink

class InkModel:
    def __init__(self):
        import torch
        from model_resnet3d_3d_decoder import load_model
        from stage2_ink_smoke import verify_device, hann2d

        self.torch = torch
        self.device = verify_device()
        self.wrapper = load_model(str(CKPT_PATH), device=self.device, num_frames=END - START)
        self.wrapper.eval()
        self.win = hann2d(TILE, TILE)

    def predict(self, vol62: np.ndarray) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as F
        _, H, W = vol62.shape
        acc = np.zeros((H, W), np.float32)
        wacc = np.zeros((H, W), np.float32)
        x = np.clip(vol62, 0, MAX_CLIP).astype(np.float32) / MAX_CLIP
        with torch.inference_mode():
            for yy in range(0, H - TILE + 1, STRIDE):
                for xx in range(0, W - TILE + 1, STRIDE):
                    t = torch.from_numpy(np.ascontiguousarray(x[:, yy:yy + TILE, xx:xx + TILE]))[None, None]
                    p = torch.sigmoid(self.wrapper.forward(t.to(self.device)))
                    p = F.interpolate(p.float(), size=(TILE, TILE), mode="bilinear", align_corners=False)
                    acc[yy:yy + TILE, xx:xx + TILE] += p[0, 0].cpu().numpy() * self.win
                    wacc[yy:yy + TILE, xx:xx + TILE] += self.win
        return acc / np.maximum(wacc, 1e-6)


def ink_metrics(p: np.ndarray) -> dict:
    gy, gx = np.gradient(gaussian_filter(p, 1.0))
    # Otsu separability: between-class variance share of the best 2-class split (label-free bimodality)
    hist, edges = np.histogram(p, bins=64, range=(0, 1))
    w = hist / hist.sum()
    mids = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(w)
    m0 = np.cumsum(w * mids)
    mt = m0[-1]
    between = (mt * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-9)
    return {
        "mean": float(p.mean()),
        "p50": float(np.percentile(p, 50)),
        "p99": float(np.percentile(p, 99)),
        "contrast_p99_p50": float(np.percentile(p, 99) - np.percentile(p, 50)),
        "area_gt_0p5": float((p > 0.5).mean()),
        "area_gt_0p3": float((p > 0.3).mean()),
        "edge_sharpness_x1000": float(np.hypot(gx, gy).mean() * 1000),
        "otsu_separability": float(between.max() / max(p.var(), 1e-9)),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


VARIANTS = ("baseline", "fine", "drift")
SWEEP = (-8, -6, -4, -2, 0, 2, 4)  # uniform layer-window offsets for the sensitivity sweep

INK_1UM_KEY = (
    "PHerc0139/segments/20260311000000-w023_202603112281/ink-detection/downsampled/"
    "PHerc0139-20260311000000-1.129um-0.22m-59keV-volume-20260413113053-L1-20260709123958-"
    "mrg20736-1um-s1z2-tile256-stride128-ds8.jpg"
)


def load_ink1um(shape: tuple[int, int]) -> np.ndarray:
    """Independent reference: the published 1.129 um-scan ink prediction (different scan, different model)
    of the same segment. Its UV grid is the 2.4 um grid scaled by 2.399 / (2 x 1.129); a plain resize
    aligns it (best integer shift search on ds8 = (0, 0), global r = 0.49)."""
    p = CACHE / "ink1um_ds8.jpg"
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(f"{BUCKET}/{INK_1UM_KEY}", str(p))
    im = Image.open(p).convert("L").resize((shape[1], shape[0]), Image.BILINEAR)
    return np.array(im).astype(np.float32)


# ------------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--no-ink", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    grp = open_layers()
    a5 = load_level5(grp)
    print(f"L5 {a5.shape}")
    scan = scan_level5(a5)
    z_ref, dz5, v = scan["z_ref"], scan["dz5"], scan["valid"]
    print(f"segment-wide median edge layer z_ref = {z_ref:.2f} (nominal {NOMINAL})")
    pct = np.nanpercentile(np.abs(dz5[v]), [50, 90, 99])
    frac3 = float(np.mean(np.abs(dz5[v]) > 3))
    frac6 = float(np.mean(np.abs(dz5[v]) > 6))
    bound = float(scan["at_bound"][v].mean())
    print(f"L5 |dz| p50/p90/p99 = {pct.round(2)}, share |dz|>3 = {frac3:.3f}, >6 = {frac6:.3f}, "
          f"search-bound hits = {bound:.3f}")

    ink8 = np.array(Image.open(INK_DS8).convert("L")).astype(np.float32)
    ink5 = np.array(Image.fromarray(ink8).resize((a5.shape[2], a5.shape[1]), Image.BILINEAR))
    ink1um = load_ink1um(ink8.shape)
    cover5 = np.array(Image.fromarray(ink1um).resize((a5.shape[2], a5.shape[1]), Image.BILINEAR)) > 2
    print(f"1.1 um reference covers {cover5[v].mean():.0%} of the segment")
    rois = pick_rois(scan, ink5, args.crop, cover5=cover5)
    # plus the letter-rich ROI the stage-2 smoke test picked (readable strokes in the official map)
    from stage2_ink_smoke import pick_letter_rich_roi
    ly, lx = pick_letter_rich_roi(ink8.astype(np.uint8), a5.shape[1] * 32, a5.shape[2] * 32, args.crop)
    ly5, lx5 = ly // 32, lx // 32
    b5 = args.crop // 32
    d = dz5[ly5:ly5 + b5, lx5:lx5 + b5]
    rois["letters"] = {"y0": ly, "x0": lx, "l5_median_dz": round(float(np.nanmedian(d)), 2),
                       "l5_std_dz": round(float(np.nanstd(d)), 2)}
    print("ROIs:", json.dumps(rois))
    overview_figure(dz5, ink8, rois, z_ref, frac3, args.crop)

    model = None if args.no_ink else InkModel()
    results = {"z_ref": z_ref, "l5_abs_dz_p50_p90_p99": pct.tolist(), "l5_share_absdz_gt3": frac3,
               "l5_share_absdz_gt6": frac6, "l5_search_bound_share": bound, "rois": {}}
    panels = {}
    for name, r in rois.items():
        y0, x0 = r["y0"], r["x0"]
        print(f"\n== ROI {name}: Y[{y0}:{y0 + args.crop}] X[{x0}:{x0 + args.crop}]")
        vol = load_crop(grp, y0, x0, args.crop)
        t0 = time.perf_counter()
        snap = dz_field_for_crop(vol, z_ref)
        rr = dict(r, snap_seconds=round(time.perf_counter() - t0, 2),
                  low_conf_block_share=float(np.mean(snap["conf_blk"] < 0.2)))
        res_before, bound_b = measure_residual(vol, z_ref)
        ok_b = ~bound_b
        rr["baseline"] = {
            "residual_abs_median": float(np.median(np.abs(res_before[ok_b]))),
            "residual_abs_p90": float(np.percentile(np.abs(res_before[ok_b]), 90)),
            "share_blocks_absres_gt3": float(np.mean(np.abs(res_before[ok_b]) > 3)),
            "residual_signed_median": float(np.median(res_before[ok_b])),
        }
        vols = {"baseline": vol.astype(np.float32)}
        for vname in ("fine", "drift"):
            dz = snap[vname]
            vols[vname] = reslice(vol, dz)
            res_after, bound_a = measure_residual(vols[vname], z_ref)
            ok = ~bound_b & ~bound_a
            rr[vname] = {
                "dz_mean": float(dz.mean()), "dz_abs_mean": float(np.abs(dz).mean()),
                "dz_min": float(dz.min()), "dz_max": float(dz.max()),
                "share_px_absdz_gt3": float(np.mean(np.abs(dz) > 3)),
                "residual_abs_median_after": float(np.median(np.abs(res_after[ok]))),
                "residual_abs_p90_after": float(np.percentile(np.abs(res_after[ok]), 90)),
                "share_blocks_absres_gt3_after": float(np.mean(np.abs(res_after[ok]) > 3)),
            }
        # control for "is it the spatial snap or just a global window shift?": uniform dz = mean(drift dz)
        c = float(np.round(snap["drift"].mean() * 2) / 2)
        vols["const"] = reslice(vol, np.full_like(snap["drift"], c))
        rr["const"] = {"dz_const": c}
        pnl = {"vols": vols, "dz": snap}
        if model is not None:
            t0 = time.perf_counter()
            preds = {k: model.predict(vols[k][START:END]) for k in VARIANTS + ("const",)}
            rr["ink_seconds_all"] = round(time.perf_counter() - t0, 1)
            sl = (slice(y0 // 8, (y0 + args.crop) // 8), slice(x0 // 8, (x0 + args.crop) // 8))
            off, ref1 = ink8[sl] / 255.0, ink1um[sl] / 255.0

            def score(p_):
                d8 = np.array(Image.fromarray(p_).resize(off.shape[::-1], Image.BILINEAR))
                return {"ink": ink_metrics(p_), "corr_vs_official_ds8": corr(off, d8),
                        "corr_vs_1um_ds8": corr(ref1, d8), "corr_vs_baseline": corr(preds["baseline"], p_)}

            for k in VARIANTS + ("const",):
                rr[k].update(score(preds[k]))
            # ink sensitivity to a uniform depth offset of the 62-layer window
            rr["shift_sweep"] = {}
            for s in SWEEP:
                p_ = preds["baseline"] if s == 0 else model.predict(vol[START + s:END + s].astype(np.float32))
                sc = score(p_)
                rr["shift_sweep"][str(s)] = {"corr_vs_1um_ds8": sc["corr_vs_1um_ds8"],
                                             "corr_vs_official_ds8": sc["corr_vs_official_ds8"],
                                             "corr_vs_baseline": sc["corr_vs_baseline"],
                                             "mean": sc["ink"]["mean"], "area_gt_0p5": sc["ink"]["area_gt_0p5"]}
            pnl.update(preds=preds, official=off, ref1=ref1)
        print(json.dumps(rr, indent=1))
        results["rois"][name] = rr
        panels[name] = pnl

    make_figure(panels, results, z_ref, args.crop)
    print("\nRESULTS_JSON " + json.dumps(results))


def overview_figure(dz5, ink8, rois, z_ref, frac3, crop):
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), dpi=110)
    im = ax[0].imshow(dz5, cmap="RdBu_r", vmin=-10, vmax=10)
    ax[0].set_title(f"Segment-wide recto-edge offset dz (L5, 77 um px, noisy)\n"
                    f"z_ref={z_ref:.1f}; |dz|>3 layers on {frac3:.0%} of L5 px")
    plt.colorbar(im, ax=ax[0], fraction=0.04, label="layers (2.4 um)")
    ax[1].imshow(ink8, cmap="gray")
    ax[1].set_title("Official ink_canonical_2um prediction (ds8)")
    cols = {"drift_a": "#ff3d00", "drift_b": "#ffab00", "control": "#00e676", "letters": "#d500f9"}
    for name, r in rois.items():
        for a, s in ((ax[0], 32), (ax[1], 8)):
            a.add_patch(plt.Rectangle((r["x0"] / s, r["y0"] / s), crop / s, crop / s, fill=False,
                                      ec=cols.get(name, "w"), lw=2))
            a.text(r["x0"] / s, r["y0"] / s - 10, name, color=cols.get(name, "w"), fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "candidate_b_overview.png", bbox_inches="tight")
    plt.close(fig)


def make_figure(panels: dict, results: dict, z_ref: float, crop: int) -> None:
    names = list(panels)
    has_ink = "preds" in panels[names[0]]
    mosaic = []
    for i in range(len(names)):
        mosaic.append([f"xz{i}"] * 4 + [f"xs{i}"] * 3)
        mosaic.append([f"dz{i}", f"b54{i}", f"s54{i}", f"r1{i}", f"ib{i}", f"if{i}", f"id{i}"])
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(28, 8.2 * len(names)), dpi=80,
                                 gridspec_kw={"height_ratios": [0.75, 1] * len(names)})
    for i, name in enumerate(names):
        p, rr = panels[name], results["rois"][name]
        vol, dzf, dzd = p["vols"]["baseline"], p["dz"]["fine"], p["dz"]["drift"]
        row = 64 + int(np.argmax(np.ptp(dzd[64:-64], axis=1)))
        a = ax[f"xz{i}"]
        a.imshow(vol[:, row, :], cmap="gray", aspect="auto", vmin=0, vmax=200)
        a.axhline(NOMINAL, color="#ff3d00", lw=1.2, label="nominal surface z=54")
        a.axhline(z_ref, color="#ff3d00", lw=0.8, ls=":", label=f"z_ref={z_ref:.1f} (median edge)")
        a.plot(z_ref + dzf[row], color="#ffd600", lw=1.0, label="snapped edge (fine)")
        a.plot(z_ref + dzd[row], color="#00e5ff", lw=1.4, label="snapped edge (drift-only)")
        a.axhspan(START, END - 1, color="#2979ff", alpha=0.07)
        b = rr["baseline"]
        a.set_title(f"{name}: raw CT XZ at row {row} (layer axis = mesh normal; blue band = ink window [23,85))\n"
                    f"edge residual |res| median {b['residual_abs_median']:.2f} -> fine "
                    f"{rr['fine']['residual_abs_median_after']:.2f} / drift "
                    f"{rr['drift']['residual_abs_median_after']:.2f} layers", fontsize=10)
        a.legend(loc="lower right", fontsize=7)
        a = ax[f"xs{i}"]
        a.imshow(p["vols"]["drift"][:, row, :], cmap="gray", aspect="auto", vmin=0, vmax=200)
        a.axhline(z_ref, color="#00e5ff", lw=0.8)
        a.set_title("same row after drift-only re-slicing", fontsize=10)
        im = ax[f"dz{i}"].imshow(dzd, cmap="RdBu_r", vmin=-10, vmax=10)
        ax[f"dz{i}"].axhline(row, color="k", lw=0.6, ls="--")
        ax[f"dz{i}"].set_title(f"drift-only dz [{dzd.min():.1f}, {dzd.max():.1f}]", fontsize=10)
        plt.colorbar(im, ax=ax[f"dz{i}"], fraction=0.046)
        ax[f"b54{i}"].imshow(vol[NOMINAL], cmap="gray", vmin=0, vmax=200)
        ax[f"b54{i}"].set_title("baseline layer 54", fontsize=10)
        ax[f"s54{i}"].imshow(p["vols"]["drift"][NOMINAL], cmap="gray", vmin=0, vmax=200)
        ax[f"s54{i}"].set_title("drift-snapped layer 54", fontsize=10)
        for key, var in (("ib", "baseline"), ("if", "fine"), ("id", "drift")):
            a = ax[f"{key}{i}"]
            if has_ink:
                m = rr[var]["ink"]
                a.imshow(p["preds"][var], cmap="gray", vmin=0, vmax=1)
                a.set_title(f"ink {var}: p99 {m['p99']:.2f}, area>0.5 {m['area_gt_0p5']:.3f}\n"
                            f"r(official)={rr[var]['corr_vs_official_ds8']:.2f}, "
                            f"r(1um ref)={rr[var]['corr_vs_1um_ds8']:.2f}, r(base)={rr[var]['corr_vs_baseline']:.2f}", fontsize=10)
            else:
                a.set_visible(False)
        a = ax[f"r1{i}"]
        if has_ink:
            a.imshow(p["ref1"], cmap="gray", vmin=0, vmax=1)
            a.set_title("independent ref: published\n1.129 um-scan ink map (ds8)", fontsize=10)
        else:
            a.set_visible(False)
        for k in ("dz", "b54", "s54", "r1", "ib", "if", "id"):
            ax[f"{k}{i}"].set_xticks([])
            ax[f"{k}{i}"].set_yticks([])
    fig.suptitle("Candidate B prototype: raw-CT recto-edge snapping -> ink_canonical_2um A/B "
                 "(PHerc0139 seg 20260311000000, 2.399 um)", fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "candidate_b_proto.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'candidate_b_proto.png'}")


if __name__ == "__main__":
    main()
