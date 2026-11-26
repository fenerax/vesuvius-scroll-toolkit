"""
Full-Scroll Benchmark across the Top 10 Z-Chunks of PHercParis4 (Z = 8,450 .. 16,920)
covering ~850+ human ground-truth relative-winding pairs across the full scroll height.

Compares:
  1. Naive 2D Peak Counter (baseline)
  2. scroll_winding_tracer (preset="balanced": +-14px 3D ribbon)
  3. scroll_winding_tracer (preset="strict":   +-24px 3D wide delamination guard)

Also runs an unbiased unsupervised cross-check (matching auto-generated curved chains
against human ground-truth points) and saves a summary ablation figure + JSON report.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scroll_winding_tracer.config import PRESETS, GateConfig
from scroll_winding_tracer.streaming import (
    interpolate_umbilicus_xy,
    load_umbilicus_points,
    open_zarr_array,
    resolve_scroll_assets,
    stream_z_slab,
)
from scroll_winding_tracer.tracer import (
    SlabFields,
    build_slab_fields,
    find_polyline_crossings,
    generate_curved_chains,
    measure_spatial_coverage,
    verify_step_7gate,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "paris4_spiral"
SLAB_DIR = CACHE_DIR / "slabs"
OUT_DIR = ROOT / "outputs"

# Top 10 Z-chunks covering bottom, middle, and top of PHercParis4 (~850+ human pairs!)
# Chunks 46, 58, 59, 61 already have cached slabs; we stream only the 6 remaining chunks (~3 min once).
TARGET_CHUNKS = [44, 45, 46, 53, 58, 59, 61, 79, 80, 81, 82, 88]


def eval_preset_on_slabs(
    slabs_by_z: dict[int, SlabFields],
    rays_by_z: dict[int, list[tuple[str, np.ndarray, np.ndarray]]],
    cfg: GateConfig,
    unsup_by_z: dict[int, list[dict]] | None = None,
) -> dict:
    res = {
        "rays": 0, "pairs": 0,
        "naive_exact": 0, "naive_under": 0, "naive_over": 0,
        "accepted": 0, "exact": 0, "under": 0, "over": 0,
        "span2_chains": 0, "span2_exact": 0,
        "span5_chains": 0, "span5_exact": 0,
        "max_exact_span": 0,
        "unsup_matched": 0, "unsup_exact": 0,
        "rejects": {},
    }

    for z_med, sf in slabs_by_z.items():
        for cid, coords, winds in rays_by_z.get(z_med, []):
            res["rays"] += 1
            all_dists = [
                float(np.linalg.norm(coords[k + 1, :2] - coords[k, :2]))
                for k in range(len(coords) - 1)
                if abs(winds[k + 1] - winds[k] - 1.0) < 1e-6
            ]
            ray_med = float(np.median(all_dists)) if len(all_dists) >= 3 else None

            run_true = 0
            run_pred = 0

            def flush_run():
                nonlocal run_true, run_pred
                if run_true >= 2:
                    res["span2_chains"] += 1
                    if run_pred == run_true:
                        res["span2_exact"] += 1
                        res["max_exact_span"] = max(res["max_exact_span"], run_true)
                if run_true >= 5:
                    res["span5_chains"] += 1
                    if run_pred == run_true:
                        res["span5_exact"] += 1
                run_true = 0
                run_pred = 0

            for i in range(len(coords) - 1):
                dw_true = int(round(winds[i + 1] - winds[i]))
                if dw_true < 1:
                    flush_run()
                    continue
                p_a, p_b = coords[i, :2], coords[i + 1, :2]
                vec = p_b - p_a
                seg_len = float(np.linalg.norm(vec))
                if seg_len < 3.0:
                    flush_run()
                    continue
                dx, dy = vec[0] / seg_len, vec[1] / seg_len
                pad = 4.0
                poly = np.array([[p_a[0] - pad * dx, p_a[1] - pad * dy], [p_b[0] + pad * dx, p_b[1] + pad * dy]])
                s_grid, pts_grid, center_s, entry_s, exit_s = find_polyline_crossings(sf, poly)
                res["pairs"] += 1

                if len(center_s) == 0:
                    dw_pred = -1
                    i_s = i_e = -1
                else:
                    i_s = int(np.argmin(np.abs(center_s - pad)))
                    i_e = int(np.argmin(np.abs(center_s - (pad + seg_len))))
                    if abs(center_s[i_s] - pad) > 5.0 or abs(center_s[i_e] - (pad + seg_len)) > 5.0:
                        dw_pred = -1
                    else:
                        dw_pred = i_e - i_s

                if dw_pred == dw_true:
                    res["naive_exact"] += 1
                elif dw_pred < dw_true:
                    res["naive_under"] += 1
                else:
                    res["naive_over"] += 1

                if dw_pred < 1:
                    res["rejects"]["endpoint_miss"] = res["rejects"].get("endpoint_miss", 0) + 1
                    flush_run()
                    continue

                c_pts = np.column_stack([
                    np.interp(center_s, s_grid, pts_grid[:, 0]),
                    np.interp(center_s, s_grid, pts_grid[:, 1]),
                ])
                all_ok = True
                fail_r = "ok"
                for k in range(i_s, i_e):
                    ok, reason, _ = verify_step_7gate(sf, c_pts[k], c_pts[k + 1], float(entry_s[k + 1] - exit_s[k]), ray_med, cfg)
                    if not ok:
                        all_ok = False
                        fail_r = reason
                        break

                if all_ok:
                    res["accepted"] += 1
                    run_true += dw_true
                    run_pred += dw_pred
                    if dw_pred == dw_true:
                        res["exact"] += 1
                    elif dw_pred < dw_true:
                        res["under"] += 1
                    else:
                        res["over"] += 1
                else:
                    res["rejects"][fail_r] = res["rejects"].get(fail_r, 0) + 1
                    flush_run()
            flush_run()

            if unsup_by_z is not None:
                h_xy = coords[:, :2]
                for ch in unsup_by_z.get(z_med, []):
                    c_xy = ch["points_xyz"][:, :2]
                    dmat = np.linalg.norm(c_xy[:, None, :] - h_xy[None, :, :], axis=-1)
                    matches = []
                    for h_idx in range(len(h_xy)):
                        c_idx = int(np.argmin(dmat[:, h_idx]))
                        if dmat[c_idx, h_idx] <= 6.0:
                            matches.append((c_idx, h_idx))
                    if len(matches) >= 2:
                        for a in range(len(matches) - 1):
                            c1, h1 = matches[a]
                            c2, h2 = matches[a + 1]
                            dw_h = abs(int(round(winds[h2] - winds[h1])))
                            dw_c = abs(c2 - c1)
                            if dw_h >= 1:
                                res["unsup_matched"] += 1
                                if dw_c == dw_h:
                                    res["unsup_exact"] += 1
    return res


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SLAB_DIR.mkdir(parents=True, exist_ok=True)

    with open(CACHE_DIR / "relative_windings.json") as f:
        rw = json.load(f)

    # Group all planar human rays by Z slice
    rays_by_z: dict[int, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for cid, c in rw["collections"].items():
        pts = list(c.get("points", {}).values())
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p["wind_a"])
        coords = np.array([p["p"] for p in pts], dtype=np.float64)
        winds = np.array([p["wind_a"] for p in pts], dtype=np.float64)
        if coords[:, 2].max() - coords[:, 2].min() >= 2.0:
            continue
        z_med = int(round(np.median(coords[:, 2])))
        rays_by_z.setdefault(z_med, []).append((cid, coords, winds))

    # Pick the richest Z slice in each of the 12 target chunks spanning Z = 8,450 .. 16,920!
    bench_zs = []
    for ch in TARGET_CHUNKS:
        zs_in_ch = [z for z in rays_by_z if (z // 192) == ch]
        if zs_in_ch:
            best_z = max(zs_in_ch, key=lambda z: sum(len(r[1]) - 1 for r in rays_by_z[z]))
            bench_zs.append(best_z)
    bench_zs.sort()
    total_target_pairs = sum(sum(len(r[1]) - 1 for r in rays_by_z[z]) for z in bench_zs)
    print(f"=== Full-Scroll PHercParis4 Benchmark: {len(bench_zs)} Z-slabs ({bench_zs[0]}..{bench_zs[-1]}), {total_target_pairs} human pairs ===")

    assets = resolve_scroll_assets("PHercParis4", ROOT / "data" / "s3_metadata.min.json")
    arr = open_zarr_array(assets["surface_zarr"], level="0")
    umb_pts = load_umbilicus_points(CACHE_DIR / "umbilicus.json")

    y_slice = (1344, 6912)
    x_slice = (1728, 6912)
    slabs_by_z: dict[int, SlabFields] = {}

    for i, z in enumerate(bench_zs, 1):
        cf = SLAB_DIR / f"slab_z_{z}.npz"
        t0 = time.perf_counter()
        slab, y0, x0, z0 = stream_z_slab(arr, z, half_z=8, y_slice=y_slice, x_slice=x_slice, cache_file=cf)
        dt_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        sf = build_slab_fields(slab, y0, x0, z0, z)
        dt_f = time.perf_counter() - t1
        slabs_by_z[z] = sf
        npairs = sum(len(r[1]) - 1 for r in rays_by_z[z])
        print(f"  [{i:2d}/{len(bench_zs)}] Z={z:5d} ({npairs:3d} human pairs): load={dt_s:.1f}s, tensor={dt_f:.1f}s")

    # Run unsupervised curved generation for strict preset across all 12 slabs + coverage
    print("\n=== Generating Unsupervised Curved Chains (strict preset) across all 12 Z-slabs ===")
    unsup_strict: dict[int, list[dict]] = {}
    covs_strict = []
    for z in bench_zs:
        sf = slabs_by_z[z]
        ux, uy = interpolate_umbilicus_xy(umb_pts, z)
        ch = generate_curved_chains(sf, ux, uy, PRESETS["strict"])
        unsup_strict[z] = ch
        covs_strict.append(measure_spatial_coverage(ch, sf))

    summary = {}
    print("\n=== Full-Scroll Ablation Results ===")
    for name in ("balanced", "strict"):
        cfg = PRESETS[name]
        ev = eval_preset_on_slabs(slabs_by_z, rays_by_z, cfg, unsup_by_z=unsup_strict if name == "strict" else None)
        n_pairs = ev["pairs"]
        naive_p = 100.0 * ev["naive_exact"] / max(n_pairs, 1)
        prec = 100.0 * ev["exact"] / max(ev["accepted"], 1)
        rec = 100.0 * ev["exact"] / max(n_pairs, 1)
        s2_p = 100.0 * ev["span2_exact"] / max(ev["span2_chains"], 1)
        s5_p = 100.0 * ev["span5_exact"] / max(ev["span5_chains"], 1)
        summary[name] = {
            **ev,
            "naive_precision_pct": round(naive_p, 2),
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "span2_exact_pct": round(s2_p, 2),
            "span5_exact_pct": round(s5_p, 2),
            "avg_coverage_pct": round(float(np.mean(covs_strict)), 2) if name == "strict" else None,
        }
        print(
            f"  [{name.upper():<8}] precision={prec:.2f}% ({ev['exact']}/{ev['accepted']}) | recall={rec:.1f}% "
            f"| under={ev['under']}, over={ev['over']} "
            f"| span>=2: {ev['span2_exact']}/{ev['span2_chains']} ({s2_p:.1f}%) "
            f"| span>=5: {ev['span5_exact']}/{ev['span5_chains']} ({s5_p:.1f}%) | max_exact_span={ev['max_exact_span']}w"
        )
        if name == "strict":
            up = 100.0 * ev["unsup_exact"] / max(ev["unsup_matched"], 1)
            print(f"             UNBIASED unsup cross-check: {ev['unsup_exact']}/{ev['unsup_matched']} ({up:.2f}%) | avg scroll coverage={np.mean(covs_strict):.1f}%")
            print(f"             Naive baseline: {naive_p:.2f}% ({ev['naive_exact']}/{n_pairs}, under={ev['naive_under']}, over={ev['naive_over']})")

    report_path = OUT_DIR / "full_scroll_benchmark_report.json"
    report_path.write_text(json.dumps({"bench_zs": bench_zs, "results": summary}, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark report: {report_path}")

    # Clean bar chart summarizing Naive vs Balanced vs Strict across the full scroll!
    st = summary["strict"]
    ba = summary["balanced"]
    labels = ["Naive 2D\nPeak Counter", "Ours (Balanced)\n+-14px 3D Slab", "Ours (Strict)\n+-24px 3D + Delam"]
    precs = [st["naive_precision_pct"], ba["precision_pct"], st["precision_pct"]]
    errs = [100.0 - p for p in precs]
    recs = [st["naive_precision_pct"], ba["recall_pct"], st["recall_pct"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=140)
    colors = ["#ef5350", "#29b6f6", "#00e676"]

    bars = axes[0].bar(labels, precs, color=colors, width=0.55)
    axes[0].set_ylim(75, 102)
    axes[0].set_ylabel("Step Precision (%)")
    axes[0].set_title(f"Full-Scroll Step Precision ({len(bench_zs)} Z-Slabs, {st['pairs']} Human Pairs)")
    axes[0].grid(axis="y", alpha=0.25)
    for b, val in zip(bars, precs):
        axes[0].text(b.get_x() + b.get_width() / 2, val + 0.6, f"{val:.2f}%", ha="center", fontweight="bold")

    bars2 = axes[1].bar(labels, errs, color=colors, width=0.55)
    axes[1].set_ylabel("Error Rate (%) — Lower is Better")
    axes[1].set_title("Counting Error Rate Reduction (vs. Naive Baseline)")
    axes[1].grid(axis="y", alpha=0.25)
    for b, val in zip(bars2, errs):
        axes[1].text(b.get_x() + b.get_width() / 2, val + 0.35, f"{val:.2f}%", ha="center", fontweight="bold")

    fig.tight_layout()
    chart_out = OUT_DIR / "full_scroll_ablation.png"
    fig.savefig(chart_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation chart: {chart_out}")


if __name__ == "__main__":
    main()
