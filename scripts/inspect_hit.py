"""Deep-inspect a top First-Letters hit: exact CT-layer contour overlay + adjacent Z-window check."""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr
from zarr.storage import FsspecStore

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fl_radar_sweep as fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", default="PHerc0125")
    ap.add_argument("--window-z", default="8144")
    ap.add_argument("--wrap", default="w065")
    ap.add_argument("--col-start", type=int, default=200)
    ap.add_argument("--col-end", type=int, default=1500)
    ap.add_argument("--side", choices=("fwd", "rev"), default="fwd")
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "fl_sweep" / f"inspect_{args.scroll}_z{args.window_z}_{args.wrap}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (ROOT / "cache" / "eligible_meshes" / "data" / "index.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["scroll"] == args.scroll]

    target = [r for r in rows if r["window_z"] == args.window_z and r["wrap"] == args.wrap][0]
    surf = fl.UprightTifXYZSurface(ROOT / "cache" / "eligible_meshes" / target["path"], auto_upright=True)
    fh, fw = surf.full_resolution_shape
    c0 = max(0, args.col_start)
    c1 = min(fw, args.col_end)
    r1 = min(fh, 1200)

    xyz_p, _, val_p = surf.tile_geometry((0, r1, c0, c1))
    center_xyz = xyz_p[val_p].mean(axis=0)
    print(f"Hit 3D center (x,y,z) = ({center_xyz[0]:.1f}, {center_xyz[1]:.1f}, {center_xyz[2]:.1f})", flush=True)

    # Find nearest adjacent Z windows in index.csv
    z_vals = sorted({int(r["window_z"]) for r in rows if r["window_z"].isdigit()})
    cur_z = int(args.window_z)
    adj_zs = [z for z in z_vals if z != cur_z]
    adj_zs.sort(key=lambda z: abs(z - cur_z))
    adj_zs = sorted(adj_zs[:2])

    adj = []
    for wz in adj_zs:
        wzs = str(wz)
        best_d, best_r, best_c = 1e9, None, 0
        for r in rows:
            if r["window_z"] != wzs:
                continue
            p = ROOT / "cache" / "eligible_meshes" / r["path"]
            if not (p / "meta.json").exists():
                continue
            sf = fl.UprightTifXYZSurface(p, auto_upright=True)
            d2 = (sf.x - center_xyz[0]) ** 2 + (sf.y - center_xyz[1]) ** 2
            d2[~sf.valid_grid] = 1e18
            idx = int(np.argmin(d2))
            d = float(np.sqrt(d2.flat[idx]))
            if d < best_d:
                best_d, best_r = d, r
                _, ci = divmod(idx, sf.shape[1])
                best_c = int(round(ci / sf.scale_x))
        if best_r is not None:
            print(f"Closest in z{wzs}: {best_r['wrap']} (xy_dist={best_d:.1f} vox, col~{best_c})", flush=True)
            adj.append((wzs, best_r, best_c, best_d))

    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    vpath, _ = fl.SCROLL_VOLUMES[args.scroll]
    arr = zarr.open_group(FsspecStore(fs, path=vpath, read_only=True), mode="r")["0"]
    cvol = fl.ChunkCachedVolume(arr, max_chunks=768)
    r3 = fl.FlatModelRunner("reader_v3", fl.MODEL_SPECS[0][1])

    # 1. Exact pixel contour overlay across CT layers
    l8, m8, dt = fl.render_surface_cached(surf, cvol, bounds=(0, r1, c0, c1), layers=17, workers=16)
    print(f"Rendered target crop ({r1}x{c1-c0}) in {dt:.1f}s", flush=True)
    p8 = r3.infer(l8, m8, overlap=0.5, reverse=(args.side == "rev"), batch_size=2)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    lo, hi = np.percentile(l8[8][m8], [2, 98])
    for idx, lz in enumerate((5, 7, 8, 9, 11)):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(l8[lz], cmap="gray", vmin=lo, vmax=hi)
        ax.contour(p8, levels=[0.5, 0.75], colors=["yellow", "red"], linewidths=1.2)
        ax.set_title(f"z{args.window_z}_{args.wrap} CT layer {lz} ({lz-8:+d} vox) + reader_v3 contour")
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, 2].imshow(p8, cmap="gray", vmin=0, vmax=0.95)
    axes[1, 2].set_title(f"reader_v3 ({args.side}) [0:{r1}, {c0}:{c1}]")
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "exact_contour_overlay.png", dpi=150)
    plt.close(fig)

    # 2. Adjacent Z windows above & below at same 3D (x,y)
    if adj:
        fig, axes = plt.subplots(len(adj), 3, figsize=(16, 5 * len(adj)))
        if len(adj) == 1:
            axes = axes[None, :]
        for row_i, (wz, r, c_est, dist) in enumerate(adj):
            sf = fl.UprightTifXYZSurface(ROOT / "cache" / "eligible_meshes" / r["path"], auto_upright=True)
            afh, afw = sf.full_resolution_shape
            ac0 = max(0, c_est - 768)
            ac1 = min(afw, ac0 + 1536)
            la, ma, dta = fl.render_surface_cached(sf, cvol, bounds=(0, afh, ac0, ac1), layers=17, workers=16)
            pf = r3.infer(la, ma, overlap=0.5, reverse=False, batch_size=2)
            pr = r3.infer(la, ma, overlap=0.5, reverse=True, batch_size=2)
            print(f"Adj z{wz}_{r['wrap']} ({afh}x{ac1-ac0}) rendered in {dta:.1f}s | fwd_p99={np.percentile(pf[ma],99):.3f} rev_p99={np.percentile(pr[ma],99):.3f}", flush=True)
            lo2, hi2 = np.percentile(la[8][ma], [2, 98]) if ma.any() else (0, 255)
            axes[row_i, 0].imshow(la[8], cmap="gray", vmin=lo2, vmax=hi2)
            axes[row_i, 0].set_title(f"z{wz}_{r['wrap']} CT (xy_dist={dist:.0f} vox)")
            axes[row_i, 1].imshow(pf, cmap="gray", vmin=0, vmax=0.9)
            axes[row_i, 1].set_title(f"z{wz}_{r['wrap']} reader_v3 FWD (p99={np.percentile(pf[ma],99):.2f})")
            axes[row_i, 2].imshow(pr, cmap="gray", vmin=0, vmax=0.9)
            axes[row_i, 2].set_title(f"z{wz}_{r['wrap']} reader_v3 REV (p99={np.percentile(pr[ma],99):.2f})")
            for c in range(3):
                axes[row_i, c].set_xticks([]); axes[row_i, c].set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / "adjacent_z_check.png", dpi=140)
        plt.close(fig)
    print("DONE:", out_dir, flush=True)


if __name__ == "__main__":
    main()
