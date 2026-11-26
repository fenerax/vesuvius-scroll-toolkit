"""Command-line interface for streaming whole-scroll winding constraint generation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import PRESETS
from .exporters import export_vc3d_json, export_winding_inference_store_v1
from .streaming import (
    interpolate_umbilicus_xy,
    load_umbilicus_points,
    open_zarr_array,
    resolve_scroll_assets,
    stream_z_slab,
)
from .tracer import (
    build_slab_fields,
    extract_vertical_same_winding_tracks,
    generate_curved_chains,
    measure_spatial_coverage,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="scroll-winding-tracer",
        description="Zero-training curved + 3D Z-slab relative & same-winding constraint generator for Vesuvius scrolls.",
    )
    p.add_argument("--scroll", type=str, help="Scroll ID in catalog (e.g., PHercParis4, PHerc0125, PHerc0800)")
    p.add_argument("--volume-id", type=str, default=None, help="Optional volume ID override")
    p.add_argument("--catalog", type=Path, default=Path("data/s3_metadata.min.json"), help="Local S3 metadata catalog JSON")
    p.add_argument("--surface-zarr", type=str, default=None, help="Direct S3 key or local path to surface-prediction Zarr")
    p.add_argument("--umbilicus", type=str, default=None, help="Direct S3 key or local path to umbilicus JSON")
    p.add_argument("--z-slices", type=int, nargs="+", default=None, help="Explicit Z slices to process")
    p.add_argument("--z-start", type=int, default=None, help="Start Z slice (used with --z-end and --z-stride)")
    p.add_argument("--z-end", type=int, default=None, help="End Z slice (inclusive)")
    p.add_argument("--z-stride", type=int, default=384, help="Z stride between sampled chunks (default: 384 = every 2 Z-chunks)")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="balanced", help="Gate preset (default: balanced)")
    p.add_argument("--out", type=Path, required=True, help="Output directory for VC3D JSONs & WindingInferenceStore v1")
    p.add_argument("--cache-dir", type=Path, default=Path("cache/cli_slabs"), help="Optional bit-packed slab cache directory")
    p.add_argument("--no-cache", action="store_true", help="Stream without caching slabs to disk")
    p.add_argument("--no-same-winding", action="store_true", help="Skip emitting vertical same-winding tracks")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = PRESETS[args.preset]

    if args.surface_zarr and args.umbilicus:
        surf_src = args.surface_zarr
        umb_src = args.umbilicus
        tag = args.scroll or "custom"
    elif args.scroll:
        assets = resolve_scroll_assets(args.scroll, args.catalog, args.volume_id)
        surf_src = assets["surface_zarr"]
        umb_src = assets["umbilicus_json"]
        tag = f"{args.scroll}_{assets['volume_id']}"
        print(f"Resolved {args.scroll} (vol={assets['volume_id']}, {assets['pixel_size_um']} um):")
        print(f"  surface  : {surf_src}")
        print(f"  umbilicus: {umb_src}")
    else:
        raise SystemExit("Provide either --scroll <ID> or both --surface-zarr and --umbilicus.")

    cache_root = None if args.no_cache else (args.cache_dir / tag)
    umb_pts = load_umbilicus_points(umb_src, cache_dir=cache_root)
    arr = open_zarr_array(surf_src, level="0")
    cz = int(arr.chunks[0])
    print(f"Opened Zarr L0: shape={arr.shape}, chunks={arr.chunks} (umbilicus Z=[{umb_pts[:,2].min():.0f}, {umb_pts[:,2].max():.0f}])")

    if args.z_slices:
        z_list = list(args.z_slices)
    else:
        z_lo = args.z_start if args.z_start is not None else int(umb_pts[:, 2].min()) + cz // 2
        z_hi = args.z_end if args.z_end is not None else int(umb_pts[:, 2].max()) - cz // 2
        # Snap each sampled Z to the center of its 192-slice Z-chunk so every 17-slice slab hits ONLY 1 Z-chunk!
        first_c = (z_lo // cz) * cz + cz // 2
        z_list = list(range(first_c, z_hi + 1, max(cz, args.z_stride)))

    print(f"Processing {len(z_list)} Z-slab(s) with preset='{args.preset}' (ribbon=+-{cfg.ribbon_half_w}px, half_z={cfg.half_z})...")
    all_rw: list[dict] = []
    all_sw = []
    t_total = time.perf_counter()

    for idx, z in enumerate(z_list, 1):
        z = max(cfg.half_z, min(int(z), int(arr.shape[0]) - cfg.half_z - 1))
        cache_file = (cache_root / f"slab_z_{z}.npz") if cache_root is not None else None
        t0 = time.perf_counter()
        slab, y0, x0, z0 = stream_z_slab(arr, z, half_z=cfg.half_z, cache_file=cache_file)
        dt_stream = time.perf_counter() - t0

        t1 = time.perf_counter()
        sf = build_slab_fields(slab, y0, x0, z0, z)
        ux, uy = interpolate_umbilicus_xy(umb_pts, z)
        ch = generate_curved_chains(sf, ux, uy, cfg)
        sw = [] if args.no_same_winding else extract_vertical_same_winding_tracks(sf, ch, stride_pts=4)
        cov = measure_spatial_coverage(ch, sf)
        dt_proc = time.perf_counter() - t1

        all_rw.extend(ch)
        all_sw.extend(sw)
        n_pts = sum(len(c["points_xyz"]) for c in ch)
        max_w = max((len(c["points_xyz"]) for c in ch), default=0)
        print(
            f"  [{idx:2d}/{len(z_list)}] Z={z:5d}: {len(ch):4d} RW chains ({n_pts:5d} pts, max={max_w:2d}w) "
            f"+ {len(sw):4d} SW tracks | cov={cov:4.1f}% | stream={dt_stream:.1f}s, cpu={dt_proc:.1f}s"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    rw_json = export_vc3d_json(all_rw, args.out / "relative_windings.json", is_same_winding=False)
    if not args.no_same_winding:
        sw_json = export_vc3d_json(all_sw, args.out / "same_windings.json", is_same_winding=True)
    else:
        sw_json = None
    manifest = export_winding_inference_store_v1(all_rw, args.out / "winding_inference")

    print(f"\n=== Complete in {time.perf_counter() - t_total:.1f}s ===")
    print(f"  VC3D Relative Windings : {rw_json} ({len(all_rw):,} chains, {sum(len(c['points_xyz']) for c in all_rw):,} pts)")
    if sw_json:
        print(f"  VC3D Same Windings     : {sw_json} ({len(all_sw):,} vertical 3D tracks)")
    print(
        f"  WindingInferenceStore  : {args.out / 'winding_inference'} "
        f"({manifest['num_rays']:,} crossing rays, {manifest['num_crossings']:,} crossings, fp={manifest['fingerprint'][:16]}...)"
    )


if __name__ == "__main__":
    main()
