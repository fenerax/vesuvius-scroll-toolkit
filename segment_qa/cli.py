"""Command line: ``python -m segment_qa {calibrate,audit,verify}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import OUT_DIR

DEFAULT_CAL = OUT_DIR / "ink_tolerance_curve.json"


def _yx(text: str) -> tuple[int, int]:
    y, x = (int(v) for v in text.replace(" ", "").split(","))
    return y, x


def _threshold_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--calibration", default=str(DEFAULT_CAL),
                   help="ink_tolerance_curve.json from `calibrate` (use 'none' for built-in defaults)")
    p.add_argument("--switch-jump-um", type=float, default=None, help="lateral step that counts as a sheet switch")
    p.add_argument("--air-gap-fail-um", type=float, default=None, help="nominal-in-air distance that counts as off-sheet")


def _thresholds(args):
    from .runs import load_thresholds

    cal = None if str(args.calibration).lower() == "none" else args.calibration
    return load_thresholds(cal, switch_jump_um=args.switch_jump_um, air_gap_fail_um=args.air_gap_fail_um)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m segment_qa", description=__doc__)
    ap.add_argument("--catalog", default=None,
                    help="S3 metadata catalog (default: $SEGMENT_QA_CATALOG or data/s3_metadata.min.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="measure ink_canonical_2um tolerance to window shifts (GPU)")
    c.add_argument("--n-crops", type=int, default=48)
    c.add_argument("--shifts", default="-20,-16,-12,-8,-6,-4,-2,0,2,4,6,8,12,16,20,24",
                   help="window shifts in layers (stack allows -23..+24)")
    c.add_argument("--min-sep", type=int, default=1024, help="minimum L0 distance between crops (px)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default=str(OUT_DIR))
    c.add_argument("--reanalyze", action="store_true",
                   help="recompute statistics, thresholds and figure from the saved raw arrays (no GPU)")

    a = sub.add_parser("audit", help="classify cells of one segment")
    a.add_argument("--sample", default="PHerc0139")
    a.add_argument("--segment", default=None, help="segment id (built-in for PHerc0139, PHerc1667, PHercParis4)")
    a.add_argument("--volume", default=None, help="target volume id of the layers-zarr")
    a.add_argument("--layers-key", default=None,
                   help="exact layers-zarr S3 key when the segment has several variants (e.g. a -t2 render)")
    mode = a.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true", help="sweep the whole segment on a pyramid level")
    mode.add_argument("--crop", type=_yx, action="append", help="L0 crop origin 'Y,X' (repeatable, 128-aligned)")
    a.add_argument("--level", type=int, default=3, help="pyramid level for --full (3 = 8x8 L0 cells)")
    a.add_argument("--size", type=int, default=256, help="crop size in L0 px for --crop")
    a.add_argument("--l0-check-tiles", type=int, default=24, help="L0 tile grid cross-check of --full")
    a.add_argument("--mark", type=_yx, action="append", default=[], help="L0 'Y,X' point to report and mark")
    a.add_argument("--out", default=str(OUT_DIR))
    _threshold_args(a)

    v = sub.add_parser("verify", help="three-scroll hotspot verification figure")
    v.add_argument("--out", default=str(OUT_DIR))
    _threshold_args(v)

    args = ap.parse_args(argv)
    if args.catalog:
        from .data import set_catalog

        set_catalog(args.catalog)
    if args.cmd == "calibrate":
        from .ink_tolerance import calibrate, reanalyze, validate_shifts

        if args.reanalyze:
            reanalyze(Path(args.out), seed=args.seed)
            return 0
        try:
            shifts = validate_shifts(float(s) for s in args.shifts.split(","))
        except ValueError as e:
            ap.error(f"--shifts: {e}")
        calibrate(n_crops=args.n_crops, shifts=shifts, out_dir=Path(args.out), seed=args.seed, min_sep=args.min_sep)
        return 0

    thr = _thresholds(args)
    print(f"thresholds ({thr.source}): {json.dumps(thr.to_dict())}", flush=True)
    if args.cmd == "verify":
        from .runs import verify

        verify(thr, Path(args.out))
        return 0

    from .data import resolve_segment
    from .runs import audit_crop, audit_full

    ref = resolve_segment(args.sample, args.segment, args.volume, args.layers_key)
    out = Path(args.out)
    if args.full:
        cal = None
        if str(args.calibration).lower() != "none" and Path(args.calibration).exists():
            cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        marks = [(y, x, f"({y}, {x})") for y, x in args.mark]
        audit_full(ref, thr, out, level=args.level, l0_check_tiles=args.l0_check_tiles,
                   mark_points=marks, calibration=cal)
        return 0

    from .auditor import find_hotspots, save_audit
    from .data import BoundedCache, LayersVolume, check_crop, output_tag

    vol = LayersVolume(ref)          # metadata only; no CT is read before every crop passes the bounds check
    cache = BoundedCache()
    try:
        shape = vol.shape(0)
        for y0, x0 in args.crop:
            try:
                check_crop(shape, y0, x0, args.size, 8)
            except ValueError as e:
                ap.error(f"--crop {y0},{x0}: {e}")
        for y0, x0 in args.crop:
            res, _ = audit_crop(vol, y0, x0, thr, size=args.size, cache=cache)
            prefix = f"{output_tag(ref)}_crop_{y0}_{x0}"
            summary = save_audit(res, out, prefix, hotspots=find_hotspots(res))
            for y, x in args.mark:
                try:
                    print(json.dumps(res.cell_report(y, x)), flush=True)
                except IndexError:
                    pass
            print(prefix, json.dumps(summary["class_counts"]), flush=True)
    finally:
        vol.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
