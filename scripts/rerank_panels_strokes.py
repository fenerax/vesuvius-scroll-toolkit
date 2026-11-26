"""Re-rank saved First-Letters panel PNGs by thin-stroke Greek-letter morphology (rejecting broad blobs)."""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label


def score_panel_png(p: pathlib.Path) -> tuple[float, dict]:
    im = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    H, W = im.shape
    y0, y1 = int(H * 0.14), int(H * 0.92)
    sub_dir = im[y0:y1, int(W * 0.815) : int(W * 0.985)]
    valid = sub_dir > 0.03
    if valid.mean() < 0.35:
        return 0.0, {}

    # Band-pass DoG matched to crisp Greek stroke width (~0.2-0.7mm -> 2-9px on ~460px panel crop)
    dog = np.clip(gaussian_filter(sub_dir, 1.8) - gaussian_filter(sub_dir, 9.0), 0.0, 1.0)
    # Threshold DoG directly (not raw brightness!) so broad blobs vanish and crisp strokes pop!
    bw = (dog > 0.055) & valid
    lbl, ncomp = label(bw)
    sizes = np.bincount(lbl.ravel())[1:] if ncomp > 0 else np.array([], dtype=np.int64)
    # Crisp letter-stroke components on ~460x460 crop: ~25..1400 px
    stroke_comps = int(((sizes >= 25) & (sizes <= 1400)).sum())
    # Raw giant-blob fraction (broad delamination glow penalty)
    raw_bw = (sub_dir > max(0.35, float(np.percentile(sub_dir[valid], 88)))) & valid
    rlbl, rn = label(raw_bw)
    rsizes = np.bincount(rlbl.ravel())[1:] if rn > 0 else np.array([], dtype=np.int64)
    giant_frac = float(rsizes[rsizes > 2200].sum()) / max(1, int(valid.sum()))

    # 4x4 grid spread of crisp DoG stroke energy
    gh, gw = dog.shape[0] // 4, dog.shape[1] // 4
    cell_hits = 0
    for r in range(4):
        for c in range(4):
            cell = dog[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw]
            if float(np.percentile(cell, 98)) > 0.06:
                cell_hits += 1

    dog_p99 = float(np.percentile(dog[valid], 99))
    dog_p95 = float(np.percentile(dog[valid], 95))
    score = (
        dog_p99
        * (0.5 + 4.0 * dog_p95)
        * (min(stroke_comps, 40) / 10.0)
        * (0.25 + cell_hits / 16.0)
        / (1.0 + 8.0 * giant_frac)
    )
    return float(score), {
        "n_strokes": stroke_comps,
        "cells": cell_hits,
        "dog99": round(dog_p99, 3),
        "dog95": round(dog_p95, 3),
        "giant": round(giant_frac, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    pdir = pathlib.Path(args.panels_dir)
    files = sorted(pdir.glob("*.png"))
    scored = []
    for p in files:
        sc, info = score_panel_png(p)
        scored.append((sc, p.stem, str(p), info))
    scored.sort(key=lambda x: x[0], reverse=True)

    out_prefix = pathlib.Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps([{"rank": i + 1, "id": s[1], "stroke_score": round(s[0], 5), **s[3]} for i, s in enumerate(scored[:50])], indent=2),
        encoding="utf-8",
    )

    print(f"Scored {len(files)} panels. Top 20 by thin-stroke morphology:")
    for i, (sc, jid, _, info) in enumerate(scored[:20], 1):
        print(f"  #{i:2d} {jid:32s} stroke={sc:.4f} {info}")

    # Save 4 clean 5-row montages (CT + reader_v3 + Directional cropped from each panel)
    top = scored[: args.top_k]
    for page in range(0, len(top), 5):
        chunk = top[page : page + 5]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(13, 3.3 * len(chunk)))
        if len(chunk) == 1:
            axes = axes[None, :]
        for r, (sc, jid, pstr, info) in enumerate(chunk):
            im = np.asarray(Image.open(pstr).convert("RGB"))
            H, W, _ = im.shape
            y0, y1 = int(H * 0.12), int(H * 0.94)
            axes[r, 0].imshow(im[y0:y1, int(W * 0.01) : int(W * 0.20)])
            axes[r, 0].set_title(f"#{page+r+1} {jid}\nCT", fontsize=9)
            axes[r, 1].imshow(im[y0:y1, int(W * 0.205) : int(W * 0.395)])
            axes[r, 1].set_title(f"reader_v3 (strokes={info.get('n_strokes')} cells={info.get('cells')})", fontsize=9)
            axes[r, 2].imshow(im[y0:y1, int(W * 0.805) : int(W * 0.995)])
            axes[r, 2].set_title(f"Directional (stroke_score={sc:.4f}, giant={info.get('giant')})", fontsize=9)
            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        fig.suptitle(f"Stroke-Ranked Top {page+1}-{page+len(chunk)}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_part{page // 5 + 1}.png", dpi=125)
        plt.close(fig)


if __name__ == "__main__":
    main()
