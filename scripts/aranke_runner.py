"""Reusable fast BF16/FP32 runner for Priyank Aranke's $50k-winning studio-softloss-20261023.

Runs cleanly on 1024x1024 candidate tiles (either hits where reader-v3 >= thresh, or matched
reader-v3 negative tiles for disagreement mining) and saves crisp side-by-side comparison PNGs.
"""
from __future__ import annotations

import pathlib
import sys
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _hann96() -> np.ndarray:
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96)
    return (w[:, None] * w[None, :]).astype(np.float32)


def _crop128(arr: np.ndarray, y: int, x: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    tile = np.asarray(arr[:, max(0, y) : min(h, y + 128), max(0, x) : min(w, x + 128)])
    pad = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
    return np.pad(tile, pad, mode="reflect") if any(a or b for a, b in pad) else tile


class ArankeStudioRunner:
    def __init__(self, model_dir: pathlib.Path | None = None, bf16: bool = True):
        mdir = model_dir or (ROOT / "cache" / "models" / "aranke_studio")
        if str(mdir) not in sys.path:
            sys.path.insert(0, str(mdir))
        from model import DeepResidualUNet

        ckpt = mdir / "studio-softloss-20261023-ema.safetensors"
        assert ckpt.exists(), f"Missing Aranke checkpoint: {ckpt}"
        self.model = DeepResidualUNet(deeper=False)
        self.model.load_state_dict(load_file(str(ckpt), device="cpu"), strict=True)
        self.model = self.model.eval().to("cuda")
        self.use_bf16 = bool(bf16 and torch.cuda.is_bf16_supported())
        self.kern = _hann96()

    @torch.inference_mode()
    def infer_one_order(self, vol16: np.ndarray, mask: np.ndarray, batch_size: int = 16, stride: int = 64) -> np.ndarray:
        h, w = vol16.shape[-2:]
        res = np.zeros((h, w), dtype=np.float32)
        wt = np.zeros((h, w), dtype=np.float32)
        ys = sorted(set(list(range(0, h, stride)) + [max(0, h - 96)]))
        xs = sorted(set(list(range(0, w, stride)) + [max(0, w - 96)]))
        coords = [(y, x) for y in ys for x in xs if mask[y : y + 96, x : x + 96].any()]
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext()
        for i in range(0, len(coords), batch_size):
            batch = coords[i : i + batch_size]
            raw = np.stack([_crop128(vol16, y - 16, x - 16) for y, x in batch], axis=0)
            x = torch.from_numpy(raw[:, None]).to("cuda", dtype=torch.float32, non_blocking=True)
            x.clamp_(0, 200).div_(255.0)
            with ctx:
                logits = self.model(x)
            probs = logits[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
            for (y, x0), p in zip(batch, probs):
                hh, ww = min(96, h - y), min(96, w - x0)
                res[y : y + hh, x0 : x0 + ww] += p[:hh, :ww] * self.kern[:hh, :ww]
                wt[y : y + hh, x0 : x0 + ww] += self.kern[:hh, :ww]
        out = np.divide(res, wt, out=np.zeros_like(res), where=wt > 1e-12)
        out[~mask] = 0.0
        return out

    def score_tile(
        self,
        slab17: np.ndarray,
        mask: np.ndarray,
        reader_best_tile_map: np.ndarray,
        vum: float,
        batch_size: int = 16,
        stride: int = 64,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Run both depth orders of Aranke studio-softloss on one tile and compute stroke score + agreement."""
        import stroke_scorer as ss

        sf = self.infer_one_order(slab17[:16], mask, batch_size=batch_size, stride=stride)
        sr = self.infer_one_order(np.ascontiguousarray(slab17[1:][::-1]), mask, batch_size=batch_size, stride=stride)
        sc = ss.score_full_mesh_tiles(sf, sr, mask, vum, win_px=min(1024, mask.shape[0], mask.shape[1]), stride_px=512, top_k=1)["best_tile"]
        s_best = sf if (sc and sc["side"] == "fwd") else sr
        s_opp = sr if (sc and sc["side"] == "fwd") else sf
        s_dir = np.clip(s_best - 0.5 * s_opp, 0.0, 1.0)

        m = mask.astype(bool)
        if m.any() and reader_best_tile_map[m].std() > 1e-5:
            r_ff = float(np.corrcoef(reader_best_tile_map[m], sf[m])[0, 1]) if sf[m].std() > 1e-5 else 0.0
            r_fr = float(np.corrcoef(reader_best_tile_map[m], sr[m])[0, 1]) if sr[m].std() > 1e-5 else 0.0
            r_agree = max(r_ff, r_fr)
        else:
            r_agree = 0.0

        info = {
            "studio_score": sc["stroke_score"] if sc else 0.0,
            "studio_dir": sc["dir_score"] if sc else 0.0,
            "studio_raw": sc["raw_score"] if sc else 0.0,
            "studio_n": sc["n_strokes"] if sc else 0,
            "studio_cells": sc["cells"] if sc else 0,
            "studio_p99": sc["p99"] if sc else 0.0,
            "studio_side": sc["side"] if sc else "?",
            "r_reader_studio": round(r_agree, 4),
        }
        return info, s_best, s_dir


def save_aranke_confirm_panel(
    out_path: pathlib.Path,
    title: str,
    ct_tile: np.ndarray,
    reader_tile: np.ndarray,
    studio_tile: np.ndarray,
    studio_dir: np.ndarray,
    r_score: float,
    s_info: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    lo, hi = np.percentile(ct_tile[ct_tile > 0], [2, 98]) if (ct_tile > 0).any() else (0, 255)
    axes[0].imshow(ct_tile, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Upright CT", fontsize=9)
    axes[1].imshow(reader_tile, cmap="gray", vmin=0, vmax=max(0.45, float(np.percentile(reader_tile, 99.5))))
    axes[1].set_title(f"reader-v3\nstroke={r_score:.3f}", fontsize=9)
    axes[2].imshow(studio_tile, cmap="gray", vmin=0, vmax=max(0.45, float(np.percentile(studio_tile, 99.5))))
    axes[2].set_title(f"Aranke studio-softloss\nstroke={s_info['studio_score']:.3f} r={s_info['r_reader_studio']:.2f}", fontsize=9)
    axes[3].imshow(studio_dir, cmap="gray", vmin=0, vmax=max(0.40, float(np.percentile(studio_dir, 99.5))))
    axes[3].set_title(f"Studio Directional\nn={s_info['studio_n']} c={s_info['studio_cells']} p99={s_info['studio_p99']:.2f}", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
