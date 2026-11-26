"""Verify exported WindingInferenceStore v1 against REAL upstream ScrollPrize/villa/spiral-fitting/winding_supervision.py."""

from __future__ import annotations

import json
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np

UPSTREAM_URL = "https://raw.githubusercontent.com/ScrollPrize/villa/main/spiral-fitting/winding_supervision.py"


def load_upstream_store_class():
    code = urllib.request.urlopen(UPSTREAM_URL, timeout=20).read().decode("utf-8")
    for mod_name in ("loss_maps", "sample_spiral"):
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.record_loss_samples = lambda *a, **k: None
            m.get_theta_and_radii = lambda *a, **k: None
            m.unwrap_shifted_radii = lambda *a, **k: None
            sys.modules[mod_name] = m
    ns: dict = {}
    exec(compile(code, "winding_supervision.py", "exec"), ns)
    return ns["WindingInferenceStore"]


def verify_store(store_dir: Path, umbilicus_json: Path) -> None:
    WindingInferenceStore = load_upstream_store_class()
    with open(umbilicus_json, "r", encoding="utf-8") as f:
        u_pts = np.array([[p["x"], p["y"], p["z"]] for p in json.load(f)["control_points"]], dtype=np.float64)
    order = np.argsort(u_pts[:, 2])
    u_pts = u_pts[order]

    def z_to_umb_yx(z_arr):
        z_arr = np.asarray(z_arr, dtype=np.float64)
        return np.column_stack([
            np.interp(z_arr, u_pts[:, 2], u_pts[:, 1]),
            np.interp(z_arr, u_pts[:, 2], u_pts[:, 0]),
        ])

    store = WindingInferenceStore(str(store_dir), device="cpu", verify=True, z_to_umbilicus_yx=z_to_umb_yx)
    assert len(store.origin) > 0, "No rays loaded"
    assert store.num_inward_rays == 0, f"Unexpected inward rays: {store.num_inward_rays}"
    assert len(store.density_rays) == len(store.origin), "Not all rays eligible for density sampling"

    adj = store.sample_adjacent(32)
    rel = store.sample_relative(32, min_delta=2, max_delta=10)
    assert isinstance(adj, dict) and isinstance(rel, dict)
    print(f"[PASS] Upstream WindingInferenceStore verified cleanly on {store_dir}:")
    print(f"       rays={len(store.origin):,}, crossings={len(store.crossing_t):,}, inward_rays={store.num_inward_rays}")
    print(f"       sample_adjacent(32): { {k: tuple(v.shape) for k, v in adj.items()} }")
    print(f"       sample_relative(32, 2..10): { {k: tuple(v.shape) for k, v in rel.items()} }")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    verify_store(root / "outputs" / "paris4_winding_inference_v3", root / "cache" / "paris4_spiral" / "umbilicus.json")
