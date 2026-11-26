"""Clean Runpod MCP lifecycle runner for First-Letters Tranche 1 sweep.

Reads RUNPOD_API_KEY from the environment (never prints or logs it).
Subcommands:
  create    - refresh live stock, create 1 Secure Cloud Pod, wait until nvidia-smi answers
  bootstrap - upload fl_radar_sweep.py, install deps, clone villa + eligible_meshes,
              download 3 models, and launch tranche1 sweep in background
  status    - tail sweep log + read current leaderboard progress + pod status
  download  - pull tranche1_leaderboard.json, montage PNG, and top panels locally
  delete    - delete the Pod and verify via list-pods that 0 pods remain
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "fl_sweep"
POD_META = OUT_DIR / "active_pod.json"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


def _load_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set the RUNPOD_API_KEY environment variable before running.")
    return key


def mcp(name: str, args: dict | None = None, timeout: int = 35) -> tuple[bool, str]:
    key = _load_key()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args or {}}}
    req = urllib.request.Request(
        "https://mcp.runpod.io/",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read().decode("utf-8"))["result"]
    txt = res["content"][0]["text"]
    return bool(res.get("isError", False)), txt


def exec_pod(pod_id: str, cmd: list[str], timeout_sec: int = 25, cwd: str = "/workspace") -> tuple[bool, dict | str]:
    err, txt = mcp("exec-pod", {"id": pod_id, "body": {"command": cmd, "timeoutSeconds": timeout_sec, "cwd": cwd}}, timeout=timeout_sec + 15)
    try:
        return err, json.loads(txt)
    except Exception:
        return err, txt


def get_pod_id() -> str:
    return json.loads(POD_META.read_text(encoding="utf-8"))["id"]


def cmd_create(_args):
    candidates = [
        "NVIDIA RTX PRO 4500 Blackwell",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
        "NVIDIA A40",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA L40S",
        "NVIDIA RTX A6000",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb",
    ]
    available = []
    for gid in candidates:
        err, txt = mcp("get-gpu-type", {
            "id": gid, "include": ["AVAILABILITY"], "product": ["POD"],
            "cloud": "SECURE", "minCudaVersion": "12.8",
        })
        if err:
            continue
        g = json.loads(txt)
        dcs = g.get("dataCenters") or []
        us_dcs = [d["id"] for d in dcs if d["id"].startswith(("US-", "CA-")) and d.get("availability") not in (None, "NONE", "OUT")]
        all_dcs = [d["id"] for d in dcs if d.get("availability") not in (None, "NONE", "OUT")]
        price = (g.get("price") or {}).get("secure")
        av = g.get("availability")
        print(f"STOCK: {gid:<55s} | ${price}/hr | stock={av} | US/CA={us_dcs} | ALL={all_dcs}", flush=True)
        if us_dcs:
            available.append((0, price or 99, gid, us_dcs, av))
        elif all_dcs:
            available.append((1, price or 99, gid, all_dcs, av))

    if not available:
        print("ERROR: No GPU candidates in stock.", flush=True)
        sys.exit(1)

    available.sort()
    # Try top candidate (and if 400 race on LOW stock, try next available once)
    pod = None
    for _, price, gid, dcs, av in available[:3]:
        print(f"\nCREATING POD: {gid} (${price}/hr, stock={av}, DCs={dcs}, image={IMAGE})", flush=True)
        body = {
            "name": "vesuvius-tranche1-sweep",
            "cloud": "SECURE",
            "image": IMAGE,
            "disk": 30,
            "dataCenterIds": dcs,
            "gpu": {"id": gid, "count": 1, "minCudaVersion": "12.8"},
            "startJupyter": False,
            "startSsh": False,
        }
        is_err, txt = mcp("create-pod", {"body": body})
        print("CREATE_POD isError=", is_err, flush=True)
        print(txt[:1200], flush=True)
        if not is_err:
            pod = json.loads(txt)
            pod["_picked_stock"] = av
            pod["_picked_price_hr"] = price
            pod["_picked_dcs"] = dcs
            break
    if pod is None:
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POD_META.write_text(json.dumps(pod, indent=2), encoding="utf-8")
    pod_id = pod["id"]
    print(f"\nPOD_ID: {pod_id} — polling until RUNNING + nvidia-smi answers...", flush=True)

    for i in range(60):
        time.sleep(5)
        err, gtxt = mcp("get-pod", {"id": pod_id})
        if err:
            print(f"  [{(i+1)*5:3d}s] get-pod err: {gtxt[:200]}", flush=True)
            continue
        gp = json.loads(gtxt)
        st = gp.get("status") or gp.get("desiredStatus")
        dc = gp.get("dataCenterId") or gp.get("machine", {}).get("dataCenterId")
        print(f"  [{(i+1)*5:3d}s] status={st} dc={dc} costPerHr={gp.get('costPerHr')}", flush=True)
        if st == "RUNNING":
            eerr, eres = exec_pod(pod_id, ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout_sec=15)
            print("  EXEC_CHECK isError=", eerr, str(eres)[:300], flush=True)
            if not eerr and isinstance(eres, dict) and eres.get("exitCode") == 0:
                POD_META.write_text(json.dumps(gp, indent=2), encoding="utf-8")
                print("POD_READY!", flush=True)
                return
    print("WARNING: Pod did not become exec-ready within 5 min.", flush=True)
    sys.exit(2)


def cmd_bootstrap(args):
    pod_id = get_pod_id()
    script_bytes = (ROOT / "scripts" / "fl_radar_sweep.py").read_bytes()
    b64 = base64.b64encode(script_bytes).decode("ascii")

    # 1. Create workspace structure and upload fl_radar_sweep.py in 16 KB chunks
    print(f"Uploading fl_radar_sweep.py ({len(script_bytes)} bytes) to pod {pod_id}...", flush=True)
    err, res = exec_pod(pod_id, ["sh", "-c", "mkdir -p /workspace/vesuvius/scripts /workspace/vesuvius/cache/models /workspace/vesuvius/outputs/fl_sweep && rm -f /workspace/vesuvius/scripts/fl_radar_sweep.py.b64"], timeout_sec=15)
    assert not err, res
    chunk_sz = 3500
    for i in range(0, len(b64), chunk_sz):
        part = b64[i : i + chunk_sz]
        err, res = exec_pod(pod_id, ["sh", "-c", f"printf '%s' '{part}' >> /workspace/vesuvius/scripts/fl_radar_sweep.py.b64"], timeout_sec=15)
        assert not err, res
    err, res = exec_pod(pod_id, ["sh", "-c", "base64 -d /workspace/vesuvius/scripts/fl_radar_sweep.py.b64 > /workspace/vesuvius/scripts/fl_radar_sweep.py && rm -f /workspace/vesuvius/scripts/fl_radar_sweep.py.b64 && wc -l /workspace/vesuvius/scripts/fl_radar_sweep.py"], timeout_sec=15)
    print("UPLOAD:", res, flush=True)

    # 2. Write & launch background bootstrap + sweep script so exec-pod doesn't time out during pip/git/HF downloads
    precision_flag = "--bf16" if args.bf16 else ("--fp16" if args.fp16 else "")
    max_dim = args.max_dim
    workers = args.workers
    batch_size = args.batch_size
    max_chunks = args.max_chunks
    panel_thresh = args.panel_thresh

    setup_sh = f"""#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vesuvius
echo "=== [1/4] System & GPU check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nproc && free -h

echo "=== [2/4] Install Python deps ==="
pip install -q --no-cache-dir timm einops imagecodecs zarr s3fs scipy matplotlib tifffile pillow huggingface_hub

echo "=== [3/4] Clone villa (sparse) + eligible_meshes (shallow) ==="
if [ ! -d cache/villa_src/vesuvius/src ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/ScrollPrize/villa.git cache/villa_src
  git -C cache/villa_src sparse-checkout set vesuvius/src/vesuvius
fi
if [ ! -f cache/eligible_meshes/data/index.csv ]; then
  git clone --depth 1 https://github.com/pscamillo/vesuvius-eligible-meshes.git cache/eligible_meshes
fi

echo "=== [4/4] Download 3 independent 9 um models from HuggingFace ==="
python3 - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

dst = Path("/workspace/vesuvius/cache/models")
dst.mkdir(parents=True, exist_ok=True)
specs = [
    ("domenicor046/reader-v3", "reader-v3.pth", "reader-v3.pth"),
    ("ruiconte/argus-v3", "argus-v3.pth", "argus-v3.pth"),
    ("scrollprize/ink_9um", "hybrid_3d2d-seed42/step-075000.pth", "ink_9um_seed42_step075000.pth"),
]
for repo, fname, out_name in specs:
    target = dst / out_name
    if target.exists() and target.stat().st_size > 10_000_000:
        print("Cached:", target)
        continue
    p = hf_hub_download(repo_id=repo, filename=fname)
    shutil.copy2(p, target)
    print("Downloaded:", target, f"{{target.stat().st_size/1e6:.1f}} MB")
PY

echo "=== [RUN] Launching Tranche 1 sweep (528 unclaimed meshes) ==="
python3 -u scripts/fl_radar_sweep.py \\
  --mode tranche1 \\
  --workers {workers} \\
  --batch-size {batch_size} \\
  --max-chunks {max_chunks} \\
  --max-dim {max_dim} \\
  --panel-thresh {panel_thresh} \\
  --top-k-montage 20 \\
  {precision_flag} \\
  --out /workspace/vesuvius/outputs/fl_sweep
echo "=== SWEEP_COMPLETE_OK ==="
"""
    b64_sh = base64.b64encode(setup_sh.encode("utf-8")).decode("ascii")
    err, res = exec_pod(pod_id, ["sh", "-c", "rm -f /workspace/vesuvius/run_tranche1.sh.b64"], timeout_sec=15)
    assert not err, res
    for i in range(0, len(b64_sh), 3500):
        part = b64_sh[i : i + 3500]
        err, res = exec_pod(pod_id, ["sh", "-c", f"printf '%s' '{part}' >> /workspace/vesuvius/run_tranche1.sh.b64"], timeout_sec=15)
        assert not err, res
    err, res = exec_pod(pod_id, ["sh", "-c", "base64 -d /workspace/vesuvius/run_tranche1.sh.b64 > /workspace/vesuvius/run_tranche1.sh && rm -f /workspace/vesuvius/run_tranche1.sh.b64 && chmod +x /workspace/vesuvius/run_tranche1.sh && nohup bash /workspace/vesuvius/run_tranche1.sh > /workspace/vesuvius/sweep.log 2>&1 < /dev/null & sleep 2 && ps aux | grep run_tranche1 | grep -v grep && head -n 20 /workspace/vesuvius/sweep.log"], timeout_sec=20)
    print("LAUNCH isError=", err, json.dumps(res, indent=2) if isinstance(res, dict) else res, flush=True)


def cmd_status(_args):
    pod_id = get_pod_id()
    err, gtxt = mcp("get-pod", {"id": pod_id})
    gp = json.loads(gtxt) if not err else {"err": gtxt}
    print(f"POD {pod_id}: status={gp.get('status') or gp.get('desiredStatus')} costPerHr={gp.get('costPerHr')}", flush=True)
    eerr, res = exec_pod(pod_id, ["sh", "-c", "ps aux | grep -E 'run_tranche1|fl_radar_sweep' | grep -v grep || echo 'NO_ACTIVE_SWEEP_PROC'; echo '--- PROGRESS ---'; python3 -c \"import json, pathlib; p=pathlib.Path('/workspace/vesuvius/outputs/fl_sweep/tranche1_leaderboard.json'); d=json.loads(p.read_text()) if p.exists() else None; print('DONE:', d['completed'], '/', d['total_jobs'], 'elapsed:', d['total_seconds'], 's top5:', [(r['id'], (r.get('best_window') or {}).get('radar_score')) for r in d['leaderboard'][:5]]) if d else print('NO_LB_YET')\"; echo '--- LOG TAIL ---'; tail -n 25 /workspace/vesuvius/sweep.log 2>/dev/null || true"], timeout_sec=20)
    if isinstance(res, dict):
        print(res.get("stdout", ""), flush=True)
        if res.get("stderr"):
            print("STDERR:", res["stderr"], flush=True)
    else:
        print(res, flush=True)


def _pull_file(pod_id: str, remote_path: str, local_path: pathlib.Path):
    err, res = exec_pod(pod_id, ["sh", "-c", f"stat -c %s '{remote_path}'"], timeout_sec=15)
    assert not err and isinstance(res, dict) and res.get("exitCode") == 0, f"stat failed: {res}"
    size = int(res["stdout"].strip())
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Stream in 24 KB raw chunks via dd | base64 (safe for exec-pod stdout limits)
    chunk = 24576
    buf = bytearray()
    offset = 0
    while offset < size:
        for attempt in range(5):
            err, res = exec_pod(pod_id, ["sh", "-c", f"dd if='{remote_path}' bs={chunk} skip={offset // chunk} count=1 status=none | base64 -w 0"], timeout_sec=20)
            if not err and isinstance(res, dict) and res.get("exitCode") == 0 and not res.get("truncated"):
                break
            if attempt == 4:
                raise AssertionError(f"read failed at {offset}: {res}")
            time.sleep(1.5 * (attempt + 1))
        buf.extend(base64.b64decode(res["stdout"].strip()))
        offset += chunk
    local_path.write_bytes(bytes(buf))
    print(f"PULLED {remote_path} -> {local_path} ({len(buf):,} bytes)", flush=True)


def cmd_download(args):
    pod_id = get_pod_id()
    dl_dir = OUT_DIR / "tranche1"
    dl_dir.mkdir(parents=True, exist_ok=True)
    _pull_file(pod_id, "/workspace/vesuvius/outputs/fl_sweep/tranche1_leaderboard.json", dl_dir / "tranche1_leaderboard.json")
    _pull_file(pod_id, "/workspace/vesuvius/outputs/fl_sweep/tranche1_top_candidates.png", dl_dir / "tranche1_top_candidates.png")
    _pull_file(pod_id, "/workspace/vesuvius/sweep.log", dl_dir / "sweep.log")

    # Pull top-N individual panels too
    lb = json.loads((dl_dir / "tranche1_leaderboard.json").read_text(encoding="utf-8"))
    top_ids = [r["id"] for r in lb.get("leaderboard", [])[: args.top_panels]]
    pdir = dl_dir / "panels"
    pdir.mkdir(parents=True, exist_ok=True)
    for jid in top_ids:
        rpath = f"/workspace/vesuvius/outputs/fl_sweep/panels/{jid}.png"
        err, res = exec_pod(pod_id, ["test", "-f", rpath], timeout_sec=10)
        if not err and isinstance(res, dict) and res.get("exitCode") == 0:
            _pull_file(pod_id, rpath, pdir / f"{jid}.png")


def cmd_delete(_args):
    pod_id = get_pod_id()
    err, txt = mcp("delete-pod", {"id": pod_id})
    print("DELETE_POD isError=", err, txt[:500], flush=True)
    time.sleep(2)
    lerr, ltxt = mcp("list-pods", {})
    print("VERIFY list-pods isError=", lerr, ltxt[:500], flush=True)
    b_err, b_txt = mcp("get-balance", {})
    print("VERIFY get-balance isError=", b_err, b_txt[:300], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create")
    bp = sub.add_parser("bootstrap")
    bp.add_argument("--bf16", action="store_true", default=True)
    bp.add_argument("--fp16", action="store_true")
    bp.add_argument("--max-dim", type=int, default=0)
    bp.add_argument("--workers", type=int, default=24)
    bp.add_argument("--batch-size", type=int, default=32)
    bp.add_argument("--max-chunks", type=int, default=4096)
    bp.add_argument("--panel-thresh", type=float, default=0.38)
    sub.add_parser("status")
    dp = sub.add_parser("download")
    dp.add_argument("--top-panels", type=int, default=15)
    sub.add_parser("delete")
    args = ap.parse_args()
    {"create": cmd_create, "bootstrap": cmd_bootstrap, "status": cmd_status, "download": cmd_download, "delete": cmd_delete}[args.cmd](args)


if __name__ == "__main__":
    main()
