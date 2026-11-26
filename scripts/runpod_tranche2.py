"""Clean Runpod MCP lifecycle runner for Tranche 1B-Upright (PHerc1203 Z-derotated) + Tranche 2 (VC3D L0 harvest)."""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runpod_tranche1 as rt

OUT_DIR = ROOT / "outputs" / "tranche2"
POD_META = OUT_DIR / "active_pod_t2.json"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
VC3D_APPIMAGE_URL = "https://github.com/ScrollPrize/villa/releases/download/vc3d-archive-latest-456772923/VC3D-b949105-2026-11-24-linux-x86_64.AppImage"


def get_pod_id() -> str:
    return json.loads(POD_META.read_text(encoding="utf-8"))["id"]


def upload_file(pod_id: str, local_path: pathlib.Path, remote_path: str):
    b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    tmp = remote_path + ".b64"
    err, res = rt.exec_pod(pod_id, ["sh", "-c", f"mkdir -p '{pathlib.PurePosixPath(remote_path).parent}' && rm -f '{tmp}'"], timeout_sec=15)
    assert not err, res
    for i in range(0, len(b64), 3500):
        part = b64[i : i + 3500]
        err, res = rt.exec_pod(pod_id, ["sh", "-c", f"printf '%s' '{part}' >> '{tmp}'"], timeout_sec=15)
        assert not err, res
    err, res = rt.exec_pod(pod_id, ["sh", "-c", f"base64 -d '{tmp}' > '{remote_path}' && rm -f '{tmp}' && wc -l '{remote_path}'"], timeout_sec=15)
    assert not err, res
    print(f"UPLOADED {local_path.name} -> {remote_path}: {res.get('stdout','').strip()}", flush=True)


def cmd_create(_args):
    candidates = [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA A40",
        "NVIDIA RTX PRO 4500 Blackwell",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
        "NVIDIA RTX A6000",
        "NVIDIA L40S",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb",
    ]
    available = []
    for gid in candidates:
        err, txt = rt.mcp("get-gpu-type", {
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
    pod = None
    for _, price, gid, dcs, av in available[:3]:
        print(f"\nCREATING T2 POD: {gid} (${price}/hr, stock={av}, DCs={dcs}, disk=80GB, image={IMAGE})", flush=True)
        body = {
            "name": "vesuvius-tranche2-harvest",
            "cloud": "SECURE",
            "image": IMAGE,
            "disk": 80,
            "dataCenterIds": dcs,
            "gpu": {"id": gid, "count": 1, "minCudaVersion": "12.8"},
            "startJupyter": False,
            "startSsh": False,
        }
        is_err, txt = rt.mcp("create-pod", {"body": body})
        print("CREATE_POD isError=", is_err, txt[:900], flush=True)
        if not is_err:
            pod = json.loads(txt)
            pod["_picked_stock"] = av
            pod["_picked_price_hr"] = price
            break
    if pod is None:
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POD_META.write_text(json.dumps(pod, indent=2), encoding="utf-8")
    pod_id = pod["id"]
    print(f"\nT2 POD_ID: {pod_id} — polling until RUNNING + nvidia-smi answers...", flush=True)
    for i in range(60):
        time.sleep(5)
        err, gtxt = rt.mcp("get-pod", {"id": pod_id})
        if err:
            continue
        gp = json.loads(gtxt)
        st = gp.get("status") or gp.get("desiredStatus")
        dc = gp.get("dataCenterId")
        print(f"  [{(i+1)*5:3d}s] status={st} dc={dc}", flush=True)
        if st == "RUNNING":
            eerr, eres = rt.exec_pod(pod_id, ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout_sec=15)
            print("  EXEC_CHECK isError=", eerr, str(eres)[:300], flush=True)
            if not eerr and isinstance(eres, dict) and eres.get("exitCode") == 0:
                POD_META.write_text(json.dumps(gp, indent=2), encoding="utf-8")
                print("T2_POD_READY!", flush=True)
                return
    sys.exit(2)


def cmd_bootstrap(_args):
    pod_id = get_pod_id()
    for fname in ("fl_radar_sweep.py", "fl_sweep_s3_segments.py", "rerank_panels_strokes.py", "tranche2_vc3d_harvest.py"):
        upload_file(pod_id, ROOT / "scripts" / fname, f"/workspace/vesuvius/scripts/{fname}")

    setup_sh = f"""#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vesuvius
mkdir -p cache/models outputs/fl_sweep_1b_upright outputs/tranche2 /workspace/t2_scratch

echo "=== [1/5] GPU & Disk ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /workspace && nproc && free -h

echo "=== [2/5] Python deps + Villa sparse clone ==="
pip install -q --no-cache-dir timm einops imagecodecs zarr s3fs scipy matplotlib tifffile pillow huggingface_hub
if [ ! -d cache/villa_src/vesuvius/src ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/ScrollPrize/villa.git cache/villa_src
  git -C cache/villa_src sparse-checkout set vesuvius/src/vesuvius
fi

echo "=== [3/5] Download 3 independent 9 um models ==="
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
    shutil.copy2(hf_hub_download(repo_id=repo, filename=fname), target)
    print("Downloaded:", target, f"{{target.stat().st_size/1e6:.1f}} MB")
PY

echo "=== [4/5] Download & extract official VC3D Linux AppImage (vc_grow_seg_from_seed) ==="
if [ ! -d /workspace/vc3d_extracted/squashfs-root ]; then
  mkdir -p /workspace/vc3d_extracted
  curl -fsSL -L "{VC3D_APPIMAGE_URL}" -o /workspace/vc3d_extracted/VC3D.AppImage
  chmod +x /workspace/vc3d_extracted/VC3D.AppImage
  (cd /workspace/vc3d_extracted && ./VC3D.AppImage --appimage-extract > /dev/null)
  rm -f /workspace/vc3d_extracted/VC3D.AppImage
fi
cat <<'WRAP' > /workspace/vc3d_extracted/vc_grow_wrapper.sh
#!/usr/bin/env bash
exec /workspace/vc3d_extracted/squashfs-root/AppRun vc_grow_seg_from_seed "$@"
WRAP
chmod +x /workspace/vc3d_extracted/vc_grow_wrapper.sh
VC_BIN="/workspace/vc3d_extracted/vc_grow_wrapper.sh"
echo "FOUND VC_BIN: $VC_BIN"
"$VC_BIN" --help | head -n 20 || true

echo "=== [5A] Skip Tranche 1B-Upright if already complete ==="
if [ ! -f /workspace/vesuvius/outputs/fl_sweep_1b_upright/tranche1b_PHerc1203_leaderboard.json ]; then
  python3 -u scripts/fl_sweep_s3_segments.py \\
    --scroll PHerc1203 \\
    --workers 24 --batch-size 32 --max-chunks 4096 --bf16 \\
    --out /workspace/vesuvius/outputs/fl_sweep_1b_upright
  python3 -u scripts/rerank_panels_strokes.py \\
    --panels-dir /workspace/vesuvius/outputs/fl_sweep_1b_upright/panels \\
    --out-prefix /workspace/vesuvius/outputs/fl_sweep_1b_upright/stroke_top10 \\
    --top-k 10
fi
echo "=== T1B_UPRIGHT_OK ==="

echo "=== [5B] Tranche 2: Automated VC3D L0 harvest + Z-derotated 3-model sweep on 4 virgin scrolls ==="
python3 -u scripts/tranche2_vc3d_harvest.py \\
  --scrolls PHerc1545 PHerc0191 PHerc0175A PHerc0175B \\
  --vc-bin "$VC_BIN" \\
  --n-seeds 32 --generations 65 --grow-procs 4 --grow-timeout 360 \\
  --sync-workers 48 --workers 24 --batch-size 32 --max-chunks 4096 --bf16 \\
  --scratch /workspace/t2_scratch \\
  --out /workspace/vesuvius/outputs/tranche2
echo "=== ALL_TRANCHE2_COMPLETE_OK ==="
"""
    b64_sh = base64.b64encode(setup_sh.encode("utf-8")).decode("ascii")
    err, res = rt.exec_pod(pod_id, ["sh", "-c", "rm -f /workspace/vesuvius/run_tranche2.sh.b64"], timeout_sec=15)
    assert not err, res
    for i in range(0, len(b64_sh), 3500):
        part = b64_sh[i : i + 3500]
        err, res = rt.exec_pod(pod_id, ["sh", "-c", f"printf '%s' '{part}' >> /workspace/vesuvius/run_tranche2.sh.b64"], timeout_sec=15)
        assert not err, res
    err, res = rt.exec_pod(pod_id, ["sh", "-c", "base64 -d /workspace/vesuvius/run_tranche2.sh.b64 > /workspace/vesuvius/run_tranche2.sh && rm -f /workspace/vesuvius/run_tranche2.sh.b64 && chmod +x /workspace/vesuvius/run_tranche2.sh && nohup bash /workspace/vesuvius/run_tranche2.sh > /workspace/vesuvius/t2.log 2>&1 < /dev/null & sleep 2 && ps aux | grep run_tranche2 | grep -v grep && head -n 15 /workspace/vesuvius/t2.log"], timeout_sec=20)
    print("LAUNCH_T2 isError=", err, json.dumps(res, indent=2) if isinstance(res, dict) else res, flush=True)


def cmd_status(_args):
    pod_id = get_pod_id()
    err, gtxt = rt.mcp("get-pod", {"id": pod_id})
    gp = json.loads(gtxt) if not err else {"err": gtxt}
    print(f"T2 POD {pod_id}: status={gp.get('status') or gp.get('desiredStatus')} dc={gp.get('dataCenterId')}", flush=True)
    _, res = rt.exec_pod(pod_id, ["sh", "-c", "ps aux | grep -E 'run_tranche2|fl_sweep|tranche2_vc3d|vc_grow' | grep -v grep || echo NO_ACTIVE_T2_PROC; echo '--- DISK ---'; df -h /workspace | tail -n 1; echo '--- T2 LOG TAIL ---'; tail -n 30 /workspace/vesuvius/t2.log 2>/dev/null || true"], timeout_sec=20)
    print(res.get("stdout", res) if isinstance(res, dict) else res, flush=True)


def cmd_download(_args):
    pod_id = get_pod_id()
    # Pull PHerc1203 upright results + all completed Tranche 2 scroll leaderboards & montages
    _, res = rt.exec_pod(pod_id, ["sh", "-c", "find /workspace/vesuvius/outputs/fl_sweep_1b_upright /workspace/vesuvius/outputs/tranche2 -maxdepth 2 \\( -name '*.json' -o -name '*top_candidates.png' -o -name '*stroke_top10*.png' \\) | sort"], timeout_sec=15)
    files = [l.strip() for l in (res.get("stdout", "") if isinstance(res, dict) else "").splitlines() if l.strip()]
    print(f"Found {len(files)} result artifacts to pull:", files, flush=True)
    for rpath in files:
        rel = rpath.replace("/workspace/vesuvius/outputs/", "")
        rt._pull_file(pod_id, rpath, ROOT / "outputs" / pathlib.Path(rel))
    rt._pull_file(pod_id, "/workspace/vesuvius/t2.log", OUT_DIR / "t2.log")


def cmd_delete(_args):
    pod_id = get_pod_id()
    err, txt = rt.mcp("delete-pod", {"id": pod_id})
    print("DELETE_T2_POD isError=", err, txt[:400], flush=True)
    time.sleep(2)
    _, ltxt = rt.mcp("list-pods", {})
    print("VERIFY list-pods:", ltxt[:400], flush=True)
    _, btxt = rt.mcp("get-balance", {})
    print("VERIFY get-balance:", btxt[:300], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("create", "bootstrap", "status", "download", "delete"):
        sub.add_parser(c)
    args = ap.parse_args()
    {"create": cmd_create, "bootstrap": cmd_bootstrap, "status": cmd_status, "download": cmd_download, "delete": cmd_delete}[args.cmd](args)


if __name__ == "__main__":
    main()
