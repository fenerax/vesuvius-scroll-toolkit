"""Clean Runpod MCP lifecycle runner for Gate 1 (PHerc0211 blind recall) & Gate 2 (full-map sweeps).

Supports:
  - gate1_211: Blind recall test on all 90 PHerc0211 meshes (with optional --recenter-ct)
  - gate2a_eligible: Full-map 100%-tile reader-v3 re-sweep of on-sheet eligible_meshes
                     (PHerc0125 on_sheet>=0.70, PHerc0813, PHerc0358, PHerc0257, PHerc0826, PHerc0800)
  - gate2b_harvest: Fresh VC3D L0 harvest + full-map 100%-tile sweep + tifxyz archival on
                    lightly-covered 113keV (PHerc0257, 0358, 0826) & untouched 116keV
                    (PHerc0483A/B, 0490A/B, 1218) scrolls.
"""
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

IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
VC3D_APPIMAGE_URL = "https://github.com/ScrollPrize/villa/releases/download/vc3d-archive-latest-456772923/VC3D-b949105-2026-11-24-linux-x86_64.AppImage"


def pod_meta_path(tag: str) -> pathlib.Path:
    p = ROOT / "outputs" / tag / f"active_pod_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


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


def upload_text(pod_id: str, content: str, remote_path: str):
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    tmp = remote_path + ".b64"
    err, res = rt.exec_pod(pod_id, ["sh", "-c", f"mkdir -p '{pathlib.PurePosixPath(remote_path).parent}' && rm -f '{tmp}'"], timeout_sec=15)
    assert not err, res
    for i in range(0, len(b64), 3500):
        part = b64[i : i + 3500]
        err, res = rt.exec_pod(pod_id, ["sh", "-c", f"printf '%s' '{part}' >> '{tmp}'"], timeout_sec=15)
        assert not err, res
    err, res = rt.exec_pod(pod_id, ["sh", "-c", f"base64 -d '{tmp}' > '{remote_path}' && rm -f '{tmp}' && chmod +x '{remote_path}'"], timeout_sec=15)
    assert not err, res


def cmd_create(args):
    candidates = [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA A40",
        "NVIDIA RTX PRO 4500 Blackwell",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
        "NVIDIA RTX A6000",
        "NVIDIA L40S",
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
        print(f"\nCREATING POD [{args.tag}]: {gid} (${price}/hr, stock={av}, DCs={dcs}, disk={args.disk}GB)", flush=True)
        body = {
            "name": f"vesuvius-{args.tag}",
            "cloud": "SECURE",
            "image": IMAGE,
            "disk": args.disk,
            "dataCenterIds": dcs,
            "gpu": {"id": gid, "count": 1, "minCudaVersion": "12.8"},
            "startJupyter": False,
            "startSsh": False,
        }
        is_err, txt = rt.mcp("create-pod", {"body": body})
        print("CREATE_POD isError=", is_err, txt[:700], flush=True)
        if not is_err:
            pod = json.loads(txt)
            pod["_picked_stock"] = av
            pod["_picked_price_hr"] = price
            break
    if pod is None:
        sys.exit(1)

    meta_p = pod_meta_path(args.tag)
    meta_p.write_text(json.dumps(pod, indent=2), encoding="utf-8")
    pod_id = pod["id"]
    print(f"\nPOD_ID [{args.tag}]: {pod_id} — polling until RUNNING + nvidia-smi answers...", flush=True)
    for i in range(60):
        time.sleep(5)
        err, gtxt = rt.mcp("get-pod", {"id": pod_id})
        if err:
            continue
        gp = json.loads(gtxt)
        st = gp.get("status") or gp.get("desiredStatus")
        print(f"  [{(i+1)*5:3d}s] status={st} dc={gp.get('dataCenterId')}", flush=True)
        if st == "RUNNING":
            eerr, eres = rt.exec_pod(pod_id, ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout_sec=15)
            print("  EXEC_CHECK isError=", eerr, str(eres)[:300], flush=True)
            if not eerr and isinstance(eres, dict) and eres.get("exitCode") == 0:
                meta_p.write_text(json.dumps(gp, indent=2), encoding="utf-8")
                print(f"POD_READY [{args.tag}]!", flush=True)
                return
    sys.exit(2)


def cmd_launch(args):
    pod_id = json.loads(pod_meta_path(args.tag).read_text(encoding="utf-8"))["id"]
    for fname in ("aranke_runner.py", "stroke_scorer.py", "fl_radar_sweep.py", "tranche2_vc3d_harvest.py"):
        upload_file(pod_id, ROOT / "scripts" / fname, f"/workspace/vesuvius/scripts/{fname}")

    need_eligible = args.job in ("gate1_211", "gate2a_eligible", "run1_0125")
    need_vc3d = args.job == "gate2b_harvest"

    eligible_block = ""
    if need_eligible:
        git_cfg = (ROOT / "cache" / "eligible_meshes" / ".git" / "config").read_text(encoding="utf-8")
        repo_url = ""
        for line in git_cfg.splitlines():
            if "url =" in line:
                repo_url = line.split("=", 1)[1].strip()
                break
        eligible_block = f"""
if [ ! -d cache/eligible_meshes/data ]; then
  git clone --depth 1 "{repo_url}" cache/eligible_meshes
fi
"""

    vc3d_block = f"""
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
""" if need_vc3d else ""

    recenter_flag = "--recenter-ct" if args.recenter_ct else ""
    if args.job == "gate1_211":
        run_cmd = (
            f"python3 -u scripts/fl_radar_sweep.py --mode pherc0211 "
            f"--fast-reader-only --confirm-thresh 0.10 --aranke-thresh 0.07 --save-full-maps {recenter_flag} "
            f"--workers 24 --batch-size 32 --max-chunks 4096 --bf16 --top-k-montage 25 "
            f"--out /workspace/vesuvius/outputs/{args.tag}"
        )
    elif args.job == "run1_0125":
        run_cmd = (
            f"python3 -u scripts/fl_radar_sweep.py --mode tranche1 --scrolls PHerc0125 --only-nov17 "
            f"--fast-reader-only --confirm-thresh 0.10 --aranke-thresh 0.07 --aranke-disagree-samples 20 "
            f"--save-full-maps --recenter-ct --workers 24 --batch-size 32 --max-chunks 4096 --bf16 --top-k-montage 25 "
            f"--out /workspace/vesuvius/outputs/{args.tag}"
        )
    elif args.job == "gate2a_eligible":
        scrolls = " ".join(args.scrolls or ["PHerc0125", "PHerc0813", "PHerc0358", "PHerc0257", "PHerc0826", "PHerc0800"])
        run_cmd = (
            f"python3 -u scripts/fl_radar_sweep.py --mode tranche1 --scrolls {scrolls} "
            f"--min-on-sheet 0.70 --fast-reader-only --confirm-thresh 0.10 --aranke-thresh 0.07 --save-full-maps {recenter_flag} "
            f"--workers 24 --batch-size 32 --max-chunks 4096 --bf16 --top-k-montage 30 "
            f"--out /workspace/vesuvius/outputs/{args.tag}"
        )
    else:
        scrolls = " ".join(args.scrolls or ["PHerc0257", "PHerc0358", "PHerc0826", "PHerc0483A", "PHerc0483B", "PHerc0490A", "PHerc0490B", "PHerc1218"])
        run_cmd = (
            f"python3 -u scripts/tranche2_vc3d_harvest.py --scrolls {scrolls} "
            f"--vc-bin /workspace/vc3d_extracted/vc_grow_wrapper.sh "
            f"--n-seeds 32 --generations 65 --grow-procs 4 --grow-timeout 360 "
            f"--fast-reader-only --confirm-thresh 0.10 --aranke-thresh 0.07 --save-full-maps "
            f"--sync-workers 48 --workers 24 --batch-size 32 --max-chunks 4096 --bf16 "
            f"--scratch /workspace/t2_scratch --out /workspace/vesuvius/outputs/{args.tag}"
        )

    setup_sh = f"""#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vesuvius
mkdir -p cache/models/aranke_studio outputs/{args.tag}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
pip install -q --no-cache-dir timm einops imagecodecs zarr s3fs scipy matplotlib tifffile pillow huggingface_hub safetensors
if [ ! -d cache/villa_src/vesuvius/src ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/ScrollPrize/villa.git cache/villa_src
  git -C cache/villa_src sparse-checkout set vesuvius/src/vesuvius
fi
{eligible_block}
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
        continue
    shutil.copy2(hf_hub_download(repo_id=repo, filename=fname), target)
    print("Downloaded:", target, f"{{target.stat().st_size/1e6:.1f}} MB")
PY
if [ ! -f cache/models/aranke_studio/model.py ]; then
  curl -fsSL https://raw.githubusercontent.com/priyankaranke/vesuvius-studio-multiscroll-cpt/main/studio-multiscroll-cpt-20261023/hf/model.py -o cache/models/aranke_studio/model.py
fi
if [ ! -f cache/models/aranke_studio/studio-softloss-20261023-ema.safetensors ]; then
  curl -fsSL https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/studio-softloss-20261023/studio-softloss-20261023-ema.safetensors -o cache/models/aranke_studio/studio-softloss-20261023-ema.safetensors
fi
{vc3d_block}
echo "=== STARTING {args.job} ==="
{run_cmd}
echo "=== JOB_{args.tag}_COMPLETE_OK ==="
"""
    upload_text(pod_id, setup_sh, f"/workspace/vesuvius/run_{args.tag}.sh")
    err, res = rt.exec_pod(
        pod_id,
        ["sh", "-c", f"nohup bash /workspace/vesuvius/run_{args.tag}.sh > /workspace/vesuvius/{args.tag}.log 2>&1 < /dev/null & sleep 2 && ps aux | grep run_{args.tag} | grep -v grep && head -n 10 /workspace/vesuvius/{args.tag}.log"],
        timeout_sec=20,
    )
    print("LAUNCH isError=", err, json.dumps(res, indent=2) if isinstance(res, dict) else res, flush=True)


def cmd_status(args):
    pod_id = json.loads(pod_meta_path(args.tag).read_text(encoding="utf-8"))["id"]
    err, gtxt = rt.mcp("get-pod", {"id": pod_id})
    gp = json.loads(gtxt) if not err else {"err": gtxt}
    print(f"POD [{args.tag}] {pod_id}: status={gp.get('status') or gp.get('desiredStatus')} dc={gp.get('dataCenterId')}", flush=True)
    _, res = rt.exec_pod(
        pod_id,
        ["sh", "-c", f"ps aux | grep -E 'run_{args.tag}|fl_radar|tranche2_vc3d|vc_grow' | grep -v grep || echo NO_ACTIVE_PROC; echo '--- LOG TAIL ---'; tail -n 30 /workspace/vesuvius/{args.tag}.log 2>/dev/null || true"],
        timeout_sec=20,
    )
    print(res.get("stdout", res) if isinstance(res, dict) else res, flush=True)


def _pull_file_parallel(pod_id: str, remote_path: str, local_path: pathlib.Path, workers: int = 2):
    from concurrent.futures import ThreadPoolExecutor
    if local_path.exists() and local_path.stat().st_size > 0:
        err_s, res_s = rt.exec_pod(pod_id, ["sh", "-c", f"stat -c %s '{remote_path}'"], timeout_sec=15)
        if not err_s and isinstance(res_s, dict) and res_s.get("exitCode") == 0 and int(res_s["stdout"].strip()) == local_path.stat().st_size:
            print(f"CACHED {local_path} ({local_path.stat().st_size:,} bytes)", flush=True)
            return
    err, res = rt.exec_pod(pod_id, ["sh", "-c", f"stat -c %s '{remote_path}'"], timeout_sec=15)
    assert not err and isinstance(res, dict) and res.get("exitCode") == 0, f"stat failed: {res}"
    size = int(res["stdout"].strip())
    local_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = 24576
    n_chunks = (size + chunk - 1) // chunk if size > 0 else 0

    def fetch(idx: int) -> bytes:
        for attempt in range(8):
            err_i, res_i = rt.exec_pod(
                pod_id,
                ["sh", "-c", f"dd if='{remote_path}' bs={chunk} skip={idx} count=1 status=none | base64 -w 0"],
                timeout_sec=25,
            )
            if not err_i and isinstance(res_i, dict) and res_i.get("exitCode") == 0 and not res_i.get("truncated"):
                time.sleep(0.35)
                return base64.b64decode(res_i["stdout"].strip())
            if attempt == 7:
                raise AssertionError(f"read failed at chunk {idx}: {res_i}")
            # Back off longer on 429 rate limits!
            is_429 = isinstance(res_i, dict) and res_i.get("status") == 429
            time.sleep(10.0 + 3.0 * attempt if is_429 else 2.0 * (attempt + 1))
        return b""

    if n_chunks <= 2 or workers <= 1:
        parts = [fetch(i) for i in range(n_chunks)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(fetch, range(n_chunks)))
    data = b"".join(parts)
    local_path.write_bytes(data)
    print(f"PULLED {remote_path} -> {local_path} ({len(data):,} bytes)", flush=True)


def cmd_download(args):
    pod_id = json.loads(pod_meta_path(args.tag).read_text(encoding="utf-8"))["id"]
    pack_cmd = (
        f"cd /workspace/vesuvius/outputs/{args.tag} && "
        f"( [ -d full_maps ] && tar -cf full_maps.tar full_maps || true ) && "
        f"for d in */; do [ -d \"${{d}}full_maps\" ] && tar -cf \"${{d}}full_maps.tar\" -C \"$d\" full_maps || true; done && "
        f"find /workspace/vesuvius/outputs/{args.tag} -maxdepth 3 \\( -name '*.json' -o -name '*.png' -o -name '*.tar' -o -name '*_tifxyz.tar.gz' \\) | sort"
    )
    _, res = rt.exec_pod(pod_id, ["sh", "-c", pack_cmd], timeout_sec=60)
    files = [l.strip() for l in (res.get("stdout", "") if isinstance(res, dict) else "").splitlines() if l.strip()]
    print(f"Pulling {len(files)} files for [{args.tag}]:", files, flush=True)
    for rpath in files:
        rel = rpath.replace("/workspace/vesuvius/outputs/", "")
        _pull_file_parallel(pod_id, rpath, ROOT / "outputs" / pathlib.Path(rel))
    _pull_file_parallel(pod_id, f"/workspace/vesuvius/{args.tag}.log", ROOT / "outputs" / args.tag / f"{args.tag}.log")


def cmd_delete(args):
    pod_id = json.loads(pod_meta_path(args.tag).read_text(encoding="utf-8"))["id"]
    err, txt = rt.mcp("delete-pod", {"id": pod_id})
    print(f"DELETE_POD [{args.tag}] isError=", err, txt[:400], flush=True)
    time.sleep(2)
    _, ltxt = rt.mcp("list-pods", {})
    print("VERIFY list-pods:", ltxt[:400], flush=True)
    _, btxt = rt.mcp("get-balance", {})
    print("VERIFY get-balance:", btxt[:300], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Run tag (e.g. gate1_211, gate2a, gate2b)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_cr = sub.add_parser("create")
    p_cr.add_argument("--disk", type=int, default=40)
    p_ln = sub.add_parser("launch")
    p_ln.add_argument("--job", choices=("gate1_211", "run1_0125", "gate2a_eligible", "gate2b_harvest"), required=True)
    p_ln.add_argument("--scrolls", nargs="*", default=None)
    p_ln.add_argument("--recenter-ct", action="store_true")
    sub.add_parser("status")
    sub.add_parser("download")
    sub.add_parser("delete")
    args = ap.parse_args()
    {"create": cmd_create, "launch": cmd_launch, "status": cmd_status, "download": cmd_download, "delete": cmd_delete}[args.cmd](args)


if __name__ == "__main__":
    main()
