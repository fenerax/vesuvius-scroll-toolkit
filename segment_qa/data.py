"""Catalog lookup, anonymous S3 streaming of layers-zarr surface volumes, and a bounded cache.

Paths
-----
Inside a repository checkout every path is relative to the repository root. When the package is
installed elsewhere (no ``data/s3_metadata.min.json`` next to it), the working directory is the
root. Each path can be overridden with an environment variable:

* ``SEGMENT_QA_ROOT``     root for the defaults below
* ``SEGMENT_QA_CATALOG``  catalog JSON (default ``<root>/data/s3_metadata.min.json``)
* ``SEGMENT_QA_CACHE``    bounded S3 cache (default ``<root>/cache/segment_qa``)
* ``SEGMENT_QA_OUT``      outputs (default ``<root>/outputs/segment_qa``)
* ``SEGMENT_QA_MODELS``   model checkpoints (default ``<root>/cache/models``)

``python -m segment_qa --catalog PATH`` sets the catalog for one run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _default_root() -> Path:
    env = os.environ.get("SEGMENT_QA_ROOT")
    if env:
        return Path(env).resolve()
    repo = Path(__file__).resolve().parents[1]
    if (repo / "data" / "s3_metadata.min.json").exists():
        return repo
    return Path.cwd()


ROOT = _default_root()
BUCKET = "vesuvius-challenge-open-data"
CATALOG_PATH = Path(os.environ.get("SEGMENT_QA_CATALOG", ROOT / "data" / "s3_metadata.min.json"))
CACHE_DIR = Path(os.environ.get("SEGMENT_QA_CACHE", ROOT / "cache" / "segment_qa"))
CACHE_LIMIT_BYTES = 450 * 1024 * 1024      # hard ceiling below the 500 MB budget
OUT_DIR = Path(os.environ.get("SEGMENT_QA_OUT", ROOT / "outputs" / "segment_qa"))
MODELS_DIR = Path(os.environ.get("SEGMENT_QA_MODELS", ROOT / "cache" / "models"))
CACHE_SUFFIXES = {".npz", ".jpg", ".png", ".npy", ".tif"}


@dataclass(frozen=True)
class SegmentRef:
    sample: str
    key: str          # segment timestamp id
    volume: str       # target volume id of the 2.4 um layers-zarr
    layers_key: str | None = None   # exact layers-zarr S3 key (default: first catalog match)

    @property
    def tag(self) -> str:
        return f"{self.sample}_{self.key}"


SEGMENTS = {
    "PHerc0139": SegmentRef("PHerc0139", "20260311000000", "20260102150214"),
    "PHerc1667": SegmentRef("PHerc1667", "20251212185248", "20251217075048"),
    "PHercParis4": SegmentRef("PHercParis4", "20260623144957", "20260411134726"),
}
# Independent reference for PHerc0139: the 1.129 um-scan ink map (different scan, model mrg20736).
REFERENCE_1UM_VOLUME = {"PHerc0139": "20260413113053"}


_catalog_cache: dict | None = None
_catalog_path: Path | None = None


def set_catalog(path: str | Path) -> None:
    global CATALOG_PATH, _catalog_cache
    CATALOG_PATH = Path(path)
    _catalog_cache = None


def catalog() -> dict:
    global _catalog_cache, _catalog_path
    if _catalog_cache is None or _catalog_path != CATALOG_PATH:
        if not CATALOG_PATH.exists():
            raise FileNotFoundError(f"segment catalog not found: {CATALOG_PATH} (set SEGMENT_QA_CATALOG or "
                                    "pass --catalog)")
        _catalog_cache = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["samples"]
        _catalog_path = CATALOG_PATH
    return _catalog_cache


def segment_record(ref: SegmentRef) -> dict:
    return catalog()[ref.sample]["segments"][ref.key]


def asset_keys(ref: SegmentRef, kind: str, target_volume: str | None = None) -> list[str]:
    """S3 keys of every catalog asset of ``kind`` on ``target_volume`` (default: ``ref.volume``)."""
    vol = target_volume or ref.volume
    out = []
    for item in segment_record(ref)["data"]:
        params = item.get("parameters") or {}
        if item["type"] == kind and params.get("target_volume") == vol:
            out.append(item["origins"][0]["path"].strip("/"))
    return out


def asset_key(ref: SegmentRef, kind: str, target_volume: str | None = None) -> str:
    """S3 key of the first catalog asset of ``kind`` on ``target_volume`` (default: ``ref.volume``).
    For ``layers-zarr``, ``ref.layers_key`` pins the exact asset."""
    if kind == "layers-zarr" and ref.layers_key and target_volume in (None, ref.volume):
        return ref.layers_key.strip("/")
    keys = asset_keys(ref, kind, target_volume)
    if not keys:
        raise KeyError(f"{ref.sample}/{ref.key}: no {kind} on volume {target_volume or ref.volume}")
    return keys[0]


def resolve_segment(sample: str, key: str | None = None, volume: str | None = None,
                    layers_key: str | None = None) -> SegmentRef:
    if key is None and volume is None and sample in SEGMENTS:
        ref = SEGMENTS[sample]
        return ref if layers_key is None else SegmentRef(ref.sample, ref.key, ref.volume, layers_key)
    if key is None or volume is None:
        raise ValueError("give --segment and --volume for samples without a built-in default")
    return SegmentRef(sample, key, volume, layers_key)


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def asset_tag(ref: SegmentRef, zarr_key: str | None = None) -> str:
    """Cache namespace of one resolved layers-zarr: scroll, segment, volume and a hash of the full path."""
    zarr_key = zarr_key or asset_key(ref, "layers-zarr")
    return f"{ref.sample}_{ref.key}_{ref.volume}_{short_hash(f'{BUCKET}/{zarr_key}')}"


def output_tag(ref: SegmentRef) -> str:
    """Default output filename stem: sample, segment and volume (plus a path hash for a pinned
    non-default layers-zarr variant)."""
    tag = f"{ref.sample.lower()}_{ref.key}_{ref.volume}"
    if ref.layers_key:
        default = asset_keys(ref, "layers-zarr")
        if not default or ref.layers_key.strip("/") != default[0]:
            tag += f"_{short_hash(ref.layers_key.strip('/'), 8)}"
    return tag


class CacheEntryTooLarge(ValueError):
    pass


class BoundedCache:
    """Files under ``cache/segment_qa``; oldest-first eviction keeps the total under ``limit``.

    Writes go to a unique ``.part`` temp file in the target directory and are renamed into place,
    so a crash or a concurrent writer never leaves a truncated entry. A single entry larger than
    the limit is rejected (not cached) instead of evicting everything else."""

    def __init__(self, root: Path = CACHE_DIR, limit: int = CACHE_LIMIT_BYTES):
        self.root = Path(root)
        self.limit = int(limit)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def files(self) -> list[Path]:
        return [p for p in self.root.rglob("*") if p.is_file() and p.suffix in CACHE_SUFFIXES]

    def size(self) -> int:
        return sum(p.stat().st_size for p in self.files())

    def enforce(self, protect: Path | None = None) -> None:
        files = sorted(self.files(), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for p in files:
            if total <= self.limit:
                break
            if protect is not None and p.resolve() == protect.resolve():
                continue
            total -= p.stat().st_size
            p.unlink(missing_ok=True)

    def _tmp(self, p: Path) -> Path:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.parent / f".{p.name}.{uuid.uuid4().hex[:8]}.part"

    def commit(self, tmp: Path, p: Path) -> bool:
        """Move a finished temp file into the cache. Returns False (and deletes it) when it is larger
        than the whole cache."""
        if tmp.stat().st_size > self.limit:
            tmp.unlink(missing_ok=True)
            return False
        os.replace(tmp, p)
        self.enforce(protect=p)
        return True

    def load_array(self, name: str) -> np.ndarray | None:
        p = self.path(name)
        if not p.exists():
            return None
        try:
            with np.load(p) as z:
                return z["a"]
        except Exception:
            p.unlink(missing_ok=True)
            return None

    def save_array(self, name: str, a: np.ndarray) -> bool:
        """Cache ``a``; returns False when the compressed entry would exceed the cache limit."""
        p = self.path(name)
        tmp = self._tmp(p)
        try:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, a=a)
            return self.commit(tmp, p)
        finally:
            tmp.unlink(missing_ok=True)


class LayersVolume:
    """Anonymous read-only access to one segment's layers-zarr pyramid (zarr >= 3, v2 or v3 arrays)."""

    def __init__(self, ref: SegmentRef):
        import s3fs
        import zarr
        if int(zarr.__version__.split(".")[0]) < 3:
            raise ImportError(f"segment_qa needs zarr >= 3 (found {zarr.__version__}): pip install '.[segment-qa]'")
        from zarr.storage import FsspecStore

        self.ref = ref
        self.key = asset_key(ref, "layers-zarr")
        self.cache_tag = asset_tag(ref, self.key)
        self._fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
        store = FsspecStore(self._fs, path=f"{BUCKET}/{self.key}", read_only=True)
        self.group = zarr.open_group(store, mode="r")
        meta = self.group.attrs["multiscales"][0]
        self._scales = {}
        for ds in meta["datasets"]:
            scale = next(t["scale"] for t in ds["coordinateTransformations"] if t["type"] == "scale")
            self._scales[str(ds["path"])] = scale
        self.downsampling = meta.get("metadata", {}).get("downsampling_method")

    def array(self, level: int = 0):
        return self.group[str(level)]

    def shape(self, level: int = 0) -> tuple[int, int, int]:
        return tuple(self.array(level).shape)

    @property
    def pixel_size_um(self) -> float:
        """Spacing along the layer (mesh-normal) axis; identical on every pyramid level."""
        return float(self._scales["0"][0])

    def lateral_um(self, level: int) -> float:
        return float(self._scales[str(level)][1])

    def read(self, level: int, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        return np.asarray(self.array(level)[:, y0:y0 + h, x0:x0 + w])

    def close(self) -> None:
        try:
            from zarr.core.sync import sync
            if getattr(self._fs, "_s3", None) is not None:
                sync(self._fs._s3creator.__aexit__(None, None, None))
        except Exception:
            pass


def check_crop(shape_zyx: tuple[int, int, int], y0: int, x0: int, size: int, cell: int = 1) -> None:
    """Raise ``ValueError`` unless the ``size`` x ``size`` crop at ``(y0, x0)`` lies inside the level."""
    _, h, w = shape_zyx
    if size <= 0 or size % cell:
        raise ValueError(f"crop size {size} must be a positive multiple of the {cell} px cell")
    if y0 < 0 or x0 < 0 or y0 + size > h or x0 + size > w:
        raise ValueError(f"crop ({y0}, {x0}) + {size} px lies outside the {h} x {w} level")


def read_crop(vol: LayersVolume, y0: int, x0: int, size: int = 256, level: int = 0,
              cache: BoundedCache | None = None) -> np.ndarray:
    """Full-depth crop, cached as compressed npz under the resolved asset's namespace."""
    check_crop(vol.shape(level), y0, x0, size)
    name = f"{vol.cache_tag}/L{level}_{y0}_{x0}_{size}.npz"
    if cache is not None:
        a = cache.load_array(name)
        if a is not None:
            return a
    a = vol.read(level, y0, x0, size, size)
    if cache is not None:
        cache.save_array(name, a)
    return a


def fetch_object(key: str, cache: BoundedCache, name: str) -> Path:
    """Download one S3 object into the bounded cache (no-op if already present).

    Downloads to a temp file and renames it into place; objects larger than the cache limit are
    refused before any byte is transferred."""
    p = cache.path(name)
    if p.exists():
        return p
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    size = fs.size(f"{BUCKET}/{key}")
    if size is not None and size > cache.limit:
        raise CacheEntryTooLarge(f"{key} is {size / 1e6:.0f} MB, larger than the {cache.limit / 1e6:.0f} MB cache")
    tmp = cache._tmp(p)
    t0 = time.perf_counter()
    try:
        fs.get(f"{BUCKET}/{key}", str(tmp))
        if not cache.commit(tmp, p):
            raise CacheEntryTooLarge(f"{key} is larger than the cache limit")
    finally:
        tmp.unlink(missing_ok=True)
    print(f"  downloaded {key.rsplit('/', 1)[-1]} ({p.stat().st_size / 1e6:.1f} MB) in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    return p


def ink_preview_ds8(ref: SegmentRef, cache: BoundedCache, target_volume: str | None = None) -> np.ndarray:
    """Published ds8 ink-prediction JPEG (grey uint8). Reuses the stage-1 cache when present."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    key = asset_key(ref, "ink-detection-downsampled", target_volume)
    stage1 = ROOT / "cache" / "stage1" / key
    path = stage1 if stage1.exists() else fetch_object(key, cache, f"{ref.tag}/{Path(key).name}")
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def reference_1um_on_grid(ref: SegmentRef, cache: BoundedCache, shape: tuple[int, int]) -> np.ndarray:
    """Independent 1.129 um-scan ink map resized onto the 2.4 um ds8 grid ``shape`` (float 0..255).

    The 1.129 um UV grid is the 2.4 um grid scaled by 2.399 / (2 x 1.129); a plain resize aligns
    it (Candidate B found the best integer ds8 shift to be (0, 0), global r = 0.49)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    vol = REFERENCE_1UM_VOLUME[ref.sample]
    key = asset_key(ref, "ink-detection-downsampled", vol)
    path = fetch_object(key, cache, f"{ref.tag}/{Path(key).name}")
    with Image.open(path) as im:
        g = im.convert("L").resize((shape[1], shape[0]), Image.BILINEAR)
        return np.asarray(g).astype(np.float32)
