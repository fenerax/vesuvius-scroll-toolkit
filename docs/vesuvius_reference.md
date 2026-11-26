# Vesuvius Challenge — Research & Streaming Data Reference

> Compiled: 2026-11-23 | Local catalog snapshot: [data/s3_metadata.min.json](../data/s3_metadata.min.json) (45 samples, 74 OME-Zarr volumes, 330 segments; ~2 MB JSON metadata, **0 bytes of CT voxel data downloaded**).

---

## 1. Current State of the Challenge (as of Late 2026)

- **First Complete Scroll Read End-to-End (June 25, 2026):** **PHerc. 1667** became the first sealed Herculaneum scroll ever virtually unwrapped and read from beginning to end (~1.4 m of papyrus, ~22 columns of Greek text). Preprint: *Complete virtual unwrapping and reading of a rolled Herculaneum papyrus* (`arXiv:2606.29085`).
- **Other Major Decipherments:**
  - **PHerc. 172** (Bodleian Library, Oxford): >70 columns recovered; identified as Philodemus' *On Vices*.
  - **PHerc. 139**: Identified as Philodemus' *On Gods, Book 8*.
  - **PHerc. Paris 4 (Scroll 1)**: 2023 Grand Prize scroll (82 segments in S3); its **title/author** is still an open $50,000 prize.
  - **First Letters Wins**: **PHerc. 343** (Erwin Nieuwlaar), **PHerc. 211** (Priyank Aranke), with active work on **PHerc. 306B** and other unread scrolls.
- **Core Goal Now:** Scaling from "we proved one scroll can be read" to **automated, reliable, low-human-input unwrapping and ink detection** across the entire library of hundreds of remaining carbonized scrolls.

---

## 2. Active Prizes (2026–2027 Cycle)

All major prize deadlines run through **June 25, 2027 (11:59 PM Pacific)**. Full terms at [scrollprize.org/prizes](https://scrollprize.org/prizes).

| Prize | Pool / Award | Goal & Key Rules |
| --- | --- | --- |
| **2027 Grand Prize** | **$1,000,000** ($800k 1st, $100k 2nd, $50k 3rd, $50k 4th) | Fully unroll and render readable an eligible scroll (≥70% legible characters per column, ≤8 hrs human input, MIT/permissive open source integrated into VC3D). |
| **First Letters Prizes** | **$50,000 per scroll** (up to 10 scrolls / $500k pool) | Uncover ≥10 readable letters within a single 4 cm² area of an eligible unread scroll. |
| **PHerc. Paris 4 Title Prize** | **$50,000** | Discover the title of PHerc. Paris 4 (Scroll 1). |
| **Monthly Progress Prizes** | **$20,000+ / month** (guaranteed) | Open-ended awards for open-source contributions tackling any of the [2026 Open Problems](https://scrollprize.org/2026_open_problems). |

---

## 3. Streaming Data Without Downloading Terabytes

Yes — **you do not need to download full multi-TB scans.** All production volumes are published on the **AWS Registry of Open Data** as cloud-optimized **multiscale OME-Zarr** pyramids (`0/` = native resolution, `1/`, `2/`, ... = downsampled pyramids). Clients fetch only the compressed 3D chunks intersecting your requested bounding box over HTTPS/S3 with **no AWS account required** (`anon=True` / `--no-sign-request`).

### Primary Data & Asset Hubs

| Hub | URL / Endpoint | Contents |
| --- | --- | --- |
| **AWS Open Data Bucket** | `s3://vesuvius-challenge-open-data` ([web index](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html)) | 45 samples (35 scrolls, 10 fragments), 74 OME-Zarr volumes, 330 segments, surface predictions. License: `CC BY-NC 4.0`. |
| **Interactive Data Browser** | [scrollprize.org/data_browser](https://scrollprize.org/data_browser) | Browse scrolls, volumes, segments, representations, and ink predictions in-browser. |
| **Hugging Face Models** | [huggingface.co/scrollprize](https://huggingface.co/scrollprize) | Pretrained checkpoints: `ink_canonical_2um`, `ink_3d_dino_guided`, `dinovol_v2_ps8_with_paris4_352500`, `surface_recto_3dunet`, `PHerc.1667-iteration-0..5`, `timesformer_*`, `resnet50_*`. |
| **Hugging Face Datasets** | [huggingface.co/buckets/scrollprize/datasets](https://huggingface.co/buckets/scrollprize/datasets) & `scrollprize/profilometer` | Curated ink & spiral training datasets, profilometer ground truth. |
| **Legacy / Community Server** | `https://dl.ash2txt.org/` | Raw dev files, intermediate datasets, legacy downloads. |

### Streaming Option A — Official `vesuvius` Python Package (Recommended)

Lives in [`ScrollPrize/villa/vesuvius`](https://github.com/ScrollPrize/villa/tree/main/vesuvius), streams remote OME-Zarr from S3/HTTP with local chunk caching, and includes CLI entrypoints (`vesuvius.predict`, `vesuvius.train`, `vesuvius.compute_st`, `vesuvius.napari_trainer`, `vc_wgraph`, etc.).

```bash
pip install vesuvius
vesuvius.accept_terms --yes
```

```python
from vesuvius import Volume

# Stream a 256^3 subvolume directly from S3 (only overlapping chunks are fetched)
url = "s3://vesuvius-challenge-open-data/PHerc0009B/volumes/20250521125136-8.640um-1.2m-116keV-masked.zarr/0"
vol = Volume(type="zarr", path=url, anon=True)
print(vol.shape(), vol.dtype)

crop = vol[4000:4256, 4000:4256, 4000:4256]
```

### Streaming Option B — Direct `zarr` + `s3fs` / `fsspec` (Zero Extra Dependencies)

```python
import fsspec, zarr

s3_path = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/"
store = fsspec.get_mapper(s3_path, anon=True)
root = zarr.open(store, mode="r")

# Level "0" = native resolution; "2" = 4x downsampled for fast browsing/prototyping
arr = root["2"]
crop = arr[1000:1050, 2000:2500, 2000:2500]
```

### Streaming Option C — AWS CLI Inspection

```bash
aws s3 ls --no-sign-request s3://vesuvius-challenge-open-data/
aws s3 ls --no-sign-request s3://vesuvius-challenge-open-data/PHerc1667/
```

---

## 4. S3 Bucket Structure & Sample Inventory

Each sample directory on `s3://vesuvius-challenge-open-data/<SampleID>/` follows:

```text
<SampleID>/
├── volumes/                         # Multiscale OME-Zarr 3D CT scans (0/, 1/, 2/, ...)
├── segments/                        # Extracted papyrus surfaces (tifxyz coords, .obj meshes, ink renders)
├── representations/
│   └── predictions/surfaces/        # Precomputed neural surface-prediction Zarrs
└── scans/                           # Scan metadata
```

### Notable Scrolls in the S3 Catalog (35 Scrolls Total)

| Scroll | Volumes | Segments | Notable Resolutions / Assets |
| --- | ---: | ---: | --- |
| **PHercParis4** (Scroll 1) | 7 | 82 | 1.13 µm, 7.91 µm, 45.5 µm — 2023 GP scroll; open Title Prize |
| **PHerc0172** | 2 | 53 | 7.91 µm / 53 keV — Philodemus *On Vices* (>70 cols) |
| **PHerc0139** | 10 | 39 | 2.40 µm & 9.36 µm + surface predictions — Philodemus *On Gods Bk 8* |
| **PHerc1667** | 4 | 20 | 1.13 µm, 2.40 µm, 3.24 µm, 7.91 µm — First scroll read end-to-end |
| **PHerc0814** | 3 | 19 | 1.13 µm, 2.40 µm, 9.36 µm + surface predictions |
| **PHerc1447** | 1 | 18 | 8.64 µm / 116 keV + surface predictions |
| **PHerc0800** | 1 | 6 | 8.64 µm / 116 keV + surface predictions |
| **PHerc0841** | 2 | 3 | 2.40 µm & 9.37 µm + surface predictions |
| **PHerc0332** | 4 | 2 | 2.40 µm, 3.24 µm, 7.91 µm + surface predictions |
| **PHerc0211**, **PHerc0306B**, **PHerc0343**, **PHercMAN5** | 1 ea | 1 ea | 2.40–9.36 µm + surface predictions (active First Letters targets/wins) |
| **22 Unsegmented Scrolls** | 1–2 ea | 0 | `PHerc0125`, `0175A/B`, `0191`, `0257`, `0268`, `0358`, `0483A/B`, `0490A/B`, `0813`, `0826`, `0846A/B`, `1203`, `1218`, `1299`, `1451`, `1545`, `MANB`, `Paris3` |

### Detached Fragments (10 Total — Ink Ground Truth)

- **With volumes & segments:** `PHerc0009B` (3 vols, 19 segs), `PHerc0500P2` (4 vols, 46 segs), `PHercMANBp` (2 vols, 11 segs), `PHerc0343P` (2 vols, 8 segs).
- **Metadata / legacy fragment entries:** `PHerc1667Cr1Fr3`, `PHerc51Cr4Fr8`, `PHercParis1Fr34`, `PHercParis1Fr39`, `PHercParis2Fr143`, `PHercParis2Fr47`.

---

## 5. Key Code Repositories & Pipeline

Almost everything is consolidated under the **[ScrollPrize](https://github.com/ScrollPrize)** GitHub org:

1. **[ScrollPrize/villa](https://github.com/ScrollPrize/villa)** *(active monorepo, MIT)* — Houses:
   - `vesuvius/` — Python streaming, training, nnUNetv2/custom 3D inference, structure tensors, Napari trainer, winding-graph (`vc_wgraph`).
   - `volume-cartographer/` (**VC3D**) — C++ GUI & toolkit for loading remote/local Zarrs, viewing 3D slices, and growing/editing `tifxyz` papyrus surface meshes.
   - `vesuvius-c/`, `scroll-renderer/`, and the full `scrollprize.org/` documentation site.
2. **[ScrollPrize/open-data](https://github.com/ScrollPrize/open-data)** — AWS Open Data docs and starter notebook (`examples/get-to-know-a-dataset.ipynb`).
3. **[ScrollPrize/dinovol](https://github.com/ScrollPrize/dinovol)** — 3D DINOv2 self-supervised pretraining directly on larger-than-memory OME-Zarr volumes.
4. **[ScrollPrize/vesuvius-gui](https://github.com/ScrollPrize/vesuvius-gui)** — GUI for browsing Vesuvius project data.

### End-to-End Pipeline & 6 Open Research Bottlenecks (Oct 2026)

```text
Synchrotron Micro-CT (ESRF / DLS)
  → 8-bit Masked Multiscale OME-Zarr Volumes
    → 3D Surface Prediction (e.g., 3D U-Net / nnUNetv2)
      → Sheet Tracing & Virtual Unwrapping (VC3D / tifxyz / winding-graph / spiral prior)
        → Ink Detection (3D ResNet / TimeSformer / DINO-guided / multi-teacher ensembles)
          → Papyrological Reading
```

The official **[2026 Open Problems](https://scrollprize.org/2026_open_problems)** highlights 6 high-impact contribution areas:
1. **3D Surface Prediction** — sharper, cleaner sheet probability maps across varying scan energies/resolutions.
2. **Mesh Tracing & Connectivity (Geometry/C++)** — eliminating holes, sheet merges, and sheet switches without manual VC3D repair.
3. **3D Label Quality & Active Learning** — higher-precision supervision in compressed/damaged regions.
4. **Conservative Fiber Tracing** — following individual papyrus fibers across long distances as connectivity clues.
5. **Global Spiral Fit** — better losses, evaluation suites, and priors for global scroll winding geometry.
6. **Direct 3D Ink Segmentation & Self-Supervised 3D Representations** (`dinovol`) — generalizing ink detection across scrolls without per-scroll retraining.
