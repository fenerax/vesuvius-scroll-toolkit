# Path 3 Scouting Report — High-Leverage Progress Prize Targets ($20k+/mo)

> Compiled: 2026-11-23 from `ScrollPrize/villa` (`37_2026_open_problems.md`, `38_tutorial_spiral.md`, `39_winding_annotations.md`, `15_winners.md`, `spiral-fitting/autoresearch.md`) + live S3 catalog audit (`data/s3_metadata.min.json`).

---

## 1. What Is Already Crowded (Avoid Direct Head-On Races)

Recent September–November 2026 winners & core team efforts heavily cover:
- **Bottom-up meshing / VC3D patch growth:** ScrollFiesta (`@hari_seldon` et al.), HercUNet/HercuViT (`@jaimelomro_79060`, $20k Oct 2026), `winding-field` (`@LimeGS`), VC3D CT patch extension (`@Chase`).
- **Neural fiber extraction models:** Samir Lerognon (`@qualzz`, $20k Sept 2026 + $2.5k Oct 2026) actively iterating on multi-scroll fiber ML + VC3D integration.
- **Core `fit_spiral.py` GPU optimizer:** Already a 230 KB codebase in `ScrollPrize/villa/spiral-fitting/` with custom Triton kernels (`flow_triton.py`, `gap_triton.py`) and an internal `autoresearch` loop.

---

## 2. S3 Data Readiness (Ready to Stream Today)

From [data/s3_metadata.min.json](../data/s3_metadata.min.json):
- **11 scrolls** already have precomputed **`umbilicus` JSON + `surface-prediction-zarr` + `lasagna` (+ `normal-grids` on most)**:
  - `PHercParis4` (`20260411134726`, 2.4 µm) — **82 segments** + **`spiral-dataset`** + `ink-detection-3d-zarr` + `umbilicus` + `normal-grids` + `surface-prediction-zarr` *(the gold-standard benchmark scroll!)*
  - `PHerc0139` (`20260102150214`, 2.399 µm) — **39 segments** + `umbilicus` + `lasagna` + `fibers` + `surface-prediction-zarr`
  - `PHerc0800` (6 segs), `PHerc0332` (2 segs), `PHerc0211` (1 seg), plus 6 unsegmented scrolls (`PHerc0125`, `0358`, `0813`, `0826`, `1203`, `1451`).
- Remember from Stage 1: precomputed `surface-prediction-zarr` is Blosc-zstd compressed and streams a native `256³` cube in **0.40s**.

---

## 3. Top 3 Uncrowded, CPU-Friendly Gaps

### Candidate A — Automated High-Precision Relative Winding Constraints (Spiral Fit Input)
- **Why the team wants it (`39_winding_annotations.md`):** Explicitly bolded as their strategic focus: *"the fastest way to unroll scrolls at scale is to develop methods for creating winding constraints that are precise and fast enough to use widely."*
- **What we'd build:** A fast, conservative (precision >> recall) streaming Python tool that takes `umbilicus` + precomputed `surface-prediction-zarr` + `normal-grids`, traces clean radial rays / multi-winding strips in uncrushed sectors, rejects compressed/ambiguous zones via normal-consistency & peak-spacing checks, and emits verified relative winding constraints (`Δw = +1, +2, ...`) in VC3D / `spiral-fitting` format.
- **Benchmarking:** Validate directly against ground-truth multi-wrap `tifxyz` segments on **`PHercParis4`** (82 segments + `spiral-dataset`) and **`PHerc0139`** (39 segments), then run on unsegmented scrolls (`PHerc0125`, `0358`, `0813`, `1203`, `1451`).
- **Prize ceiling:** High ($5k–$20k/mo) — directly unlocks collection-scale spiral fitting with minimal human hours.

### Candidate B — CT-Guided Label Snapping & Sheet-Switch / Drift QA Tool
- **Why the team wants it (`37_2026_open_problems.md`):** Flagged in a red `⚠️ Open problem` callout: *"label quality is now one of the main unwrapping bottlenecks."* Explicitly calls out **label snapping** (moving approximate meshes back onto the exact physical recto fiber boundary using raw CT + local geometry) and **active-learning QA** (flagging drift / sheet switches for targeted human review).
- **What we'd build:** A streaming Python tool taking any `tifxyz` segment + CT Zarr + surface-prediction Zarr that:
  1. Scores every patch for **recto offset drift**, **sheet-switch discontinuities**, and **layer-merger ambiguity** (producing a ranked QA queue / heatmap).
  2. Performs **sub-voxel CT-guided recto snapping** along local surface normals (constrained by smoothness/regularization to prevent jumping wraps).
  3. **Killer proof:** Uses our already-working local GTX 1070 Ti `ink_canonical_2um` pipeline (0.58s/tile!) to measure **recovered ink contrast & clarity BEFORE vs. AFTER snapping** on real segments.
- **Prize ceiling:** High ($5k–$20k/mo) — uncrowded, zero recent Progress Prize winners in this niche, and directly addresses a stated top bottleneck.

### Candidate C — Spiral-Fit Topology & Duplicate-Winding Evaluation Suite
- **Why the team wants it (`spiral-fitting/autoresearch.md` + `37_2026_open_problems.md`):** Their own `autoresearch.md` explicitly admits a major metric blind spot: *duplicated windings inflate `total_fg_pixels` without dropping `overall_fg_fraction`*, and `37_2026_open_problems.md` asks for better evaluation suites and losses.
- **What we'd build:** A standalone geometry/topology audit & penalty library that detects duplicated wraps, self-intersections, non-monotonic radial winding order, and ground-truth `tifxyz` recall/precision.
- **Prize ceiling:** Medium ($1k–$5k standalone, or $10k–$20k if paired with a concrete loss PR into `fit_spiral.py`).
