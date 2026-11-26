# segment_qa: raw-CT surface-placement QA for ink training data

`segment_qa` checks every 8×8 px cell of a layers-zarr segment, using only the raw CT. For each
cell it asks whether the papyrus sheet sits where the `ink_canonical_2um` model expects it inside
its 62-layer window. It produces:

- a class map;
- a hotspot list, ranked by summed severity;
- a training keep-mask.

Its thresholds come from a GPU calibration of the ink model's tolerance to window shifts. They
are **provisional** (§5).

All numbers below come from the round-2 runs on 2026-11-24, saved in `outputs/segment_qa/`
(gitignored). The S3 cache in `cache/segment_qa/` is bounded at 500 MB.

## 1. Method

### 1.1 Band detection (`segment_qa/band_detector.py`)

**Input and padding**

- Input: a layers slab `(Z=109, Y, X)` at any pyramid level. Pyramid levels mean-pool only in Y
  and X, so a level-3 pixel equals one 8×8 L0 cell (19.19 µm). Z keeps 2.399 µm layers.
- **Pad mask.** At L0, a cell is valid when all of its pixels have a nonzero layer. Mean pooling
  hides padding at coarser levels: one valid L0 pixel makes the pooled pixel nonzero. Coarse
  audits therefore apply the **exact L0 `all` validity** before any lateral-jump classification:
  - For levels 1–3 (`f = 2**level ≤ 10`), candidates are the level-valid cells that touch an
    invalid cell (8-neighbourhood) or the grid edge, because every 20×20 L0 `tifxyz` invalid quad
    contains at least one full aligned `f×f` cell. For levels ≥ 4 (`f > 10`), a 20×20 quad can
    fit inside a single pooled cell or straddle two cells with no fully invalid neighbour, so all
    level-valid cells are checked against L0.
  - The L0 chunks under those candidates are read and pooled with `pool_all`, so a coarse pixel is
    valid only if every L0 pixel under it is.
  - Invalid cells are `INVALID_PAD`. An empty sweep returns an all-255 map.

**Per-cell profile and band rules**

- Per cell, the 8×8-mean z profile is smoothed (σ = 3.0 µm). It is thresholded at
  `p15 + max(12, 0.30·(p90 − p15))` of that cell's own profile.
- Runs above the cutoff become candidate bands when:
  - their **interpolated face-to-face thickness is ≥ 9.5 µm** (`min_band_rule="faces_um"`); and
  - they peak above a slightly higher floor.
- The thickness rule works on the sub-voxel faces, not on a sample count. A 9.5 µm minimum
  therefore means the same physical thickness at any `pixel_size_um`. Tests cover 2.4 µm and
  1.2 µm. The old sample-count rule is still available as `min_band_rule="samples"`.

**Nominal anchor and band selection**

- The **nominal anchor** is `z_nom = D // 2` = layer **54**: the mesh surface itself. The ink
  window is layers `[23, 85)`, i.e. −31..+30 layers from the anchor. Its geometric midpoint is
  53.5, half a layer below the anchor.
- Every offset in this report, the calibration included, is measured from the anchor (54), not
  from the midpoint. A band at the anchor has `delta_z = 0`. The window spans −74.4 to +72.0 µm,
  and both edges are inclusive (tested).
- The band nearest the anchor is selected. The anchor is "in air" when it lies ≥ 4.8 µm outside
  every band. The window is "empty" when no band overlaps it.
- Per-cell outputs:
  - geometry: `z_top`, `z_bot`, `z_mid`, `thickness_um`, `delta_z_vx`/`delta_z_um`;
  - anchor position: `nominal_in_air`, `gap_um`;
  - band context: `competing_bands`, `band_spacing_um`, `window_empty`, `clipped`, `no_band`;
  - lateral steps: `switch_jump_um`, `flip_jump_um`, `uncertain_jump_um`, `midpoint_step_um`.

**Lateral continuity**

For each 4-neighbour edge where the selected bands are disjoint in z (a midpoint step):

1. **Prefilter:** a uint64 occupancy bitmask of all accepted bands (2 layers per bit). This step
   can only say that a band *may* continue.
2. **Exact test:** each selected band continues if some accepted band in the other cell overlaps
   its closed `[z_top, z_bot]` interval.
3. The step is then classified:

   | prefilter | exact | type | meaning |
   |---|---|---|---|
   | both continue | both continue | **flip** (`COMPETING_BANDS`) | the papyrus did not move; only the nearest-band choice changed |
   | both continue | at least one ends | **uncertain switch** (`LATERAL_JUMP` + `SHEET_ID_UNCERTAIN`) | the bitmask put two near-touching bands in one bit, but their exact intervals are disjoint; the sheet identity cannot be resolved at this z resolution |
   | at least one ends | – | **switch** (`LATERAL_JUMP`) | a real mesh step: the selected band ends |

   Switches, including uncertain ones, feed `FAIL_SHEET_SWITCH`. The uncertain ones also carry
   `SHEET_ID_UNCERTAIN`, so they can be reviewed separately. A flip of ≥ 40 µm in a region of
   ≥ 4 cells is a WARN (`COMPETING_BANDS`). Tests:
   - near-touching bands (faces 50.7 | 51.3, same bitmask bit) give an uncertain step at L0 and at
     L3;
   - bands in different bits give a certain switch;
   - touching bands continue.

**Sweeps**

- Blocks of 512×512 cells with a 1-cell halo, cropped back to the block core. This makes the seam
  flags identical to a single-block run (tested). The detector is fully vectorized.
- The whole PHerc0139 segment (3370×3100 L3 cells, 49 blocks) sweeps in 137 s. The exact L0
  validity pass adds 164 s (853 L0 chunks).

### 1.2 Classes (`segment_qa/auditor.py`)

**Class precedence.** The first match wins, from the top:

| code | class | rule |
|---|---|---|
| 255 | INVALID_PAD | exact L0 validity fails |
| 2 | FAIL_OFF_SHEET | `drift` beyond a **crossed** fail edge, or air gap ≥ 25.53 µm, or window empty, or no band |
| 3 | FAIL_SHEET_SWITCH | `switch_jump_um ≥ 40 µm` in an 8-connected switch region of ≥ 4 cells; uncertain switches also carry `SHEET_ID_UNCERTAIN` |
| 1 | WARN_DRIFT | `drift` outside the safe range, anchor in air (`NOMINAL_IN_AIR`), drift beyond a **censored** fail edge (`DRIFT_BEYOND_MEASURED`), or a flip region (`COMPETING_BANDS`, ≥ 40 µm, ≥ 4 cells) |
| 0 | OK | otherwise |

Notes:

- The class codes are historical, so OFF_SHEET (2) outranks SWITCH (3) despite the lower code.
- Every cell also carries a uint16 **reason bitmask**:
  - DRIFT_WARN, DRIFT_FAIL, DRIFT_BEYOND_MEASURED;
  - NOMINAL_IN_AIR, AIR_GAP_FAIL, WINDOW_EMPTY, NO_BAND;
  - LATERAL_JUMP, COMPETING_BANDS, SHEET_ID_UNCERTAIN.
- **Censored edges.** The calibration never saw the retention cross the fail level below
  −76.8 µm (§2). Drift beyond that side is therefore WARN + `DRIFT_BEYOND_MEASURED`, not FAIL.
  The crossed edge (+25.53 µm) is a FAIL. The flags travel with the thresholds as
  `fail_drift_crossed`.
- **`drift`.** The same-sheet 5×5-cell nan-median of `delta_z_um`:
  - only window cells whose selected band overlaps the centre cell's band in z are used;
  - with fewer than 8 such cells, the raw value is kept;
  - `drift_raw_um` keeps the unsmoothed value.

**Outputs**

- `*_hotspots.json` lists the 8-connected FAIL regions **ranked by summed severity**. A cell's severity
  is the largest of `drift/fail edge`, `switch jump/40 µm`, `gap/air-gap fail` (when in air),
  2 (window empty) and 1 (no band), so it is 1 at a threshold and larger beyond it. A region's
  score is the sum over its cells, so a small region with a large jump can outrank a larger mild
  one. Each entry has:
  - rank, score, L0 peak `[Y, X]`, class;
  - `drift_um`, `drift_raw_um`, `jump_um`, `flip_jump_um`, `midpoint_step_um`, `gap_um`;
  - reasons, cell count and mm².
- `train_keep_mask` = OK or WARN, minus a 1-cell (8-neighbour) margin around FAIL cells. PAD
  is dropped.

### 1.3 Ink tolerance calibration (`segment_qa/ink_tolerance.py`)

**Setup**

- **Model:** `cache/models/r152_3ddec_v2_l5_epoch13.ckpt` (ResNet3D-152, 3D decoder,
  `new_canon_autoresearch_recipe`). FP32, batch 1, on the local GTX 1070 Ti.
- **Crops:** 48 text-rich 256 px crops of PHerc0139 `20260311000000-w023_202603112281` on volume
  `20260102150214` (2.399 µm).
  - They were **chosen from the independent 1.129 µm reference ink map**, resized to the 2.4 µm
    ds8 grid and ranked by text richness, at least 1024 px apart.
  - They are therefore not chosen from the 2.4 µm predictions under test.
- **Shifts:** the window `[23+dz, 85+dz)` for dz ∈ {−20, −16, −12, −8, −6, −4, −2, 0, 2, 4, 6, 8,
  12, 16, 20, 24} layers.
  - This is a superset of the requested −16..+16, needed to reach a fail crossing.
  - Shifts must be sorted, unique, include 0 and fit the stack (−23..+24). The CLI checks this
    before any S3 read.
- **Reference:** the published 1.129 µm-scan ink map (model mrg20736, volume `20260413113053`),
  resized to the ds8 grid. Metric: Pearson r on the 32×32 ds8 cells of each crop.

**Metrics.** For **every** dz, each with a 2000-sample crop bootstrap 95% CI and IQR:

- agreement r vs the reference, and retention = r(dz)/r(0);
- paired Δr vs dz = 0;
- self-consistency r(pred(dz), pred(0));
- foreground retention: the share of the dz = 0 foreground still above 0.5.

**Window frame**

- Retention vs window shift.
- Safe: bootstrap lower bound ≥ 0.90. Fail: point retention < 0.70, interpolated.

**Band frame** (`thresholds.band_frame`, used by the auditor)

Agreement is measured against the band position each prediction actually saw.

- Each crop × shift pair has an effective offset from the anchor:
  `e = crop median band offset − dz·2.399 µm`. The crop offsets are **re-measured with the current
  detector** on every analysis.
- The model `r = a_crop + g(e bin)`, with 9.6 µm bins, is fitted by alternating means.
- Bins with fewer than 20 pairs are **unsupported**: they are excluded from the fit and get no
  value, and the same guard applies inside every bootstrap replicate.
- The **support** is the contiguous run of supported bins around the peak.
- The rules stop at the first unsupported bin:
  - **safe:** contiguous bins around the best bin with point retention ≥ 0.90;
  - **proven-safe core:** the same, with the bootstrap lower bound ≥ 0.90;
  - **fail:** point retention < 0.70, interpolated. Without a crossing, the edge is **censored**
    at the support edge and marked `crossed = false`.
- **Bootstrapped peak:** the best bin is re-selected in every crop-paired bootstrap replicate,
  which gives a peak CI and a share per bin.
- **Air-gap fail** = the smallest crossed |fail edge|.
- **Bin-width sensitivity** repeats the fit with 7.2, 9.6 and 12 µm bins (500 draws each).
- The fit and analysis run on the saved raw arrays (`ink_tolerance_raw.npz`); `--reanalyze` needs
  no GPU. The CT crops come from the cache or, when evicted, are re-read from S3. They are never
  skipped.

**Detector-noise rules**

Fixed from physical scale before the final run:

- switch ≥ 40 µm (about 0.63 of the 63.5 µm wrap pitch);
- switch region ≥ 4 cells;
- drift smoothing over 5 cells (96 µm).

## 2. Calibration results

The GPU measure took 402.3 s; the analysis took 110.3 s (CPU, including CT re-reads and the
bootstrap). `r0` (agreement at dz = 0): median 0.377, 68.8% of crops above 0.3, range −0.343 to
0.898.

| dz (vx) | dz (µm) | mean r [95% CI] | retention [95% CI] | paired Δr [95% CI] | self-consistency | fg retention [95% CI] |
|---|---|---|---|---|---|---|
| −20 | −47.98 | 0.157 [0.055, 0.251] | 0.463 [0.179, 0.757] | −0.181 [−0.294, −0.072] | 0.432 | 0.642 [0.521, 0.766] |
| −16 | −38.38 | 0.212 [0.134, 0.286] | 0.628 [0.454, 0.805] | −0.126 [−0.194, −0.058] | 0.694 | 0.764 [0.651, 0.865] |
| −12 | −28.79 | 0.309 [0.242, 0.377] | 0.914 [0.735, 1.173] | ns | 0.677 | 0.826 [0.740, 0.904] |
| −8 | −19.19 | 0.279 [0.184, 0.371] | 0.826 [0.687, 0.936] | −0.059 [−0.099, −0.022] | 0.876 | 0.873 [0.799, 0.935] |
| −6 | −14.39 | 0.324 [0.241, 0.406] | 0.960 [0.820, 1.115] | ns | 0.808 | 0.837 [0.739, 0.918] |
| −4 | −9.60 | 0.361 [0.287, 0.434] | 1.069 [0.913, 1.300] | ns | 0.765 | 0.851 [0.773, 0.919] |
| −2 | −4.80 | 0.372 [0.291, 0.454] | 1.101 [0.989, 1.262] | ns | 0.844 | 0.853 [0.781, 0.919] |
| 0 | 0 | 0.338 [0.247, 0.428] | 1.000 | – | 1.000 | 1.000 |
| +2 | +4.80 | 0.345 [0.259, 0.428] | 1.022 [0.911, 1.164] | ns | 0.885 | 0.894 [0.810, 0.960] |
| +4 | +9.60 | 0.358 [0.280, 0.431] | 1.059 [0.900, 1.283] | ns | 0.762 | 0.832 [0.741, 0.910] |
| +6 | +14.39 | 0.354 [0.274, 0.433] | 1.047 [0.911, 1.235] | ns | 0.793 | 0.761 [0.654, 0.859] |
| +8 | +19.19 | 0.319 [0.231, 0.410] | 0.945 [0.840, 1.049] | ns | 0.904 | 0.788 [0.692, 0.872] |
| +12 | +28.79 | 0.313 [0.228, 0.394] | 0.927 [0.732, 1.174] | ns | 0.671 | 0.776 [0.677, 0.871] |
| +16 | +38.38 | 0.309 [0.222, 0.394] | 0.916 [0.741, 1.109] | ns | 0.759 | 0.701 [0.595, 0.800] |
| +20 | +47.98 | 0.264 [0.179, 0.344] | 0.781 [0.565, 1.037] | ns | 0.607 | 0.677 [0.552, 0.793] |
| +24 | +57.58 | 0.287 [0.198, 0.374] | 0.851 [0.632, 1.102] | ns | 0.665 | 0.635 [0.510, 0.757] |

- "ns" means the paired-Δr CI spans 0.
- Foreground retention is computed at every shift on n = 43 crops. The other 5 crops have < 1%
  foreground at dz = 0.
- IQRs and medians are in `ink_tolerance_curve.json`.

**Window-frame thresholds** (CI rule, fail interpolated):

- `safe_dz_range_um` = [−9.6, +4.8] (dz −4..+2 vx);
- `fail_dz_um` = −35.96 (−14.99 vx, reached) / +57.58 (+24 vx, the sweep edge, not reached).

**Band-offset curve** (the thresholds the auditor applies; offsets from the anchor z = 54):

| bin centre (µm) | pairs | used | fitted r [95% CI] | retention [95% CI] |
|---|---|---|---|---|
| −81.6 | 17 | no (< 20) | – | – |
| −72.0 | 28 | yes | 0.287 [0.196, 0.375] | 0.795 [0.593, 0.963] |
| −62.4 | 43 | yes | 0.297 | 0.824 [0.576, 1.000] |
| −52.8 | 50 | yes | 0.309 | 0.857 [0.677, 0.968] |
| −43.2 | 59 | yes | 0.303 | 0.839 [0.629, 0.974] |
| −33.6 | 70 | yes | 0.326 | 0.903 [0.732, 1.000] |
| −24.0 | 85 | yes | 0.342 | 0.949 [0.796, 1.000] |
| **−14.4 (peak)** | 92 | yes | 0.361 [0.283, 0.436] | 1.000 [0.891, 1.000] |
| −4.8 | 84 | yes | 0.332 | 0.920 [0.788, 1.000] |
| +4.8 | 74 | yes | 0.342 | 0.948 [0.807, 1.000] |
| +14.4 | 58 | yes | 0.268 | 0.743 [0.540, 0.900] |
| +24.0 | 43 | yes | 0.269 | 0.746 [0.538, 0.933] |
| +33.6 | 32 | yes | 0.166 | 0.460 [0.185, 0.703] |
| +43.2 | 20 | yes | 0.146 | 0.405 [0.060, 0.710] |
| +52.8 | 5 | no | – | – |

**Thresholds and peak**

- Support: [−76.8, +48.0] µm.
- Safe drift: [−38.4, +9.6] µm. Proven-safe core: [−19.2, −9.6] µm, which **excludes 0**.
- Fail edges:
  - **+25.53 µm (crossed)**;
  - **−76.8 µm (censored, `crossed = false`):** the support edge, never crossed.
- Air-gap fail = 25.53 µm.
- **Bootstrapped peak:**
  - the point peak −14.4 µm is the best bin in 56.9% of replicates;
  - −24.0 µm in 14.5%, +4.8 µm in 11.8%;
  - peak 95% CI [−62.4, +4.8] µm.
  - The location of the optimum is therefore only loosely determined. The asymmetry (a steep fall
    above +5 µm) is better supported than the exact peak.
- **Offset recompute:** 15 of the 48 crop offsets changed, by at most 0.23 µm. Crop offsets at
  dz = 0: median −12.33 µm, p10–p90 [−30.15, −0.63].
- Mirroring the window-frame ranges around −12.33 gives safe [−4.8, +9.6] and fail
  [−57.58, +35.96]. These are kept as `band_frame_window_mirror` for comparison only.

**Bin-width sensitivity** (500 draws per width):

| bin width | peak (µm) | peak 95% CI | safe | proven-safe core | fail −/+ |
|---|---|---|---|---|---|
| 7.2 µm | −18.0 | [−61.2, +21.8] | [−36.0, +14.4] | [−21.6, −14.4] | −72.0 (censored) / +16.90 |
| **9.6 µm** | −14.4 | [−62.4, +4.8] | [−38.4, +9.6] | [−19.2, −9.6] | −76.8 (censored) / +25.53 |
| 12 µm | −6.0 | [−78.0, +6.0] | [−36.0, +12.0] | [−12.0, 0.0] | −84.0 (censored) / +20.71 |

- The positive fail edge moves between 16.9 and 25.5 µm, and the peak between −18 and −6 µm.
- The negative edge is censored at every width.

**Informative-crop subset** (35 crops with mean r ≥ 0.2)

| frame | peak | safe | proven-safe core | fail |
|---|---|---|---|---|
| band | −24.0 µm (bootstrap CI [−43.2, +4.8]) | [−57.6, +9.6] | [−28.8, −19.2] | +22.51 (negative side censored at −76.8) |
| window | – | [−4.8, 0.0] µm | – | −33.12 µm |

**Per-crop optimum.** Spearman ρ of best dz vs crop band offset = 0.196 (p = 0.18, n = 48). This
is in the expected direction but not significant.

**Calibration flag share** (formerly "specificity"). This is the share of each class when the
auditor runs on the same 48 ink-bearing calibration crops, measured on the CT.

- It is **not** a specificity: the meshes are not verified-correct, and the number of truly clean
  cells is unknown.
- Every crop is audited (the CT is re-read when the cache was evicted).
- These crops were used to set the thresholds, so this is in-sample.

| rules | OK | WARN | OFF_SHEET | SWITCH | crops > 10% FAIL |
|---|---|---|---|---|---|
| audit rules (smoothing + region size) | 68.58% | 26.58% | 3.90% | 0.94% | 7 / 48 |
| per-cell rules, no smoothing | 63.73% | 30.07% | 4.99% | 1.20% | 10 / 48 |

Reason shares under the audit rules:

| reason | share |
|---|---|
| NOMINAL_IN_AIR | 11.27% |
| DRIFT_FAIL | 3.31% |
| COMPETING_BANDS | 2.80% |
| LATERAL_JUMP | 1.12% |
| AIR_GAP_FAIL | 0.74% |
| SHEET_ID_UNCERTAIN | 0.04% |
| DRIFT_BEYOND_MEASURED | 0.01% |

WARN rose from 22.5% (round 1) to 26.6% because `NOMINAL_IN_AIR` is now always at least a WARN.

## 3. Full-segment audit: PHerc0139 w023 (level 3)

`python -m segment_qa audit --sample PHerc0139 --full --level 3 --mark 12620,15708`

| class | cells | share of valid |
|---|---|---|
| 0 OK | 5,697,076 | 66.02% |
| 1 WARN_DRIFT | 2,317,678 | 26.86% |
| 2 FAIL_OFF_SHEET | 535,557 | 6.21% |
| 3 FAIL_SHEET_SWITCH | 78,472 | 0.91% |
| 255 INVALID_PAD | 1,818,217 | – |

- Valid cells: 8,628,783. FAIL share: 7.1%. `train_keep_mask` keeps 7,487,852 cells (OK or WARN,
  minus a 1-cell margin around FAIL).
- **Runtime.** The sweep took 305.6 s: 137 s for the 49 blocks and 164 s for the exact L0
  validity pass.
- **Exact L0 validity.**
  - 13,960 boundary candidates were checked, using 853 L0 regions of 128 px.
  - **13,708 cells that the mean-pooled level counted as valid are L0 padding.** They are now
    `INVALID_PAD` and take no part in lateral-step classification.

**Reasons** (cells; a cell can have several):

| reason | cells | of which in FAIL classes |
|---|---|---|
| DRIFT_WARN | 2,200,626 | 517,410 |
| NOMINAL_IN_AIR | 1,056,661 | 309,635 |
| DRIFT_FAIL | 434,605 | 434,605 |
| COMPETING_BANDS | 199,615 | 43,605 |
| AIR_GAP_FAIL | 128,144 | 128,144 |
| LATERAL_JUMP | 103,310 | 103,310 |
| WINDOW_EMPTY | 24,598 | 24,598 |
| NO_BAND | 13,207 | 13,207 |
| DRIFT_BEYOND_MEASURED | 4,649 | 4,637 |
| SHEET_ID_UNCERTAIN | 4,115 | 4,115 |

- `NOMINAL_IN_AIR`: 747,026 WARN cells carry it. Since round 1 it always makes a cell at least
  WARN.
- `DRIFT_BEYOND_MEASURED`: only 12 cells are WARN for this reason alone. In the other 4,637 the
  band is so far below the anchor that the air-gap or empty-window rule already fails them.
- `SHEET_ID_UNCERTAIN`: 4,115 cells carry an uncertain switch. 3,369 of them are in the switch
  class (4.3% of it), and OFF_SHEET outranks the other 746.

**Band summary**

- 8,334,107 cells measured.
- |Δz| p50 / p90 / p95 / max = 18.09 / 43.10 / 50.35 / 119.73 µm.
- Signed median −11.49 µm, which matches the calibration crops' −12.33. Median spacing to the next
  band is 66.59 µm.

**Coarse vs L0 cross-check** (deterministic sampling)

- Tile origins lie on a 16-cell lattice in raster order. Evenly spaced ranks are taken from 32,872
  interior candidates and 1,559 padding-boundary candidates.
- Each tile is a 512 px L0 crop audited at 8×8 px cells. The 128 px halo is trimmed, and the
  central 256 px is compared with the L3 map.

| sample | tiles | cells | exact agreement | Cohen κ [95% CI, tile bootstrap] | fail/non-fail |
|---|---|---|---|---|---|
| interior | 24 | 24,576 | 99.79% | 0.9957 [0.9933, 0.9977] | 99.97% |
| padding boundary: valid mask | 8 | 8,192 | 100.00% (4,757 invalid, 3,435 valid in both) | – | – |
| padding boundary: classes, cells valid in both | 8 | 3,435 | 99.77% | 0.9957 | 100.00% |

- Interior cells that are coarse-valid but L0 padding: 0.
- Padding is reported separately from the class agreement, so it no longer inflates κ.

**Row B mark (12620, 15708)**

- FAIL_OFF_SHEET at L3, as at L0: drift +29.36 µm, switch jump 99.80 µm, gap 13.87 µm.
- Its FAIL region has only 4 cells, with a summed severity of 8.23. That ranks it about 11,700th of
  29,535 FAIL regions, so it is not in the saved top 500.

**Top hotspots** (ranked by summed severity):

| rank | class | cells | mm² | score | peak (L0 Y, X) | key reasons |
|---|---|---|---|---|---|---|
| 1 | OFF_SHEET | 13,813 | 5.09 | 29,089 | 7452, 5180 | window empty, gap 83 µm, drift −105 µm (beyond measured), jump 190 µm |
| 2 | OFF_SHEET | 6,629 | 2.44 | 14,616 | 17004, 3812 | window empty, gap 120 µm |
| 3 | OFF_SHEET | 6,645 | 2.45 | 11,260 | 24020, 10628 | window empty, drift +102 µm, gap 74 µm |
| 4 | OFF_SHEET | 3,959 | 1.46 | 8,683 | 564, 12236 | window empty, drift +103 µm, jump 195 µm |
| 5 | OFF_SHEET | 3,803 | 1.40 | 7,319 | 22380, 5092 | window empty, gap 113 µm |

Rank 1 is the red blob at the left of panel (a).

**Independent ink concordance, held out**

- Per 256 px tile, this is the Pearson r between the published 2.4 µm ink map and the independent
  1.129 µm reference, grouped by the tile's FAIL share.
- Every 1024 px block that touches a calibration crop is dropped: 57 blocks, 704 tiles, including
  the 87 tiles that overlap a crop directly.
- That leaves 1,823 tiles in 161 blocks. CIs come from a block bootstrap (2000 draws of whole
  blocks), because neighbouring tiles are not independent.

| FAIL share in tile | tiles | blocks | mean r [95% CI, block bootstrap] | median r |
|---|---|---|---|---|
| no FAIL cell | 312 | 125 | 0.354 [0.303, 0.403] | 0.401 |
| < 5% | 949 | 155 | 0.337 [0.306, 0.367] | 0.393 |
| 5–20% | 461 | 136 | 0.310 [0.272, 0.350] | 0.351 |
| ≥ 20% | 101 | 52 | 0.201 [0.104, 0.285] | 0.213 |

- No-FAIL minus ≥ 20%: Δr = +0.154 [0.067, 0.256].
- Spearman ρ of tile r vs share, with block-bootstrap 95% CIs:

  | share | ρ [95% CI] |
  |---|---|
  | FAIL | −0.084 [−0.139, −0.028] |
  | off-sheet | −0.096 [−0.146, −0.043] |
  | WARN | +0.006 [−0.052, +0.066] (ns) |

- The trend is monotone on held-out tiles: heavily flagged tiles lose about 43% of their
  agreement. The per-tile correlation is weak, though, because most tiles have few FAIL cells and
  r is noisy.

## 4. Hotspot verification (three scrolls)

`python -m segment_qa verify` audits a 256 px L0 crop around each documented hotspot, checks the
documented neighbour pair, and audits a **clean control**. Each check must pass on its own, and
`all_passed` is true.

**Control selection.** Candidates are taken in a fixed a-priori order:

1. priority controls;
2. published ds8 ink-preview text richness within 8192 px;
3. the whole segment.

Each candidate must be fully valid CT and ≥ 1024 px from the hotspot. The control is the first
candidate whose L0 CT audit has **FAIL < 2%**. Every tried candidate is recorded.

| case | point (L0 Y, X) | class at point (reasons) | key numbers | pair check | crop FAIL | control (Y, X): FAIL / WARN, rank |
|---|---|---|---|---|---|---|
| PHerc0139 w023, Row B | 12620, 15708 | FAIL_OFF_SHEET (DRIFT_FAIL, NOMINAL_IN_AIR, LATERAL_JUMP) | drift +29.4 µm; switch jump 99.82 µm (= 41.61 vx, as documented); gap 13.87 µm | (12620, 15708)→(12620, 15716): **switch**, 99.82 µm, bands end, flagged ✓ | 7.4% | (20608, 19328): 1.66% / 34.7%, #1 (priority) |
| PHerc1667 w029 flatboi, Row G | 18980, 15740 | **WARN_DRIFT (COMPETING_BANDS**, NOMINAL_IN_AIR, DRIFT_WARN) | drift −41.0 µm (cell −51.5); flip 122.95 µm (= 51.25 vx, as documented); switch 0; gap 16.2 µm | (18980, 15740)→(18980, 15748): **flip**, both bands continue (exact), flagged ✓ | 5.9% | (21504, 23168): 0.49% / 68.4%, #3 |
| PHercParis4 w053-058, Row I | 24484, 93396 | **FAIL_OFF_SHEET (WINDOW_EMPTY, AIR_GAP_FAIL**, NOMINAL_IN_AIR, DRIFT_BEYOND_MEASURED) | drift −95.3 µm (cell −109.3); air gap 95.65 µm (doc 95.6); window empty | (24468, 93388)→(24476, 93388): **flip**, 183.62 µm (doc 76.5 vx = 183.5 µm), both continue, flagged ✓ | 48.7% | (30336, 95616): 1.17% / 29.6%, #10 |

**Row B.** Reproduced. The documented 41.61 vx jump is measured exactly, and the pair is a true
switch: the selected bands are 24 layers apart and neither continues. The point itself is
FAIL_OFF_SHEET, because OFF outranks SWITCH.

**Row G.** The documented 51.25 vx step is a nearest-band flip. Both sheets continue across the
edge (exact interval test), and the anchor lies in the air between them. The cell is therefore
WARN_DRIFT with `COMPETING_BANDS`, not a switch.

- The crop's rank-1 hotspot is FAIL_OFF_SHEET (26 cells), peak (18980, 15708), 32 px away.
- The first two control candidates had 16.6% and 6.2% FAIL. The accepted control (#3) is
  FAIL-clean but **WARN-heavy (68.4%)**: its band sits low, near z ≈ 30. It is a clean control
  for FAIL, not for WARN.

**Row I.** Two separate checks:

1. **The off-sheet flag at (24484, 93396).** FAIL comes from the empty window and the 95.65 µm
   air gap. The −95 µm drift is beyond the censored −76.8 µm edge, so by itself it would only add
   `DRIFT_BEYOND_MEASURED`.
2. **The handoff (24468, 93388) → (24476, 93388).** It is a flip between continuing sheets, at
   +86.44 µm and −97.18 µm. Both cells are FAIL_OFF_SHEET in their own right.

The first nine Paris4 candidates were not clean (5.7%–40.1% FAIL). Candidate #10, still within
8192 px, has 1.17% FAIL.

**Coarse vs L0 at the hotspots.** The level-3 class at each hotspot cell matches L0: B OFF,
G WARN, I OFF. Level-3 κ against L0 per crop:

| | hotspot crop | control crop |
|---|---|---|
| B | 0.994 | 1.000 |
| G | 0.993 | 1.000 |
| I | 0.996 | 0.996 |

## 5. Caveats

1. **The thresholds are provisional**, in two ways (`thresholds.provisional_scope`):
   - **Cross-scroll.** Measured on one PHerc0139 segment, one 2.399 µm scan and one ink model.
     Applying them to PHerc1667, PHercParis4 or other scans assumes the same tolerance and is
     untested.
   - **Crop-to-cell.** Measured for a coherent shift of whole 256 px (614 µm) crops. The auditor
     applies them to same-sheet 5×5-cell (96 µm) medians, a scale the calibration did not test.
   - Every threshold block and figure is labelled provisional.
2. **Noisy reference.**
   - At dz = 0 the median crop r is only 0.377, and the minimum is −0.343.
   - The 1.129 µm map is aligned to the 2.4 µm grid only at ds8 (19 µm cells), and it covers
     about 31% of the segment.
   - Retention therefore measures relative loss, not absolute ink accuracy.
3. **Asymmetric tolerance, and why the two frames mirror.**
   - Agreement peaks with the band about 14 µm **below** the anchor, but the bootstrapped peak CI
     is wide ([−62.4, +4.8]).
   - Agreement falls quickly once the band moves **above** the anchor: crossed fail at +25.53 µm.
     It stays above 0.70 down to the support edge at −76.8 µm.
   - A window shift of +dz moves the band to −dz relative to the window. The tolerant side
     therefore appears at positive dz in the window frame.
   - The auditor uses the directly measured band frame, so no sign flip is involved.
4. **The negative fail edge is censored.** −76.8 µm is the edge of the supported offsets, not a
   crossing. Drift beyond it is a WARN (`DRIFT_BEYOND_MEASURED`). The only FAIL constraints on
   that side are the window-empty and air-gap rules.
5. **Post-hoc rule choice.** The band frame first used the CI rule. That gave [−19.2, −9.6] µm,
   which excludes 0 and depends on the bin width. It was switched to the point rule after seeing
   this. The CI core is still reported, and the window frame still uses the CI rule.
6. **Noise rules were set after reading the hotspot documentation.** The 40 µm switch, the
   ≥ 4-cell region and the 5-cell smoothing came from physical scale, but only after reading
   Sol's jump magnitudes (41.6, 51.25, 76.5 vx). They are not fitted to the calibration data.
7. **Self-consistency is non-monotonic.** For example, 0.876 at −8 but 0.677 at −12. The output
   texture changes with the window even where agreement does not drop. It is reported, not used.
8. **Weak per-crop signal.** The best-dz vs offset correlation is ρ = 0.20 (ns). Only the pooled
   curve is informative.
9. **Controls are clean for FAIL only.** "Clean" means FAIL < 2% on the CT audit. The Row G
   control is 68% WARN.
10. **Resolution.** The full map is at L3 (19 µm cells), so features narrower than one cell can
    be missed. Padding is exact at L0; the interior agreement is sampled (§3).
11. **Calibration flag share is in-sample.** The calibration crops are the crops the thresholds
    were fitted on. The held-out concordance in §3 is the out-of-sample check.

## 6. Reproduce

Install, then run from the repo root with `.venv\Scripts\python.exe` (Windows). Outputs go to
`outputs/segment_qa/`, and S3 data is cached in `cache/segment_qa/`.

```bash
pip install -e ".[segment-qa]"              # zarr>=3, pillow; add segment-qa-calibrate for torch

# GPU calibration (about 7 min on a GTX 1070 Ti), then statistics from the saved raw arrays (no GPU)
.venv/Scripts/python.exe -m segment_qa calibrate --n-crops 48
.venv/Scripts/python.exe -m segment_qa calibrate --reanalyze

# Whole-segment level-3 audit with the Row B mark (exact L0 padding, cross-check, held-out concordance)
.venv/Scripts/python.exe -m segment_qa audit --sample PHerc0139 --full --level 3 --mark 12620,15708

# Single 256 px L0 crop audit (bounds checked before any S3 read)
.venv/Scripts/python.exe -m segment_qa audit --sample PHercParis4 --crop 24448,93312 --mark 24484,93396

# Three-scroll hotspot verification (pair checks, clean controls)
.venv/Scripts/python.exe -m segment_qa verify

# Tests
.venv/Scripts/python.exe -m pytest -q tests/test_segment_qa.py
```

**CLI options**

- `--catalog PATH` (global) selects the segment catalog.
- `--layers-key KEY` (audit) selects another layers asset.
- `audit` reads `outputs/segment_qa/ink_tolerance_curve.json` by default:
  - `--calibration none` uses the built-in defaults;
  - `--switch-jump-um` and `--air-gap-fail-um` override single thresholds.

**Environment variables**

- `SEGMENT_QA_ROOT` sets the root for all defaults below.
- `SEGMENT_QA_CATALOG`, `SEGMENT_QA_CACHE`, `SEGMENT_QA_OUT` and `SEGMENT_QA_MODELS` override
  single paths.

**Cache**

- The crop cache is keyed by the resolved asset: scroll, segment, volume and a hash of the full
  layers-zarr path.
- Writes are atomic (temp file + rename).
- The hard ceiling is 450 MB, below the 500 MB budget. An entry larger than the ceiling is
  rejected rather than evicting everything.

## 7. Files

- **Code:**
  - `segment_qa/band_detector.py`
  - `segment_qa/auditor.py`
  - `segment_qa/ink_tolerance.py`
  - `segment_qa/runs.py`
  - `segment_qa/viz.py`
  - `segment_qa/data.py`
  - `segment_qa/cli.py`
  - `segment_qa/__main__.py`
- **Tests:** `tests/test_segment_qa.py` (58 tests).
- **Output names** are qualified by segment and volume: `<sample>_<segment-id>_<volume-id>`, e.g.
  `pherc0139_20260311000000_20260102150214`.
- **Figures:**
  - `ink_tolerance_curve.png`
  - `pherc0139_20260311000000_20260102150214_full_qa_overlay.png`
  - `hotspot_verification.png`
- **Data:**
  - `ink_tolerance_curve.json`
  - `ink_tolerance_raw.npz`
  - `ink_tolerance_crops.json`
  - `pherc0139_20260311000000_20260102150214_full_{summary.json, hotspots.json, fields.npz, train_keep_mask.png, qa_classes.png}`
  - `hotspot_verification.json`
  - `verification/verify_<tag>_{B,G,I}*`
