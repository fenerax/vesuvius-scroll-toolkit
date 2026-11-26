# Candidate B: Raw-CT Recto Snapping and Sheet-Switch QA

Status: first-pass prototype run on 2026-11-24. Verdict: **the snapper works geometrically, but it did not improve `ink_canonical_2um` on any test crop, and it made agreement with the independent reference worse on every crop.** Plain ink-window snapping should not be the submission. The useful parts are the placement-tolerance measurement and the QA signal (see "Next steps").

- Script: [candidate_b_proto.py](<C:/Users/User/.t3/projects/vesuvius-scrolls-deciphering/scripts/candidate_b_proto.py>)
- Main figure: [candidate_b_proto.png](<C:/Users/User/.t3/projects/vesuvius-scrolls-deciphering/outputs/candidate_b_proto.png>)
- ROI overview: [candidate_b_overview.png](<C:/Users/User/.t3/projects/vesuvius-scrolls-deciphering/outputs/candidate_b_overview.png>)

Run: `.venv\Scripts\python.exe scripts\candidate_b_proto.py` (about 3 minutes on the GTX 1070 Ti; streams about 70 MB of L5 data and 4 × 28.6 MB L0 crops; caches go to `%TEMP%\candidate_b_cache`, outside the repo).

## 1. Upstream reality check

| Tool | Where | What it does | Raw CT? | Regularized? | Measures ink impact? |
|---|---|---|---|---|---|
| `refine_labels.py` | villa `vesuvius/src/vesuvius/label_processing/` | Inverse-EDT dilation and ridge thinning of binary labels | No | No | No |
| `vc_objrefine` / `ObjAlphaCompRefinement.cpp` (VC3D `AlphaCompRefineRequest`, `scripts/batch_objrefine.sh`) | villa `volume-cartographer/apps/src/` | Moves each vertex along its normal by an alpha-composited CT offset (start −2, stop 30, step 2, opacity ramp 118–165/255, 7×7 blur) | **Yes** | No: per-vertex only, no lateral solve, no air-gap or adjacent-wrap gate | No |
| `lasagna/snap_surf` (+ `*_snap*.json` configs) | villa `lasagna/` | Snaps fitted surfaces to **predicted** normal and winding-density volumes, with smoothness, bend and Jacobian terms | No (ML fields) | Yes | No |
| `ink-detection/optimized_inference` | villa | Fixed `[start_z, end_z)` window plus `is_reverse_segment`; no per-pixel offset and no placement QA | – | – | – |
| **`tifxyz_snap`** (`src/flatten/tifxyz_recto.{h,c}`, October 2026) | ScrollFiesta, `Hob3rMallow/scrollfiesta_public` | Snaps any tifxyz segment onto the papyrus **recto edge in raw CT**. Per node it reads a smoothed profile along the inward normal and picks the nearest band crest. The recto edge is the steepest descent into the first trough that passes the gate. A node abstains when its edge is out of reach and never takes another band's edge. A per-scroll prior from rays sets the wrap spacing, gap and seam depth, and edge width. A ±7 median removes fibre weave, then a global CG solve (μ=3) runs with Cauchy IRLS. Moves are capped at 2 vox per iteration and 7 vox in total. | **Yes** | **Yes** | **No.** It reports only geometry: on its demo, 82.3% of nodes find the edge and the median offset falls from +0.44 to +0.01 vox |

The open-problems page (`scrollprize.org/docs/37_2026_open_problems.md`, updated 2026-10-14) says:

- "Label quality is now one of the main unwrapping bottlenecks."
- Label snapping ("using the raw CT signal … to move approximate labels back onto the most plausible papyrus surface") is a named direction.
- One listed cause of ink failure is "the surface may be slightly misplaced."

The page's snapping request is mainly about **surface-model training labels** (3D recto labels and meshes), not ink windows.

**Conclusion:** a regularized raw-CT recto snapper that respects air gaps already exists (ScrollFiesta `tifxyz_snap`). Building another one is reinvention. Three gaps are still open:
1. Nobody has published whether recto snapping actually changes or improves `ink_canonical_2um` output. This prototype measured it (below).
2. No upstream tool reports **placement or sheet-switch QA** as a per-segment map (where the mesh leaves its band or crosses wraps) before training or ink inference.
3. Nobody has shown that snapping **3D surface-model training labels** measurably improves surface-model topology.

## 2. Prototype method

Data: PHerc0139 segment `20260311000000-w023_202603112281`, volume `20260102150214` (2.399 µm), layers-zarr L0 `(109, 26960, 24800)`. The pyramid levels downsample only Y and X, so L5 `(109, 843, 775)` keeps the full 109-layer depth and supports a cheap scan of the whole segment.

Observation: the nominal surface (layer 54) sits near the papyrus → air-gap edge. Papyrus is at lower layer index, the gap at about 60–75, and the next wrap starts around 80.

Snapper (`snap_edges`, `dz_field_for_crop`):
1. Smooth with a Gaussian (σ = 0.75 in z, 2 px laterally) and average over 8×8 px blocks.
2. Compute a polarity-aware step score `S(z) = mean(I[z-4:z]) − mean(I[z:z+4])`, clipped at 0, over candidate layers 54 ± 12. Only papyrus→gap descents score, so the rising gap→next-wrap edge can never win. The ±12 range is under half the observed wrap pitch of about 25–30 layers.
3. Run 8-path semi-global matching (P1 = 0.15 for 1-layer steps, P2 = 1.2 for larger jumps), then a sub-layer parabola fit, a 3×3 median and confidence-weighted Gaussian smoothing. Variants are **fine** (σ = 1.5 blocks ≈ 12 px, which follows part of the fibre weave) and **drift-only** (σ = 5 blocks ≈ 40 px).
4. Measure the offset against the **segment-wide median edge layer `z_ref = 58.85`**, not against 54. The mesh's typical placement, about 4.85 layers (≈11.6 µm) on the papyrus side of the steepest descent, is the convention the ink model saw, so only local deviations get removed.
5. Re-slice `out[k] = V(23 + k + dz)` with linear interpolation, then run `ink_canonical_2um` (ResNet3D-152 + 3D decoder, FP32, tile 256, stride 128, Hann blending) on each variant.

Controls:
- **const:** a uniform shift equal to the mean of the drift-only dz. This separates the effect of the spatial field from a global window shift.
- **Uniform shift sweep:** window offsets of −8 to +4 layers.

References:
- The official published 2.4 µm ds8 map. It came from the *baseline* geometry, so it is biased toward the baseline.
- **An independent reference:** the published **1.129 µm-scan** ds8 ink map of the same segment (different scan, different model `mrg20736`). Its UV grid is the 2.4 µm grid scaled by 2.399/(2 × 1.129); after a plain resize the best shift is (0, 0) and the global r is 0.49. It covers 37% of the segment, so the drift ROIs were chosen inside that coverage.

ROIs (512 × 512, aligned to 128-px chunks):
- **drift_a** (Y 11904, X 21248) and **drift_b** (Y 14976, X 2176): coherent L5 offsets of −4.8 and −4.6 layers, std < 2, with text-bearing ink.
- **control** (Y 7680, X 7552): offset ≈ 0.
- **letters** (Y 10752, X 13184): the stage-2 letter-rich ROI.

## 3. Results (exact numbers from the 2026-11-24 run)

Segment-wide L5 scan (77 µm px, noisy): |dz| p50/p90/p99 = 3.5 / 7.1 / 14.8 layers. |dz| > 3 on 56% of pixels and > 6 on 23%. 20% of pixels hit the ±12 search bound, meaning no edge was found or it lies out of range. The map has visible row-wise streaks, so treat it as a coarse QA heat map, not as a measurement.

### Geometry: snapping works

Edge residual |res| in layers, measured with the same detector before and after re-slicing:

| ROI | baseline median / p90 | fine median / p90 | drift-only median / p90 | blocks with \|res\| > 3: base → fine → drift |
|---|---|---|---|---|
| drift_a | 5.79 / 11.86 | 0.99 / 3.09 | 2.34 / 6.56 | 78% → 11% → 40% |
| drift_b | 4.81 / 8.48 | 0.51 / 1.68 | 1.29 / 3.65 | 70% → 4% → 15% |
| control | 1.77 / 4.52 | 0.47 / 1.49 | 1.15 / 3.10 | 25% → 3% → 11% |
| letters | 4.71 / 10.75 | 1.19 / 5.32 | 2.67 / 7.93 | 69% → 22% → 45% (25% of blocks low-confidence) |

The snap takes 0.8 s per 512² crop on the CPU. Self-measured residuals are optimistic because the same detector judges itself. The visual XZ check in the main figure is the stronger evidence: the edge is flattened and does not jump to the neighbouring wrap.

### Ink: snapping did not help

Pearson r of the ink map (downsampled to ds8) against each reference:

| ROI | variant | r vs **1.1 µm ref** (independent) | r vs official 2.4 µm | r vs baseline | mean p | area p > 0.5 |
|---|---|---|---|---|---|---|
| drift_a | baseline | **−0.093** | 0.862 | 1 | 0.334 | 0.211 |
| | fine | −0.271 | 0.465 | 0.641 | 0.546 | 0.535 |
| | drift-only | −0.184 | 0.604 | 0.755 | 0.412 | 0.326 |
| | const (−5.5) | −0.204 | 0.669 | 0.862 | 0.342 | 0.190 |
| drift_b | baseline | **0.340** | 0.808 | 1 | 0.174 | 0.018 |
| | fine | 0.098 | 0.567 | 0.603 | 0.372 | 0.280 |
| | drift-only | 0.213 | 0.649 | 0.856 | 0.208 | 0.060 |
| | const (−4.5) | 0.331 | 0.693 | 0.930 | 0.149 | 0.000 |
| control | baseline | **0.301** | 0.799 | 1 | 0.760 | 0.808 |
| | fine | 0.142 | 0.532 | 0.729 | 0.902 | 0.988 |
| | drift-only | 0.222 | 0.727 | 0.916 | 0.824 | 0.890 |
| | const (−1.0) | 0.239 | 0.726 | 0.883 | 0.740 | 0.799 |
| letters | baseline | **0.448** | 0.945 | 1 | 0.535 | 0.523 |
| | fine | 0.304 | 0.810 | 0.818 | 0.594 | 0.593 |
| | drift-only | 0.377 | 0.893 | 0.916 | 0.594 | 0.599 |
| | const (−1.0) | 0.447 | 0.930 | 0.991 | 0.537 | 0.528 |

Uniform-shift sweep on **letters**, r vs the 1.1 µm reference: −8: 0.113, −6: 0.214, −4: 0.429, −2: 0.442, **0: 0.448**, +2: 0.392, +4: 0.333. The sweep curve is smooth. The original placement is the optimum, and it holds within about 0.02 from −4 to 0. On the low-signal crops (drift_b, control) the sweep is erratic: r vs baseline jumps between 0.55 and 0.99 for 2-layer changes. With no real text there, the model's output is not stable enough to judge.

What the numbers say:
1. **The fine snap hurts on every ROI.** It is worst on all four against the 1.1 µm reference, and on control it pushes the output to 99% "ink". This fits the ScrollFiesta warning: the fibre weave of about ±2 voxels over 8–12 px is texture the ink model uses, and flattening it removes signal.
2. **The drift-only snap also lowers independent agreement on every ROI** (−0.09 → −0.18, 0.34 → 0.21, 0.30 → 0.22, 0.45 → 0.38). The uniform "const" shift matches or beats the spatial field on 3 of 4 ROIs. The spatially varying correction is the harmful part.
3. Mean ink probability and area above 0.5 usually *rise* after snapping. These are not quality gains: the independent agreement drops at the same time, so the extra "ink" is very likely papyrus or weave texture.
4. **Placement tolerance:** on the only crop with readable text, `ink_canonical_2um` tolerates about −4 to 0 layers (−9.6 to 0 µm) of window offset with no loss, and degrades beyond +2 or −6. A mesh 5 layers off the median edge convention is therefore not, by itself, an ink failure.
5. On these crops, the recto-edge convention (`z_ref`) is not what the ink model wants. Its 62-layer window spans about ±31 layers and it seems to have learned the mesher's own placement noise. A geometric "truth" target does not translate into better ink.

Caveats:
- The evidence comes from 4 crops of 512² on one segment.
- The independent reference is weak: a different model, global r = 0.49, and r ≈ −0.1 on drift_a, where it carries almost no text signal.
- There is no ground-truth ink label.

These results are strong enough to reject the "snap the ink window" pitch. They are not enough to claim that snapping can never help, for example on real sheet switches where the mesh is more than 10 layers off.

## 4. Next steps (recommended pivot)

1. **Ink placement-tolerance curve (cheap, novel, directly answers an open-problem question).** Run the uniform sweep on about 50–100 text-rich ROIs across 2–3 scrolls, scored against the 1.1 µm reference where it exists. Publish "how far can a mesh sit off its band before `ink_canonical_2um` loses text", with a per-scroll curve. This addresses the page's "we do not always know which part of the pipeline is limiting us", and it decides whether placement correction is ever worth doing for ink.
2. **Placement and sheet-switch QA map, as a report and not a correction.** Reuse the bounded SGM evidence: search-bound hits, low edge confidence, |dz| beyond the tolerance from step 1, and lateral jumps above half the wrap pitch. The output is a per-segment flag raster plus a ranked list of regions to review before training or inference. drift_a row 332 already shows the kind of multi-sheet crossing this should flag. Calibrate it against the parallel agent's label-drift cases in `docs/label_drift_analysis.md`, which this agent did not read or touch.
3. **If snapping is still wanted, use it for 3D surface-model training labels and not ink windows,** and build on ScrollFiesta `tifxyz_snap` (it is already regularized and air-gap-safe) instead of re-implementing it. The contribution would be the evaluation harness: snapped versus original labels, then topology and sheet-merge metrics of the resulting surface model. That costs far more GPU time and needs coordination with ScrollFiesta's author.
4. If ink-window snapping is ever revisited, do it only in the drift-only form, only where |dz| exceeds the measured tolerance, and only with an independent reference or labels. Never do it at fibre-weave scale.
