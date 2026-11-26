# Label drift in published Vesuvius surface volumes

Across ten targeted CT crops from three segments, the median distance from nominal `z=54` to the nearest measurable papyrus-band midpoint was **6.63 voxels (15.90 µm)**. It exceeded 25 voxels in **288 of 8,866 measured cells (3.25%)**. The strongest off-sheet evidence is PHercParis4 w053–058 at **(Y, X) = (24484, 93396)**. The nearest band ends 39.85 voxels (95.64 µm) short of the nominal slice, and the full `[23, 85)` input window falls in the gap between detectable bands. In PHerc0139 w023, offset had essentially zero pooled rank correlation with the published ink preview.

## Crops and coordinates

All coordinates are level-0 surface-volume coordinates in **(Y, X)** order. They refer to the flattened segment, so a coordinate needs its sample, segment and volume ID. Each crop reads 109 layers and 256 × 256 XY voxels in four aligned chunks. The run reads 40 chunks representing **68.125 MiB of decoded CT** in total. It reads the PHerc0139 ds8 JPEG from the existing local cache.

| Sample | Published segment | Target volume | Level-0 shape, ZYX | Z spacing |
|---|---|---|---|---:|
| PHerc0139 | `20260311000000-w023_202603112281` | `20260102150214` | 109 × 26960 × 24800 | 2.399 µm |
| PHerc1667 | `20251212185248-w029_20251212185248662_flatboi` | `20251217075048` | 109 × 38000 × 31320 | 2.399 µm |
| PHercParis4 | `20260623144957-w053-058` | `20260411134726` | 109 × 48640 × 186640 | 2.400 µm |

The table reports absolute band-midpoint offset in the order **median / p90 / p95 / max**. Boundary entries are the median signed offsets of the low- and high-z band faces from `z=54`. Gap counts use all supported cells, including cells where the nearest band is clipped by the stack. Midpoint statistics use only cells where both boundaries are measurable.

| Crop | Sample and region | Crop origin (Y, X) | Measured / supported | Median boundary Δz (vx), low / high | Absolute midpoint Δz (vx) | Absolute midpoint Δz (µm) | Nominal gap cells |
|---|---|---|---:|---:|---:|---:|---:|
| A | PHerc0139 upper, high preview contrast | (5120, 16128) | 1011 / 1024 | −4.38 / +16.66 | 7.98 / 15.25 / 20.04 / 29.05 | 19.15 / 36.59 / 48.07 / 69.68 | 205 |
| B | PHerc0139 middle, high preview contrast | (12544, 15616) | 999 / 1024 | −12.72 / +4.81 | 6.32 / 18.67 / 20.40 / 32.15 | 15.17 / 44.79 / 48.95 / 77.13 | 90 |
| C | PHerc0139 lower, high preview contrast | (20096, 19456) | 1024 / 1024 | −20.57 / +6.23 | 6.75 / 12.76 / 13.78 / 19.56 | 16.20 / 30.60 / 33.07 / 46.92 | 6 |
| D | PHerc0139 lower, preview-dark | (20608, 19328) | 1024 / 1024 | −8.85 / +12.76 | 4.91 / 9.81 / 10.83 / 13.91 | 11.77 / 23.54 / 25.99 / 33.38 | 9 |
| E | PHerc0139 right mesh edge | (7680, 21632) | 526 / 544 | −13.01 / +2.49 | 5.48 / 12.46 / 14.27 / 23.14 | 13.14 / 29.88 / 34.24 / 55.51 | 69 |
| F | PHerc0139 left mesh edge | (16000, 1280) | 443 / 473 | −7.55 / +6.87 | 4.81 / 11.56 / 14.56 / 28.04 | 11.54 / 27.72 / 34.92 / 67.27 | 41 |
| G | PHerc1667 interior | (18944, 15616) | 966 / 1024 | −8.62 / +7.54 | 6.34 / 17.25 / 20.89 / 29.78 | 15.22 / 41.39 / 50.11 / 71.43 | 136 |
| H | PHerc1667 left side | (18944, 2048) | 1020 / 1024 | −14.80 / +4.85 | 7.19 / 18.37 / 20.53 / 32.04 | 17.26 / 44.07 / 49.25 / 76.86 | 125 |
| I | PHercParis4 interior, steep folds | (24320, 93312) | 830 / 1024 | −11.54 / +8.18 | 14.29 / 32.22 / 35.58 / 45.54 | 34.31 / 77.32 / 85.38 / 109.29 | 641 |
| J | PHercParis4 west interior | (24320, 46464) | 1023 / 1024 | −13.57 / +8.14 | 7.50 / 20.43 / 22.30 / 31.34 | 17.99 / 49.03 / 53.51 / 75.22 | 385 |
| **Pooled** | | | **8866 / 9209** | | **6.63 / 18.41 / 21.86 / 45.54** | **15.90 / 44.16 / 52.47 / 109.29** | **1707** |

Of the 8,866 measured cells, **22.0%** were centered within 3 voxels and **38.5%** fell in the 8–25 voxel band. Among the 9,209 supported cells, **18.5%** put the nominal plane in a gap. The sampled PHerc0139 cells had **0.82%** severe midpoint offsets and **8.2%** nominal gaps. The sampled PHercParis4 cells had **12.2%** severe midpoint offsets and **50.1%** nominal gaps. These fractions describe deliberately chosen crops, and adjacent cells share spatial structure. Segment-wide rates require broader sampling.

## Hotspots for Candidate B

| Case | Sample and crop | Exact (Y, X) | Raw CT evidence |
|---|---|---|---|
| Centered control | PHerc0139 D | **(20740, 19460)** | One detected band, `z=47.75–64.39`, midpoint `56.07`, Δz **+2.07 vx (+4.97 µm)**. The region is dark in the ink preview. |
| Offset along a continuous band | PHerc0139 C | **(20228, 19588)** | Midpoint `43.98`, Δz **−10.02 vx (−24.04 µm)**. The low-z face is at `27.49` and the high-z face at `60.48`. The nominal plane is **6.48 vx (15.55 µm)** inward from the high-z face. |
| Suspected switch | PHerc0139 B | **(12620, 15708) → (12620, 15716)** | The nearest CT band midpoint jumps from `66.24` to `24.63` in 8 X voxels, a **41.61 vx (99.82 µm)** change. The XZ cut shows two distinct bands on opposite sides of the nominal plane. |
| Competing bands | PHerc1667 G | **(18980, 15740) → (18980, 15748)** | The chosen midpoint jumps from `32.53` to `83.78` in 8 X voxels, **51.25 vx (122.95 µm)**. Both bands are visible around a dark gap in the XZ cut. |
| Largest measured offset | PHercParis4 I | **(24484, 93396)** | The nearest band spans `z=2.78–14.15`, midpoint `8.46`, Δz **−45.54 vx (−109.29 µm)**. The cut shows a second band near the upper stack limit. Neither band intersects `[23, 85)`. |
| Largest midpoint handoff | PHercParis4 I | **(24468, 93388) → (24476, 93388)** | The selected midpoint drops from `90.02` to `13.51` in 8 Y voxels, **76.51 vx (183.62 µm)**. The nearby YZ cut shows steeply folded bands. The topology is ambiguous. |
| Off-sheet strip | PHercParis4 J | **(24324, 46564) → (24332, 46564)** | The strongest handoff in J is `30.94` to `85.34`, **54.40 vx (130.56 µm)**. A nominal-plane gap occurs in 385 of 1,024 supported cells in this crop. |

The gallery overlays the sampled midpoints and boundaries on unfiltered, contrast-stretched XZ and YZ cuts. Markers use the surrounding 8 × 8 CT cell and the image is a single raw plane. Jumps prove a local handoff in the nearest-band measurement. Confirming a true mesh switch requires following the intended winding in world coordinates. Folded sheets, separate wraps and delamination can all produce competing bands at this scale.

Crop I contains **89 / 1,024 cells (8.69%)** where no detected band intersects the 62-slice window. Its p95 offset stayed between **35.32 and 36.44 voxels** across the tested thresholds, and the nominal-gap count ranged from **616 to 676**. Every supported cell in the six sampled PHerc0139 crops had at least one detected band intersecting that window. Band overlap with the window leaves open whether the model received the correct sheet. [Model input window](<C:/Users/User/.t3/projects/vesuvius-scrolls-deciphering/scripts/stage2_ink_smoke.py>)

## Published ink preview

The PHerc0139 `20260417190342` ds8 prediction matched the CT grid at one JPEG pixel per 8 × 8 cell. Over 5,027 paired cells, Spearman **ρ = +0.012**. Ranking within each crop before pooling gave **r = +0.006**. The four interior crops alone gave **ρ = +0.108** over 4,058 cells. Individual crop coefficients ranged from **−0.416** (A) to **+0.252** (D). The edge crops had mean intensities near saturation even after CT padding was masked.

| Absolute midpoint Δz | Paired cells | Median JPEG intensity | JPEG pixels ≥128 |
|---|---:|---:|---:|
| ≤3 vx | 1061 | 52.0 | 44.5% |
| >3 to <8 vx | 2227 | 52.0 | 43.4% |
| 8–25 vx | 1698 | 114.5 | 49.0% |
| >25 vx | 41 | 45.0 | 46.3% |

Preview brightness has no consistent decline with drift in these crops. The severe group is small and the JPEG records model output. Actual ink dropout would require an independent ink reference or a paired inference run on manually verified bands. The centered, preview-dark D control illustrates why an area can look blank in the JPEG even when the nominal slice passes through the sheet. [Ink plot](<C:/Users/User/.t3/projects/vesuvius-scrolls-deciphering/outputs/label_drift_ink.png>)

## Measurement and reproduction

The script averages CT in nonoverlapping 8 × 8 XY cells, then smooths each profile along z with a 1.25-voxel Gaussian. It masks cells containing any all-zero z column. For each remaining profile it identifies continuous bands at `p15 + max(12, 0.30 × (p90 − p15))`. A band needs at least four samples above that threshold and a peak at least `max(22, 0.42 × (p90 − p15))` above p15. Linear interpolation gives the two threshold crossings. The selected band has the nearest boundary or interior to `z=54`, with a tie resolved by nearest midpoint. A gap requires a below-threshold nominal slice at least two voxels from the closest accepted band. In 343 supported cells the chosen band touched a stack limit, so those cells contribute to gap statistics and have no midpoint estimate.

The threshold and choice of nearest band are audit heuristics. An interval can enclose several touching sheets. The two boundaries are signed along the rendered stack. The physical recto faces inward in the scroll, and the renderer supports reversing stack polarity. The audited OME-Zarr metadata give spacing and axes but do not establish the recto-facing direction. Assigning the recto or verso at a hotspot requires an oriented world-coordinate normal or independent fiber evidence. [Vesuvius recto definition](https://scrollprize.org/2026_open_problems#surface-prediction) · [Renderer](https://github.com/ScrollPrize/villa/blob/main/volume-cartographer/apps/src/vc_render_tifxyz.cpp)

```powershell
.venv\Scripts\python.exe -B scripts\inspect_label_drift.py
```

The command uses anonymous S3 reads and prints per-crop coordinates, boundary distributions, handoffs and threshold sensitivity. It regenerates the gallery and ink plot. The margins above support a local geometry test on D, C, B and I, with the severe Paris4 gap as a case where an automatic snap needs to retain ambiguity about the correct sheet. [Audit script](<C:/Users/User/.t3/projects/vesuvius-scrolls-deciphering/scripts/inspect_label_drift.py>) · [CT gallery](<C:/Users/User/.t3/projects/vesuvius-scrolls-deciphering/outputs/label_drift_gallery.png>)
