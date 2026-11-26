"""segment_qa: ink-calibrated raw-CT segment placement and sheet-switch QA.

Modules
-------
band_detector  Vectorized papyrus band / air-gap detector on (D, H, W) layers-zarr slabs.
ink_tolerance  Empirical ink_canonical_2um drift-tolerance calibration (window-shift sweep).
auditor        Per-cell QA classes, ranked hotspots and a training keep-mask.
runs           Crop / full-sweep audits, L0 cross-check, ink-reference concordance, verification.
data           Catalog lookup, anonymous S3 streaming and a size-bounded local cache.
viz            Publication figures (tolerance curve, full-segment overlay, hotspot verification).
"""

from .band_detector import BandParams, BandField, LateralSteps, detect_bands, lateral_jumps_um, lateral_steps
from .auditor import (
    OK, WARN_DRIFT, FAIL_OFF_SHEET, FAIL_SHEET_SWITCH, INVALID_PAD, CLASS_NAMES, REASONS, REASON_BIT,
    Thresholds, classify, find_hotspots, train_keep_mask,
)

__all__ = [
    "BandParams", "BandField", "LateralSteps", "detect_bands", "lateral_jumps_um", "lateral_steps",
    "OK", "WARN_DRIFT", "FAIL_OFF_SHEET", "FAIL_SHEET_SWITCH", "INVALID_PAD", "CLASS_NAMES", "REASONS", "REASON_BIT",
    "Thresholds", "classify", "find_hotspots", "train_keep_mask",
]

__version__ = "0.1.0"
