"""Zero-training curved + 3D Z-slab conservative winding-constraint generator."""

from .config import GateConfig, PRESETS
from .exporters import export_vc3d_json, export_winding_inference_store_v1
from .tracer import (
    SlabFields,
    build_slab_fields,
    extract_vertical_same_winding_tracks,
    generate_curved_chains,
    measure_spatial_coverage,
)

__all__ = [
    "GateConfig",
    "PRESETS",
    "SlabFields",
    "build_slab_fields",
    "generate_curved_chains",
    "extract_vertical_same_winding_tracks",
    "measure_spatial_coverage",
    "export_vc3d_json",
    "export_winding_inference_store_v1",
]
