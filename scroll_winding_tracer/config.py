"""Configuration dataclasses and tuned presets for conservative winding tracing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateConfig:
    """7-Gate 3D spatiotemporal verification thresholds."""

    min_valley_px: float = 3.5
    min_spacing_px: float = 8.0
    max_spacing_px: float = 55.0
    max_spacing_jump_ratio: float = 1.85
    min_cos_angle: float = 0.78
    min_coherence: float = 0.58
    ribbon_half_w: int = 24             # 49 px perpendicular width -> catches blind-ended torn flaps
    inner_half_w: int = 10              # 21 px inner core for strict 3D gap-intruder check
    min_wide_span_frac: float = 0.82    # sheet must span >=82% of wide ribbon (no dead ends)
    min_z_persistence: float = 0.75     # sheet must persist across >=75% of Z-slab frames
    max_intruder_frac: float = 0.12     # max persistent 3D intruder footprint in inner gap
    min_chain_len: int = 3
    stream_step_px: float = 2.0
    max_turn_cos: float = 0.90
    seed_spacing_px: int = 160
    min_radial_trend: float = 0.15
    half_z: int = 8                     # 17 contiguous Z-slices (Z-8 .. Z+8)


PRESETS: dict[str, GateConfig] = {
    "strict": GateConfig(
        ribbon_half_w=24,
        min_wide_span_frac=0.82,
        min_z_persistence=0.75,
        min_cos_angle=0.78,
        min_coherence=0.58,
    ),
    "balanced": GateConfig(
        ribbon_half_w=14,
        min_wide_span_frac=0.78,
        min_z_persistence=0.70,
        min_cos_angle=0.76,
        min_coherence=0.55,
    ),
}
