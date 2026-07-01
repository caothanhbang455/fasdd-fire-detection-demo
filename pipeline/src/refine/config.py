"""
src/refine/config.py

Central place for every threshold used by the 4-tier mask refinement
cascade. Import this instead of hard-coding numbers in notebooks/scripts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RefineConfig:
    # SAM2 checkpoint name passed to ultralytics.SAM(...)
    sam2_model: str = "sam2.1_l.pt"

    # Tier 2: crop + resize for small objects
    crop_area_thr: float = 0.015      # bbox area (normalized) below this -> use Tier 2
    crop_target_size: int = 512
    crop_pad_ratio: float = 0.30

    # Tier 3 (fire, HSV) / Tier 4 (smoke, GrabCut)
    hsv_fire_min_sat: int = 90
    hsv_fire_min_val: int = 75
    grabcut_iters: int = 3

    # Whether a candidate mask is worth keeping at all
    min_coverage_keep: float = 0.15
    min_improvement: float = 0.05     # only overwrite an existing mask if it beats this margin

    # Matching: max normalized centroid distance to consider a polygon
    # already matched to a GT box (else treat the box as unmatched)
    match_max_centroid_dist: float = 0.30

    # Checkpointing
    save_every: int = 500

    seed: int = 42
