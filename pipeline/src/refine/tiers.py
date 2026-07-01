"""
src/refine/tiers.py

The four fallback tiers used to generate a pseudo-mask for a GT box
that has no matching polygon yet. Tiers are tried in order; the loop
in pipeline.py stops at the first tier whose coverage clears
`min_coverage_keep` (or keeps the best of all four if none do).

Tier 1 -- Box + midpoint prompt, full image            (general fallback)
Tier 2 -- Crop + resize around the box, then SAM2       (small objects)
Tier 3 -- HSV color threshold                            (fire only)
Tier 4 -- GrabCut foreground extraction                  (smoke only)
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import RefineConfig
from .io_utils import compute_coverage

FIRE_CLS = 0
SMOKE_CLS = 1


def tier1_box_midpoint(
    sam2,
    img_path: str,
    x1: int, y1: int, x2: int, y2: int,
) -> Optional[np.ndarray]:
    """Prompt SAM2 with the GT box plus its midpoint, on the full image."""
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    try:
        res = sam2.predict(
            source=img_path,
            bboxes=[[x1, y1, x2, y2]],
            points=[[mid_x, mid_y]],
            labels=[1],
            verbose=False,
        )
        if res and res[0].masks is not None and len(res[0].masks.data) > 0:
            return res[0].masks.data[0].cpu().numpy().astype(bool)
    except Exception:
        pass
    return None


def tier2_crop_resize(
    sam2,
    img_cv: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    cfg: RefineConfig,
) -> Optional[np.ndarray]:
    """
    Crop a padded region around the box, resize to `crop_target_size`,
    run SAM2 there, then map the resulting mask back to original
    image coordinates. Helps small/tiny fire boxes that get lost when
    SAM2 sees the whole (much larger) frame.
    """
    img_h, img_w = img_cv.shape[:2]
    bw_px, bh_px = x2 - x1, y2 - y1
    pad_x = int(bw_px * cfg.crop_pad_ratio) + 1
    pad_y = int(bh_px * cfg.crop_pad_ratio) + 1

    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = img_cv[cy1:cy2, cx1:cx2]
    crop_h, crop_w = crop.shape[:2]
    scale = cfg.crop_target_size / max(crop_h, crop_w)
    resized = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)))

    # box in resized-crop coordinates
    bx1, by1 = int((x1 - cx1) * scale), int((y1 - cy1) * scale)
    bx2, by2 = int((x2 - cx1) * scale), int((y2 - cy1) * scale)

    try:
        res = sam2.predict(source=resized, bboxes=[[bx1, by1, bx2, by2]], verbose=False)
        if not (res and res[0].masks is not None and len(res[0].masks.data) > 0):
            return None
        mask_small = res[0].masks.data[0].cpu().numpy().astype(np.uint8)
    except Exception:
        return None

    # map back: resized-crop -> crop -> full image
    mask_crop = cv2.resize(mask_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    full_mask = np.zeros((img_h, img_w), dtype=bool)
    full_mask[cy1:cy2, cx1:cx2] = mask_crop.astype(bool)
    return full_mask


def tier3_hsv_fire(
    img_cv: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    cfg: RefineConfig,
) -> Optional[np.ndarray]:
    """Classical HSV threshold for fire-colored pixels inside the box."""
    img_h, img_w = img_cv.shape[:2]
    crop = img_cv[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # fire hues: red->yellow, roughly H in [0,35] or [340,360]->[0,35] after wrap
    lower1 = np.array([0, cfg.hsv_fire_min_sat, cfg.hsv_fire_min_val])
    upper1 = np.array([35, 255, 255])
    mask_crop = cv2.inRange(hsv, lower1, upper1)
    mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    full_mask = np.zeros((img_h, img_w), dtype=bool)
    full_mask[y1:y2, x1:x2] = mask_crop.astype(bool)
    return full_mask


def tier4_grabcut_smoke(
    img_cv: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    cfg: RefineConfig,
) -> Optional[np.ndarray]:
    """GrabCut foreground extraction seeded with the GT box (smoke only)."""
    img_h, img_w = img_cv.shape[:2]
    bw_px, bh_px = x2 - x1, y2 - y1
    if bw_px < 8 or bh_px < 8:
        return None

    mask = np.zeros((img_h, img_w), np.uint8)
    bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (x1, y1, bw_px, bh_px)
    try:
        cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, cfg.grabcut_iters, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(bool)
    return fg if fg.any() else None


def run_cascade(
    sam2,
    img_cv: np.ndarray,
    img_path: str,
    cls_id: int,
    x1: int, y1: int, x2: int, y2: int,
    bbox_area_norm: float,
    cfg: RefineConfig,
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Try tiers in order, stopping at the first one whose coverage clears
    `cfg.min_coverage_keep`. Returns (best_mask, best_coverage, tier_used)
    where tier_used in {0 (none worked), 1, 2, 3, 4}.
    """
    cx = (x1 + x2) / 2 / img_cv.shape[1]
    cy = (y1 + y2) / 2 / img_cv.shape[0]
    bw = (x2 - x1) / img_cv.shape[1]
    bh = (y2 - y1) / img_cv.shape[0]

    best_mask, best_cov, tier_used = None, 0.0, 0

    candidates = [(1, lambda: tier1_box_midpoint(sam2, img_path, x1, y1, x2, y2))]
    if bbox_area_norm < cfg.crop_area_thr:
        candidates.append((2, lambda: tier2_crop_resize(sam2, img_cv, x1, y1, x2, y2, cfg)))
    if cls_id == FIRE_CLS:
        candidates.append((3, lambda: tier3_hsv_fire(img_cv, x1, y1, x2, y2, cfg)))
    if cls_id == SMOKE_CLS:
        candidates.append((4, lambda: tier4_grabcut_smoke(img_cv, x1, y1, x2, y2, cfg)))

    for tier_id, fn in candidates:
        mask = fn()
        if mask is None:
            continue
        cov = compute_coverage(mask, cx, cy, bw, bh, img_cv.shape[1], img_cv.shape[0])
        if cov > best_cov:
            best_mask, best_cov, tier_used = mask, cov, tier_id
        if best_cov >= cfg.min_coverage_keep:
            break

    return best_mask, best_cov, tier_used
