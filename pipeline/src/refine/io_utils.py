"""
src/refine/io_utils.py

Low-level conversions between binary masks, YOLO polygons, and the
on-disk segmentation label format used across the refinement pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..data.annotations import yolo_to_pixel_xyxy  # re-exported for convenience

__all__ = [
    "yolo_to_pixel_xyxy",
    "mask_to_yolo_polygon",
    "compute_coverage",
    "parse_seg_file",
]


def mask_to_yolo_polygon(
    binary_mask: np.ndarray,
    img_w: int,
    img_h: int,
    approx_eps: float = 0.005,
) -> List[float]:
    """
    Convert a binary mask to a single normalized YOLO polygon
    (largest external contour, simplified with approxPolyDP).
    Returns [] if no usable contour is found.
    """
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, closed=True)
    if peri < 1:
        return []

    approx = cv2.approxPolyDP(cnt, epsilon=approx_eps * peri, closed=True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return []

    norm: List[float] = []
    for x, y in pts:
        norm.append(round(max(0.0, min(1.0, float(x) / img_w)), 6))
        norm.append(round(max(0.0, min(1.0, float(y) / img_h)), 6))
    return norm


def compute_coverage(
    mask_bool: np.ndarray,
    cx: float, cy: float, bw: float, bh: float,
    img_w: int, img_h: int,
) -> float:
    """Fraction of the GT bbox pixel area covered by the predicted mask."""
    x1, y1, x2, y2 = yolo_to_pixel_xyxy(cx, cy, bw, bh, img_w, img_h)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    inside = int(mask_bool[y1:y2, x1:x2].sum())
    return float(inside) / bbox_area


def parse_seg_file(seg_file: Path) -> List[Dict]:
    """
    Parse an existing segmentation label file into a list of entries:
        {'cls': int, 'line': str, 'cx': float, 'cy': float}
    `cx`/`cy` are the polygon centroid, used later for GT matching.
    """
    result: List[Dict] = []
    seg_file = Path(seg_file)
    if not seg_file.exists() or seg_file.stat().st_size == 0:
        return result

    for line in seg_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            cls_id = int(parts[0])
            pts = list(map(float, parts[1:]))
            xs, ys = pts[0::2], pts[1::2]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            result.append({"cls": cls_id, "line": line, "cx": cx, "cy": cy})
        except (ValueError, ZeroDivisionError):
            continue
    return result
