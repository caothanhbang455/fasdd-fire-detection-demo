"""
src/quality/eda.py

Dataset-wide EDA over segmentation pseudo-labels:
  - per-image coverage status (perfect / partial / empty / over)
  - exact-duplicate polygon line detection (this is what ultralytics
    silently drops at load time -- see `n_exact_dup` below)
  - per-polygon validity (degenerate, out-of-range coordinates)

This factors out the logic used in FASDD_CV_Seg_EDA_Train (S5) so it can
be reused by both the Kaggle training notebook and the standalone
Colab quality-EDA notebook.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..data.annotations import GTBox


def poly_area_normalized(coords: List[float]) -> float:
    """Shoelace formula on normalized polygon coordinates."""
    xs, ys = coords[0::2], coords[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    area = abs(
        sum(xs[i] * ys[i + 1] - xs[i + 1] * ys[i] for i in range(n - 1))
        + xs[-1] * ys[0] - xs[0] * ys[-1]
    ) / 2
    return float(area)


def coverage_status(n_gt: int, n_seg: int) -> str:
    if n_seg == 0:
        return "empty"
    if n_seg == n_gt:
        return "perfect"
    if n_seg > n_gt:
        return "over"
    return "partial"


def scan_image_coverage(
    stem: str,
    gt_boxes: List[GTBox],
    seg_file: Path,
    split: str = "unknown",
) -> Dict:
    """
    One row of dataset-level coverage stats for a single image, including
    exact-duplicate-line count -- the silent-drop issue ultralytics has
    at dataloader time (`Counter(raw_lines)` finds copies of the exact
    same line, which collapse to one annotation when ultralytics reads
    the file, effectively losing a GT box's supervision).
    """
    seg_file = Path(seg_file)
    n_gt = len(gt_boxes)
    raw_lines: List[str] = []
    if seg_file.exists() and seg_file.stat().st_size > 0:
        raw_lines = [l.strip() for l in seg_file.read_text().splitlines() if l.strip()]

    n_seg = len(raw_lines)
    line_counts = Counter(raw_lines)
    n_exact_dup = sum(c - 1 for c in line_counts.values())  # extra copies beyond the first

    return {
        "stem": stem,
        "n_gt": n_gt,
        "n_seg": n_seg,
        "n_exact_dup": n_exact_dup,
        "status": coverage_status(n_gt, n_seg),
        "has_file": seg_file.exists(),
        "split": split,
    }


def scan_polygon_quality(
    stem: str,
    gt_boxes: List[GTBox],
    seg_file: Path,
) -> List[Dict]:
    """
    Per-polygon-line validity rows: degenerate (< 3 points or odd coord
    count), out-of-range coordinates, exact-duplicate flag, normalized
    area, and whether the polygon centroid actually falls inside a GT
    box of the same class.
    """
    seg_file = Path(seg_file)
    rows: List[Dict] = []
    if not seg_file.exists() or seg_file.stat().st_size == 0:
        return rows

    raw_lines = [l.strip() for l in seg_file.read_text().splitlines() if l.strip()]
    line_counts = Counter(raw_lines)

    for line in raw_lines:
        parts = line.split()
        n_tok = len(parts)
        n_coords = n_tok - 1
        is_dup = line_counts[line] > 1
        is_degen = (n_tok < 7) or (n_coords % 2 != 0)

        try:
            cls_id = int(parts[0])
        except (ValueError, IndexError):
            cls_id = -1

        coords_ok = False
        poly_area = 0.0
        inside_ratio = 0.0

        if not is_degen and cls_id >= 0:
            try:
                coords = list(map(float, parts[1:]))
                coords_ok = all(0.0 <= v <= 1.0 for v in coords)
                poly_area = poly_area_normalized(coords)
                xs, ys = coords[0::2], coords[1::2]
                px, py = sum(xs) / len(xs), sum(ys) / len(ys)
                for b_cls, b_cx, b_cy, b_bw, b_bh in gt_boxes:
                    if b_cls != cls_id:
                        continue
                    bx1, by1 = b_cx - b_bw / 2, b_cy - b_bh / 2
                    bx2, by2 = b_cx + b_bw / 2, b_cy + b_bh / 2
                    if bx1 <= px <= bx2 and by1 <= py <= by2:
                        inside_ratio = 1.0
                        break
            except (ValueError, ZeroDivisionError):
                coords_ok = False

        rows.append({
            "stem": stem, "cls_id": cls_id, "n_tok": n_tok,
            "is_degenerate": is_degen, "coords_ok": coords_ok,
            "is_dup": is_dup, "poly_area": poly_area,
            "centroid_inside_gt": inside_ratio,
        })
    return rows


def scan_dataset_coverage(
    ann_stems: List[str],
    img_ann_map: Dict[str, List[GTBox]],
    labels_seg_dir: Path,
    split_map: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the full per-image coverage DataFrame used by every plot."""
    labels_seg_dir = Path(labels_seg_dir)
    split_map = split_map or {}
    rows = [
        scan_image_coverage(
            stem, img_ann_map[stem], labels_seg_dir / f"{stem}.txt",
            split_map.get(stem, "unknown"),
        )
        for stem in ann_stems
    ]
    return pd.DataFrame(rows)


def scan_dataset_polygons(
    ann_stems: List[str],
    img_ann_map: Dict[str, List[GTBox]],
    labels_seg_dir: Path,
) -> pd.DataFrame:
    """Build the full per-polygon-line quality DataFrame."""
    labels_seg_dir = Path(labels_seg_dir)
    rows = []
    for stem in ann_stems:
        rows.extend(scan_polygon_quality(stem, img_ann_map[stem], labels_seg_dir / f"{stem}.txt"))
    return pd.DataFrame(rows)


def still_missing_after_refinement(df_cov: pd.DataFrame) -> pd.DataFrame:
    """
    Images that are still `partial` or `empty` after the full 4-tier
    cascade -- i.e. boxes the cascade could not fill above
    `min_coverage_keep`. These are exactly the cases the standalone
    quality-EDA notebook should visualize.
    """
    return df_cov[df_cov["status"].isin(["partial", "empty"])].copy()
