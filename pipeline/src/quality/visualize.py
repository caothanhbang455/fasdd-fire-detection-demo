"""
src/quality/visualize.py

Visual inspection helpers used by both the Kaggle training notebooks
and the standalone Colab quality-EDA notebook:
  - draw_overlay              : GT boxes + mask polygons on one image
  - visualize_still_missing   : grid of images that are still
                                 partial/empty after the full 4-tier
                                 refinement cascade
  - visualize_duplicates      : grid of images that have exact-duplicate
                                 polygon lines (the ones ultralytics
                                 silently collapses at load time)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.annotations import GTBox

CLS_NAMES = {0: "fire", 1: "smoke"}
CLS_COLORS_RGB = {0: (255, 80, 0), 1: (30, 120, 255)}  # GT box colors per class
MASK_COLOR_RGB = (0, 220, 0)                            # mask polygon outline


def draw_overlay(
    img_path: Path,
    gt_boxes: List[GTBox],
    seg_lines: List[str],
) -> np.ndarray:
    """Return an RGB image with GT boxes (colored by class) and any mask
    polygons (green outline) drawn on top."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img_h, img_w = img_bgr.shape[:2]
    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()

    for cls_id, cx, cy, bw, bh in gt_boxes:
        x1, y1 = int((cx - bw / 2) * img_w), int((cy - bh / 2) * img_h)
        x2, y2 = int((cx + bw / 2) * img_w), int((cy + bh / 2) * img_h)
        color = CLS_COLORS_RGB.get(cls_id, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, CLS_NAMES.get(cls_id, "?"), (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for line in seg_lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            coords = list(map(float, parts[1:]))
        except ValueError:
            continue
        pts = np.array(
            [[int(coords[i] * img_w), int(coords[i + 1] * img_h)] for i in range(0, len(coords), 2)],
            dtype=np.int32,
        )
        cv2.polylines(vis, [pts], isClosed=True, color=MASK_COLOR_RGB, thickness=2)

    return vis


def visualize_still_missing(
    df_missing: pd.DataFrame,
    img_ann_map: Dict[str, List[GTBox]],
    img_dir: Path,
    labels_seg_dir: Path,
    n_show: int = 16,
    save_path: Path | None = None,
):
    """
    Grid of images that are still `partial`/`empty` after the full
    4-tier cascade. Each panel shows GT boxes vs whatever mask (if any)
    exists, so it's obvious which boxes the cascade could not fill.
    """
    img_dir, labels_seg_dir = Path(img_dir), Path(labels_seg_dir)
    sample = df_missing.sample(min(n_show, len(df_missing)), random_state=42)

    n_cols = 4
    n_rows = (len(sample) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, sample.iterrows()):
        stem = row["stem"]
        seg_file = labels_seg_dir / f"{stem}.txt"
        seg_lines = seg_file.read_text().splitlines() if seg_file.exists() else []
        vis = draw_overlay(img_dir / f"{stem}.jpg", img_ann_map.get(stem, []), seg_lines)
        ax.imshow(vis)
        ax.set_title(f"{stem[-16:]}\nn_gt={row['n_gt']} n_seg={row['n_seg']} ({row['status']})", fontsize=8)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.suptitle("Still missing a mask after all 4 tiers (green = existing mask, colored box = GT)", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def visualize_duplicates(
    df_cov: pd.DataFrame,
    img_ann_map: Dict[str, List[GTBox]],
    img_dir: Path,
    labels_seg_dir: Path,
    n_show: int = 12,
    save_path: Path | None = None,
):
    """
    Grid of images that contain exact-duplicate polygon lines. These
    duplicates are silently collapsed by ultralytics' dataloader at
    load time (it dedupes identical lines), which means the *visible*
    seg-file line count overstates how many distinct masks YOLO will
    actually see during training -- this view makes that gap visible.
    """
    img_dir, labels_seg_dir = Path(img_dir), Path(labels_seg_dir)
    dup_rows = df_cov[df_cov["n_exact_dup"] > 0]
    if dup_rows.empty:
        print("No images with exact-duplicate polygon lines found.")
        return

    sample = dup_rows.sample(min(n_show, len(dup_rows)), random_state=42)
    n_cols = 4
    n_rows = (len(sample) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, sample.iterrows()):
        stem = row["stem"]
        seg_file = labels_seg_dir / f"{stem}.txt"
        seg_lines = seg_file.read_text().splitlines() if seg_file.exists() else []
        vis = draw_overlay(img_dir / f"{stem}.jpg", img_ann_map.get(stem, []), seg_lines)
        ax.imshow(vis)
        ax.set_title(f"{stem[-16:]}\n{row['n_exact_dup']} dup line(s) of {row['n_seg']}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.suptitle("Images with exact-duplicate polygon lines (silently dropped by ultralytics)", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
