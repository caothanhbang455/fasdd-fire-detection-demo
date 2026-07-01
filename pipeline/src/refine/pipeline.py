"""
src/refine/pipeline.py

Orchestrates the full refinement loop:
  1. identify_refinement_targets -- which images need work
  2. run_refinement_loop         -- match existing polygons, cascade
                                    through tiers 1-4 for unmatched
                                    boxes, write results, checkpoint

This mirrors the logic in FASDD_CV_SAM2_MaskRefine_v2.ipynb (S4 + S7),
factored into importable functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm.auto import tqdm

from ..data.annotations import GTBox, yolo_to_pixel_xyxy
from .config import RefineConfig
from .io_utils import mask_to_yolo_polygon, parse_seg_file
from .matching import match_segs_to_gt, unmatched_gt_indices
from .progress import (done_set, load_progress, mark_done, save_progress,
                        sync_labels_to_drive)
from .tiers import run_cascade

RefineTarget = Tuple[str, int, int, int]  # (stem, n_gt, n_seg, n_missing)


def identify_refinement_targets(
    img_ann_map: Dict[str, List[GTBox]],
    labels_seg_dir: Path,
) -> Tuple[List[RefineTarget], List[str]]:
    """
    Criterion: n_seg_lines < n_gt_boxes -> at least one box has no mask.
    Includes images with zero seg lines (empty files).

    Returns (refine_targets, ok_stems).
    """
    labels_seg_dir = Path(labels_seg_dir)
    refine_targets: List[RefineTarget] = []
    ok_stems: List[str] = []

    ann_stems = [s for s, boxes in img_ann_map.items() if boxes]
    for stem in ann_stems:
        n_gt = len(img_ann_map[stem])
        seg_file = labels_seg_dir / f"{stem}.txt"
        n_seg = 0
        if seg_file.exists() and seg_file.stat().st_size > 0:
            n_seg = sum(1 for l in seg_file.read_text().splitlines() if l.strip())

        missing = n_gt - n_seg
        if missing > 0:
            refine_targets.append((stem, n_gt, n_seg, missing))
        else:
            ok_stems.append(stem)

    return refine_targets, ok_stems


def run_refinement_loop(
    refine_targets: List[RefineTarget],
    img_ann_map: Dict[str, List[GTBox]],
    img_dir: Path,
    labels_seg_orig_dir: Path,
    labels_seg_ref_dir: Path,
    sam2,
    cfg: RefineConfig,
    local_progress_json: Path,
    drive_progress_json: Path | None = None,
    drive_labels_dir: Path | None = None,
) -> Dict:
    """
    Run (or resume) the 4-tier cascade over every target image, writing
    refined polygons to `labels_seg_ref_dir` and checkpointing every
    `cfg.save_every` images.
    """
    img_dir = Path(img_dir)
    labels_seg_orig_dir = Path(labels_seg_orig_dir)
    labels_seg_ref_dir = Path(labels_seg_ref_dir)
    labels_seg_ref_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(local_progress_json, drive_progress_json)
    done = done_set(progress)
    stats = progress["stats"]

    todo = [t for t in refine_targets if t[0] not in done]
    updated_this_batch: List[str] = []

    for batch_start in range(0, len(todo), cfg.save_every):
        batch = todo[batch_start: batch_start + cfg.save_every]

        for stem, n_gt, _n_seg_orig, _n_missing in tqdm(
            batch, desc=f"Refine [{batch_start}/{len(todo)}]", ncols=85
        ):
            img_path = img_dir / f"{stem}.jpg"
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                mark_done(progress, stem)
                continue
            img_h, img_w = img_cv.shape[:2]
            gt_boxes = img_ann_map[stem]

            seg_file_orig = labels_seg_orig_dir / f"{stem}.txt"
            seg_file_ref = labels_seg_ref_dir / f"{stem}.txt"
            active_seg = seg_file_ref if seg_file_ref.exists() else seg_file_orig

            existing = parse_seg_file(active_seg)
            matched = match_segs_to_gt(existing, gt_boxes, cfg.match_max_centroid_dist)
            unmatched_gi = unmatched_gt_indices(matched, n_gt)

            if not unmatched_gi:
                mark_done(progress, stem)
                continue

            stats["total_missing"] += len(unmatched_gi)
            final_lines = {gi: entry["line"] for gi, entry in matched.items()}
            any_new = False

            for gi in unmatched_gi:
                cls_id, cx, cy, bw, bh = gt_boxes[gi]
                x1, y1, x2, y2 = yolo_to_pixel_xyxy(cx, cy, bw, bh, img_w, img_h)
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue

                bbox_area_norm = bw * bh
                mask, cov, tier_used = run_cascade(
                    sam2, img_cv, str(img_path), cls_id, x1, y1, x2, y2,
                    bbox_area_norm, cfg,
                )

                if mask is None or cov < cfg.min_coverage_keep:
                    stats["no_improve"] += 1
                    continue

                poly = mask_to_yolo_polygon(mask, img_w, img_h)
                if not poly:
                    stats["no_improve"] += 1
                    continue

                line = str(cls_id) + " " + " ".join(f"{v:.6f}" for v in poly)
                final_lines[gi] = line
                any_new = True
                stats["newly_filled"] += 1
                stats[f"t{tier_used}_improved"] += 1

            if any_new:
                out_lines = [final_lines[gi] for gi in sorted(final_lines)]
                seg_file_ref.write_text("\n".join(out_lines) + "\n")
                updated_this_batch.append(stem)

            mark_done(progress, stem)

        # checkpoint after every batch
        save_progress(progress, local_progress_json, drive_progress_json)
        if drive_labels_dir:
            sync_labels_to_drive(updated_this_batch, labels_seg_ref_dir, drive_labels_dir)
        updated_this_batch = []

    return progress
