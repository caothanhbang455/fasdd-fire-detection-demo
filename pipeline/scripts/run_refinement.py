#!/usr/bin/env python3
"""
scripts/run_refinement.py

CLI entrypoint for the 4-tier SAM2 mask refinement cascade.

Example (Colab, paths matching MaskRefine_v2 conventions):

    python scripts/run_refinement.py \
        --img-dir /content/<dataset>/images \
        --label-dir /content/<dataset>/annotations/YOLO_CV/labels \
        --labels-seg-orig /content/labels_seg \
        --labels-seg-ref /content/labels_seg_refined \
        --progress-json /content/refine_progress.json \
        --drive-progress-json /content/drive/MyDrive/Project-Phase1-VDT/DATA/refine_progress.json \
        --drive-labels-dir /content/drive/MyDrive/Project-Phase1-VDT/DATA/labels_seg_refined
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.annotations import load_img_ann_map
from src.refine.config import RefineConfig
from src.refine.pipeline import identify_refinement_targets, run_refinement_loop


def parse_args():
    p = argparse.ArgumentParser(description="Run the 4-tier SAM2 mask refinement cascade.")
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--label-dir", required=True, type=Path, help="GT YOLO detection labels dir")
    p.add_argument("--labels-seg-orig", required=True, type=Path)
    p.add_argument("--labels-seg-ref", required=True, type=Path)
    p.add_argument("--progress-json", required=True, type=Path)
    p.add_argument("--drive-progress-json", type=Path, default=None)
    p.add_argument("--drive-labels-dir", type=Path, default=None)
    p.add_argument("--sam2-model", default="sam2.1_l.pt")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--min-coverage-keep", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import SAM

    cfg = RefineConfig(
        sam2_model=args.sam2_model,
        save_every=args.save_every,
        min_coverage_keep=args.min_coverage_keep,
    )

    print(f"Loading SAM2: {cfg.sam2_model}")
    sam2 = SAM(cfg.sam2_model)
    sam2.model.eval()

    print("Loading GT annotations...")
    img_ann_map = load_img_ann_map(args.img_dir, args.label_dir)

    print("Identifying refinement targets (n_seg < n_gt)...")
    refine_targets, ok_stems = identify_refinement_targets(img_ann_map, args.labels_seg_orig)
    print(f"  OK already       : {len(ok_stems):,}")
    print(f"  Need refinement  : {len(refine_targets):,}")

    run_refinement_loop(
        refine_targets=refine_targets,
        img_ann_map=img_ann_map,
        img_dir=args.img_dir,
        labels_seg_orig_dir=args.labels_seg_orig,
        labels_seg_ref_dir=args.labels_seg_ref,
        sam2=sam2,
        cfg=cfg,
        local_progress_json=args.progress_json,
        drive_progress_json=args.drive_progress_json,
        drive_labels_dir=args.drive_labels_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
