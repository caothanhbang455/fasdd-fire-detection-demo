#!/usr/bin/env python3
"""
scripts/run_quality_eda.py

CLI entrypoint for mask-quality EDA + cleaning:
  - coverage / duplicate / polygon-validity scan
  - labels_seg_clean/ output (dedup, degenerate removal, coord clamp,
    optional bbox-rect fallback)
  - visual grids for "still missing after 4 tiers" and "has duplicate lines"

Example:

    python scripts/run_quality_eda.py \
        --img-dir /content/<dataset>/images \
        --label-dir /content/<dataset>/annotations/YOLO_CV/labels \
        --labels-seg-dir /content/labels_seg_refined \
        --labels-seg-clean-dir /content/labels_seg_clean \
        --out-dir /content/quality_report \
        --bbox-fallback
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.annotations import load_img_ann_map
from src.quality.cleaning import clean_dataset, cross_split_leak_check, dedup_split_file
from src.quality.eda import scan_dataset_coverage, scan_dataset_polygons, still_missing_after_refinement
from src.quality.visualize import visualize_duplicates, visualize_still_missing


def parse_args():
    p = argparse.ArgumentParser(description="Mask quality EDA + cleaning + visualization.")
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--label-dir", required=True, type=Path, help="GT YOLO detection labels dir")
    p.add_argument("--labels-seg-dir", required=True, type=Path, help="refined seg labels to audit")
    p.add_argument("--labels-seg-clean-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path, help="where to write report PNGs/CSVs")
    p.add_argument("--train-list", type=Path, default=None)
    p.add_argument("--val-list", type=Path, default=None)
    p.add_argument("--test-list", type=Path, default=None)
    p.add_argument("--bbox-fallback", action="store_true")
    p.add_argument("--n-show", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GT annotations...")
    img_ann_map = load_img_ann_map(args.img_dir, args.label_dir)
    ann_stems = [s for s, boxes in img_ann_map.items() if boxes]
    print(f"  Annotated images: {len(ann_stems):,}")

    print("\nScanning coverage status + duplicates...")
    df_cov = scan_dataset_coverage(ann_stems, img_ann_map, args.labels_seg_dir)
    df_cov.to_csv(args.out_dir / "coverage_audit.csv", index=False)
    print(df_cov["status"].value_counts())
    print(f"  Images with exact-duplicate lines: {(df_cov['n_exact_dup'] > 0).sum():,}")

    print("\nScanning per-polygon validity...")
    df_poly = scan_dataset_polygons(ann_stems, img_ann_map, args.labels_seg_dir)
    df_poly.to_csv(args.out_dir / "polygon_quality.csv", index=False)
    print(f"  Degenerate : {df_poly['is_degenerate'].sum():,}")
    print(f"  Out-of-range coords (and not degenerate): "
          f"{((~df_poly['coords_ok']) & (~df_poly['is_degenerate'])).sum():,}")

    df_missing = still_missing_after_refinement(df_cov)
    print(f"\nStill missing a mask after 4-tier refinement: {len(df_missing):,} images")
    visualize_still_missing(
        df_missing, img_ann_map, args.img_dir, args.labels_seg_dir,
        n_show=args.n_show, save_path=args.out_dir / "still_missing_grid.png",
    )
    visualize_duplicates(
        df_cov, img_ann_map, args.img_dir, args.labels_seg_dir,
        n_show=args.n_show, save_path=args.out_dir / "duplicates_grid.png",
    )

    print("\nCleaning -> labels_seg_clean/ ...")
    clean_stats = clean_dataset(
        ann_stems, args.labels_seg_dir, args.labels_seg_clean_dir,
        img_ann_map=img_ann_map, bbox_fallback_for_missing=args.bbox_fallback,
    )
    for k, v in clean_stats.items():
        print(f"  {k:24s}: {v:,}")

    if args.train_list and args.val_list and args.test_list:
        print("\nDeduplicating split list files + cross-split leak check...")
        for split_file in [args.train_list, args.val_list, args.test_list]:
            n = dedup_split_file(split_file)
            print(f"  {split_file.name}: removed {n} duplicate path(s)")
        leaks = cross_split_leak_check(args.train_list, args.val_list, args.test_list)
        for k, v in leaks.items():
            print(f"  {k}: {'LEAK ' + str(len(v)) if v else 'ok'}")

    print(f"\nReport written to: {args.out_dir}")


if __name__ == "__main__":
    main()
