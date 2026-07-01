#!/usr/bin/env python3
"""
scripts/run_training.py

CLI entrypoint for YOLO11m-seg training (fresh or resumed), wrapping
src/data + src/train. Mirrors S7-S9 of project-phase1-vdt.ipynb.

Example (fresh run):

    python scripts/run_training.py \
        --train-list /kaggle/working/train_images.txt \
        --val-list /kaggle/working/val_images.txt \
        --test-list /kaggle/working/test_images.txt \
        --img-dir /kaggle/working/images \
        --labels-seg-clean-dir /kaggle/working/labels_seg_clean \
        --work-dir /kaggle/working \
        --project-dir /kaggle/working/runs \
        --run-name yolo11m_seg_v1 \
        --epochs 100 --batch 16 --imgsz 640

Example (resume):

    python scripts/run_training.py ... --resume \
        --resume-last-pt /kaggle/input/<prev-run>/weights/last.pt \
        --resume-run-input /kaggle/input/<prev-run>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_yaml import build_symlinks, write_dataset_yaml
from src.train.logging_utils import init_wandb, save_run_config
from src.train.model import load_for_resume, load_fresh
from src.train.train import train_seg


def parse_args():
    p = argparse.ArgumentParser(description="Train (or resume) YOLO11m-seg on FASDD_CV.")
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--labels-seg-clean-dir", required=True, type=Path)
    p.add_argument("--train-list", required=True, type=Path)
    p.add_argument("--val-list", required=True, type=Path)
    p.add_argument("--test-list", type=Path, default=None)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--run-name", required=True)
    p.add_argument("--weights", default="yolo11m-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-last-pt", type=Path, default=None)
    p.add_argument("--resume-run-input", type=Path, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="fasdd-cv-seg")
    return p.parse_args()


def main():
    args = parse_args()

    work_images = args.work_dir / "images"
    work_labels = args.work_dir / "labels"
    build_symlinks(args.work_dir, args.img_dir, args.labels_seg_clean_dir)

    yaml_path = args.work_dir / "fasdd_cv_seg.yaml"
    write_dataset_yaml(
        yaml_path, base_path=args.work_dir,
        train_list=args.train_list, val_list=args.val_list, test_list=args.test_list,
    )

    run_dir = args.project_dir / args.run_name
    save_run_config(run_dir, extra={
        "run_name": args.run_name, "weights": args.weights,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "resume": args.resume,
    })

    init_wandb(args.wandb, args.wandb_project, args.run_name)

    if args.resume:
        assert args.resume_last_pt and args.resume_run_input, \
            "--resume requires --resume-last-pt and --resume-run-input"
        model = load_for_resume(
            args.resume_last_pt, args.resume_run_input,
            data_yaml=yaml_path, project_dir=args.project_dir, run_name=args.run_name,
            run_dir=run_dir, epochs=args.epochs, batch=args.batch,
            workers=args.workers, device=args.device,
        )
    else:
        model = load_fresh(args.weights)

    results = train_seg(
        model, yaml_path, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        project=args.project_dir, name=args.run_name, resume=args.resume,
    )

    print("Training complete.")
    print(f"Run dir: {run_dir}")
    return results


if __name__ == "__main__":
    main()
