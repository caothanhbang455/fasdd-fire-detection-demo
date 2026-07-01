"""
src/train/model.py

Load a YOLO11m-seg model either fresh (from a pretrained checkpoint) or
resumed from a previous run's last.pt, and patch the resumed run's
args.yaml so paths stay valid across Kaggle sessions (Kaggle wipes
/kaggle/working on every new session, so a resumed run must be told
where the current session's data.yaml / project dir / run name live).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO


def load_fresh(weights: str = "yolo11m-seg.pt") -> YOLO:
    """Load a YOLO11m-seg model from a pretrained checkpoint."""
    return YOLO(weights)


def patch_resume_args_yaml(
    args_yaml: Path,
    data_yaml: Path,
    project_dir: Path,
    run_name: str,
    run_dir: Path,
    epochs: int,
    batch: int,
    workers: int,
    device: str,
) -> None:
    """
    Rewrite the copied args.yaml of a resumed run so ultralytics picks up
    the *current* session's writable paths instead of the stale paths
    baked in from the original (now-deleted) Kaggle session.
    """
    args_yaml = Path(args_yaml)
    if not args_yaml.exists():
        return

    with open(args_yaml) as f:
        args = yaml.safe_load(f) or {}

    args.update({
        "data": str(data_yaml),
        "project": str(project_dir),
        "name": run_name,
        "save_dir": str(run_dir),
        "epochs": epochs,
        "batch": batch,
        "workers": workers,
        "device": device,
    })

    with open(args_yaml, "w") as f:
        yaml.safe_dump(args, f, sort_keys=False)


def load_for_resume(last_pt: Path, run_input_dir: Path, **patch_kwargs) -> YOLO:
    """
    Resume training from a previous run's last.pt. `run_input_dir` is the
    read-only Kaggle-input copy of the original run folder (containing
    args.yaml next to weights/last.pt) -- patch its args.yaml in place
    before constructing the YOLO object so `.train(resume=True)` works.
    """
    args_yaml = Path(run_input_dir) / "args.yaml"
    patch_resume_args_yaml(args_yaml, **patch_kwargs)
    return YOLO(str(last_pt))


def load_for_eval(weights_path: Path) -> YOLO:
    """Load a trained checkpoint purely for `.val()` / `.predict()`."""
    return YOLO(str(weights_path))


def find_best_checkpoint(
    run_dir: Path,
    project_dir: Path,
    run_name: str,
    resume_best_pt: Optional[Path] = None,
) -> Optional[Path]:
    """Look in the usual places for best.pt, in priority order."""
    candidates = [
        Path(run_dir) / "weights" / "best.pt",
        Path(project_dir) / run_name / "weights" / "best.pt",
    ]
    if resume_best_pt is not None:
        candidates.append(Path(resume_best_pt))
    for c in candidates:
        if c.exists():
            return c
    return None
