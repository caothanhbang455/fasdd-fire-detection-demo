"""
src/train/train.py

Thin wrapper around ultralytics' `.train()` for YOLO11m-seg, covering
both fresh and resume modes. Keeps the actual training call in one
place so notebooks/scripts don't duplicate the argument list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def train_seg(
    model: YOLO,
    data_yaml: Path,
    epochs: int,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 4,
    device: str = "0",
    project: Optional[Path] = None,
    name: Optional[str] = None,
    resume: bool = False,
    **extra_kwargs,
):
    """
    Run (or resume) YOLO11m-seg training.

    `extra_kwargs` passes straight through to ultralytics, e.g.
    lr0=, mosaic=, patience=, save_period=, seed=.
    """
    kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        resume=resume,
        **extra_kwargs,
    )
    if project is not None:
        kwargs["project"] = str(project)
    if name is not None:
        kwargs["name"] = name

    return model.train(**kwargs)
