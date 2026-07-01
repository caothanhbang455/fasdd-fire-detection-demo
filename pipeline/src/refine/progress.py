"""
src/refine/progress.py

Crash-safe progress tracking for the refinement loop. SAM2 inference
over ~56k images is slow enough that the loop must be resumable across
disconnects (Colab) or session limits (Kaggle), and must support a
local + Drive/remote mirror of the same progress.json.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

DEFAULT_STATS = {
    "t1_improved": 0, "t2_improved": 0,
    "t3_improved": 0, "t4_improved": 0,
    "no_improve": 0, "total_missing": 0,
    "newly_filled": 0,
}


def load_progress(local_path: Path, drive_path: Path | None = None) -> Dict:
    """
    Load progress.json, preferring the local copy. If absent locally but
    present remotely (Drive), restore it first.
    """
    local_path = Path(local_path)
    if not local_path.exists() and drive_path and Path(drive_path).exists():
        shutil.copy(drive_path, local_path)

    if local_path.exists():
        progress = json.loads(local_path.read_text())
    else:
        progress = {"done": [], "stats": dict(DEFAULT_STATS)}

    progress.setdefault("done", [])
    progress.setdefault("stats", dict(DEFAULT_STATS))
    return progress


def save_progress(
    progress: Dict,
    local_path: Path,
    drive_path: Path | None = None,
) -> None:
    """Write progress.json locally and mirror it to Drive if configured."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(progress, indent=2))
    if drive_path:
        Path(drive_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, drive_path)


def done_set(progress: Dict) -> Set[str]:
    return set(progress.get("done", []))


def mark_done(progress: Dict, stem: str) -> None:
    progress.setdefault("done", []).append(stem)


def sync_labels_to_drive(
    updated_stems: List[str],
    labels_local_dir: Path,
    labels_drive_dir: Path,
) -> None:
    """Copy just the .txt files touched this run into the Drive mirror folder."""
    labels_drive_dir = Path(labels_drive_dir)
    labels_drive_dir.mkdir(parents=True, exist_ok=True)
    for stem in updated_stems:
        src = Path(labels_local_dir) / f"{stem}.txt"
        if src.exists():
            shutil.copy(src, labels_drive_dir / src.name)
