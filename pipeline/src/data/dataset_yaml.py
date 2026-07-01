"""
src/data/dataset_yaml.py

Build the ultralytics data.yaml and the images/labels symlinks that
ultralytics' YOLODataset expects to find next to the split list files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

DEFAULT_NAMES = {0: "fire", 1: "smoke"}


def make_symlink(link: Path, target: Path) -> None:
    """(Re)create a symlink, removing any existing file/link/dir at `link`."""
    link = Path(link)
    target = Path(target)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def build_symlinks(work_dir: Path, img_dir: Path, label_dir: Path) -> None:
    """Create work_dir/images -> img_dir and work_dir/labels -> label_dir."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    make_symlink(work_dir / "images", img_dir)
    make_symlink(work_dir / "labels", label_dir)


def write_dataset_yaml(
    yaml_path: Path,
    base_path: Path,
    train_list: Path,
    val_list: Path,
    test_list: Optional[Path] = None,
    nc: int = 2,
    names: Optional[Dict[int, str]] = None,
) -> Path:
    """Write an ultralytics-compatible data.yaml for the seg dataset."""
    names = names or DEFAULT_NAMES
    data = {
        "path": str(base_path),
        "train": str(train_list),
        "val": str(val_list),
        "nc": nc,
        "names": names,
    }
    if test_list is not None:
        data["test"] = str(test_list)

    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    return yaml_path
