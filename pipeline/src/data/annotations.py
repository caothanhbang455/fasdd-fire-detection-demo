"""
src/data/annotations.py

Load GT YOLO detection labels and dataset split files for FASDD_CV.

Conventions
-----------
GT detection label line (5 tokens):
    class_id cx cy bw bh        (class 0 = fire, class 1 = smoke)

Segmentation pseudo-label line (>=7 tokens, odd count after class_id):
    class_id x1 y1 x2 y2 x3 y3 ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

GTBox = Tuple[int, float, float, float, float]  # (cls_id, cx, cy, bw, bh)


def read_yolo_label(label_path: Path) -> List[GTBox]:
    """Parse a YOLO detection .txt file into a list of GT boxes."""
    boxes: List[GTBox] = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(x) for x in parts[1:])
        except ValueError:
            continue
        boxes.append((cls_id, cx, cy, bw, bh))
    return boxes


def load_img_ann_map(
    img_dir: Path,
    label_dir: Path,
    img_ext: str = "*.jpg",
) -> Dict[str, List[GTBox]]:
    """Build {stem -> [GTBox, ...]} for every image under img_dir."""
    img_ann_map: Dict[str, List[GTBox]] = {}
    for img_path in sorted(Path(img_dir).glob(img_ext)):
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        img_ann_map[img_path.stem] = read_yolo_label(label_path)
    return img_ann_map


def load_split_stems(split_txt: Path) -> List[str]:
    """Read a train/val/test list file (one image path per line) -> stems."""
    if not Path(split_txt).exists():
        return []
    stems = []
    for line in Path(split_txt).read_text().splitlines():
        line = line.strip()
        if line:
            stems.append(Path(line).stem)
    return stems


def build_split_map(
    train_txt: Path, val_txt: Path, test_txt: Path
) -> Dict[str, str]:
    """Build {stem -> 'train'|'val'|'test'} from the three split list files."""
    split_map: Dict[str, str] = {}
    for split_name, txt_path in [
        ("train", train_txt),
        ("val", val_txt),
        ("test", test_txt),
    ]:
        for stem in load_split_stems(txt_path):
            split_map[stem] = split_name
    return split_map


def write_img_list(stems: List[str], img_dir: Path, out_path: Path, ext: str = ".jpg") -> None:
    """Write an image-path-per-line list file for a given set of stems."""
    lines = [str(Path(img_dir) / f"{stem}{ext}") for stem in stems]
    Path(out_path).write_text("\n".join(lines) + ("\n" if lines else ""))


def yolo_to_pixel_xyxy(
    cx: float, cy: float, bw: float, bh: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Convert a normalized YOLO box to pixel-space (x1, y1, x2, y2)."""
    x1 = max(0, int((cx - bw / 2) * img_w))
    y1 = max(0, int((cy - bh / 2) * img_h))
    x2 = min(img_w, int((cx + bw / 2) * img_w))
    y2 = min(img_h, int((cy + bh / 2) * img_h))
    return x1, y1, x2, y2
