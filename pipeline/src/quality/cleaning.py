"""
src/quality/cleaning.py

Produces labels_seg_clean/ from labels_seg_refined/:
  1. drop exact-duplicate lines (the ones ultralytics silently collapses)
  2. drop degenerate polygons (< 3 points or odd coordinate count)
  3. clamp out-of-range coordinates into [0, 1] instead of discarding
  4. optionally write a 4-corner bbox-as-polygon fallback for images
     that still have zero usable seg lines after refinement, so YOLO
     detection supervision is not silently lost for those boxes
  5. dedup the train/val/test split list files + cross-split leak check
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..data.annotations import GTBox

CleanStats = Dict[str, int]


def _empty_stats() -> CleanStats:
    return {
        "files_written": 0, "lines_in": 0, "lines_out": 0,
        "removed_dup": 0, "removed_degen": 0, "coords_clamped": 0,
        "bbox_fallback_images": 0,
    }


def bbox_to_rect_polygon(cx: float, cy: float, bw: float, bh: float) -> List[float]:
    """YOLO bbox -> 4-corner rectangle polygon (normalized coords)."""
    x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
    x2, y2 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def clean_seg_lines(raw_lines: List[str]) -> tuple[List[str], CleanStats]:
    """Apply dedup -> degenerate filter -> coordinate clamp to one file's lines."""
    stats = _empty_stats()
    stats["lines_in"] = len(raw_lines)

    seen: set[str] = set()
    clean_lines: List[str] = []

    for line in raw_lines:
        if line in seen:
            stats["removed_dup"] += 1
            continue
        seen.add(line)

        parts = line.split()
        n_tok = len(parts)
        n_crd = n_tok - 1

        if n_tok < 7 or n_crd % 2 != 0:
            stats["removed_degen"] += 1
            continue

        try:
            coords = list(map(float, parts[1:]))
            cls_id = int(parts[0])
        except ValueError:
            stats["removed_degen"] += 1
            continue

        if not all(0.0 <= v <= 1.0 for v in coords):
            coords = [round(max(0.0, min(1.0, v)), 6) for v in coords]
            line = str(cls_id) + " " + " ".join(f"{v:.6f}" for v in coords)
            stats["coords_clamped"] += 1

        clean_lines.append(line)

    stats["lines_out"] = len(clean_lines)
    return clean_lines, stats


def clean_dataset(
    ann_stems: List[str],
    labels_seg_raw_dir: Path,
    labels_seg_clean_dir: Path,
    img_ann_map: Dict[str, List[GTBox]] | None = None,
    bbox_fallback_for_missing: bool = False,
) -> CleanStats:
    """
    Run `clean_seg_lines` over every annotated image and write
    labels_seg_clean/. If `bbox_fallback_for_missing` is set, images
    that end up with zero clean lines get a rectangle-polygon fallback
    built from their GT boxes (requires img_ann_map).
    """
    labels_seg_raw_dir = Path(labels_seg_raw_dir)
    labels_seg_clean_dir = Path(labels_seg_clean_dir)
    labels_seg_clean_dir.mkdir(parents=True, exist_ok=True)

    total = _empty_stats()

    for stem in ann_stems:
        raw_file = labels_seg_raw_dir / f"{stem}.txt"
        raw_lines = []
        if raw_file.exists() and raw_file.stat().st_size > 0:
            raw_lines = [l.strip() for l in raw_file.read_text().splitlines() if l.strip()]

        clean_lines, stats = clean_seg_lines(raw_lines)
        for k in total:
            if k in stats:
                total[k] += stats[k]

        if not clean_lines and bbox_fallback_for_missing and img_ann_map is not None:
            gt_boxes = img_ann_map.get(stem, [])
            if gt_boxes:
                for cls_id, cx, cy, bw, bh in gt_boxes:
                    pts = bbox_to_rect_polygon(cx, cy, bw, bh)
                    clean_lines.append(str(cls_id) + " " + " ".join(f"{v:.6f}" for v in pts))
                total["bbox_fallback_images"] += 1

        out_path = labels_seg_clean_dir / f"{stem}.txt"
        out_path.write_text("\n".join(clean_lines) + ("\n" if clean_lines else ""))
        total["files_written"] += 1

    return total


def dedup_split_file(split_txt: Path) -> int:
    """Remove duplicate paths from a split list file in place. Returns count removed."""
    split_txt = Path(split_txt)
    if not split_txt.exists():
        return 0
    original = [l.strip() for l in split_txt.read_text().splitlines() if l.strip()]
    deduped = list(dict.fromkeys(original))
    n_removed = len(original) - len(deduped)
    if n_removed > 0:
        split_txt.write_text("\n".join(deduped) + "\n")
    return n_removed


def cross_split_leak_check(
    train_txt: Path, val_txt: Path, test_txt: Path
) -> Dict[str, set[str]]:
    """Return any stems that appear in more than one split (should all be empty)."""
    def stems(p: Path) -> set[str]:
        return {Path(l.strip()).stem for l in Path(p).read_text().splitlines() if l.strip()} \
            if Path(p).exists() else set()

    train_set, val_set, test_set = stems(train_txt), stems(val_txt), stems(test_txt)
    return {
        "train_val": train_set & val_set,
        "train_test": train_set & test_set,
        "val_test": val_set & test_set,
    }
