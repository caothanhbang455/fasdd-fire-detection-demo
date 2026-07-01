"""
src/refine/matching.py

Match existing segmentation polygons back to GT boxes so the refinement
loop only touches GT boxes that genuinely have no mask yet, instead of
relying on file-line order (which SAM2 does not preserve when given
multiple boxes at once -- this was the root cause of the index-offset
bug found during the original mask generation run).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..data.annotations import GTBox

Match = Dict[int, Dict]  # gt_index -> seg entry (see io_utils.parse_seg_file)


def match_segs_to_gt(
    seg_entries: List[Dict],
    gt_boxes: List[GTBox],
    max_centroid_dist: float = 0.30,
) -> Match:
    """
    Greedily match each GT box to the closest unused polygon of the same
    class (L1 centroid distance, normalized image coordinates).

    Returns {gt_index -> seg_entry}. GT indices absent from the result
    have no matching polygon and are candidates for refinement.
    """
    matched: Match = {}
    used_seg: set[int] = set()

    for gi, (cls_id, cx, cy, _bw, _bh) in enumerate(gt_boxes):
        best_i, best_d = -1, float("inf")
        for si, seg in enumerate(seg_entries):
            if si in used_seg or seg["cls"] != cls_id:
                continue
            d = abs(seg["cx"] - cx) + abs(seg["cy"] - cy)
            if d < best_d:
                best_d, best_i = d, si

        if best_i >= 0 and best_d < max_centroid_dist:
            matched[gi] = seg_entries[best_i]
            used_seg.add(best_i)

    return matched


def unmatched_gt_indices(matched: Match, n_gt: int) -> List[int]:
    """GT indices that still need a mask after matching."""
    return [gi for gi in range(n_gt) if gi not in matched]
