"""
flame/background.py

Stage 1 of FLAME: Background Subtraction Gate.

Following Gragnaniello et al. (Neural Computing and Applications, 2025):
"filter out fire candidates occurring in the scene's background"

For fixed surveillance cameras, MOG2 builds a per-pixel Gaussian mixture
model of the static background. Any YOLO detection whose bbox region has
insufficient foreground motion is discarded as a static FP (sunset glow,
permanently-lit sign, headlight wash on a wall).


thêm việc đo độ trễ các stage xử lý, đánh giá lại độ hiểu quả khi có và không có stage12 của Flame
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


class BackgroundGate:
    """
    MOG2-based foreground motion gate.

    Parameters
    ----------
    history        : number of frames used to build the background model
    var_threshold  : Mahalanobis distance threshold for foreground pixel
    min_fg_ratio   : minimum fraction of a detection bbox that must be
                     foreground for the detection to pass Stage 1
    warmup_frames  : frames during which detections always pass (bg model
                     not yet stable)
    """

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 25.0,
        min_fg_ratio: float = 0.05,
        warmup_frames: int = 50,
    ) -> None:
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self.min_fg_ratio = min_fg_ratio
        self.warmup_frames = warmup_frames
        self._frame_count = 0
        self._last_fg_mask: np.ndarray | None = None

    def reset(self) -> None:
        """Reset state — call this when starting a new video."""
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25.0, detectShadows=False
        )
        self._frame_count = 0
        self._last_fg_mask = None

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Update background model and return foreground mask (0/255)."""
        fg_mask = self.bg_sub.apply(frame_bgr)
        self._last_fg_mask = fg_mask
        self._frame_count += 1
        return fg_mask

    def filter(
        self, detections: sv.Detections, fg_mask: np.ndarray
    ) -> tuple[sv.Detections, int]:
        """
        Keep only detections with >= min_fg_ratio foreground pixels in
        their bounding-box region.

        Returns (filtered_detections, n_suppressed).
        """
        if len(detections) == 0:
            return detections, 0

        # During warmup, pass everything through
        if self._frame_count <= self.warmup_frames:
            return detections, 0

        h, w = fg_mask.shape[:2]
        keep: list[int] = []

        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = (
                max(0, int(bbox[0])), max(0, int(bbox[1])),
                min(w, int(bbox[2])), min(h, int(bbox[3])),
            )
            region = fg_mask[y1:y2, x1:x2]
            if region.size == 0:
                continue
            fg_ratio = float((region > 128).sum()) / region.size
            if fg_ratio >= self.min_fg_ratio:
                keep.append(i)

        n_suppressed = len(detections) - len(keep)
        return detections[np.array(keep, dtype=int)] if keep else sv.Detections.empty(), n_suppressed

    @property
    def is_warmed_up(self) -> bool:
        return self._frame_count > self.warmup_frames
