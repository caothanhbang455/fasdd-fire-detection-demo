"""
flame/pipeline.py  (fixed: sv.ByteTrack, not deprecated sv.ByteTracker)
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from .background import BackgroundGate
from .trajectory import TrajectoryFilter


@dataclass
class FLAMEResult:
    detections:   sv.Detections
    raw_count:    int
    fp_stage1:    int
    fp_stage2:    int
    new_alerts:   list[int]
    bg_warmed_up: bool
    fg_mask:      np.ndarray | None = field(default=None, repr=False)


def _make_tracker() -> sv.ByteTrack:
    """supervision >= 0.28 uses sv.ByteTrack (not sv.ByteTracker)."""
    return sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )


class FLAMEPipeline:
    def __init__(
        self,
        bg_history: int = 300,
        bg_var_threshold: float = 25.0,
        bg_min_fg_ratio: float = 0.05,
        bg_warmup_frames: int = 50,
        min_track_frames: int = 4,
        max_area_cv: float = 0.45,
        min_mean_conf: float = 0.30,
        history_window: int = 10,
    ) -> None:
        self._bg = BackgroundGate(
            history=bg_history,
            var_threshold=bg_var_threshold,
            min_fg_ratio=bg_min_fg_ratio,
            warmup_frames=bg_warmup_frames,
        )
        self._traj = TrajectoryFilter(
            min_track_frames=min_track_frames,
            max_area_cv=max_area_cv,
            min_mean_conf=min_mean_conf,
            history_window=history_window,
        )
        self._tracker = _make_tracker()

    def reset(self) -> None:
        self._bg.reset()
        self._traj.reset()
        self._tracker = _make_tracker()

    def process(
        self,
        frame_bgr: np.ndarray,
        detections: sv.Detections,
        return_fg_mask: bool = False,
    ) -> FLAMEResult:
        raw_count = len(detections)

        fg_mask = self._bg.update(frame_bgr)
        after_s1, fp_s1 = self._bg.filter(detections, fg_mask)

        if len(after_s1) > 0:
            after_tracked = self._tracker.update_with_detections(after_s1)
        else:
            after_tracked = sv.Detections.empty()

        confirmed, new_alerts, fp_s2 = self._traj.update(after_tracked)

        return FLAMEResult(
            detections=confirmed,
            raw_count=raw_count,
            fp_stage1=fp_s1,
            fp_stage2=fp_s2,
            new_alerts=new_alerts,
            bg_warmed_up=self._bg.is_warmed_up,
            fg_mask=fg_mask if return_fg_mask else None,
        )

    @property
    def total_alerts(self) -> int:
        return self._traj.total_alerts
