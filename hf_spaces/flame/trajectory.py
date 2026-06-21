"""
flame/trajectory.py

Stage 2 of FLAME: Motion Trajectory Analysis.

Following Gragnaniello et al. (Neural Computing and Applications, 2025):
"filter out fire candidates whose movements differ from those of the fire"

Real fire and smoke exhibit characteristic temporal dynamics:
  - Persistence: detected for multiple consecutive frames
  - Growth: bbox area tends to increase or remain stable (not random)
  - Stability: confidence is sustained (not a single-frame spike)

False positives (headlights, single-frame flares) fail one or more of
these criteria and are suppressed before an alert is raised.

The motion model is deliberately tunable (all thresholds as parameters)
so it can be calibrated per deployment environment without retraining.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import supervision as sv


class TrajectoryFilter:
    """
    Validates per-track trajectory history for fire-like dynamics.

    Parameters
    ----------
    min_track_frames   : minimum number of consecutive frames a detection
                         must appear before it can trigger an alert
    max_area_cv        : max coefficient of variation of bbox area — fire
                         grows or holds; random FP areas jump erratically
    min_mean_conf      : minimum average confidence over track history
    history_window     : how many recent frames to consider per track
    lost_track_frames  : how many frames of absence before resetting a
                         track's history (avoids stale state for scene cuts)
    """

    def __init__(
        self,
        min_track_frames: int = 4,
        max_area_cv: float = 0.45,
        min_mean_conf: float = 0.30,
        history_window: int = 10,
        lost_track_frames: int = 15,
    ) -> None:
        self.min_track_frames = min_track_frames
        self.max_area_cv = max_area_cv
        self.min_mean_conf = min_mean_conf
        self.history_window = history_window
        self.lost_track_frames = lost_track_frames

        # track_id -> list of frame records
        self._history: dict[int, list[dict]] = defaultdict(list)
        self._last_seen: dict[int, int] = {}
        self._alerted: set[int] = set()
        self._frame_idx: int = 0

    def reset(self) -> None:
        self._history.clear()
        self._last_seen.clear()
        self._alerted.clear()
        self._frame_idx = 0

    def update(
        self, detections: sv.Detections
    ) -> tuple[sv.Detections, list[int], int]:
        """
        Update track histories and return:
          (confirmed_detections, new_alert_track_ids, n_suppressed)

        `confirmed_detections` is the subset of detections whose tracks
        have passed all trajectory criteria.
        """
        self._frame_idx += 1

        if len(detections) == 0 or detections.tracker_id is None:
            return detections, [], 0

        # Expire stale tracks
        for tid in list(self._last_seen):
            if self._frame_idx - self._last_seen[tid] > self.lost_track_frames:
                del self._history[tid]
                del self._last_seen[tid]
                self._alerted.discard(tid)

        keep: list[int] = []
        new_alerts: list[int] = []

        for i, (bbox, tid, conf) in enumerate(
            zip(detections.xyxy, detections.tracker_id, detections.confidence)
        ):
            if tid is None:
                continue

            tid = int(tid)
            x1, y1, x2, y2 = bbox
            area = max(0.0, (x2 - x1) * (y2 - y1))

            self._history[tid].append({
                "frame":  self._frame_idx,
                "area":   area,
                "conf":   float(conf),
                "cy":     (y1 + y2) / 2.0,
            })
            self._last_seen[tid] = self._frame_idx

            # Trim to sliding window
            if len(self._history[tid]) > self.history_window:
                self._history[tid] = self._history[tid][-self.history_window:]

            hist = self._history[tid]

            # Need at least min_track_frames observations
            if len(hist) < self.min_track_frames:
                continue

            recent = hist[-self.min_track_frames:]
            areas  = [h["area"] for h in recent]
            confs  = [h["conf"] for h in recent]

            # Criterion 1: area coefficient of variation (fire grows steadily)
            area_mean = np.mean(areas) + 1e-6
            area_cv   = float(np.std(areas) / area_mean)

            # Criterion 2: sustained confidence
            mean_conf = float(np.mean(confs))

            if area_cv <= self.max_area_cv and mean_conf >= self.min_mean_conf:
                keep.append(i)
                if tid not in self._alerted:
                    new_alerts.append(tid)
                    self._alerted.add(tid)

        n_suppressed = len(detections) - len(keep)
        if keep:
            filtered = detections[np.array(keep, dtype=int)]
        else:
            filtered = sv.Detections.empty()

        return filtered, new_alerts, n_suppressed

    @property
    def total_alerts(self) -> int:
        return len(self._alerted)
