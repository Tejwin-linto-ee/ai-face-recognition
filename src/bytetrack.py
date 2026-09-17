"""Pure Python + NumPy implementation of ByteTrack (Zhang et al., 2022).

ByteTrack associates both high-confidence and low-confidence detection boxes
across video frames to maintain persistent track IDs through partial occlusions,
motion blur, and extreme angles without requiring heavy optical flow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple
import numpy as np


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes: shape [N, 4] and [M, 4].

    Box format: [x1, y1, x2, y2]
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


class KalmanBoxTracker:
    """Kalman filter tracking bounding box state [cx, cy, aspect_ratio, height]."""

    _count = 0

    def __init__(self, bbox_xywh: Tuple[int, int, int, int]):
        x, y, w, h = bbox_xywh
        self.state = np.array([x + w / 2.0, y + h / 2.0, w / max(float(h), 1e-4), float(h), 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
        self.covariance[4:, 4:] *= 100.0

    def predict(self) -> None:
        """Constant-velocity motion model prediction."""
        dt = 1.0
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = dt

        Q = np.eye(8, dtype=np.float32) * 1.0
        Q[4:, 4:] *= 0.1

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, bbox_xywh: Tuple[int, int, int, int]) -> None:
        """Correct state with new detection observation."""
        x, y, w, h = bbox_xywh
        z = np.array([x + w / 2.0, y + h / 2.0, w / max(float(h), 1e-4), float(h)], dtype=np.float32)

        H = np.zeros((4, 8), dtype=np.float32)
        H[:4, :4] = np.eye(4, dtype=np.float32)

        R = np.eye(4, dtype=np.float32) * 1.0
        R[2, 2] *= 10.0

        y_res = z - (H @ self.state)
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y_res)
        self.covariance = (np.eye(8, dtype=np.float32) - (K @ H)) @ self.covariance

    def get_bbox_xywh(self) -> Tuple[int, int, int, int]:
        """Return estimated [x, y, w, h]."""
        cx, cy, r, h = self.state[:4]
        w = r * h
        x = cx - w / 2.0
        y = cy - h / 2.0
        return int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))

    def get_bbox_xyxy(self) -> Tuple[float, float, float, float]:
        x, y, w, h = self.get_bbox_xywh()
        return float(x), float(y), float(x + w), float(y + h)


@dataclass
class STrack:
    track_id: int
    bbox_xywh: Tuple[int, int, int, int]
    score: float
    state: TrackState = TrackState.NEW
    kalman: KalmanBoxTracker = field(init=False)
    frame_id: int = 0
    tracklet_len: int = 0
    user_id: str = "Scanning"
    similarity: float = 0.0
    quality_score: float = 0.0
    motion_verified: bool = False
    motion_reference: Tuple[int, int, int, int] | None = None
    deep_liveness: str = "not checked"

    def __post_init__(self):
        self.kalman = KalmanBoxTracker(self.bbox_xywh)

    def predict(self) -> None:
        self.kalman.predict()
        self.bbox_xywh = self.kalman.get_bbox_xywh()

    def update(self, bbox_xywh: Tuple[int, int, int, int], score: float, frame_id: int) -> None:
        self.kalman.update(bbox_xywh)
        self.bbox_xywh = self.kalman.get_bbox_xywh()
        self.score = score
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.TRACKED

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    @property
    def tlbr(self) -> Tuple[float, float, float, float]:
        x, y, w, h = self.bbox_xywh
        return float(x), float(y), float(x + w), float(y + h)


class ByteTracker:
    """Robust multi-face association using ByteTrack logic."""

    def __init__(
        self,
        track_thresh: float = 0.50,
        high_thresh: float = 0.60,
        match_thresh: float = 0.70,
        max_lost_frames: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.next_id = 1

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> List[STrack]:
        """Update tracks given list of ((x, y, w, h), confidence)."""
        self.frame_id += 1

        # 1. Predict current locations with Kalman Filter
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()

        # 2. Split detections into high and low confidence sets
        detections_high = []
        detections_low = []
        for box, score in detections:
            if score >= self.high_thresh:
                detections_high.append((box, score))
            elif score >= self.track_thresh:
                detections_low.append((box, score))

        # 3. First Association: Match high-confidence detections with active tracks
        strack_pool = [t for t in self.tracked_stracks if t.state == TrackState.TRACKED] + self.lost_stracks
        matched_tracks_a, unmatched_tracks_a, unmatched_dets_high = self._associate(
            strack_pool, detections_high, iou_thresh=self.match_thresh
        )

        for track, (box, score) in matched_tracks_a:
            track.update(box, score, self.frame_id)
            if track in self.lost_stracks:
                self.lost_stracks.remove(track)
                self.tracked_stracks.append(track)

        # 4. Second Association: Match low-confidence detections with remaining active tracks
        unmatched_active_tracks = [t for t in unmatched_tracks_a if t.state == TrackState.TRACKED]
        matched_tracks_b, unmatched_tracks_b, _ = self._associate(
            unmatched_active_tracks, detections_low, iou_thresh=0.50
        )

        for track, (box, score) in matched_tracks_b:
            track.update(box, score, self.frame_id)

        # 5. Handle remaining unmatched tracks: mark as LOST
        for track in unmatched_tracks_b:
            if track.state != TrackState.LOST:
                track.mark_lost()
                if track in self.tracked_stracks:
                    self.tracked_stracks.remove(track)
                self.lost_stracks.append(track)

        # 6. Initialize new tracks from remaining high-confidence detections
        for box, score in unmatched_dets_high:
            new_track = STrack(
                track_id=self.next_id,
                bbox_xywh=box,
                score=score,
                state=TrackState.TRACKED,
                frame_id=self.frame_id,
            )
            self.next_id += 1
            self.tracked_stracks.append(new_track)

        # 7. Remove dead tracks that exceeded max_lost_frames
        retained_lost = []
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_lost_frames:
                track.mark_removed()
                self.removed_stracks.append(track)
            else:
                retained_lost.append(track)
        self.lost_stracks = retained_lost

        # Return all currently active tracks
        return [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]

    def _associate(
        self,
        tracks: List[STrack],
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[STrack, Tuple[Tuple[int, int, int, int], float]]], List[STrack], List[Tuple[Tuple[int, int, int, int], float]]]:
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        track_boxes = np.array([t.tlbr for t in tracks], dtype=np.float32)
        det_boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b, _ in detections], dtype=np.float32)

        ious = box_iou(track_boxes, det_boxes)

        matched_tracks = []
        unmatched_track_indices = set(range(len(tracks)))
        unmatched_det_indices = set(range(len(detections)))

        # Greedy association sorted by highest IoU overlap
        candidate_matches = sorted(
            [
                (ious[t_idx, d_idx], t_idx, d_idx)
                for t_idx in range(len(tracks))
                for d_idx in range(len(detections))
                if ious[t_idx, d_idx] >= iou_thresh
            ],
            reverse=True,
            key=lambda item: item[0],
        )

        for overlap, t_idx, d_idx in candidate_matches:
            if t_idx in unmatched_track_indices and d_idx in unmatched_det_indices:
                matched_tracks.append((tracks[t_idx], detections[d_idx]))
                unmatched_track_indices.discard(t_idx)
                unmatched_det_indices.discard(d_idx)

        unmatched_tracks = [tracks[idx] for idx in unmatched_track_indices]
        unmatched_dets = [detections[idx] for idx in unmatched_det_indices]

        return matched_tracks, unmatched_tracks, unmatched_dets
