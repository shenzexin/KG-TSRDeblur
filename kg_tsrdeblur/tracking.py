"""Causal IoU/Hungarian Kalman tracking and equations (1)--(6), NumPy only."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackerConfig:
    confirm_hits: int = 3
    max_missing: int = 5
    iou_threshold: float = 0.01
    innovation_threshold: float = 9.4877 / 4
    process_accel_std: float = 300.0
    measurement_std: float = 3.0
    initial_velocity_std: float = 100.0
    initial_displacement_fraction: float = 1.0
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.5
    min_confidence: float = 0.25
    nms_iou: float = 0.5


def xyxy_to_geometry(box):
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])


def geometry_to_xyxy(g):
    cx, cy, w, h = g[:4]
    return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])


def box_iou(a, b):
    lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = np.maximum(rb-lt, 0).prod()
    area_a, area_b = np.maximum(a[2:]-a[:2], 0).prod(), np.maximum(b[2:]-b[:2], 0).prod()
    return float(intersection / max(area_a+area_b-intersection, 1e-12))


def filter_detections(detections, cfg):
    """Class-agnostic confidence filtering and NMS; input [x1,y1,x2,y2,score]."""
    dets = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
    valid = np.isfinite(dets).all(1) & (dets[:, 4] >= cfg.min_confidence) & (dets[:, 4] <= 1)
    valid &= (dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])
    dets = dets[valid]
    kept = []
    for i in np.argsort(-dets[:, 4], kind="stable"):
        if all(box_iou(dets[i, :4], x[:4]) <= cfg.nms_iou for x in kept):
            kept.append(dets[i])
    return np.asarray(kept, dtype=np.float64).reshape(-1, 5)


class KalmanTrack:
    def __init__(self, track_id, detection, cfg, dt=1/30):
        self.id, self.cfg = track_id, cfg
        self.x = np.r_[xyxy_to_geometry(detection[:4]), np.zeros(4)]
        # Initial motion is unknown: allow one box dimension per frame, in px/s.
        velocity_std = np.maximum(cfg.initial_velocity_std,
            cfg.initial_displacement_fraction * np.tile(self.x[2:4], 2) / dt)
        self.P = np.diag(np.r_[[cfg.measurement_std**2]*4, velocity_std**2])
        self.H = np.eye(4, 8)
        self.R = np.eye(4) * cfg.measurement_std**2
        self.hits, self.missing = 1, 0
        self.confidence, self.innovation = float(detection[4]), 0.0
        self.selected_box = detection[:4].copy()

    def predict(self, dt):
        F = np.eye(8)
        F[:4, 4:] = dt*np.eye(4)
        G = np.vstack([0.5*dt**2*np.eye(4), dt*np.eye(4)])
        Q = self.cfg.process_accel_std**2 * G @ G.T
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.missing += 1
        self.selected_box = geometry_to_xyxy(self.x)

    def residual(self, box):
        v = xyxy_to_geometry(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return v, S, float(v @ np.linalg.solve(S, v) / 4)

    def update(self, detection):
        v, S, score = self.residual(detection[:4])
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x += K @ v
        # Joseph update maintains symmetry and PSD under finite precision.
        A = np.eye(8) - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T)/2
        self.hits += 1
        self.missing = 0
        self.confidence, self.innovation = float(detection[4]), score
        # Current observed box for ROI; predicted box only during a gap.
        self.selected_box = detection[:4].copy()

    @property
    def confirmed(self):
        return self.hits >= self.cfg.confirm_hits

    def prior(self, frame_width, frame_height, dt):
        g = xyxy_to_geometry(self.selected_box)
        w, h = np.maximum(g[2:4], 1e-6)
        scales = np.array([1/w, 1/h, 1/w, 1/h, dt/w, dt/h, dt/w, dt/h])
        diagonal = np.maximum(np.diag(self.P) * scales**2, 0)
        u = diagonal.mean()
        cfg = self.cfg
        r = float(self.confirmed) * self.confidence * np.exp(-cfg.alpha*u - cfg.beta*self.innovation - cfg.gamma*self.missing)
        normalized = self.x / np.array([frame_width, frame_height, frame_width, frame_height, w/dt, h/dt, w/dt, h/dt])
        descriptor = np.r_[normalized, np.log1p(diagonal)].astype(np.float32)
        return descriptor, float(np.clip(r, 0, 1))


class MultiObjectTracker:
    def __init__(self, config=None):
        self.config = config or TrackerConfig()
        self.tracks = []
        self.next_id = 1

    def step(self, detections, frame_size, dt=1/30):
        """One call per frame, including empty frames. frame_size=(width,height)."""
        width, height = frame_size
        if not np.isfinite(dt) or dt <= 0 or min(width, height) <= 0:
            raise ValueError("Positive finite dt and frame dimensions required")
        cfg = self.config
        detections = filter_detections(detections, cfg)
        for track in self.tracks:
            track.predict(dt)
        # Remove geometrically invalid predictions before association.
        self.tracks = [t for t in self.tracks if np.isfinite(t.x).all() and (t.x[2:4] > 0).all()]
        cost = np.full((len(self.tracks), len(detections)), 1e6)
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou = box_iou(track.selected_box, det[:4])
                _, _, innovation = track.residual(det[:4])
                if iou >= cfg.iou_threshold and innovation <= cfg.innovation_threshold:
                    cost[i, j] = 1-iou
        matched = set()
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e6:
                    self.tracks[i].update(detections[j])
                    matched.add(j)
        for j, det in enumerate(detections):
            if j not in matched:
                self.tracks.append(KalmanTrack(self.next_id, det, cfg, dt))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missing <= cfg.max_missing and
                       not (t.missing and not t.confirmed)]
        candidates = []
        for t in self.tracks:
            box = t.selected_box.copy()
            clipped = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            if (clipped[2:]-clipped[:2] <= 0).any():
                continue
            descriptor, reliability = t.prior(width, height, dt)
            candidates.append(dict(track_id=t.id, box=box.tolist(), descriptor=descriptor,
                reliability=reliability, missing=t.missing, confirmed=t.confirmed, confidence=t.confidence))
        # Current observations first, then predicted ROIs. Limit is applied by pipeline.
        return sorted(candidates, key=lambda x: (x['missing'] > 0, -x['confidence'], x['track_id']))
