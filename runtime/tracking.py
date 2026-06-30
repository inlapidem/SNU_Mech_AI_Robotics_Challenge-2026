"""Lightweight multi-object tracker (IoU association) for multi-frame voting.

Keeps a short classification history per tracked object so the decision policy can
require several consistent observations before committing to a pickup. Set-agnostic.
"""

from collections import deque


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    _next_id = 0

    def __init__(self, bbox, frame_idx, window):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.last_seen = frame_idx
        self.hits = 1
        self.history = deque(maxlen=window)   # list of dicts: {cls, conf, margin, frame}

    def update(self, bbox, frame_idx):
        self.bbox = bbox
        self.last_seen = frame_idx
        self.hits += 1

    def add_obs(self, cls, conf, margin, frame_idx):
        self.history.append({"cls": cls, "conf": conf, "margin": margin, "frame": frame_idx})


class Tracker:
    def __init__(self, iou_thr=0.3, max_age=10, window=7):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.window = window
        self.tracks = []

    def update(self, detections, frame_idx):
        """detections: list of bbox (x0,y0,x1,y1). Returns matched [(track, bbox), ...]."""
        unmatched = list(range(len(detections)))
        matched = []
        # Greedy IoU association, best pairs first.
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di in unmatched:
                v = iou(tr.bbox, detections[di])
                if v >= self.iou_thr:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            self.tracks[ti].update(detections[di], frame_idx)
            matched.append((self.tracks[ti], detections[di]))
        for di in unmatched:
            if di not in used_d:
                tr = Track(detections[di], frame_idx, self.window)
                self.tracks.append(tr)
                matched.append((tr, detections[di]))
        # Age out stale tracks.
        self.tracks = [t for t in self.tracks if frame_idx - t.last_seen <= self.max_age]
        return matched
