import time

import config


class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox                # (x1, y1, x2, y2)
        self.last_seen = time.time()
        self.recognized_id = None       # set once confidently recognized
        self.recognized_name = None
        self.recognition_confidence = 0.0
        self.attendance_event = None
        self.last_logged_time = 0.0     # last time we wrote to Firestore for this track
        self.misses = 0


class CentroidTracker:
    """
    Simple IoU-matching tracker. Not as robust as SORT/DeepSORT/ByteTrack,
    but dependency-free and enough for a fixed attendance camera. Swap in
    ByteTrack later if people cross paths a lot or move quickly.
    """

    def __init__(self, max_misses=None, iou_threshold=None):
        self.next_id = 1
        self.tracks = {}
        self.max_misses = max_misses or config.TRACKER_MAX_MISSES
        self.iou_threshold = iou_threshold or config.TRACKER_IOU_THRESHOLD

    @staticmethod
    def _iou(box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / float(area_a + area_b - inter)

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2, conf)
        Returns {track_id: Track} for every track active in this frame
        (newly created or matched to an existing one).
        """
        unmatched = list(range(len(detections)))
        matched_ids = []

        for tid, track in list(self.tracks.items()):
            best_iou, best_idx = 0.0, -1
            for di in unmatched:
                iou = self._iou(track.bbox, detections[di][:4])
                if iou > best_iou:
                    best_iou, best_idx = iou, di

            if best_iou >= self.iou_threshold:
                track.bbox = detections[best_idx][:4]
                track.last_seen = time.time()
                track.misses = 0
                unmatched.remove(best_idx)
                matched_ids.append(tid)
            else:
                track.misses += 1
                if track.misses > self.max_misses:
                    del self.tracks[tid]

        for di in unmatched:
            new_track = Track(self.next_id, detections[di][:4])
            self.tracks[self.next_id] = new_track
            matched_ids.append(self.next_id)
            self.next_id += 1

        return {tid: self.tracks[tid] for tid in matched_ids if tid in self.tracks}
