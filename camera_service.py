import datetime
import threading
import time

import cv2

import config
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from entry_exit_manager import EntryExitManager
from tracker import CentroidTracker


class CameraService:
    """Run attendance recognition and expose the annotated frame as MJPEG."""

    def __init__(self, db):
        self.db = db
        self.running = False
        self.status = "stopped"
        self.error = None
        self.latest_jpeg = None
        self.condition = threading.Condition()
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.status = "starting"
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.running = False
        self.status = "stopped"
        with self.condition:
            self.condition.notify_all()

    def frames(self):
        last_frame = None
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.latest_jpeg is not last_frame
                    or self.status in {"stopped", "error"},
                    timeout=2,
                )
                frame = self.latest_jpeg
            if frame is not None and frame is not last_frame:
                last_frame = frame
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            elif self.status in {"stopped", "error"}:
                break

    def _run(self):
        capture = None
        entry_exit = EntryExitManager(self.db)
        try:
            ids, names, embeddings = self.db.load_known_embeddings()
            detector = FaceDetector()
            recognizer = FaceRecognizer(ids, names, embeddings)
            tracker = CentroidTracker()
            capture = cv2.VideoCapture(config.CAMERA_SOURCE)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera source {config.CAMERA_SOURCE}")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.running = True
            self.status = "running"
            frame_index = 0
            last_refresh = time.monotonic()

            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                frame_index += 1
                if time.monotonic() - last_refresh >= config.EMBEDDING_CACHE_REFRESH_SECONDS:
                    ids, names, embeddings = self.db.load_known_embeddings()
                    recognizer.set_known(ids, names, embeddings)
                    last_refresh = time.monotonic()

                if frame_index % config.FRAME_SKIP == 0:
                    detections = detector.detect(frame)
                    tracker.update(detections)
                    self._recognize_tracks(frame, tracker, recognizer, entry_exit)
                self._draw_tracks(frame, tracker)
                encoded, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                if encoded:
                    with self.condition:
                        self.latest_jpeg = buffer.tobytes()
                        self.condition.notify_all()
        except Exception as error:
            self.error = str(error)
            self.status = "error"
        finally:
            if capture is not None:
                capture.release()
            self.running = False
            if self.status != "error":
                self.status = "stopped"
            with self.condition:
                self.condition.notify_all()

    def _recognize_tracks(self, frame, tracker, recognizer, entry_exit):
        for track_id, track in tracker.tracks.items():
            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            now = time.time()
            if (
                track.recognized_id is None
                and now - track.last_logged_time > config.RE_LOG_COOLDOWN_SECONDS
            ):
                crop = frame[y1:y2, x1:x2]
                embedding = recognizer.get_embedding(crop)
                if embedding is None:
                    track.recognized_name = "Unknown"
                    self.db.log_flagged(
                        track_id, 0.0, crop, datetime.datetime.now(datetime.timezone.utc)
                    )
                    track.last_logged_time = now
                else:
                    employee_id, name, similarity = recognizer.recognize(embedding)
                    if employee_id is not None:
                        track.recognized_id = employee_id
                        track.recognized_name = name
                        track.recognition_confidence = similarity
                        event, _ = entry_exit.toggle(
                            employee_id,
                            name,
                            similarity,
                            datetime.datetime.now(datetime.timezone.utc),
                        )
                        track.attendance_event = event
                        track.last_logged_time = now
                    else:
                        track.recognized_name = "Unknown"
                        self.db.log_flagged(
                            track_id,
                            similarity,
                            crop,
                            datetime.datetime.now(datetime.timezone.utc),
                        )
                        track.last_logged_time = now

    @staticmethod
    def _draw_tracks(frame, tracker):
        for track_id, track in tracker.tracks.items():
            x1, y1, x2, y2 = track.bbox
            color = (0, 190, 110) if track.recognized_id else (30, 45, 220)
            label = track.recognized_name or f"Detecting · {track_id}"
            if track.attendance_event in {"entry", "exit"}:
                label += f" · {track.attendance_event.upper()}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x2, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 7, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
