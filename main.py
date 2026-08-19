import datetime
import time

import cv2

import config
from firebase_db import FirebaseDB
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from tracker import CentroidTracker
from video_stream import VideoStream


def main():
    print("Connecting to Firebase and loading known embeddings...")
    db = FirebaseDB()
    ids, names, embeddings = db.load_known_embeddings()
    print(f"Loaded {len(ids)} known faces.")

    detector = FaceDetector()
    recognizer = FaceRecognizer(ids, names, embeddings)
    tracker = CentroidTracker()

    cap = VideoStream(config.CAMERA_SOURCE)
    frame_idx = 0
    last_cache_refresh = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying...")
            time.sleep(1)
            continue

        # periodically refresh known-face cache so new enrollments show up
        if time.time() - last_cache_refresh > config.EMBEDDING_CACHE_REFRESH_SECONDS:
            ids, names, embeddings = db.load_known_embeddings()
            recognizer.set_known(ids, names, embeddings)
            last_cache_refresh = time.time()
            print(f"Refreshed known-face cache: {len(ids)} people.")

        frame_idx += 1
        run_detection = (frame_idx % config.FRAME_SKIP == 0)

        if run_detection:
            detections = detector.detect(frame)
            tracker.update(detections)

            for tid, track in tracker.tracks.items():
                x1, y1, x2, y2 = track.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]

                # Only attempt recognition if this track hasn't been confidently
                # identified yet, and we're not still inside the cooldown from
                # its last attempt -- avoids re-running InsightFace on every
                # single processed frame for the same unresolved face.
                now = time.time()
                should_attempt = (
                    track.recognized_id is None
                    and now - track.last_logged_time > config.RE_LOG_COOLDOWN_SECONDS
                )
                if should_attempt:
                    emb = recognizer.get_embedding(crop)
                    if emb is not None:
                        emp_id, name, sim = recognizer.recognize(emb)

                        if emp_id is not None:
                            track.recognized_id = emp_id
                            track.recognized_name = name
                            db.log_attendance(emp_id, name, sim, datetime.datetime.utcnow())
                            track.last_logged_time = now
                            print(f"[LOGGED] Track {tid}: {name} ({sim:.2f})")
                        else:
                            db.log_flagged(tid, sim, crop, datetime.datetime.utcnow())
                            track.last_logged_time = now
                            print(f"[FLAGGED] Track {tid}: sim={sim:.2f}")

        # Draw every currently active track on every frame (not just frames
        # where we ran detection) so the box doesn't flicker between frames.
        for tid, track in tracker.tracks.items():
            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            color = (0, 255, 0) if track.recognized_id else (0, 0, 255)
            label = track.recognized_name or f"ID {tid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()