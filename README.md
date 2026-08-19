# Facial Recognition Attendance System

Pipeline: **YOLOv11s** (face detection) -> **IoU tracker** (stable per-face ID)
-> **InsightFace** (embedding + recognition against Firestore) -> **Firestore /
Firebase Storage** (attendance logs + flagged low-confidence crops).

## 1. Setup

```bash
pip install -r requirements.txt
```

- Put your trained YOLOv11s weights at `models/yolov11s_face.pt` (or set
  `YOLO_MODEL_PATH` env var).
- Download a Firebase service account key (Project Settings -> Service
  Accounts -> Generate new private key) and save it as
  `firebase_credentials.json` (or set `FIREBASE_CRED_PATH`).
- Set `FIREBASE_STORAGE_BUCKET` in `config.py` to your project's storage
  bucket (usually `<project-id>.appspot.com`), used for flagged-face images.

## 2. Firestore schema

**`employees` collection** (one doc per person, doc ID = employee ID):
```json
{
  "name": "Jane Doe",
  "embedding": [0.0123, -0.045, ...]   // 512 floats from InsightFace
}
```

**`attendance_logs`** (auto-generated doc IDs, one per confirmed recognition):
```json
{
  "employee_id": "EMP001",
  "name": "Jane Doe",
  "confidence": 0.71,
  "timestamp": "2026-08-11T09:03:12Z"
}
```

**`flagged_faces`** (low-confidence detections for human review):
```json
{
  "track_id": 14,
  "confidence": 0.31,
  "image_url": "https://storage.googleapis.com/.../flagged/14_....jpg",
  "timestamp": "2026-08-11T09:04:55Z",
  "reviewed": false
}
```

## 3. Enroll people

```bash
python enroll.py photos/jane.jpg EMP001 "Jane Doe"
```

Run this once per person, ideally with a clear front-facing photo. This
populates the `employees` collection that `main.py` reads at startup.

## 4. Run the live system

```bash
python main.py
```

Opens the default webcam (`CAMERA_SOURCE = 0` in `config.py` — change to an
RTSP/HTTP URL for an IP camera), draws green boxes for recognized faces and
red boxes for unrecognized/flagged ones, and logs to Firestore as described
above. Press `q` to quit.

## 5. Key thresholds to tune (in `config.py`)

- `DETECTION_CONF_THRESHOLD` — YOLO detection confidence.
- `RECOGNITION_SIM_THRESHOLD` — cosine similarity cutoff for a "confident"
  match. Start at 0.45 and adjust based on false-accept/false-reject rates
  on your own footage — this matters more than any other setting.
- `RE_LOG_COOLDOWN_SECONDS` — prevents duplicate logs/flags for the same
  person within a short window.
- `FRAME_SKIP` — trade off latency vs. CPU/GPU load.

## Next steps worth considering

- Move Firestore/Storage writes onto a background thread or queue so they
  never block the camera loop.
- Swap the IoU tracker for ByteTrack/DeepSORT if people cross paths often.
- Add a liveness/anti-spoof check before logging a match.
- Batch-refresh the embedding cache on a schedule (already stubbed via
  `EMBEDDING_CACHE_REFRESH_SECONDS`) so new enrollments show up without a
  restart — already wired into `main.py`.
