import os
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"

# Keep third-party caches/settings project-local. This is especially useful on
# managed Windows machines where the roaming profile may not be writable.
ULTRALYTICS_CONFIG_ROOT = RUNTIME_DIR / "ultralytics"
ULTRALYTICS_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

# ---- Model paths ----
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH", str(BASE_DIR / "models" / "yolov11_detection.pt")
)
INSIGHTFACE_MODEL_NAME = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")
INSIGHTFACE_MODEL_ROOT = os.getenv(
    "INSIGHTFACE_MODEL_ROOT", str(BASE_DIR / "models" / "insightface")
)

_GPU_AVAILABLE = torch.cuda.is_available()
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda:0" if _GPU_AVAILABLE else "cpu")
INSIGHTFACE_CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID", "0" if _GPU_AVAILABLE else "-1"))

# ---- Firebase ----
FIREBASE_CRED_PATH = os.getenv(
    "FIREBASE_CRED_PATH", str(BASE_DIR / "firebase_credentials.json")
)
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "itco-attendance-tracking.firebasestorage.app")


FIRESTORE_USERS_COLLECTION = "users"          # holds name + embedding per person
FIRESTORE_ATTENDANCE_COLLECTION = "attendance_logs"     # confirmed recognitions
FIRESTORE_SESSIONS_COLLECTION = "attendance_sessions"
FIRESTORE_REPORTS_COLLECTION = "attendance_reports"
FIRESTORE_FLAGGED_COLLECTION = "flagged_faces"          # low-confidence crops for human review

# ---- Thresholds (tune these against your own data) ----
DETECTION_CONF_THRESHOLD = 0.5      # YOLO face detection confidence
RECOGNITION_SIM_THRESHOLD = 0.45    # cosine similarity for a "confident" match
RE_LOG_COOLDOWN_SECONDS = 300       # don't re-log/re-flag the same track within this window
EMBEDDING_CACHE_REFRESH_SECONDS = 60  # pick up admin-labeled faces quickly

# ---- Attendance policy ----
CLINIC_TIMEZONE = os.getenv("CLINIC_TIMEZONE", "Asia/Dubai")
DEFAULT_REQUIRED_DAILY_HOURS = float(
    os.getenv("DEFAULT_REQUIRED_DAILY_HOURS", "8")
)
DEFAULT_WORKING_WEEKDAYS = [
    int(day) for day in os.getenv("DEFAULT_WORKING_WEEKDAYS", "0,1,2,3,4").split(",")
    if day.strip()
]
ENTRY_EXIT_COOLDOWN_SECONDS = int(os.getenv("ENTRY_EXIT_COOLDOWN_SECONDS", "30"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports")))

# ---- Camera / performance ----
_CAMERA_SOURCE_VALUE = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = (
    int(_CAMERA_SOURCE_VALUE)
    if _CAMERA_SOURCE_VALUE.isdigit()
    else _CAMERA_SOURCE_VALUE
)  # numeric webcam index or an RTSP/HTTP stream URL
FRAME_SKIP = 2       # only run detection on every Nth frame
TRACKER_MAX_MISSES = 15   # frames a track can go undetected before being dropped
TRACKER_IOU_THRESHOLD = 0.3
