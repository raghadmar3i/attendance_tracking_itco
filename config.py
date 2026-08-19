import os
import torch


# ---- Model paths ----
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", r"D:\itco_internship\attendance _tracking_system\files\yolov11_detection.pt")
INSIGHTFACE_MODEL_NAME = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")

_GPU_AVAILABLE = torch.cuda.is_available()
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda:0" if _GPU_AVAILABLE else "cpu")
INSIGHTFACE_CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID", "0" if _GPU_AVAILABLE else "-1"))

# ---- Firebase ----
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH", r"D:\itco_internship\attendance _tracking_system\files\itco-attendance-tracking-firebase-adminsdk-fbsvc-9390482806.json")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "itco-attendance-tracking.firebasestorage.app")


FIRESTORE_USERS_COLLECTION = "users"          # holds name + embedding per person
FIRESTORE_ATTENDANCE_COLLECTION = "attendance_logs"     # confirmed recognitions
FIRESTORE_FLAGGED_COLLECTION = "flagged_faces"          # low-confidence crops for human review

# ---- Thresholds (tune these against your own data) ----
DETECTION_CONF_THRESHOLD = 0.5      # YOLO face detection confidence
RECOGNITION_SIM_THRESHOLD = 0.45    # cosine similarity for a "confident" match
RE_LOG_COOLDOWN_SECONDS = 300       # don't re-log/re-flag the same track within this window
EMBEDDING_CACHE_REFRESH_SECONDS = 600  # reload known embeddings from Firestore periodically

# ---- Camera / performance ----
CAMERA_SOURCE = 0   # 0 for default webcam, or an RTSP/HTTP stream URL
FRAME_SKIP = 2       # only run detection on every Nth frame
TRACKER_MAX_MISSES = 15   # frames a track can go undetected before being dropped
TRACKER_IOU_THRESHOLD = 0.3
