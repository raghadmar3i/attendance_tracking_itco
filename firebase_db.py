import datetime

import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, storage

import config


class FirebaseDB:
    def __init__(self):
        cred = credentials.Certificate(config.FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred, {
            "storageBucket": config.FIREBASE_STORAGE_BUCKET
        })
        self.db = firestore.client()
        self.bucket = storage.bucket()

    def load_known_embeddings(self):
        """
        Loads all employee embeddings from Firestore into memory.
        Call this at startup and periodically to refresh.

        Returns: (ids: list[str], names: list[str], embeddings: np.ndarray [N, D])
        """
        docs = self.db.collection(config.FIRESTORE_USERS_COLLECTION).stream()
        ids, names, embeddings = [], [], []
        for doc in docs:
            data = doc.to_dict()
            emb = data.get("embedding")
            if emb is None:
                continue
            ids.append(data.get("user_id", doc.id))
            names.append(data.get("full_name", doc.id))
            embeddings.append(np.array(emb, dtype=np.float32))

        if len(embeddings) == 0:
            return ids, names, np.zeros((0, 512), dtype=np.float32)
        return ids, names, np.vstack(embeddings)

    def log_attendance(self, employee_id, name, confidence, timestamp=None):
        timestamp = timestamp or datetime.datetime.utcnow()
        self.db.collection(config.FIRESTORE_ATTENDANCE_COLLECTION).add({
            "employee_id": employee_id,
            "name": name,
            "confidence": float(confidence),
            "timestamp": timestamp,
        })

    def log_flagged(self, track_id, confidence, face_crop_bgr, timestamp=None):
        """
        Uploads the cropped, low-confidence face to Firebase Storage and
        writes a reference document to Firestore for human review.
        """
        timestamp = timestamp or datetime.datetime.utcnow()
        filename = f"flagged/{track_id}_{int(timestamp.timestamp())}.jpg"

        ok, buf = cv2.imencode(".jpg", face_crop_bgr)
        if not ok:
            return

        blob = self.bucket.blob(filename)
        blob.upload_from_string(buf.tobytes(), content_type="image/jpeg")
        blob.make_public()  # switch to a signed URL if these shouldn't be publicly readable

        self.db.collection(config.FIRESTORE_FLAGGED_COLLECTION).add({
            "track_id": track_id,
            "confidence": float(confidence),
            "image_url": blob.public_url,
            "timestamp": timestamp,
            "reviewed": False,
        })