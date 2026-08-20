import datetime
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, storage

import config


class FirebaseDB:
    def __init__(self):
        try:
            firebase_admin.get_app()
        except ValueError:
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
            face_templates = [emb]
            for reference in doc.reference.collection("face_references").stream():
                reference_embedding = reference.to_dict().get("embedding")
                if reference_embedding is not None:
                    face_templates.append(reference_embedding)
            for template in face_templates:
                ids.append(data.get("user_id", doc.id))
                names.append(data.get("full_name", doc.id))
                embeddings.append(np.array(template, dtype=np.float32))

        if len(embeddings) == 0:
            return ids, names, np.zeros((0, 512), dtype=np.float32)
        return ids, names, np.vstack(embeddings)

    def load_users(self):
        """Return employee metadata without loading face embeddings."""
        users = []
        for doc in self.db.collection(config.FIRESTORE_USERS_COLLECTION).stream():
            data = doc.to_dict()
            users.append({
                "user_id": data.get("user_id", doc.id),
                "full_name": data.get("full_name", doc.id),
                "required_daily_hours": float(
                    data.get("required_daily_hours")
                    or config.DEFAULT_REQUIRED_DAILY_HOURS
                ),
                "working_weekdays": data.get("working_weekdays")
                or config.DEFAULT_WORKING_WEEKDAYS,
                "active": data.get("active", True),
            })
        return users

    def log_attendance(
        self, employee_id, name, confidence, timestamp=None, event_type=None
    ):
        timestamp = timestamp or datetime.datetime.utcnow()
        payload = {
            "employee_id": employee_id,
            "name": name,
            "confidence": float(confidence),
            "timestamp": timestamp,
        }
        if event_type:
            payload["event_type"] = event_type
        self.db.collection(config.FIRESTORE_ATTENDANCE_COLLECTION).add(payload)

    def start_attendance_session(self, employee_id, name, confidence, timestamp):
        ref = self.db.collection(config.FIRESTORE_SESSIONS_COLLECTION).document()
        ref.set({
            "employee_id": employee_id,
            "name": name,
            "check_in": timestamp,
            "check_out": None,
            "last_seen": timestamp,
            "duration_seconds": 0.0,
            "status": "open",
            "recognition_confidence": float(confidence),
            "camera_source": str(config.CAMERA_SOURCE),
        })
        return ref.id

    def update_attendance_session(self, session_id, check_in, last_seen):
        duration = max(0.0, (last_seen - check_in).total_seconds())
        self.db.collection(config.FIRESTORE_SESSIONS_COLLECTION).document(
            session_id
        ).update({"last_seen": last_seen, "duration_seconds": duration})

    def close_attendance_session(self, session_id, check_in, check_out):
        duration = max(0.0, (check_out - check_in).total_seconds())
        self.db.collection(config.FIRESTORE_SESSIONS_COLLECTION).document(
            session_id
        ).update({
            "check_out": check_out,
            "last_seen": check_out,
            "duration_seconds": duration,
            "status": "closed",
        })

    def load_attendance_sessions(self):
        sessions = []
        for doc in self.db.collection(
            config.FIRESTORE_SESSIONS_COLLECTION
        ).stream():
            data = doc.to_dict()
            data["session_id"] = doc.id
            sessions.append(data)
        return sessions

    def save_report(self, report_id, payload):
        self.db.collection(config.FIRESTORE_REPORTS_COLLECTION).document(
            report_id
        ).set(payload)

    def load_reports(self):
        reports = []
        for doc in self.db.collection(config.FIRESTORE_REPORTS_COLLECTION).stream():
            data = doc.to_dict()
            data["report_id"] = doc.id
            reports.append(data)
        return reports

    def update_user_policy(
        self, user_id, required_daily_hours, working_weekdays, active
    ):
        self.db.collection(config.FIRESTORE_USERS_COLLECTION).document(
            user_id
        ).update({
            "required_daily_hours": float(required_daily_hours),
            "working_weekdays": list(working_weekdays),
            "active": bool(active),
        })

    def load_flagged_faces(self):
        faces = []
        for doc in self.db.collection(config.FIRESTORE_FLAGGED_COLLECTION).stream():
            data = doc.to_dict()
            data["flag_id"] = doc.id
            self._normalize_flagged_storage_path(data)
            faces.append(data)
        return faces

    def get_flagged_face(self, flag_id):
        doc = self.db.collection(config.FIRESTORE_FLAGGED_COLLECTION).document(
            flag_id
        ).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        data["flag_id"] = doc.id
        self._normalize_flagged_storage_path(data)
        return data

    def _normalize_flagged_storage_path(self, data):
        """Support legacy records that stored a Google Storage public URL."""
        if data.get("storage_path") or not data.get("image_url"):
            return
        path = unquote(urlparse(data["image_url"]).path).lstrip("/")
        bucket_prefix = f"{self.bucket.name}/"
        if path.startswith(bucket_prefix):
            data["storage_path"] = path[len(bucket_prefix):]

    def mark_flagged_reviewed(self, flag_id, reviewed=True):
        self.db.collection(config.FIRESTORE_FLAGGED_COLLECTION).document(
            flag_id
        ).update({
            "reviewed": bool(reviewed),
            "reviewed_at": firestore.SERVER_TIMESTAMP,
        })

    def add_employee_face_reference(self, user_id, embedding, flag_id):
        """Add a face template without replacing the employee's good samples."""
        ref = self.db.collection(config.FIRESTORE_USERS_COLLECTION).document(user_id)
        doc = ref.get()
        if not doc.exists:
            raise ValueError("Employee does not exist")
        data = doc.to_dict()
        new_template = np.asarray(embedding, dtype=np.float32)
        new_template /= max(float(np.linalg.norm(new_template)), 1e-6)
        duplicate = False
        existing_templates = []
        if data.get("embedding"):
            existing_templates.append(data["embedding"])
        reference_docs = list(ref.collection("face_references").stream())
        for reference in reference_docs:
            existing = reference.to_dict().get("embedding")
            if existing is not None:
                existing_templates.append(existing)
        for existing in existing_templates:
            vector = np.asarray(existing, dtype=np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            if float(vector @ new_template) >= 0.995:
                duplicate = True
        if not duplicate:
            if len(reference_docs) >= 9:
                raise ValueError(
                    "This employee already has the maximum of 10 face references."
                )
            ref.collection("face_references").add({
                "embedding": new_template.tolist(),
                "source_flag_id": flag_id,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
        ref.update({
            "face_reference_count": 1 + len(reference_docs) + (not duplicate),
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        self.db.collection(config.FIRESTORE_FLAGGED_COLLECTION).document(
            flag_id
        ).update({
            "reviewed": True,
            "reviewed_at": firestore.SERVER_TIMESTAMP,
            "labeled_employee_id": user_id,
            "resolution": "employee_reference_added",
        })
        return not duplicate

    def delete_flagged_face(self, flag_id):
        face = self.get_flagged_face(flag_id)
        if face is None:
            return False
        storage_path = face.get("storage_path")
        if storage_path:
            blob = self.bucket.blob(storage_path)
            if blob.exists():
                blob.delete()
        self.db.collection(config.FIRESTORE_FLAGGED_COLLECTION).document(
            flag_id
        ).delete()
        return True

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

        self.db.collection(config.FIRESTORE_FLAGGED_COLLECTION).add({
            "track_id": track_id,
            "confidence": float(confidence),
            "storage_path": filename,
            "timestamp": timestamp,
            "reviewed": False,
        })
