"""
One-off / admin script to enroll a new person into the attendance system.
Matches the existing Firestore schema (collection 'users', doc ID = user_id):

    {
      "full_name": "Feno",
      "embedding": [...],
      "sample_count": 1,
      "updated_at": <server timestamp>,
      "user_id": "usr_feno"
    }

Usage:
    python enroll.py <image_path> <user_id> "<Full Name>"

Use a clear, front-facing, well-lit photo for best embedding quality.
Note: your existing 'usr_feno' entry has sample_count: 512, which suggests
whatever you used to create it already averages the embedding over many
samples/frames rather than a single photo -- that's a stronger embedding
than this single-image version below. If that's the case, keep using your
existing enrollment process and treat this script as a fallback/reference.
"""
import sys

import cv2
from insightface.app import FaceAnalysis
from google.cloud import firestore as gcf

import config
from firebase_db import FirebaseDB


def enroll(image_path, user_id, full_name):
    app = FaceAnalysis(name=config.INSIGHTFACE_MODEL_NAME)
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    faces = app.get(img)
    if len(faces) == 0:
        print("No face found in image.")
        return
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    embedding = faces[0].normed_embedding.tolist()

    db = FirebaseDB()
    db.db.collection(config.FIRESTORE_USERS_COLLECTION).document(user_id).set({
        "full_name": full_name,
        "embedding": embedding,
        "sample_count": 1,
        "updated_at": gcf.SERVER_TIMESTAMP,
        "user_id": user_id,
    })
    print(f"Enrolled {full_name} ({user_id}).")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python enroll.py <image_path> <user_id> "<Full Name>"')
        sys.exit(1)
    enroll(sys.argv[1], sys.argv[2], sys.argv[3])