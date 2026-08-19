import numpy as np
from insightface.app import FaceAnalysis

import config


class FaceRecognizer:
    def __init__(self, known_ids, known_names, known_embeddings, ctx_id=None):
        ctx_id = config.INSIGHTFACE_CTX_ID if ctx_id is None else ctx_id
        self.app = FaceAnalysis(name=config.INSIGHTFACE_MODEL_NAME)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print(f"[FaceRecognizer] ctx_id={ctx_id} ({'GPU' if ctx_id >= 0 else 'CPU'})")
        self.set_known(known_ids, known_names, known_embeddings)

    def set_known(self, known_ids, known_names, known_embeddings):
        """Replace the in-memory known-faces cache (used at startup and on refresh)."""
        self.known_ids = known_ids
        self.known_names = known_names
        if known_embeddings.shape[0] > 0:
            known_embeddings = self._normalize(known_embeddings)
        self.known_embeddings = known_embeddings

    @staticmethod
    def _normalize(vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        return vecs / norms

    def get_embedding(self, face_crop_bgr):
        """
        face_crop_bgr: cropped face image (numpy array, BGR), from the YOLO detection.
        Returns a 512-d L2-normalized embedding, or None if InsightFace can't
        find a usable face inside the crop.
        """
        faces = self.app.get(face_crop_bgr)
        if len(faces) == 0:
            return None
        # if InsightFace finds more than one face in the crop, use the largest
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[0].normed_embedding

    def recognize(self, embedding):
        """
        Compares an embedding against all known embeddings via cosine similarity
        (single vectorized matrix multiply, since everything is L2-normalized).

        Returns (employee_id, name, similarity) for the best match if it clears
        the threshold, otherwise (None, None, best_similarity_found).
        """
        if self.known_embeddings.shape[0] == 0:
            return None, None, 0.0

        sims = self.known_embeddings @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= config.RECOGNITION_SIM_THRESHOLD:
            return self.known_ids[best_idx], self.known_names[best_idx], best_sim
        return None, None, best_sim