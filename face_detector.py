import config
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path=None, conf_threshold=None, device=None):
        self.model = YOLO(model_path or config.YOLO_MODEL_PATH)
        self.device = device or config.YOLO_DEVICE
        self.model.to(self.device)
        print(f"[FaceDetector] running on {self.device}")
        self.conf_threshold = conf_threshold or config.DETECTION_CONF_THRESHOLD

    def detect(self, frame):
        """
        Runs face detection on a single BGR frame.
        Returns list of (x1, y1, x2, y2, conf) in pixel coordinates.
        """
        results = self.model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes
