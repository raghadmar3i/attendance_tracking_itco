import threading

import cv2


class VideoStream:
    """
    Reads frames from the camera in a background thread so the main loop
    always grabs the most recent frame instead of one stuck behind a backlog
    in OpenCV's internal buffer. This is often the biggest source of
    perceived camera 'lag', independent of how fast the models run.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # not honored by all backends
        except Exception:
            pass

        self.lock = threading.Lock()
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()