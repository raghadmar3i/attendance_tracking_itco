import datetime
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from dashboard import create_app


UTC = datetime.timezone.utc


class FakeDashboardDB:
    def __init__(self):
        self.updated_policy = None
        self.deleted_flag = None
        self.added_reference = None
        ok, encoded = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
        self.bucket = FakeBucket(encoded.tobytes() if ok else b"")

    def load_users(self):
        return [{
            "user_id": "E1",
            "full_name": "Jane Doe",
            "required_daily_hours": 8.0,
            "working_weekdays": [0, 1, 2, 3, 4],
            "active": True,
        }]

    def load_attendance_sessions(self):
        return [{
            "employee_id": "E1",
            "name": "Jane Doe",
            "check_in": datetime.datetime.now(UTC),
            "check_out": None,
            "last_seen": datetime.datetime.now(UTC),
            "duration_seconds": 60,
            "status": "open",
            "recognition_confidence": 0.9,
        }]

    def load_flagged_faces(self):
        return []

    def load_reports(self):
        return []

    def update_user_policy(self, user_id, hours, weekdays, active):
        self.updated_policy = (user_id, hours, weekdays, active)

    def delete_flagged_face(self, flag_id):
        self.deleted_flag = flag_id
        return True

    def get_flagged_face(self, flag_id):
        return {"flag_id": flag_id, "storage_path": "flagged/test.jpg"}

    def add_employee_face_reference(self, user_id, embedding, flag_id):
        self.added_reference = (user_id, tuple(embedding.shape), flag_id)
        return True


class FakeBlob:
    def __init__(self, content):
        self.content = content

    def download_as_bytes(self):
        return self.content


class FakeBucket:
    def __init__(self, content):
        self.content = content

    def blob(self, _path):
        return FakeBlob(self.content)


class FakeRecognizer:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_embedding(self, _image):
        return np.ones(512, dtype=np.float32) / np.sqrt(512)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDashboardDB()
        self.password_patch = patch.dict(
            os.environ,
            {"ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "test-secret"},
        )
        self.password_patch.start()
        self.app = create_app(db=self.db, testing=True)
        self.client = self.app.test_client()

    def tearDown(self):
        self.password_patch.stop()

    def csrf(self):
        self.client.get("/login")
        with self.client.session_transaction() as state:
            return state["csrf_token"]

    def login(self):
        return self.client.post("/login", data={
            "csrf_token": self.csrf(),
            "username": "admin",
            "password": "test-secret",
        })

    def test_protected_page_redirects_to_login(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.location)

    def test_login_and_dashboard_render(self):
        response = self.login()
        self.assertEqual(response.status_code, 302)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Currently present", response.data)
        self.assertIn(b"Jane Doe", response.data)

    def test_live_camera_page_is_available_after_login(self):
        self.login()
        response = self.client.get("/camera")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Live detection camera", response.data)
        self.assertIn(b"Start camera", response.data)

    def test_employee_policy_update_requires_csrf(self):
        self.login()
        response = self.client.post("/employees/E1", data={
            "required_daily_hours": "7.5",
            "working_weekdays": ["0", "1", "2", "3", "4"],
            "active": "on",
        })
        self.assertEqual(response.status_code, 400)

        with self.client.session_transaction() as state:
            token = state["csrf_token"]
        response = self.client.post("/employees/E1", data={
            "csrf_token": token,
            "required_daily_hours": "7.5",
            "working_weekdays": ["0", "1", "2", "3", "4"],
            "active": "on",
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            self.db.updated_policy,
            ("E1", 7.5, [0, 1, 2, 3, 4], True),
        )

    def test_flagged_delete_requires_login_and_csrf(self):
        response = self.client.post("/flagged/flag-1/delete")
        self.assertEqual(response.status_code, 302)
        self.login()
        response = self.client.post("/flagged/flag-1/delete")
        self.assertEqual(response.status_code, 400)
        with self.client.session_transaction() as state:
            token = state["csrf_token"]
        response = self.client.post(
            "/flagged/flag-1/delete", data={"csrf_token": token}
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.db.deleted_flag, "flag-1")

    def test_flagged_face_can_be_labeled_as_employee(self):
        self.login()
        with self.client.session_transaction() as state:
            token = state["csrf_token"]
        with patch("dashboard.FaceRecognizer", FakeRecognizer):
            response = self.client.post("/flagged/flag-1/label", data={
                "csrf_token": token,
                "employee_id": "E1",
            })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.db.added_reference, ("E1", (512,), "flag-1"))


if __name__ == "__main__":
    unittest.main()
