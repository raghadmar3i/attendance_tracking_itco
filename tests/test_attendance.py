import datetime
import unittest
from zoneinfo import ZoneInfo

from entry_exit_manager import EntryExitManager
from report_generator import (
    build_report,
    session_seconds_by_day,
    signed_hours_minutes,
    visit_history_by_employee,
)


UTC = datetime.timezone.utc


class FakeDB:
    def __init__(self):
        self.started = []
        self.updated = []
        self.closed = []
        self.logs = []

    def load_attendance_sessions(self):
        return []

    def start_attendance_session(self, employee_id, name, confidence, timestamp):
        self.started.append((employee_id, name, confidence, timestamp))
        return "session-1"

    def update_attendance_session(self, session_id, check_in, last_seen):
        self.updated.append((session_id, check_in, last_seen))

    def close_attendance_session(self, session_id, check_in, check_out):
        self.closed.append((session_id, check_in, check_out))

    def log_attendance(
        self, employee_id, name, confidence, timestamp, event_type=None
    ):
        self.logs.append((employee_id, timestamp, event_type))


class EntryExitManagerTests(unittest.TestCase):
    def test_distinct_scans_alternate_entry_and_exit(self):
        db = FakeDB()
        manager = EntryExitManager(db)
        start = datetime.datetime(2026, 8, 20, 8, tzinfo=UTC)
        event, _ = manager.toggle("E1", "Jane", 0.9, start)
        self.assertEqual(event, "entry")
        self.assertEqual(len(db.started), 1)
        event, _ = manager.toggle(
            "E1", "Jane", 0.9, start + datetime.timedelta(hours=8)
        )
        self.assertEqual(event, "exit")
        self.assertEqual(len(db.closed), 1)
        self.assertEqual(db.closed[0][2], start + datetime.timedelta(hours=8))
        self.assertEqual([item[2] for item in db.logs], ["entry", "exit"])

        event, _ = manager.toggle(
            "E1", "Jane", 0.9, start + datetime.timedelta(hours=9)
        )
        self.assertEqual(event, "entry")
        self.assertEqual(len(db.started), 2)


class ReportTests(unittest.TestCase):
    def test_hour_difference_uses_excel_safe_signed_format(self):
        self.assertEqual(signed_hours_minutes(-4.57), "-4:34")
        self.assertEqual(signed_hours_minutes(0.25), "+0:15")

    def test_multiple_visits_are_listed_separately(self):
        visits = [
            {
                "employee_id": "E1",
                "entry": datetime.datetime(2026, 8, 20, 8),
                "exit": datetime.datetime(2026, 8, 20, 12),
                "duration_days": 4 / 24,
            },
            {
                "employee_id": "E1",
                "entry": datetime.datetime(2026, 8, 20, 13),
                "exit": datetime.datetime(2026, 8, 20, 17),
                "duration_days": 4 / 24,
            },
        ]
        history = visit_history_by_employee(visits)["E1"]
        self.assertEqual(len(history), 2)
        self.assertIn("08:00", history[0])
        self.assertIn("17:00", history[1])

    def test_session_is_split_across_local_midnight(self):
        timezone = ZoneInfo("Asia/Dubai")
        session = {
            "check_in": datetime.datetime(2026, 8, 19, 19, 30, tzinfo=UTC),
            "check_out": datetime.datetime(2026, 8, 19, 21, 30, tzinfo=UTC),
        }
        pieces = session_seconds_by_day(
            session,
            timezone,
            datetime.date(2026, 8, 19),
            datetime.date(2026, 8, 20),
        )
        self.assertEqual(pieces[datetime.date(2026, 8, 19)], 1800)
        self.assertEqual(pieces[datetime.date(2026, 8, 20)], 5400)

    def test_daily_report_marks_required_hours(self):
        users = [{
            "user_id": "E1",
            "full_name": "Jane",
            "required_daily_hours": 8,
            "working_weekdays": [0, 1, 2, 3, 4],
            "active": True,
        }]
        sessions = [{
            "employee_id": "E1",
            "check_in": datetime.datetime(2026, 8, 20, 4, tzinfo=UTC),
            "check_out": datetime.datetime(2026, 8, 20, 12, tzinfo=UTC),
        }]
        _, _, rows = build_report(
            users,
            sessions,
            "daily",
            datetime.date(2026, 8, 20),
            "Asia/Dubai",
        )
        self.assertEqual(rows[0]["worked_hours"], 8.0)
        self.assertEqual(rows[0]["status"], "ACHIEVED")

    def test_daily_report_sums_multiple_visits(self):
        users = [{
            "user_id": "E1", "full_name": "Jane",
            "required_daily_hours": 8,
            "working_weekdays": [0, 1, 2, 3, 4], "active": True,
        }]
        sessions = [
            {
                "employee_id": "E1", "status": "closed",
                "check_in": datetime.datetime(2026, 8, 20, 4, tzinfo=UTC),
                "check_out": datetime.datetime(2026, 8, 20, 8, tzinfo=UTC),
            },
            {
                "employee_id": "E1", "status": "closed",
                "check_in": datetime.datetime(2026, 8, 20, 9, tzinfo=UTC),
                "check_out": datetime.datetime(2026, 8, 20, 13, tzinfo=UTC),
            },
        ]
        _, _, rows = build_report(
            users, sessions, "daily", datetime.date(2026, 8, 20), "Asia/Dubai"
        )
        self.assertEqual(rows[0]["worked_hours"], 8.0)
        self.assertEqual(rows[0]["session_count"], 2)
        self.assertEqual(rows[0]["status"], "ACHIEVED")


if __name__ == "__main__":
    unittest.main()
