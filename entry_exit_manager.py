import datetime

import config


UTC = datetime.timezone.utc


def aware_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class EntryExitManager:
    """Treat distinct recognition events as alternating entry/exit scans."""

    def __init__(self, db):
        self.db = db
        self.active = {}
        self.last_event = {}
        sessions = sorted(
            db.load_attendance_sessions(),
            key=lambda item: aware_utc(item.get("check_in"))
            or datetime.datetime.min.replace(tzinfo=UTC),
        )
        for item in sessions:
            employee_id = item.get("employee_id")
            if not employee_id:
                continue
            check_in = aware_utc(item.get("check_in"))
            check_out = aware_utc(item.get("check_out"))
            event_time = check_out or check_in
            if event_time:
                self.last_event[employee_id] = event_time
            if item.get("status") == "open" and check_in:
                self.active[employee_id] = {
                    "session_id": item["session_id"],
                    "check_in": check_in,
                    "name": item.get("name", employee_id),
                }
            elif item.get("status") == "closed":
                self.active.pop(employee_id, None)

    @staticmethod
    def now_utc():
        return datetime.datetime.now(UTC)

    def toggle(self, employee_id, name, confidence, timestamp=None):
        timestamp = aware_utc(timestamp) or self.now_utc()
        previous = self.last_event.get(employee_id)
        if previous and (
            timestamp - previous
        ).total_seconds() < config.ENTRY_EXIT_COOLDOWN_SECONDS:
            return "cooldown", None

        current = self.active.get(employee_id)
        if current is None:
            session_id = self.db.start_attendance_session(
                employee_id, name, confidence, timestamp
            )
            self.db.log_attendance(
                employee_id, name, confidence, timestamp, event_type="entry"
            )
            self.active[employee_id] = {
                "session_id": session_id,
                "check_in": timestamp,
                "name": name,
            }
            event = "entry"
            print(f"[ENTRY] {name} at {timestamp.isoformat()}")
        else:
            self.db.close_attendance_session(
                current["session_id"], current["check_in"], timestamp
            )
            self.db.log_attendance(
                employee_id, name, confidence, timestamp, event_type="exit"
            )
            self.active.pop(employee_id, None)
            event = "exit"
            hours = (timestamp - current["check_in"]).total_seconds() / 3600
            print(f"[EXIT] {name} at {timestamp.isoformat()} ({hours:.2f} hours)")

        self.last_event[employee_id] = timestamp
        return event, self.active.get(employee_id)

    def is_inside(self, employee_id):
        return employee_id in self.active
