# Clinic Facial-Recognition Attendance System

The system detects and tracks faces from a laptop camera or CCTV stream, matches
them against enrolled employees, creates check-in/check-out sessions, calculates
worked hours, flags unknown faces, and generates daily, weekly, and monthly CSV
and Firestore reports.

## Setup

The repository's long OneDrive path can exceed the Windows path limit while
installing PyTorch. Use a short-path environment:

```powershell
python -m venv C:\tmp\itco-attendance-venv
C:\tmp\itco-attendance-venv\Scripts\python.exe -m pip install -r requirements.txt
```

Required local assets (already installed in this workspace):

- `models/yolov11_detection.pt`
- `firebase_credentials.json`

Both are ignored by Git. Never commit the Firebase service-account key.

## Employee enrollment and schedule

Enroll a person from a clear, front-facing photograph:

```powershell
C:\tmp\itco-attendance-venv\Scripts\python.exe enroll.py photos\jane.jpg EMP001 "Jane Doe"
```

The `users/{user_id}` Firestore document supports:

```json
{
  "user_id": "EMP001",
  "full_name": "Jane Doe",
  "embedding": [0.0123, -0.045],
  "required_daily_hours": 8,
  "working_weekdays": [0, 1, 2, 3, 4],
  "active": true
}
```

Weekday numbers follow Python conventions: Monday is `0`, Sunday is `6`.
Missing schedule fields default to eight hours, Monday through Friday.

## Run attendance monitoring

```powershell
C:\tmp\itco-attendance-venv\Scripts\python.exe main.py
```

Press `q` to stop. A recognized employee receives a green label. The camera is
used as an entry/exit checkpoint: the first distinct scan checks the employee
in, the next distinct scan checks them out, and a later scan starts another
visit. Repeated frames from the same appearance do not create extra events.
The employee should present their face to the checkpoint both when entering
and when leaving.

Unknown faces receive a red `Unknown` label and are stored privately under
`flagged/` in Firebase Storage, with metadata in `flagged_faces`.

## Admin dashboard

Set a private admin password in the current PowerShell window and start the
dashboard:

```powershell
$env:ADMIN_USERNAME="admin"
$env:ADMIN_PASSWORD="replace-this-with-a-strong-password"
$env:DASHBOARD_SECRET_KEY="replace-this-with-a-long-random-secret"
C:\tmp\itco-attendance-venv\Scripts\python.exe dashboard.py
```

Open `http://127.0.0.1:5000` in a browser. The dashboard provides:

- Current attendance and daily-hour metrics
- Browser-based live camera detection with Start/Stop controls
- Full check-in/check-out session history
- Employee required-hours, workday, and active-status editing
- Authenticated review of private flagged-face images
- Labeling a flagged face as an employee to add another recognition reference
- Permanent deletion of unwanted flagged images and their Firestore metadata
- Daily, weekly, and monthly report generation
- CSV report downloads and Firestore report history

`ADMIN_PASSWORD` is mandatory; the server will not accept a login when it is
missing. Keep all three values out of Git. The built-in Flask server is suitable
for local development. A clinic deployment should run behind HTTPS using a
production WSGI server and restrict access to the clinic network or VPN.

Open **Live camera** from the sidebar or use the **Open camera** button on the
overview page. The browser stream runs the same detection, recognition,
attendance-session, and unknown-face pipeline as `main.py`. Normally only one
process can own the laptop camera, so do not run `main.py` and the dashboard
camera simultaneously.

When an admin labels a flagged face, the system extracts an InsightFace
embedding and stores it as one of up to ten recognition references for that
employee. It does not retrain the YOLO detector: YOLO only finds faces, while
the employee-specific references improve identity matching. Camera processes
reload these references from Firestore within one minute. Only label clear,
correct face crops; a wrong label directly reduces recognition accuracy.

## Firestore attendance sessions

Each `attendance_sessions` document contains:

```json
{
  "employee_id": "EMP001",
  "name": "Jane Doe",
  "check_in": "Firestore timestamp",
  "check_out": "Firestore timestamp or null",
  "last_seen": "Firestore timestamp",
  "duration_seconds": 28800,
  "status": "open or closed",
  "recognition_confidence": 0.71,
  "camera_source": "0"
}
```

Open sessions remain open across camera or dashboard restarts because stopping
the software does not mean the employee left the clinic. On the next distinct
scan, the existing open session is checked out normally.

## Reports

Generate reports for the current Dubai-local day, week, or month:

```powershell
C:\tmp\itco-attendance-venv\Scripts\python.exe report_generator.py daily
C:\tmp\itco-attendance-venv\Scripts\python.exe report_generator.py weekly
C:\tmp\itco-attendance-venv\Scripts\python.exe report_generator.py monthly
```

Generate a historical period using any date inside that period:

```powershell
C:\tmp\itco-attendance-venv\Scripts\python.exe report_generator.py weekly --date 2026-08-17
```

Each report creates a formatted Excel workbook and a clean CSV in `reports/`,
then saves a snapshot to the `attendance_reports` Firestore collection. The
Excel workbook contains an **Attendance Summary** sheet with color-coded status
and readable hour durations, plus an **Entry Exit Details** sheet listing every
visit and its exact entry, exit, duration, and recognition confidence. Employees
with no attendance are included, multiple visits are summed, and sessions
crossing midnight are divided between the correct local days. Use
`--local-only` to skip the Firestore report snapshot.

## CCTV configuration

The laptop camera is `CAMERA_SOURCE = 0` in `config.py`. For an IP camera, set
an RTSP or HTTP URL, preferably through an environment variable or a secure
deployment configuration. Example source value:

```text
rtsp://camera-host:554/stream
```

Do not put camera passwords in Git.

For the laptop demonstration, each distinct face presentation alternates the
employee between entered and exited states. In production, place the checkpoint
camera at the clinic entrance and require employees to face it in both
directions. Fully automatic entry/exit without deliberate scans requires a
calibrated doorway line and direction tracking.

## Main configuration

- `CLINIC_TIMEZONE`: defaults to `Asia/Dubai`.
- `DEFAULT_REQUIRED_DAILY_HOURS`: defaults to `8`.
- `DEFAULT_WORKING_WEEKDAYS`: defaults to `0,1,2,3,4`.
- `ENTRY_EXIT_COOLDOWN_SECONDS`: ignores accidental repeated scans, defaults to `30`.
- `RECOGNITION_SIM_THRESHOLD`: recognition cutoff, defaults to `0.45`.
- `FRAME_SKIP`: detection frequency/performance tradeoff.
- `CAMERA_SOURCE`: laptop camera index or CCTV stream URL.

## Tests

```powershell
C:\tmp\itco-attendance-venv\Scripts\python.exe -m unittest discover -s tests -v
```
