import datetime
import hmac
import io
import os
import secrets
import threading
from functools import wraps
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
    jsonify,
)

import config
from camera_service import CameraService
from firebase_db import FirebaseDB
from face_recognizer import FaceRecognizer
from report_generator import build_report, generate


def create_app(db=None, testing=False):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("DASHBOARD_SECRET_KEY", secrets.token_hex(32)),
        TESTING=testing,
    )
    app.extensions["attendance_db"] = db or FirebaseDB()
    app.extensions["camera_service"] = CameraService(
        app.extensions["attendance_db"]
    )
    recognizer_holder = {}
    recognizer_lock = threading.Lock()

    def get_db():
        return app.extensions["attendance_db"]

    def get_camera():
        return app.extensions["camera_service"]

    def embedding_from_storage(storage_path):
        image_bytes = get_db().bucket.blob(storage_path).download_as_bytes()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            return None
        with recognizer_lock:
            recognizer = recognizer_holder.get("instance")
            if recognizer is None:
                recognizer = FaceRecognizer(
                    [], [], np.zeros((0, 512), dtype=np.float32)
                )
                recognizer_holder["instance"] = recognizer
            return recognizer.get_embedding(image)

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("admin_authenticated"):
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)

        return wrapped

    def csrf_token():
        token = session.get("csrf_token")
        if token is None:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    def require_csrf():
        expected = session.get("csrf_token", "")
        supplied = request.form.get("csrf_token", "")
        if not expected or not hmac.compare_digest(expected, supplied):
            abort(400, "Invalid CSRF token")

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.context_processor
    def shared_template_values():
        return {
            "now_date": datetime.datetime.now(
                ZoneInfo(config.CLINIC_TIMEZONE)
            ).date().isoformat()
        }

    @app.template_filter("datetime_local")
    def datetime_local(value):
        if value is None:
            return "—"
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(ZoneInfo(config.CLINIC_TIMEZONE)).strftime(
            "%d %b %Y, %H:%M"
        )

    @app.template_filter("hours")
    def hours(value):
        return f"{float(value or 0) / 3600:.2f} h"

    @app.template_filter("session_hours")
    def session_hours(item):
        if item.get("status") == "open" and item.get("check_in"):
            check_in = item["check_in"]
            if check_in.tzinfo is None:
                check_in = check_in.replace(tzinfo=datetime.timezone.utc)
            seconds = (
                datetime.datetime.now(datetime.timezone.utc) - check_in
            ).total_seconds()
        else:
            seconds = item.get("duration_seconds", 0)
        return f"{max(0, float(seconds or 0)) / 3600:.2f} h"

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            require_csrf()
            configured_password = os.getenv("ADMIN_PASSWORD")
            if not configured_password:
                flash("ADMIN_PASSWORD is not configured on the server.", "error")
                return render_template("login.html"), 503
            username_ok = hmac.compare_digest(
                request.form.get("username", ""),
                os.getenv("ADMIN_USERNAME", "admin"),
            )
            password_ok = hmac.compare_digest(
                request.form.get("password", ""), configured_password
            )
            if username_ok and password_ok:
                session.clear()
                session["admin_authenticated"] = True
                session["csrf_token"] = secrets.token_urlsafe(32)
                target = request.args.get("next", "")
                safe_target = target.startswith("/") and not target.startswith("//")
                return redirect(target if safe_target else url_for("index"))
            flash("Incorrect username or password.", "error")
        return render_template("login.html")

    @app.post("/logout")
    @login_required
    def logout():
        require_csrf()
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    @login_required
    def index():
        db_client = get_db()
        users = db_client.load_users()
        sessions = db_client.load_attendance_sessions()
        today = datetime.datetime.now(ZoneInfo(config.CLINIC_TIMEZONE)).date()
        _, _, rows = build_report(
            users, sessions, "daily", today, config.CLINIC_TIMEZONE
        )
        open_sessions = [item for item in sessions if item.get("status") == "open"]
        metrics = {
            "employees": sum(item.get("active", True) for item in users),
            "present": len({item.get("employee_id") for item in open_sessions}),
            "achieved": sum(item["status"] == "ACHIEVED" for item in rows),
            "flagged": sum(
                not item.get("reviewed", False)
                for item in db_client.load_flagged_faces()
            ),
        }
        recent = sorted(
            sessions,
            key=lambda item: item.get("check_in")
            or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
            reverse=True,
        )[:8]
        return render_template(
            "index.html", metrics=metrics, rows=rows, recent=recent
        )

    @app.get("/sessions")
    @login_required
    def sessions_page():
        sessions = sorted(
            get_db().load_attendance_sessions(),
            key=lambda item: item.get("check_in")
            or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
            reverse=True,
        )
        return render_template("sessions.html", sessions=sessions)

    @app.get("/camera")
    @login_required
    def camera_page():
        return render_template("camera.html", camera=get_camera())

    @app.post("/camera/start")
    @login_required
    def start_camera():
        require_csrf()
        get_camera().start()
        flash("Camera detection is starting.", "success")
        return redirect(url_for("camera_page"))

    @app.post("/camera/stop")
    @login_required
    def stop_camera():
        require_csrf()
        get_camera().stop()
        flash("Camera detection stopped.", "success")
        return redirect(url_for("camera_page"))

    @app.get("/camera/feed")
    @login_required
    def camera_feed():
        return Response(
            get_camera().frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/camera/status")
    @login_required
    def camera_status():
        camera = get_camera()
        return jsonify({"status": camera.status, "error": camera.error})

    @app.get("/employees")
    @login_required
    def employees_page():
        return render_template("employees.html", users=get_db().load_users())

    @app.post("/employees/<user_id>")
    @login_required
    def update_employee(user_id):
        require_csrf()
        try:
            required_hours = float(request.form["required_daily_hours"])
            if not 0 <= required_hours <= 24:
                raise ValueError
            weekdays = sorted({
                int(value) for value in request.form.getlist("working_weekdays")
            })
            if any(day < 0 or day > 6 for day in weekdays):
                raise ValueError
        except (KeyError, ValueError):
            flash("Invalid hours or working-day selection.", "error")
            return redirect(url_for("employees_page"))
        get_db().update_user_policy(
            user_id, required_hours, weekdays, "active" in request.form
        )
        flash("Employee policy updated.", "success")
        return redirect(url_for("employees_page"))

    @app.get("/flagged")
    @login_required
    def flagged_page():
        faces = sorted(
            get_db().load_flagged_faces(),
            key=lambda item: item.get("timestamp")
            or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
            reverse=True,
        )
        return render_template(
            "flagged.html", faces=faces, users=get_db().load_users()
        )

    @app.get("/flagged/<flag_id>/image")
    @login_required
    def flagged_image(flag_id):
        face = get_db().get_flagged_face(flag_id)
        if face is None or not face.get("storage_path"):
            abort(404)
        blob = get_db().bucket.blob(face["storage_path"])
        if not blob.exists():
            abort(404)
        return send_file(
            io.BytesIO(blob.download_as_bytes()),
            mimetype="image/jpeg",
            max_age=0,
        )

    @app.post("/flagged/<flag_id>/review")
    @login_required
    def review_flagged(flag_id):
        require_csrf()
        get_db().mark_flagged_reviewed(flag_id, True)
        flash("Flagged face marked as reviewed.", "success")
        return redirect(url_for("flagged_page"))

    @app.post("/flagged/<flag_id>/label")
    @login_required
    def label_flagged(flag_id):
        require_csrf()
        employee_id = request.form.get("employee_id", "")
        valid_ids = {user["user_id"] for user in get_db().load_users()}
        if employee_id not in valid_ids:
            abort(400, "Invalid employee")
        face = get_db().get_flagged_face(flag_id)
        if face is None or not face.get("storage_path"):
            abort(404)
        try:
            embedding = embedding_from_storage(face["storage_path"])
        except Exception:
            app.logger.exception("Could not read flagged image %s", flag_id)
            flash(
                "The flagged image could not be read from Firebase Storage.",
                "error",
            )
            return redirect(url_for("flagged_page"))
        if embedding is None:
            flash(
                "That crop does not contain a usable face. Keep it for review "
                "or delete it.",
                "error",
            )
            return redirect(url_for("flagged_page"))
        try:
            added = get_db().add_employee_face_reference(
                employee_id, embedding, flag_id
            )
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("flagged_page"))
        except Exception:
            app.logger.exception("Could not label flagged image %s", flag_id)
            flash(
                "Firebase could not save this face reference. Please try again.",
                "error",
            )
            return redirect(url_for("flagged_page"))
        message = (
            "Face reference added. Camera recognition refreshes within one minute."
            if added
            else "This face was already stored for that employee."
        )
        flash(message, "success")
        return redirect(url_for("flagged_page"))

    @app.post("/flagged/<flag_id>/delete")
    @login_required
    def delete_flagged(flag_id):
        require_csrf()
        if not get_db().delete_flagged_face(flag_id):
            abort(404)
        flash("Flagged image and metadata permanently deleted.", "success")
        return redirect(url_for("flagged_page"))

    @app.get("/reports")
    @login_required
    def reports_page():
        reports = sorted(
            get_db().load_reports(),
            key=lambda item: item.get("generated_at")
            or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
            reverse=True,
        )
        return render_template("reports.html", reports=reports)

    @app.post("/reports/generate")
    @login_required
    def generate_report():
        require_csrf()
        period = request.form.get("period")
        if period not in {"daily", "weekly", "monthly"}:
            abort(400, "Invalid report period")
        date_value = request.form.get("date")
        try:
            anchor = datetime.date.fromisoformat(date_value)
        except (TypeError, ValueError):
            abort(400, "Invalid report date")
        path, rows = generate(
            period, anchor, save_to_firestore=True, db=get_db()
        )
        flash(f"Created {path.name} for {len(rows)} employees.", "success")
        return redirect(url_for("reports_page"))

    @app.get("/reports/download/<path:filename>")
    @login_required
    def download_report(filename):
        safe_name = Path(filename).name
        path = (config.REPORTS_DIR / safe_name).resolve()
        reports_root = config.REPORTS_DIR.resolve()
        if path.parent != reports_root or not path.is_file():
            abort(404)
        return send_file(path, as_attachment=True)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
