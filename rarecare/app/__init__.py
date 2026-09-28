"""Flask application factory."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from rarecare.app.security import SECURITY_HEADERS, RateLimiter, TTLSessionStore
from rarecare.config import Config, load_config
from rarecare.kg.graph import DATA_DIR
from rarecare.pipeline import RareCarePipeline, Session
from rarecare.vision.preprocess import InvalidImageError

log = logging.getLogger("rarecare")


def create_app(config: Config | None = None, pipeline: RareCarePipeline | None = None) -> Flask:
    cfg = config or load_config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.app.max_upload_mb * 1024 * 1024
    app.json.sort_keys = False  # type: ignore[attr-defined]

    pipe = pipeline or RareCarePipeline(cfg)
    sessions: TTLSessionStore[Session] = TTLSessionStore(cfg.app.session_ttl_seconds, cfg.app.max_sessions)
    limiter = RateLimiter(cfg.app.rate_limit_per_minute)
    age_bands = set(map(str, pipe.kg.variables["age_band"].values))
    questionnaire_data = json.loads((DATA_DIR / "questionnaire.json").read_text())

    def error(status: int, code: str, message: str) -> tuple[Any, int]:
        return jsonify({"error": {"code": code, "message": message, "request_id": g.get("request_id")}}), status

    @app.before_request
    def _before() -> Any:
        g.request_id = uuid.uuid4().hex[:12]
        if request.path.startswith("/api/") and request.method == "POST":
            client = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
            if not limiter.allow(client):
                return error(429, "rate_limited", "Too many requests. Please wait a minute.")
        return None

    @app.after_request
    def _after(resp: Any) -> Any:
        resp.headers.update(SECURITY_HEADERS)
        resp.headers["X-Request-ID"] = g.get("request_id", "")
        # Never log request bodies: they may contain sensitive health information.
        log.info("%s %s %s rid=%s", request.method, request.path, resp.status_code, g.get("request_id"))
        return resp

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException) -> Any:
        return error(exc.code or 500, (exc.name or "error").lower().replace(" ", "_"), exc.description or "")

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception) -> Any:
        log.exception("unhandled error rid=%s", g.get("request_id"))
        return error(500, "internal_error", "Something went wrong on our side. Please try again.")

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/v1/questionnaire")
    def questionnaire() -> Any:
        q = dict(questionnaire_data)
        q["age_bands"] = list(pipe.kg.variables["age_band"].values)
        q.pop("_meta", None)
        return jsonify(q)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok", "components": {k: v.value for k, v in pipe.modes.items()}})

    @app.get("/model-card")
    def model_card() -> Any:
        return jsonify(
            {
                "name": "RareCare",
                "status": "research prototype, ongoing; neural models under training",
                "intended_use": "Screening triage research for breast, cervical and ovarian cancer symptoms.",
                "not_intended_for": "Diagnosis or clinical decision-making.",
                "components": {k: v.value for k, v in pipe.modes.items()},
                "conformal_calibrated": bool(pipe.conformal),
                "guideline_encoding": pipe.kg.meta["description"],
            }
        )

    @app.post("/api/v1/assess")
    def assess() -> Any:
        if request.mimetype == "multipart/form-data":
            text = request.form.get("text", "")
            age_band = request.form.get("age_band") or None
            duration = request.form.get("duration") or None
            symptoms = request.form.getlist("symptoms")
            upload = request.files.get("image")
            image_bytes = upload.read() if upload and upload.filename else None
        else:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                return error(400, "bad_request", "Send JSON or multipart form data.")
            text = str(body.get("text", ""))
            age_band = body.get("age_band") or None
            duration = body.get("duration") or None
            raw_symptoms = body.get("symptoms", [])
            if not isinstance(raw_symptoms, list):
                return error(422, "invalid_symptoms", "symptoms must be a list of ids.")
            symptoms = [str(x) for x in raw_symptoms]
            image_bytes = None

        text = text.strip()
        if not text and not symptoms:
            return error(422, "missing_input", "Please choose at least one symptom or describe it in words.")
        if len(symptoms) > 30:
            return error(422, "invalid_symptoms", "Too many symptoms selected.")
        if len(text) > cfg.app.max_text_chars:
            return error(422, "text_too_long", f"Please keep it under {cfg.app.max_text_chars} characters.")
        if age_band is not None and age_band not in age_bands:
            return error(422, "invalid_age_band", f"age_band must be one of {sorted(age_bands)}.")

        try:
            session, result = pipe.start(text, age_band, image_bytes, symptoms, duration)
        except InvalidImageError as exc:
            return error(422, "invalid_image", str(exc))
        except ValueError as exc:
            return error(422, "invalid_input", str(exc))
        sid = sessions.create(session) if result.question else None
        return jsonify({"session_id": sid, "result": result.model_dump(mode="json")})

    @app.post("/api/v1/session/<sid>/answer")
    def answer(sid: str) -> Any:
        session = sessions.get(sid)
        if session is None:
            return error(404, "session_expired", "This session has expired. Please start again.")
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or "variable" not in body or "value" not in body:
            return error(400, "bad_request", "Expected JSON with 'variable' and 'value'.")
        try:
            result = pipe.answer(session, str(body["variable"]), str(body["value"]))
        except ValueError as exc:
            return error(422, "invalid_answer", str(exc))
        if result.question is None:
            sessions.delete(sid)
            sid_out = None
        else:
            sid_out = sid
        return jsonify({"session_id": sid_out, "result": result.model_dump(mode="json")})

    @app.delete("/api/v1/session/<sid>")
    def forget(sid: str) -> Any:
        sessions.delete(sid)
        return ("", 204)

    return app
