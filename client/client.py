"""
Flask application exposing the local iPad/WebView backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_file,
)

from client.frame_extractor import FrameExtractionError, FrameExtractor
from client.mdai_client import DatasetNotReady, MDaiDatasetManager
from lib.config import ClientConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"


class ClientContext:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.dataset = MDaiDatasetManager(config)
        self.frames = FrameExtractor(config.frames_path)
        self.http_client = httpx.Client(timeout=60)

    def close(self) -> None:
        self.http_client.close()


def create_app(config: Optional[ClientConfig] = None) -> Flask:
    config = config or load_config("client")
    context = ClientContext(config)

    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )
    app.config["CLIENT_CONTEXT"] = context

    # --------------------------------------------------------------------- Routes
    @app.route("/healthz")
    def healthcheck():
        return jsonify(
            {
                "client_ready": context.dataset.dataset_ready(),
                "server_url": config.server_url,
                "video_cache": str(config.video_cache_path),
                "frames_cache": str(config.frames_path),
            }
        )

    @app.route("/")
    def home():
        """
        Placeholder home route until the WebView is wired to new APIs.
        """
        return jsonify(
            {
                "message": "Goobusters client backend is running.",
                "instructions": "Hit /api/dataset/sync to download MD.ai data, then "
                "/api/local/series to enumerate studies.",
                "server_url": config.server_url,
            }
        )

    @app.post("/api/dataset/sync")
    def sync_dataset():
        images_dir = context.dataset.sync_dataset()
        return jsonify({"images_dir": str(images_dir)})

    @app.get("/api/local/series")
    def local_series():
        try:
            series = context.dataset.list_local_series()
        except DatasetNotReady as exc:
            return jsonify({"error": str(exc)}), 503

        payload = [
            {
                "study_uid": info.study_uid,
                "series_uid": info.series_uid,
                "exam_number": info.exam_number,
                "series_number": info.series_number,
                "dataset_name": info.dataset_name,
                "video_path": str(info.video_path),
                "frames_dir": str(context.frames.frame_dir(info.study_uid, info.series_uid)),
            }
            for info in series
        ]
        return jsonify(payload)

    @app.post("/api/local/frames/<study_uid>/<series_uid>")
    def ensure_frames(study_uid: str, series_uid: str):
        try:
            video_path = context.dataset.resolve_video(study_uid, series_uid)
        except (DatasetNotReady, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 404

        try:
            frame_dir = context.frames.ensure_frames(video_path, study_uid, series_uid)
        except FrameExtractionError as exc:
            return jsonify({"error": str(exc)}), 500

        manifest_path = context.frames.manifest_path(study_uid, series_uid)
        manifest = {}
        if manifest_path.exists():
            import json

            with manifest_path.open() as f:
                manifest = json.load(f)

        return jsonify({"frames_dir": str(frame_dir), "manifest": manifest})

    @app.get("/frames/<study_uid>/<series_uid>/<int:frame_index>.png")
    def serve_frame(study_uid: str, series_uid: str, frame_index: int):
        try:
            video_path = context.dataset.resolve_video(study_uid, series_uid)
            context.frames.ensure_frames(video_path, study_uid, series_uid)
        except (DatasetNotReady, FileNotFoundError, FrameExtractionError):
            abort(404)

        frame_path = context.frames.frame_dir(study_uid, series_uid) / f"frame_{frame_index:06d}.png"
        if not frame_path.exists():
            abort(404)
        return send_file(frame_path, mimetype="image/png")

    @app.route("/proxy/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def proxy_to_server(subpath: str):
        """
        Forward requests to the centralized tracking server with MD.ai auth header.
        """
        url = f"{config.server_url.rstrip('/')}/{subpath}"
        headers = {
            key: value
            for key, value in request.headers
            if key.lower() not in {"host", "content-length", "connection"}
        }
        headers["Authorization"] = f"Bearer {config.mdai_token}"

        files = None
        if request.files:
            files = {
                name: (file.filename, file.stream, file.mimetype)
                for name, file in request.files.items()
            }

        data = request.get_data()
        json_payload = None
        if request.is_json:
            json_payload = request.get_json(silent=True)
            data = None

        resp = context.http_client.request(
            request.method,
            url,
            params=request.args,
            data=data,
            json=json_payload,
            files=files,
            headers=headers,
            timeout=60,
        )

        excluded_headers = {
            "content-length",
            "content-encoding",
            "transfer-encoding",
            "connection",
        }
        response_headers = {
            key: value
            for key, value in resp.headers.items()
            if key.lower() not in excluded_headers
        }
        return Response(resp.content, status=resp.status_code, headers=response_headers)

    @app.teardown_appcontext
    def close_context(_exc):
        context.close()

    return app


def main() -> None:
    config = load_config("client")
    app = create_app(config)
    app.run(host="0.0.0.0", port=config.client_port)


if __name__ == "__main__":
    main()
