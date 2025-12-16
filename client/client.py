"""
Flask application exposing the local iPad/WebView backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import atexit
import io
import json
import tarfile

import httpx
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    render_template,
    send_file,
)

# Support running as a module (`python -m client.client`) or as a script
if __package__:
    from .frame_extractor import FrameExtractionError, FrameExtractor
    from .mdai_client import DatasetNotReady, MDaiDatasetManager
else:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    # Add project root to import path, then import without the package prefix
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from frame_extractor import FrameExtractionError, FrameExtractor  # type: ignore
    from mdai_client import DatasetNotReady, MDaiDatasetManager  # type: ignore
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

    def _build_videos() -> list[dict]:
        """
        Build the list of locally available series for the viewer.
        """
        try:
            series = context.dataset.list_local_series()
        except DatasetNotReady:
            return []

        labels = [
            {"labelId": config.label_id, "labelName": "Fluid"},
            {"labelId": config.empty_id, "labelName": "Empty"},
        ]

        return [
            {
                "method": config.flow_method,
                "study_uid": info.study_uid,
                "series_uid": info.series_uid,
                "exam_number": info.exam_number,
                "series_number": info.series_number,
                "labels": labels,
            }
            for info in series
        ]

    def _find_series_info(study_uid: str, series_uid: str):
        for item in _build_videos():
            if item["study_uid"] == study_uid and item["series_uid"] == series_uid:
                return item
        return None

    # --------------------------------------------------------------------- Routes
    @app.route("/healthz")
    def healthcheck():
        return jsonify({
            "client_ready": context.dataset.dataset_ready(),
            "server_url": config.server_url,
            "video_cache": str(config.video_cache_path),
            "frames_cache": str(config.frames_path),
        })

    @app.route("/")
    def viewer_home():
        """
        Serve the main viewer UI using the legacy app.py frontend.
        """
        videos = _build_videos()
        if not videos:
            return render_template("no_videos.html"), 503

        selected_video = videos[0]
        return render_template(
            "viewer.html", videos=videos, selected_video=selected_video
        )

    @app.get("/api/videos")
    def api_videos():
        """
        Provide the list of available series for the front-end viewer.
        """
        videos = _build_videos()
        if not videos:
            return jsonify({"error": "Dataset not ready"}), 503
        return jsonify(videos)

    @app.get("/api/video/<method>/<study_uid>/<series_uid>")
    def api_video(method: str, study_uid: str, series_uid: str):
        """
        Return minimal metadata for a given series to drive the viewer.
        """
        info = _find_series_info(study_uid, series_uid)
        if not info:
            return jsonify({"error": "Series not found locally"}), 404

        try:
            video_path = context.dataset.resolve_video(study_uid, series_uid)
            context.frames.ensure_frames(video_path, study_uid, series_uid)
        except (DatasetNotReady, FileNotFoundError, FrameExtractionError) as exc:
            return jsonify({"error": str(exc)}), 404

        manifest_path = context.frames.manifest_path(study_uid, series_uid)
        frame_count = 0
        if manifest_path.exists():
            with manifest_path.open() as f:
                manifest = json.load(f)
                frame_count = manifest.get("frame_count", 0)
        if frame_count == 0:
            frame_dir = context.frames.frame_dir(study_uid, series_uid)
            frame_count = len(list(frame_dir.glob("frame_*.webp")))

        return jsonify({
            "total_frames": frame_count,
            "method": config.flow_method,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "exam_number": info.get("exam_number"),
            "labels": info.get("labels", []),
            "masks_annotations": [],
            "modified_frames": {},
        })

    @app.get("/api/frames/<method>/<study_uid>/<series_uid>")
    def api_frames(method: str, study_uid: str, series_uid: str):
        """
        Return URLs for frame and mask archives used by the legacy viewer JS.
        Frames are packaged on demand into a tar.gz with WebP frames.
        """
        info = _find_series_info(study_uid, series_uid)
        if not info:
            return jsonify({"error": "Series not found locally"}), 404

        try:
            video_path = context.dataset.resolve_video(study_uid, series_uid)
            context.frames.ensure_frames(video_path, study_uid, series_uid)
        except (DatasetNotReady, FileNotFoundError, FrameExtractionError) as exc:
            return jsonify({"error": str(exc)}), 404

        frames_archive_url = f"/api/frames_archive/{study_uid}/{series_uid}.tar"
        masks_archive_url = f"/proxy/api/masks/{study_uid}/{series_uid}"
        return jsonify({
            "frames_archive_url": frames_archive_url,
            "masks_archive_url": masks_archive_url,
        })

    @app.get("/api/frames_archive/<study_uid>/<series_uid>.tar")
    def frames_archive(study_uid: str, series_uid: str):
        """
        Serve cached .tar archive of WebP frames (no gzip - WebP already compressed).
        Archive is prebuilt during frame extraction.
        """
        try:
            video_path = context.dataset.resolve_video(study_uid, series_uid)
            context.frames.ensure_frames(video_path, study_uid, series_uid)
        except (DatasetNotReady, FileNotFoundError, FrameExtractionError):
            abort(404)

        tar_path = context.frames.frames_tar_path(study_uid, series_uid)
        if not tar_path.exists():
            abort(404)

        return send_file(
            tar_path,
            mimetype="application/x-tar",
            as_attachment=True,
            download_name=f"{study_uid}_{series_uid}_frames.tar",
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
                "frames_dir": str(
                    context.frames.frame_dir(info.study_uid, info.series_uid)
                ),
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
            frame_dir = context.frames.ensure_frames(
                video_path, study_uid, series_uid
            )
        except FrameExtractionError as exc:
            return jsonify({"error": str(exc)}), 500

        manifest_path = context.frames.manifest_path(study_uid, series_uid)
        manifest = {}
        if manifest_path.exists():
            import json

            with manifest_path.open() as f:
                manifest = json.load(f)

        return jsonify({"frames_dir": str(frame_dir), "manifest": manifest})

    @app.get("/frames/<study_uid>/<series_uid>/<int:frame_index>.webp")
    def serve_frame(study_uid: str, series_uid: str, frame_index: int):
        try:
            video_path = context.dataset.resolve_video(study_uid, series_uid)
            context.frames.ensure_frames(video_path, study_uid, series_uid)
        except (DatasetNotReady, FileNotFoundError, FrameExtractionError):
            abort(404)

        frame_path = (
            context.frames.frame_dir(study_uid, series_uid)
            / f"frame_{frame_index:06d}.webp"
        )
        if not frame_path.exists():
            abort(404)
        return send_file(frame_path, mimetype="image/webp")

    @app.post("/api/save_changes")
    def save_changes():
        """
        Adapter route: convert legacy viewer save format to distributed API format.
        Proxies to POST /api/masks/{study}/{series} with proper .tgz archive.
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        study_uid = data.get("study_uid")
        series_uid = data.get("series_uid")
        if not study_uid or not series_uid:
            return jsonify({"error": "study_uid and series_uid required"}), 400

        # For now, proxy directly - viewer.js will need to be updated to send
        # .tgz format. This is a placeholder that will fail gracefully.
        # TODO: Convert viewer.js JSON format to .tgz archive format
        return jsonify({
            "error": "save_changes endpoint needs conversion to .tgz format"
        }), 501

    @app.post("/api/retrack/<study_uid>/<series_uid>")
    def retrack_series(study_uid: str, series_uid: str):
        """
        Adapter route: legacy retrack endpoint.
        In distributed architecture, retrack is triggered by POST /api/masks.
        This route is kept for compatibility but should redirect to save flow.
        """
        return jsonify({
            "error": "Retrack is triggered by saving masks. Use /api/save_changes instead."
        }), 400

    @app.get("/api/retrack/status/<task_id>")
    def retrack_status_legacy(task_id: str):
        """
        Adapter route: legacy retrack status by task_id.
        Distributed API uses study/series instead of task_id.
        This is a placeholder - viewer.js needs to be updated.
        """
        return jsonify({
            "error": "Retrack status uses study/series, not task_id. Update viewer.js."
        }), 400

    @app.route(
        "/proxy/<path:subpath>",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )
    def proxy_to_server(subpath: str):
        """
        Forward requests to the centralized tracking server with MD.ai auth header.
        Preserves custom headers like X-Previous-Version-ID and X-Editor.
        """
        url = f"{config.server_url.rstrip('/')}/{subpath}"
        headers = {
            key: value
            for key, value in request.headers
            if key.lower() not in {"host", "content-length", "connection"}
        }
        headers["Authorization"] = f"Bearer {config.mdai_token}"
        
        # Add X-User-Email from config if available and not already set
        if config.user_email and "X-User-Email" not in headers and "X-Editor" not in headers:
            headers["X-User-Email"] = config.user_email

        files = None
        if request.files:
            files = {
                name: (file.filename, file.stream, file.mimetype)
                for name, file in request.files.items()
            }

        # Handle binary data (e.g., .tgz archives) - don't convert to JSON
        data = None
        json_payload = None
        if request.is_json and not request.data:
            # Only use JSON if it's actually JSON and no binary data
            json_payload = request.get_json(silent=True)
        else:
            # Binary data (e.g., .tgz archives for POST /api/masks)
            data = request.get_data()

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
        return Response(
            resp.content, status=resp.status_code, headers=response_headers
        )

    # Close resources when the process exits (avoid closing per-request)
    atexit.register(context.close)

    return app


def main() -> None:
    config = load_config("client")
    app = create_app(config)
    app.run(host="0.0.0.0", port=config.client_port)


if __name__ == "__main__":
    main()
