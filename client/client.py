"""
Flask application exposing the local iPad/WebView backend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path BEFORE any imports
# This allows running client/client.py directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import atexit
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

# Now we can import lib modules (project_root is in sys.path)
from lib.mask_archive import build_mask_archive, iso_now
from lib.config import ClientConfig, load_config

# Support running as a module (`python -m client.client`) or as a script
if __package__:
    from .frame_extractor import FrameExtractionError, FrameExtractor
    from .mdai_client import DatasetNotReady, MDaiDatasetManager
else:  # pragma: no cover - fallback for direct script execution
    # Add project root to import path, then import without the package prefix
    # (already done above, but keeping for clarity)
    from frame_extractor import FrameExtractionError, FrameExtractor  # type: ignore
    from mdai_client import DatasetNotReady, MDaiDatasetManager  # type: ignore

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
        """
        Blockingly:
        - Download/refresh the MD.ai dataset
        - Discover all local series
        - Extract frames for each series

        This is intentionally blocking so callers know that, on success, both
        dataset and frames are ready for use.
        """
        images_dir = context.dataset.sync_dataset()

        extracted = []
        series = context.dataset.list_local_series()
        for info in series:
            try:
                context.frames.ensure_frames(
                    info.video_path, info.study_uid, info.series_uid
                )
                extracted.append(f"{info.study_uid}/{info.series_uid}")
            except Exception as exc:  # pragma: no cover - surfaced to caller
                return (
                    jsonify({
                        "error": f"Failed to extract frames for {info.study_uid}/{info.series_uid}: {exc}",
                        "images_dir": str(images_dir),
                    }),
                    500,
                )

        return jsonify({
            "images_dir": str(images_dir),
            "series_count": len(series),
            "frames_extracted_for": extracted,
        })

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

    @app.get("/api/dataset/version_status")
    def dataset_version_status():
        """
        Compare dataset version between client and server.

        Uses the MD.ai annotations export mtime on each side as a simple version
        key. Intended only for warning UI; no automatic negotiation yet.
        """
        from track import find_annotations_file

        client_version = None
        server_version = None

        # Client annotations version
        try:
            client_annotations = Path(
                find_annotations_file(
                    str(config.data_dir),
                    config.project_id,
                    config.dataset_id,
                )
            )
            cstat = client_annotations.stat()
            client_version = {
                "annotations_path": str(client_annotations),
                "annotations_mtime_ns": cstat.st_mtime_ns,
                "annotations_size": cstat.st_size,
            }
        except FileNotFoundError:
            client_version = None

        # Server annotations version via API
        try:
            resp = context.http_client.get(
                f"{config.server_url.rstrip('/')}/api/dataset/version",
                timeout=10,
            )
            if resp.status_code == 200:
                server_version = resp.json()
            else:
                server_version = {"error": f"HTTP {resp.status_code}"}
        except Exception as exc:  # pragma: no cover - network failures
            server_version = {"error": str(exc)}

        in_sync = False
        if (
            client_version
            and isinstance(server_version, dict)
            and "annotations_mtime_ns" in server_version
        ):
            in_sync = (
                client_version["annotations_mtime_ns"]
                == server_version["annotations_mtime_ns"]
            )

        return jsonify({
            "client": client_version,
            "server": server_version,
            "in_sync": in_sync,
        })

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
        Builds a .tar mask archive using lib.mask_archive and POSTs it to the
        central server /api/masks/{study}/{series}.
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        method = data.get("method")
        study_uid = data.get("study_uid")
        series_uid = data.get("series_uid")
        modified_frames = data.get("modified_frames") or {}
        previous_version_id = data.get("previous_version_id") or ""

        if not study_uid or not series_uid or not method:
            return jsonify({"error": "method, study_uid and series_uid required"}), 400

        if not modified_frames:
            return jsonify({"error": "No modified frames provided"}), 400

        # Build temporary mask directory and metadata.frames
        from tempfile import TemporaryDirectory

        label_id_fluid = config.label_id
        label_id_empty = config.empty_id

        with TemporaryDirectory() as tmp:
            mask_dir = Path(tmp) / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

            frames_meta = []

            for frame_str, frame_data in modified_frames.items():
                try:
                    frame_num = int(frame_str)
                except (TypeError, ValueError):
                    continue

                is_empty = bool(frame_data.get("is_empty"))
                has_mask = not is_empty
                label_id = label_id_empty if is_empty else label_id_fluid
                filename = None

                if has_mask:
                    filename = f"frame_{frame_num:06d}_mask.webp"
                    mask_path = mask_dir / filename
                    mask_b64 = frame_data.get("mask_data") or ""
                    
                    # Use shared helper for base64→WebP conversion
                    from lib.mask_utils import decode_base64_mask_to_webp
                    if not decode_base64_mask_to_webp(mask_b64, mask_path, quality=85):
                        continue

                frames_meta.append(
                    {
                        "frame_number": frame_num,
                        "has_mask": has_mask,
                        "is_annotation": True,
                        "label_id": label_id,
                        "filename": filename,
                    }
                )

            if not frames_meta:
                return jsonify({"error": "No valid masks generated from modified_frames"}), 400

            # Use shared helper for ISO timestamp
            from lib.mask_archive import iso_now
            
            metadata = {
                "study_uid": study_uid,
                "series_uid": series_uid,
                "previous_version_id": previous_version_id or None,
                "flow_method": config.flow_method,
                "generated_at": iso_now(),
                "frame_count": len(frames_meta),
                "mask_count": sum(1 for f in frames_meta if f.get("has_mask")),
                "frames": frames_meta,
            }

            # Build .tar archive (no gzip) using shared helper
            archive_bytes = build_mask_archive(mask_dir, metadata)

        # POST archive to central server /api/masks/{study}/{series}
        server_url = config.server_url.rstrip("/")
        url = f"{server_url}/api/masks/{study_uid}/{series_uid}"

        headers = {
            "Content-Type": "application/x-tar",
            "X-Previous-Version-ID": previous_version_id or "",
        }
        # Identification for last editor
        if config.user_email:
            headers["X-Editor"] = config.user_email

        try:
            resp = context.http_client.post(
                url,
                content=archive_bytes,
                headers=headers,
                timeout=60,
            )
        except Exception as exc:
            return jsonify({"error": f"Error contacting server: {exc}"}), 502

        # Pass through server response JSON and status
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return jsonify(resp.json()), resp.status_code
        return Response(resp.content, status=resp.status_code)

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
            if key.lower() not in {"host", "content-length", "connection", "authorization"}
        }

        # Add X-User-Email from config if available and not already set
        if (
            config.user_email
            and "X-User-Email" not in headers
            and "X-Editor" not in headers
        ):
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
