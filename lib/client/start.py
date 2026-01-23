"""
Flask application exposing the local iPad/WebView backend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Add paths: project root (for lib imports) and lib/client (for client package imports)
import os

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_lib_client = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _lib_client not in sys.path:
    sys.path.insert(0, _lib_client)

import argparse
import atexit
import json
import logging
import os
import signal

from flask import Flask, Response, jsonify, request, render_template

from lib.mask_archive import build_mask_archive, iso_now
from lib.config import ClientConfig, load_config
from client.mdai_client import DatasetNotReady, MDaiDatasetManager

PROJECT_ROOT = Path(_project_root)
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"


class ClientContext:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.dataset = MDaiDatasetManager(config)
        self.user_email_override: Optional[str] = None

    def close(self) -> None:
        pass

    def current_user_email(self) -> str:
        return (
            self.user_email_override or self.config.user_email or ""
        ).strip()


def create_app(config: Optional[ClientConfig] = None) -> Flask:
    """
    Create the Flask client application.
    
    The frontend handles sync and credential checking:
    - Frontend checks credentials after connecting to backend
    - If credentials present -> frontend triggers sync via /api/dataset/sync
    - If credentials missing or sync fails -> frontend shows blocking config modal
    """
    config = config or load_config("client")
    context = ClientContext(config)

    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )
    app.config["CLIENT_CONTEXT"] = context

    # Initialize client: don't sync on startup - frontend will handle it
    logger = logging.getLogger(__name__)
    logger.info("Flask app created. Frontend will handle sync if credentials are present.")

    def _current_user_email() -> str:
        return context.current_user_email()

    def _build_videos() -> list[dict]:
        """
        Build the list of locally available series for the viewer.
        Fetches status and activity from server for each series.
        """
        try:
            series = context.dataset.list_local_series()
        except DatasetNotReady:
            return []

        labels = [
            {"labelId": config.label_id, "labelName": "Fluid"},
            {"labelId": config.empty_id, "labelName": "Empty"},
        ]

        # Fetch server series data for status/activity (activity only lives on server)
        # This is non-blocking - if it fails, we continue without activity data
        server_series_map = {}
        try:
            import requests
            server_url = f"{config.server_url}/api/series"
            resp = requests.get(server_url, timeout=5)
            if resp.ok:
                server_data = resp.json()
                activity_count = 0
                for item in server_data:
                    key = f"{item['study_uid']}|{item['series_uid']}"
                    activity = item.get("activity", {})
                    if activity:
                        activity_count += 1
                    server_series_map[key] = {
                        "status": item.get("status", "pending"),
                        "activity": activity,
                    }
                logger.debug(f"Fetched {len(server_series_map)} series from server, {activity_count} with activity")
            else:
                logger.warning(f"Server returned {resp.status_code} when fetching activity data")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to {config.server_url} for activity data")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error to {config.server_url}: {e}")
        except Exception as e:
            logger.warning(f"Failed to fetch activity data: {type(e).__name__}: {e}")

        # Build video list with server data merged
        videos = []
        for info in series:
            key = f"{info.study_uid}|{info.series_uid}"
            server_data = server_series_map.get(key, {})
            videos.append({
                "method": config.flow_method,
                "study_uid": info.study_uid,
                "series_uid": info.series_uid,
                "exam_number": info.exam_number,
                "series_number": info.series_number,
                "labels": labels,
                "status": server_data.get("status", "pending"),
                "activity": server_data.get("activity", {}),
            })

        # Sort by exam_number
        videos.sort(key=lambda v: v["exam_number"] or 0)

        return videos

    def _find_series_info(study_uid: str, series_uid: str):
        """
        Find series info efficiently - uses cache if available, otherwise builds minimal info.
        """
        # Try cache first (fast path)
        videos = _build_videos()
        for item in videos:
            if (
                item["study_uid"] == study_uid
                and item["series_uid"] == series_uid
            ):
                return item

        # If not in cache, try to get minimal info without full list scan
        # This is a fallback for when cache is invalidated
        try:
            video_path = context.dataset.resolve_video(study_uid, series_uid)
            if not video_path.exists():
                return None

            # Get minimal info from dataset manager without full scan
            images_dir = context.dataset._ensure_images_dir()
            annotations_df = context.dataset._ensure_annotations()
            studies_lookup = context.dataset._studies_lookup or {}

            # Find this specific series in annotations
            series_annotations = annotations_df[
                (annotations_df["StudyInstanceUID"] == study_uid)
                & (annotations_df["SeriesInstanceUID"] == series_uid)
            ]
            if series_annotations.empty:
                return None

            study_info = studies_lookup.get(study_uid, {})
            labels = [
                {"labelId": config.label_id, "labelName": "Fluid"},
                {"labelId": config.empty_id, "labelName": "Empty"},
            ]

            return {
                "method": config.flow_method,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "exam_number": int(study_info.get("number"))
                if study_info.get("number") not in (None, "")
                else None,
                "series_number": int(
                    study_info.get("SeriesNumber")
                    or study_info.get("seriesNumber")
                    or 0
                )
                if study_info.get("SeriesNumber")
                or study_info.get("seriesNumber")
                else None,
                "labels": labels,
            }
        except (DatasetNotReady, FileNotFoundError, Exception):
            return None

    # --------------------------------------------------------------------- Routes
    @app.route("/healthz")
    def healthcheck():
        """
        Health check endpoint. Returns client_ready=True if Flask server is running.
        Data sync status is checked separately via /api/status.
        """
        return jsonify({
            "client_ready": True,  # Server is running and ready to accept requests
            "server_url": config.server_url,
            "video_cache": str(config.video_cache_path),
            "frames_cache": str(config.frames_path),
        })

    @app.route("/")
    def viewer_home():
        """
        Serve the main viewer UI. Frontend handles credential checking and sync.
        Always serve viewer.html - frontend will check credentials and sync if needed.
        """
        # Try to build videos list, but don't fail if no data exists yet
        # Frontend will handle credential check and sync
        try:
            videos = _build_videos()
            selected_video = videos[0] if videos else None
        except DatasetNotReady:
            videos = []
            selected_video = None

        return render_template(
            "viewer.html", 
            videos=videos or [], 
            selected_video=selected_video,
            server_url=config.server_url
        )

    @app.route("/no_videos")
    def no_videos():
        """
        Serve the setup page when credentials are missing.
        This is called by JavaScript when showConfigModal() is invoked.
        """
        return render_template("no_videos.html")

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
        Verify series exists locally. Frontend calls server directly for video metadata.
        """
        # Verify series exists locally (for dataset sync check)
        info = _find_series_info(study_uid, series_uid)
        if not info:
            return jsonify({"error": "Series not found locally"}), 404

        try:
            video_path = context.dataset.resolve_video(study_uid, series_uid)
        except (DatasetNotReady, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 404

        # Return server URL for frontend to call directly
        server_url = f"{config.server_url.rstrip('/')}/api/video/{method}/{study_uid}/{series_uid}"
        return jsonify({
            "server_url": server_url,
            "local": True
        })

    @app.get("/api/frames/<method>/<study_uid>/<series_uid>")
    def api_frames(method: str, study_uid: str, series_uid: str):
        """
        Return URLs for frame and mask archives used by the legacy viewer JS.

        Frames should already exist from client startup initialization.
        This endpoint only checks existence and returns URLs - it does NOT extract frames.
        """
        info = _find_series_info(study_uid, series_uid)
        if not info:
            return jsonify({"error": "Series not found locally"}), 404

        # Return direct server URLs (no proxy)
        frames_archive_url = (
            f"{config.server_url.rstrip('/')}/api/frames_archive/{study_uid}/{series_uid}.tar"
        )
        masks_archive_url = f"{config.server_url.rstrip('/')}/api/masks/{study_uid}/{series_uid}"
        return jsonify({
            "frames_archive_url": frames_archive_url,
            "masks_archive_url": masks_archive_url,
        })

    @app.get("/api/status")
    def api_status():
        """
        Check if dataset is ready.
        Used by setup screen to determine flow.
        """
        has_data = context.dataset.dataset_ready()
        series_count = 0
        if has_data:
            try:
                series_count = len(context.dataset.list_local_series())
            except DatasetNotReady:
                has_data = False
        
        return jsonify({
            "has_data": has_data,
            "series_count": series_count,
            "server_url": config.server_url,
        })

    @app.get("/api/debug/paths")
    def debug_paths():
        """Debug endpoint to check file paths."""
        import glob as glob_mod
        video_cache = str(context.dataset.video_cache_path)
        project_id = config.project_id
        dataset_id = config.dataset_id
        
        ann_pattern = f"{video_cache}/mdai_*_project_{project_id}_annotations_dataset_{dataset_id}_*.json"
        img_pattern = f"{video_cache}/mdai_*_project_{project_id}_images_dataset_{dataset_id}_*"
        
        ann_matches = glob_mod.glob(ann_pattern)
        img_matches = [d for d in glob_mod.glob(img_pattern) if Path(d).is_dir()]
        
        return jsonify({
            "video_cache": video_cache,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "ann_pattern": ann_pattern,
            "ann_matches": ann_matches,
            "img_pattern": img_pattern,
            "img_matches": img_matches,
        })

    @app.post("/api/dataset/sync")
    def sync_dataset():
        """
        Download/refresh the MD.ai dataset. Frames are produced by the server
        tracking pipeline; the client no longer extracts or caches frames.
        """
        try:
            images_dir = context.dataset.sync_dataset()
            series = context.dataset.list_local_series()
            return jsonify({
                "images_dir": str(images_dir),
                "series_count": len(series),
                "frames_extracted_for": [],  # no local extraction
            })
        except DatasetNotReady as e:
            return jsonify({"error": str(e), "error_message": str(e)}), 503
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Dataset sync failed: {e}", exc_info=True)
            return jsonify({"error": str(e), "error_message": str(e)}), 500

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
                "method": config.flow_method,
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
        from client.mdai_client import find_annotations_file

        client_version = None
        server_version = None

        # Client annotations version - look in video_cache where sync downloads
        try:
            client_annotations = Path(
                find_annotations_file(
                    str(context.dataset.video_cache_path),
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

        # Get server annotations version
        server_version = None
        try:
            import requests
            server_resp = requests.get(
                f"{config.server_url}/api/dataset/version",
                timeout=5
            )
            if server_resp.ok:
                server_version = server_resp.json()
        except Exception:
            pass  # Server unreachable, can't compare

        # Compare by file size (mtime differs across machines)
        in_sync = False
        if client_version and server_version and "annotations_size" in server_version:
            in_sync = client_version["annotations_size"] == server_version["annotations_size"]

        return jsonify({
            "client": client_version,
            "server": server_version,
            "in_sync": in_sync,
        })

    @app.get("/api/settings")
    def get_settings():
        """
        Return current user identity and whether an MD.ai token is set.
        Token contents are never returned.
        """
        token = context.dataset._token_override or context.config.mdai_token
        has_token = bool(token and token != "not_configured_yet")
        return jsonify({
            "user_email": _current_user_email(),
            "mdai_token_present": has_token,
        })

    @app.post("/api/settings")
    def update_settings():
        """
        Update user_email and/or MD.ai token for this client process.
        Values are kept in-memory for the session.
        """
        data = request.get_json(silent=True) or {}
        user_email = data.get("user_email")
        mdai_token = data.get("mdai_token")

        if user_email is not None:
            email_clean = user_email.strip()
            context.user_email_override = email_clean or None

        if mdai_token is not None:
            token_clean = mdai_token.strip()
            context.dataset.set_token(token_clean or None)

        return jsonify({
            "user_email": _current_user_email(),
            "mdai_token_present": bool(
                context.dataset._token_override or context.config.mdai_token
            ),
        })

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
            return jsonify({
                "error": "method, study_uid and series_uid required"
            }), 400

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

                    if not decode_base64_mask_to_webp(
                        mask_b64, mask_path, quality=85
                    ):
                        continue

                frames_meta.append({
                    "frame_number": frame_num,
                    "has_mask": has_mask,
                    "is_annotation": True,
                    "label_id": label_id,
                    "filename": filename,
                })

            if not frames_meta:
                return jsonify({
                    "error": "No valid masks generated from modified_frames"
                }), 400

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

        # Return archive bytes - frontend will POST to server directly
        return Response(
            archive_bytes,
            mimetype="application/x-tar",
            headers={
                "Content-Disposition": f"attachment; filename=masks_{study_uid}_{series_uid}.tar"
            }
        )

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


    # Close resources when the process exits (avoid closing per-request)
    atexit.register(context.close)

    return app


def get_pid_file(config: ClientConfig) -> Path:
    """Get path to PID file."""
    pid_dir = config.video_cache_path.parent / "state"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir / "client.pid"


def read_pid(pid_file: Path) -> Optional[int]:
    """Read PID from file if it exists and process is still running."""
    if not pid_file.exists():
        return None

    try:
        pid = int(pid_file.read_text().strip())
        # Check if process is still running
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
        return pid
    except (ValueError, OSError, ProcessLookupError):
        # PID file exists but process is dead - remove stale file
        pid_file.unlink(missing_ok=True)
        return None


def write_pid(pid_file: Path) -> None:
    """Write current process PID to file."""
    pid_file.write_text(str(os.getpid()))


def update_latest_log_symlink(log_file: Path) -> None:
    """Create or update symlink log/client pointing to the current log file."""
    log_dir = log_file.parent
    latest_link = log_dir / "client"

    # Remove existing symlink if it exists
    if latest_link.exists() or latest_link.is_symlink():
        try:
            latest_link.unlink()
        except OSError:
            pass  # Ignore errors removing old symlink

    # Create new symlink (relative path so it works if log_dir is moved)
    try:
        latest_link.symlink_to(log_file.name)
    except OSError as e:
        # Log but don't fail if symlink creation fails
        # Use print since logger might not be set up yet
        import sys

        print(
            f"Warning: Failed to create latest log symlink: {e}",
            file=sys.stderr,
        )


def kill_existing(pid_file: Path) -> bool:
    """Kill existing process if PID file exists and process is running. Returns True if killed."""
    pid = read_pid(pid_file)
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait a bit for graceful shutdown
        import time

        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                # Process is dead
                pid_file.unlink(missing_ok=True)
                return True
        # Still running, force kill
        os.kill(pid, signal.SIGKILL)
        pid_file.unlink(missing_ok=True)
        return True
    except (OSError, ProcessLookupError):
        # Process already dead
        pid_file.unlink(missing_ok=True)
        return False


def daemonize(log_file: Path, pid_file: Path) -> None:
    """Detach process and run in background."""
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - print log location and exit
            print(f"Log file: {log_file}")
            print(f"PID file: {pid_file}")
            os._exit(0)
    except OSError as e:
        sys.stderr.write(f"Fork failed: {e}\n")
        sys.exit(1)

    # Child process continues
    os.setsid()  # Create new session

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - exit
            os._exit(0)
    except OSError as e:
        sys.stderr.write(f"Second fork failed: {e}\n")
        sys.exit(1)

    # Redirect standard file descriptors to /dev/null
    os.chdir("/")
    os.umask(0)

    # Redirect stdin, stdout, stderr to /dev/null
    with open("/dev/null", "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    with open("/dev/null", "w") as devnull:
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())

    # Close file descriptors (except 0, 1, 2 which are now /dev/null)
    import resource

    maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if maxfd == resource.RLIM_INFINITY:
        maxfd = 1024

    for fd in range(3, maxfd):
        try:
            os.close(fd)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Goobusters client backend")
    parser.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="Run as daemon (detached process)",
    )
    parser.add_argument(
        "-k",
        "--kill",
        action="store_true",
        help="Kill existing client process before starting",
    )
    args = parser.parse_args()

    config = load_config("client")
    pid_file = get_pid_file(config)

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create log directory (repo root log/)
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    log_dir = Path(project_root) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log filename: client_YYMMDD-HHMMSS.log
    from datetime import datetime

    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    log_file = log_dir / f"client_{timestamp}.log"

    # Create log file immediately so symlink works
    log_file.touch(exist_ok=True)

    # Kill existing process if requested
    if args.kill:
        if kill_existing(pid_file):
            print("Killed existing client process")
        else:
            print("No existing client process found")

    # Check if client is already running
    existing_pid = read_pid(pid_file)
    if existing_pid is not None:
        print(f"Client is already running (PID: {existing_pid})")
        print(f"Use -k to kill it, or check PID file: {pid_file}")
        sys.exit(1)

    # Daemonize if requested (must be before logging setup)
    if args.daemon:
        daemonize(log_file, pid_file)

    # Create/update symlink to latest log (after daemonize so child process creates it)
    # This must happen after daemonize so the child process creates the symlink
    update_latest_log_symlink(log_file)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (if not daemonized)
    if not args.daemon:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)

    # Write PID file
    write_pid(pid_file)

    # Log startup
    logger.info("=" * 60)
    logger.info("Starting Goobusters Client")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"PID file: {pid_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    try:
        app = create_app(config)
        logger.info(f"Client ready on 0.0.0.0:{config.client_port}")
        app.run(host="0.0.0.0", port=config.client_port)
    finally:
        # Clean up PID file on exit
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
