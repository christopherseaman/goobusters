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

import httpx
from flask import Flask, Response, jsonify, request, render_template

# Import ios_config FIRST to set up stubs before importing mdai
# The stubbing code in ios_config.py runs at module import time
try:
    import ios_config  # This sets up stubs automatically
except ImportError:
    pass  # Not iOS, stubs not needed

import mdai
from lib.mask_archive import build_mask_archive, iso_now
from lib.config import ClientConfig, load_config


# Pandas-free json_to_dataframe implementation (replaces mdai.common_utils.json_to_dataframe)
def json_to_dataframe(
    json_file: str, datasets: list[str] | None = None
) -> dict:
    """
    Load and parse MD.ai annotations JSON file (pandas-free version).

    Returns same structure as mdai.common_utils.json_to_dataframe but with lists of dicts
    instead of DataFrames. Handles label merging and study merging like the SDK version.

    Args:
        json_file: Path to MD.ai annotations JSON file
        datasets: Optional list of dataset IDs to filter (default: all)

    Returns:
        Dictionary with keys:
            - annotations: List of annotation dicts (with merged label and study data)
            - studies: List of study dicts (filtered to StudyInstanceUID, dataset, datasetId, number)
            - labels: List of label dicts
    """
    if datasets is None:
        datasets = []

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_annotations = []
    all_studies = []

    # Process each dataset
    for dataset in data.get("datasets", []):
        dataset_id = dataset.get("id", "")
        dataset_name = dataset.get("name", "Unknown")

        # Filter by dataset IDs if specified
        if datasets and dataset_id not in datasets:
            continue

        # Add dataset context to studies
        for study in dataset.get("studies", []):
            study_copy = study.copy()
            study_copy["dataset"] = dataset_name
            study_copy["datasetId"] = dataset_id
            all_studies.append(study_copy)

        # Add dataset context to annotations
        for annot in dataset.get("annotations", []):
            annot_copy = annot.copy()
            annot_copy["dataset"] = dataset_name
            annot_copy["datasetId"] = dataset_id
            all_annotations.append(annot_copy)

    # Process labels from label groups (replicate SDK's label unpacking logic)
    labels_list = []
    for label_group in data.get("labelGroups", []):
        group_id = label_group.get("id")
        group_name = label_group.get("name")

        for label in label_group.get("labels", []):
            label_entry = {
                "labelGroupId": group_id,
                "labelGroupName": group_name,
                "annotationMode": label.get("annotationMode"),
                "color": label.get("color"),
                "description": label.get("description"),
                "labelId": label.get("id"),
                "labelName": label.get("name"),
                "radlexTagIdsLabel": label.get("radlexTagIds"),
                "scope": label.get("scope"),
            }

            # Add parentId if it exists
            if "parentId" in label:
                label_entry["parentLabelId"] = label.get("parentId")

            labels_list.append(label_entry)

    # Create label lookup by labelId for merging
    labels_lookup = {label["labelId"]: label for label in labels_list}

    # Merge labels into annotations (like SDK's a.merge(labels, on=["labelId"]))
    if labels_list and all_annotations:
        for annot in all_annotations:
            label_id = annot.get("labelId")
            if label_id and label_id in labels_lookup:
                # Merge label fields into annotation
                label_data = labels_lookup[label_id]
                for key, value in label_data.items():
                    annot[key] = value

    # Filter studies to only include needed fields (like SDK's studies[["StudyInstanceUID", "dataset", "datasetId", "number"]])
    filtered_studies = []
    for study in all_studies:
        filtered_studies.append({
            "StudyInstanceUID": study.get("StudyInstanceUID"),
            "dataset": study.get("dataset"),
            "datasetId": study.get("datasetId"),
            "number": study.get("number"),
        })

    # Create study lookup for merging
    studies_lookup = {}
    for study in filtered_studies:
        key = (study["StudyInstanceUID"], study["dataset"])
        studies_lookup[key] = study

    # Merge studies into annotations (like SDK's a.merge(studies, on=["StudyInstanceUID", "dataset"]))
    if filtered_studies and all_annotations:
        for annot in all_annotations:
            key = (annot.get("StudyInstanceUID"), annot.get("dataset"))
            if key in studies_lookup:
                study_data = studies_lookup[key]
                # Merge study fields into annotation
                annot["number"] = study_data.get("number")
                # Note: dataset and datasetId already in annotation from earlier

    # Convert number to int (like SDK's studies.number.astype(int) and a.number.astype(int))
    for study in filtered_studies:
        if "number" in study and study["number"] is not None:
            try:
                study["number"] = int(study["number"])
            except (ValueError, TypeError):
                pass

    for annot in all_annotations:
        if "number" in annot and annot["number"] is not None:
            try:
                annot["number"] = int(annot["number"])
            except (ValueError, TypeError):
                pass

    # Note: SDK converts createdAt/updatedAt to datetime, but we keep as strings
    # This is fine since we're using plain dicts, not DataFrames

    return {
        "annotations": all_annotations,
        "studies": filtered_studies,
        "labels": labels_list,
    }


# Utility functions for finding cached MD.ai data (moved from track.py for iOS compatibility)
def find_annotations_file(
    data_dir: str, project_id: str, dataset_id: str
) -> str:
    """Find the most recent annotations JSON file for the given project and dataset."""
    import glob

    pattern = os.path.join(
        data_dir,
        f"mdai_*_project_{project_id}_annotations_dataset_{dataset_id}_*.json",
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No annotations file found: {project_id}, dataset {dataset_id} in {data_dir}"
        )
    return sorted(matches)[-1]


def find_images_dir(data_dir: str, project_id: str, dataset_id: str) -> str:
    """Find the most recent images directory for the given project and dataset."""
    import glob

    pattern = os.path.join(
        data_dir, f"mdai_*_project_{project_id}_images_dataset_{dataset_id}_*"
    )
    matches = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not matches:
        raise FileNotFoundError(
            f"No images directory found: {project_id}, dataset {dataset_id} in {data_dir}"
        )
    return sorted(matches)[-1]


PROJECT_ROOT = Path(_project_root)
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"


class DatasetNotReady(RuntimeError):
    """Raised when the MD.ai dataset is not present locally."""


class TokenNotConfigured(RuntimeError):
    """Raised when MD.ai token is not configured or invalid."""


def is_valid_token(token: Optional[str]) -> bool:
    """Check if token is valid (not None, not empty, not placeholder)."""
    if not token:
        return False
    token_clean = token.strip()
    return token_clean and token_clean != "not_configured_yet"


class ClientContext:
    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self._mdai_client: Optional[mdai.Client] = None
        self._token_override: Optional[str] = None
        self._images_dir: Optional[Path] = None
        self._annotations_list: Optional[list[dict]] = (
            None  # List of annotation dicts
        )
        self._studies_lookup: Optional[dict[str, dict]] = None
        # Sync status tracking
        self._sync_status: str = (
            "idle"  # "idle", "syncing", "completed", "error"
        )
        self._sync_error: Optional[str] = None
        # Frames are produced by the server tracking pipeline; no local extraction/cache needed.
        # Disable all caching - clients should never cache anything
        # httpx doesn't cache by default, but explicitly disable connection pooling
        self.http_client = httpx.Client(
            timeout=60,
            follow_redirects=True,
            limits=httpx.Limits(
                max_keepalive_connections=0, max_connections=100
            ),  # No connection reuse
        )
        self.user_email_override: Optional[str] = None

    def _mdai_client_instance(self) -> mdai.Client:
        """Get MD.ai client instance (same as server)."""
        token = self._token_override or self.config.mdai_token
        if not is_valid_token(token):
            raise TokenNotConfigured(
                "MD.ai token not configured. Set token via /api/settings before accessing MD.ai API."
            )
        if self._mdai_client is None:
            self._mdai_client = mdai.Client(
                domain=self.config.domain,
                access_token=token,
            )
        return self._mdai_client

    def set_mdai_token(self, token: Optional[str]) -> None:
        """Update the MD.ai token for subsequent SDK calls."""
        self._token_override = token.strip() if token else None
        self._mdai_client = None  # Reset client to pick up new token

    def sync_dataset(self) -> Path:
        """Download/refresh the MD.ai dataset. Returns the images directory."""
        print(
            f"[ClientContext] sync_dataset() called - project_id: {self.config.project_id}, dataset_id: {self.config.dataset_id}"
        )
        print(
            f"[ClientContext] video_cache_path: {self.config.video_cache_path}"
        )
        client = self._mdai_client_instance()
        print("[ClientContext] Calling client.project()...")
        project = client.project(
            project_id=self.config.project_id,
            dataset_id=self.config.dataset_id,
            path=str(Path(self.config.video_cache_path)),
        )
        print(f"[ClientContext] client.project() returned: {project}")
        if project is None:
            raise RuntimeError("MD.ai project download failed: project is None")
        if project.images_dir is None:
            raise RuntimeError(
                "MD.ai project download failed: images_dir is None"
            )
        self._images_dir = Path(project.images_dir)
        self._annotations_list = None  # force reload
        self._studies_lookup = None
        return self._images_dir

    def sync_dataset_async(self) -> None:
        """Start dataset sync in background thread."""
        if self._sync_status == "syncing":
            return  # Already syncing

        def _sync():
            self._sync_status = "syncing"
            self._sync_error = None
            try:
                print("[ClientContext] Starting sync_dataset()...")
                images_dir = self.sync_dataset()
                print(
                    f"[ClientContext] sync_dataset() completed, images_dir: {images_dir}"
                )
                # Verify we can list series after sync
                series = self.list_local_series()
                print(f"[ClientContext] Found {len(series)} series after sync")
                self._sync_status = "completed"
            except Exception as exc:
                print(f"[ClientContext] Sync error: {exc}")
                import traceback

                traceback.print_exc()
                self._sync_status = "error"
                self._sync_error = str(exc)

        import threading

        thread = threading.Thread(target=_sync, daemon=True)
        thread.start()

    def get_sync_status(self) -> dict:
        """Get current sync status."""
        return {
            "status": self._sync_status,
            "error": self._sync_error,
        }

    def _ensure_images_dir(self) -> Path:
        """Find images directory, checking video_cache_path first, then data_dir."""
        if self._images_dir and self._images_dir.exists():
            return self._images_dir
        search_paths = [
            Path(self.config.video_cache_path),
            Path(self.config.data_dir),
        ]
        for search_path in search_paths:
            try:
                images_dir = Path(
                    find_images_dir(
                        str(search_path),
                        self.config.project_id,
                        self.config.dataset_id,
                    )
                )
                self._images_dir = images_dir
                return images_dir
            except FileNotFoundError:
                continue
        raise DatasetNotReady(
            "MD.ai dataset not found locally. Run /api/dataset/sync first."
        )

    def _ensure_annotations(self) -> list[dict]:
        """Load annotations as list of dicts, checking video_cache_path first, then data_dir."""
        if (
            self._annotations_list is not None
            and self._studies_lookup is not None
        ):
            return self._annotations_list
        search_paths = [
            Path(self.config.video_cache_path),
            Path(self.config.data_dir),
        ]
        for search_path in search_paths:
            try:
                annotations_path = find_annotations_file(
                    str(search_path),
                    self.config.project_id,
                    self.config.dataset_id,
                )
                break
            except FileNotFoundError:
                continue
        else:
            raise DatasetNotReady(
                "Annotations JSON missing. Download dataset via /api/dataset/sync."
            )
        blob = json_to_dataframe(annotations_path)
        # blob["annotations"] and blob["studies"] are lists of dicts (pandas-free)
        annotations_list = blob["annotations"]
        studies_list = blob["studies"]
        self._annotations_list = annotations_list
        self._studies_lookup = {
            study["StudyInstanceUID"]: study for study in studies_list
        }
        return annotations_list

    def list_local_series(self) -> list[dict]:
        """List locally available series (same logic as MDaiDatasetManager)."""
        images_dir = self._ensure_images_dir()
        annotations_list = self._ensure_annotations()
        studies_lookup = self._studies_lookup or {}

        # Group annotations by (StudyInstanceUID, SeriesInstanceUID)
        from collections import defaultdict

        grouped = defaultdict(list)
        for annot in annotations_list:
            key = (
                annot.get("StudyInstanceUID"),
                annot.get("SeriesInstanceUID"),
            )
            grouped[key].append(annot)

        series = []
        for (study_uid, series_uid), _ in grouped.items():
            # Skip if study_uid or series_uid is None
            if study_uid is None or series_uid is None:
                continue
            video_path = images_dir / study_uid / f"{series_uid}.mp4"
            if not video_path.exists():
                continue
            study_info = studies_lookup.get(study_uid, {})
            exam_number = study_info.get("number")
            raw_series_number = study_info.get(
                "SeriesNumber"
            ) or study_info.get("seriesNumber")
            try:
                parsed_series_number = int(raw_series_number)
            except (TypeError, ValueError):
                parsed_series_number = None
            series.append({
                "study_uid": study_uid,
                "series_uid": series_uid,
                "exam_number": int(exam_number)
                if exam_number not in (None, "")
                else None,
                "series_number": parsed_series_number,
                "dataset_name": str(study_info.get("dataset", "Unknown")),
                "video_path": video_path,
            })
        series.sort(
            key=lambda item: (item["exam_number"] or 0, item["series_uid"])
        )
        return series

    def resolve_video(self, study_uid: str, series_uid: str) -> Path:
        """Resolve video path for a series."""
        images_dir = self._ensure_images_dir()
        video_path = images_dir / study_uid / f"{series_uid}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found for {study_uid}/{series_uid}: {video_path}"
            )
        return video_path

    def dataset_ready(self) -> bool:
        """Check if dataset is ready."""
        try:
            self._ensure_images_dir()
            self._ensure_annotations()
            return True
        except DatasetNotReady:
            return False

    def close(self) -> None:
        self.http_client.close()

    def current_user_email(self) -> str:
        return (
            self.user_email_override or self.config.user_email or ""
        ).strip()


def create_app(
    config: Optional[ClientConfig] = None, skip_startup: bool = False
) -> Flask:
    config = config or load_config("client")
    context = ClientContext(config)

    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )
    app.config["CLIENT_CONTEXT"] = context

    # Initialize client: sync dataset (frames served from server outputs)
    # This must complete before serving the viewer
    if not skip_startup:
        logger = logging.getLogger(__name__)
        logger.info("Initializing client: syncing dataset...")
        images_dir = context.sync_dataset()
        logger.info(f"Dataset synced. Images directory: {images_dir}")
        logger.info("Client initialization complete.")
    # Note: Auto-sync on startup is handled by frontend
    # Frontend will check if token is set and dataset is not ready,
    # then show a blocking modal and trigger sync via /api/dataset/sync

    def _current_user_email() -> str:
        return context.current_user_email()

    def _build_videos() -> list[dict]:
        """
        Build the list of locally available series for the viewer.
        No caching - always fetch fresh from server API.
        Fetches completion status from server for each series.
        """
        try:
            series = context.list_local_series()
        except DatasetNotReady:
            return []

        labels = [
            {"labelId": config.label_id, "labelName": "Fluid"},
            {"labelId": config.empty_id, "labelName": "Empty"},
        ]

        # Build base video list
        videos = [
            {
                "method": config.flow_method,
                "study_uid": info["study_uid"],
                "series_uid": info["series_uid"],
                "exam_number": info["exam_number"],
                "series_number": info["series_number"],
                "labels": labels,
                "status": "pending",  # Default, will be updated from server
                "activity": {},  # Default, will be updated from server (never cached locally)
            }
            for info in series
        ]

        # Don't fetch status from server during page load - it blocks if server isn't ready
        # Frontend JavaScript will fetch status asynchronously after page loads
        # This prevents the page from hanging if server isn't available

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
            video_path = context.resolve_video(study_uid, series_uid)
            if not video_path.exists():
                return None

            # Get minimal info without full scan
            images_dir = context._ensure_images_dir()
            annotations_df = context._ensure_annotations()
            studies_lookup = context._studies_lookup or {}

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
        return jsonify({
            "client_ready": context.dataset_ready(),
            "server_url": config.server_url,
            "video_cache": str(config.video_cache_path),
            "frames_cache": str(config.frames_path),
        })

    @app.route("/")
    def viewer_home():
        """
        Serve the main viewer UI using the legacy app.py frontend.
        Settings modal must be accessible even when no videos are present.
        """
        videos = _build_videos()
        # Always render viewer.html so Settings modal is accessible
        # Frontend will handle the "no videos" case
        selected_video = videos[0] if videos else None

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
        Proxy video metadata from tracking server (no local caching).
        Tracking server (hivemind.local:5000) is source of truth for masks_annotations and modified_frames.
        This client backend (localhost:8080) proxies requests to the tracking server.
        """
        # Verify series exists locally (for dataset sync check)
        info = _find_series_info(study_uid, series_uid)
        if not info:
            return jsonify({"error": "Series not found locally"}), 404

        try:
            video_path = context.resolve_video(study_uid, series_uid)
        except (DatasetNotReady, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 404

        # Proxy entire response from tracking server (tracking server is source of truth)
        tracking_server_url = f"{config.server_url.rstrip('/')}/api/video/{method}/{study_uid}/{series_uid}"
        resp = context.http_client.get(tracking_server_url, timeout=10)
        return Response(
            resp.content,
            status=resp.status_code,
            headers={"Content-Type": "application/json"},
        )

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

        frames_archive_url = (
            f"/proxy/api/frames_archive/{study_uid}/{series_uid}.tar"
        )
        masks_archive_url = f"/proxy/api/masks/{study_uid}/{series_uid}"
        return jsonify({
            "frames_archive_url": frames_archive_url,
            "masks_archive_url": masks_archive_url,
        })

    @app.post("/api/dataset/sync")
    def sync_dataset():
        """
        Download/refresh the MD.ai dataset from MD.ai API (not from server).
        Frames are produced by the server tracking pipeline; the client no longer extracts or caches frames.
        """
        print(
            "[API] POST /api/dataset/sync - Starting MD.ai dataset download..."
        )
        try:
            images_dir = context.sync_dataset()
            series = context.list_local_series()
            print(
                f"[API] Dataset sync completed: {len(series)} series downloaded from MD.ai"
            )
            return jsonify({
                "images_dir": str(images_dir),
                "series_count": len(series),
                "frames_extracted_for": [],  # no local extraction
            })
        except TokenNotConfigured as exc:
            print(f"[API] Dataset sync failed: Token not configured - {exc}")
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            print(f"[API] Dataset sync error: {exc}")
            import traceback

            traceback.print_exc()
            return jsonify({
                "error": f"Failed to sync dataset from MD.ai: {exc}"
            }), 500

    @app.get("/api/dataset/sync/status")
    def sync_status():
        """Get current sync status."""
        status = context.get_sync_status()

        # If sync completed, also return series info
        if status["status"] == "completed":
            try:
                series = context.list_local_series()
                images_dir = context._ensure_images_dir()
                status["images_dir"] = str(images_dir)
                status["series_count"] = len(series)
            except Exception:
                pass  # Ignore errors when getting series info

        return jsonify(status)

    @app.get("/api/local/series")
    def local_series():
        try:
            series = context.list_local_series()
        except DatasetNotReady as exc:
            return jsonify({"error": str(exc)}), 503

        payload = [
            {
                "study_uid": info["study_uid"],
                "series_uid": info["series_uid"],
                "exam_number": info["exam_number"],
                "series_number": info["series_number"],
                "dataset_name": info["dataset_name"],
                "video_path": str(info["video_path"]),
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
        client_version = None
        server_version = None

        # Client annotations version
        # iOS syncs to video_cache_path, check there
        try:
            client_annotations = Path(
                find_annotations_file(
                    str(config.video_cache_path),
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
            and "annotations_size" in server_version
        ):
            # Compare file size instead of mtime - mtime differs across machines
            # even when files are identical. Size is a better indicator of sync.
            in_sync = (
                client_version["annotations_size"]
                == server_version["annotations_size"]
            )

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
        token = context._token_override or context.config.mdai_token
        return jsonify({
            "user_email": _current_user_email(),
            "mdai_token_present": is_valid_token(token),
        })

    @app.post("/api/settings")
    def update_settings():
        """
        Update user_email and/or MD.ai token for this client process.
        Values are kept in-memory for the session.
        """
        print("[API] POST /api/settings called")
        data = request.get_json(silent=True) or {}
        user_email = data.get("user_email")
        mdai_token = data.get("mdai_token")

        print(
            f"[API] Received data - user_email: {bool(user_email)}, mdai_token: {bool(mdai_token)}"
        )

        # Load iOS settings file path if available
        settings_file = None
        try:
            from ios_config import (
                get_ios_paths,
                load_persisted_settings,
                save_persisted_settings,
            )

            paths = get_ios_paths()
            settings_file = paths.get("SETTINGS_FILE")
        except (ImportError, AttributeError) as e:
            print(f"[API] Could not load iOS settings functions: {e}")

        settings_to_save = {}

        if user_email is not None:
            email_clean = user_email.strip()
            context.user_email_override = email_clean or None
            if email_clean:
                settings_to_save["user_email"] = email_clean
            print(f"[API] User email set: {email_clean}")

        if mdai_token is not None:
            token_clean = mdai_token.strip()
            context.set_mdai_token(token_clean or None)
            if token_clean:
                settings_to_save["mdai_token"] = token_clean
            print(f"[API] MD.ai token set: {bool(token_clean)}")

        # Persist settings to disk (iOS only)
        if settings_file and settings_to_save:
            try:
                # Load existing settings and merge
                existing = load_persisted_settings(settings_file)
                existing.update(settings_to_save)
                save_persisted_settings(settings_file, existing)
                print(f"[API] Settings persisted to {settings_file}")
            except Exception as e:
                print(f"[API] Warning: Could not persist settings: {e}")

        token = context._token_override or context.config.mdai_token
        token_valid = is_valid_token(token)
        print(f"[API] Token valid: {token_valid}")
        return jsonify({
            "user_email": _current_user_email(),
            "mdai_token_present": token_valid,
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

        # POST archive to central server /api/masks/{study}/{series}
        server_url = config.server_url.rstrip("/")
        url = f"{server_url}/api/masks/{study_uid}/{series_uid}"

        headers = {
            "Content-Type": "application/x-tar",
            "X-Previous-Version-ID": previous_version_id or "",
        }
        # Identification for last editor - prefer X-User-Email from request (frontend)
        # Server accepts either X-Editor or X-User-Email
        user_email = request.headers.get("X-User-Email")
        if user_email:
            headers["X-User-Email"] = user_email
        else:
            current_email = _current_user_email()
            if current_email:
                headers["X-User-Email"] = current_email

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
        Forward requests to the tracking server (hivemind.local:5000).
        This client backend (localhost:8080) proxies requests to the tracking server.
        """
        try:
            # Forward to tracking server (hivemind.local:5000)
            tracking_server_url = f"{config.server_url.rstrip('/')}/{subpath}"
            headers = {
                key: value
                for key, value in request.headers
                if key.lower()
                not in {"host", "content-length", "connection", "authorization"}
            }

            if "X-User-Email" not in headers:
                frontend_email = request.headers.get("X-User-Email")
                if frontend_email:
                    headers["X-User-Email"] = frontend_email
                else:
                    current_email = _current_user_email()
                    if current_email:
                        headers["X-User-Email"] = current_email

            files = None
            if request.files:
                files = {
                    name: (file.filename, file.stream, file.mimetype)
                    for name, file in request.files.items()
                }

            data = None
            json_payload = None
            if request.is_json and not request.data:
                json_payload = request.get_json(silent=True)
            else:
                data = request.get_data()

            logger = logging.getLogger(__name__)
            logger.debug(
                f"Proxying {request.method} {subpath} to tracking server: {tracking_server_url}"
            )

            proxy_headers = headers.copy()

            # Proxy request to tracking server using the shared client
            resp = context.http_client.request(
                request.method,
                tracking_server_url,
                params=request.args,
                data=data,
                json=json_payload,
                files=files,
                headers=proxy_headers,
                timeout=60,
                follow_redirects=True,
            )
            # Ensure we read the full response content to avoid any streaming/caching issues
            status_code = resp.status_code
            response_headers_dict = dict(resp.headers)
            content = resp.content

            # Debug: Log response size and first 200 chars for /api/series
            if subpath == "api/series":
                logger.debug(
                    f"Proxy /api/series response size: {len(content)} bytes"
                )
                try:
                    import json

                    data = json.loads(content.decode("utf-8"))
                    if data and isinstance(data, list):
                        exam_429 = [
                            x for x in data if x.get("exam_number") == 429
                        ]
                        if exam_429:
                            logger.debug(
                                f"Exam 429 activity in proxy response: {exam_429[0].get('activity', {})}"
                            )
                except Exception:
                    pass

            logger.debug(f"Proxy response: {status_code} for {subpath}")

            excluded_headers = {
                "content-length",
                "content-encoding",
                "transfer-encoding",
                "connection",
            }
            response_headers = {
                key: value
                for key, value in response_headers_dict.items()
                if key.lower() not in excluded_headers
            }
            # Add no-cache headers to prevent browser caching of proxy responses
            response_headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            response_headers["Pragma"] = "no-cache"
            response_headers["Expires"] = "0"
            return Response(
                content, status=status_code, headers=response_headers
            )
        except Exception as exc:
            import traceback

            logger = logging.getLogger(__name__)
            error_msg = f"Error proxying {request.method} {subpath} to tracking server ({config.server_url}): {exc}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR in proxy: {error_msg}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return jsonify({
                "error": "Failed to connect to tracking server",
                "error_message": str(exc),
                "path": subpath,
                "tracking_server_url": config.server_url,
            }), 502

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
    # Ensure iOS environment is configured before loading config
    try:
        from ios_config import configure_environment

        configure_environment()
    except Exception:
        pass

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

    # Ensure iOS environment is configured before loading config
    try:
        from ios_config import configure_environment

        configure_environment()
    except Exception:
        pass

    config = load_config("client")
    pid_file = get_pid_file(config)

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create log directory
    # On iOS, the app bundle is read-only; use the iOS cache dir instead.
    import os

    log_dir = None
    try:
        from ios_config import get_ios_paths

        paths = get_ios_paths()
        log_dir = Path(paths["CACHE_DIR"]) / "log"
    except Exception:
        log_dir = None

    if log_dir is None:
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
