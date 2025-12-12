'"""Flask blueprint exposing the distributed tracking API."""'

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from lib.mask_archive import (
    build_mask_archive,
    build_mask_metadata,
    extract_mask_archive,
    iso_now,
    MaskArchiveError,
)
from lib.uploaded_masks import convert_uploaded_masks_to_annotations_df
from server.storage.retrack_queue import RetrackQueue
from server.storage.series_manager import SeriesManager, SeriesMetadata
from server.tracking_worker import trigger_lazy_tracking


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_series(metadata: SeriesMetadata) -> dict:
    payload = asdict(metadata)
    return payload


def _mask_series_dir(
    mask_root: Path, flow_method: str, study_uid: str, series_uid: str
) -> Path:
    return mask_root / flow_method / f"{study_uid}_{series_uid}"




def create_api_blueprint(series_manager: SeriesManager, config) -> Blueprint:
    bp = Blueprint("distributed_api", __name__)

    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method

    @bp.get("/api/status")
    def status() -> Response:
        series = series_manager.list_series()
        completed = sum(1 for item in series if item.status == "completed")
        failed = sum(1 for item in series if item.tracking_status == "failed")
        pending = sum(
            1
            for item in series
            if item.tracking_status in {"never_run", "pending"}
        )
        return jsonify({
            "ready": True,
            "series_total": len(series),
            "series_completed": completed,
            "series_failed": failed,
            "series_pending": pending,
        })

    @bp.get("/api/series")
    def list_series():
        records = [
            _serialize_series(item) for item in series_manager.list_series()
        ]
        return jsonify(records)

    @bp.get("/api/series/next")
    def next_series():
        user_email = request.headers.get("X-User-Email")
        result = series_manager.select_next_series(user_email=user_email)
        if not result:
            return jsonify({"no_available_series": True}), 200
        return jsonify(_serialize_series(result))

    @bp.get("/api/series/<study_uid>/<series_uid>")
    def series_detail(study_uid: str, series_uid: str):
        try:
            metadata = series_manager.get_series(study_uid, series_uid)
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404

        activity = series_manager.activity_history(study_uid, series_uid)
        payload = _serialize_series(metadata)
        payload["activity"] = activity
        return jsonify(payload)

    @bp.post("/api/series/<study_uid>/<series_uid>/activity")
    def series_activity(study_uid: str, series_uid: str):
        user_email = request.headers.get("X-User-Email")
        try:
            metadata = series_manager.mark_activity(
                study_uid, series_uid, user_email
            )
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404
        return jsonify(_serialize_series(metadata))

    @bp.post("/api/series/<study_uid>/<series_uid>/complete")
    def series_complete(study_uid: str, series_uid: str):
        try:
            metadata = series_manager.mark_complete(study_uid, series_uid)
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404
        return jsonify(_serialize_series(metadata))

    @bp.post("/api/series/<study_uid>/<series_uid>/reopen")
    def series_reopen(study_uid: str, series_uid: str):
        try:
            metadata = series_manager.reopen(study_uid, series_uid)
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404
        return jsonify(_serialize_series(metadata))

    @bp.get("/api/masks/<study_uid>/<series_uid>")
    def get_masks(study_uid: str, series_uid: str):
        try:
            series = series_manager.get_series(study_uid, series_uid)
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404

        mask_dir = _mask_series_dir(
            mask_root, flow_method, study_uid, series_uid
        )
        
        # Check for retracked masks first (retracked take precedence)
        retrack_masks_path = mask_dir / "retrack" / "masks"
        initial_masks_path = mask_dir / "masks"
        
        # Use retracked masks if they exist, otherwise use initial tracking masks
        if retrack_masks_path.exists() and list(retrack_masks_path.glob("*.webp")):
            masks_path = retrack_masks_path
        elif initial_masks_path.exists() and list(initial_masks_path.glob("*.webp")):
            masks_path = initial_masks_path
        else:
            masks_path = initial_masks_path  # Default to initial for lazy tracking check

        # Fallback lazy tracking: if masks don't exist, trigger tracking
        # NOTE: Per DISTRIBUTED_ARCHITECTURE.md, masks should be generated on startup.
        # This is a fallback for edge cases (e.g., new series added after startup).
        if not masks_path.exists() or not list(masks_path.glob("*.webp")):
            tracking_status = series.tracking_status
            if tracking_status == "never_run":
                # Trigger lazy tracking in background
                trigger_lazy_tracking(
                    study_uid, series_uid, config, series_manager
                )
                return (
                    jsonify({
                        "status": "pending",
                        "error_code": "TRACK_PENDING",
                    }),
                    202,
                )
            elif tracking_status == "pending":
                # Tracking in progress
                return (
                    jsonify({
                        "status": "pending",
                        "error_code": "TRACK_PROCESSING",
                    }),
                    202,
                )
            elif tracking_status == "failed":
                # Tracking failed - try to get error details from series metadata
                try:
                    # Check if there's an error message stored anywhere
                    # For now, just return generic failure
                    return (
                        jsonify({
                            "status": "failed",
                            "error_code": "TRACK_FAILED",
                            "error_message": "Tracking failed. Check server logs for details.",
                        }),
                        500,
                    )
                except Exception:
                    return (
                        jsonify({
                            "status": "failed",
                            "error_code": "TRACK_FAILED",
                        }),
                        500,
                    )
            else:
                # Unknown state, return pending
                return (
                    jsonify({
                        "status": "pending",
                        "error_code": "TRACK_PENDING",
                    }),
                    202,
                )

        # Check for pre-built archive (built on tracking/retracking completion)
        # Retracked archive takes precedence
        retrack_archive_path = mask_dir / "retrack" / "masks.tgz"
        initial_archive_path = mask_dir / "masks.tgz"
        
        if retrack_archive_path.exists():
            archive_path = retrack_archive_path
        elif initial_archive_path.exists():
            archive_path = initial_archive_path
        else:
            archive_path = initial_archive_path  # Default for fallback building
        if not archive_path.exists():
            # Fallback: build archive on-demand if it doesn't exist (for backwards compatibility)
            metadata = build_mask_metadata(series, masks_path, flow_method)
            try:
                archive_bytes = build_mask_archive(masks_path, metadata)
            except MaskArchiveError as exc:
                return jsonify({"error": str(exc)}), 500
        else:
            # Serve pre-built archive
            with archive_path.open("rb") as f:
                archive_bytes = f.read()
            # Load metadata from archive for headers
            import tarfile
            import json
            import io
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
                metadata_file = tar.extractfile("metadata.json")
                if metadata_file:
                    metadata = json.load(metadata_file)
                else:
                    # Fallback: build metadata if not in archive
                    metadata = build_mask_metadata(series, masks_path, flow_method)

        headers = {
            "Content-Type": "application/x-tar+gzip",
            "X-Version-ID": metadata.get("version_id") or "",
            "X-Mask-Count": str(metadata.get("mask_count", 0)),
            "X-Flow-Method": flow_method,
            "X-Generated-At": metadata.get("generated_at", iso_now()),
        }
        return Response(archive_bytes, headers=headers)

    @bp.post("/api/masks/<study_uid>/<series_uid>")
    def post_masks(study_uid: str, series_uid: str):
        """
        Accept edited masks as .tgz archive, validate version, and queue retrack.
        """
        try:
            series = series_manager.get_series(study_uid, series_uid)
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404

        # Get headers
        previous_version_id = request.headers.get("X-Previous-Version-ID")
        if previous_version_id == "":
            previous_version_id = None
        editor = request.headers.get("X-Editor") or request.headers.get(
            "X-User-Email"
        )

        if not editor:
            return jsonify({
                "error": "X-Editor or X-User-Email header required"
            }), 400

        # Check for active retrack
        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)
        if retrack_queue.has_active_job(study_uid, series_uid):
            return (
                jsonify({
                    "error_code": "RETRACK_IN_PROGRESS",
                    "message": "This series is currently being retracked. Wait or reload.",
                }),
                409,
            )

        # Validate version ID
        current_version = series.current_version_id
        if previous_version_id != current_version:
            return (
                jsonify({
                    "error_code": "VERSION_MISMATCH",
                    "current_version": current_version or "",
                    "your_version": previous_version_id or "",
                    "message": "Someone else edited this series. Please reset and re-apply changes.",
                }),
                409,
            )

        # Get uploaded archive
        if not request.data:
            return jsonify({"error": "No data provided"}), 400

        # Extract to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_path = Path(temp_dir) / "uploaded_masks"
            try:
                annotations_df, metadata = (
                    convert_uploaded_masks_to_annotations_df(
                        request.data, extract_path
                    )
                )
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

            # Create permanent storage for uploaded masks (will be cleaned up after retrack)
            timestamp_str = iso_now().replace(":", "-").replace(".", "-")
            uploaded_masks_dir = (
                config.server_state_path
                / "uploaded_masks"
                / f"{study_uid}_{series_uid}"
                / timestamp_str
            )
            uploaded_masks_dir.mkdir(parents=True, exist_ok=True)

            # Copy extracted files to permanent location
            for item in extract_path.iterdir():
                shutil.copy2(item, uploaded_masks_dir / item.name)

            # Enqueue retrack job
            job = retrack_queue.enqueue(
                study_uid=study_uid,
                series_uid=series_uid,
                editor=editor,
                previous_version_id=previous_version_id,
                uploaded_masks_path=uploaded_masks_dir,
            )

            # Update series metadata with temp version
            series_manager.update_version(
                study_uid, series_uid, job.new_version_id, editor
            )

            # Update tracking status to indicate retrack in progress
            series_manager.update_tracking_status(
                study_uid, series_uid, "retracking"
            )

            queue_position = retrack_queue.get_queue_position(
                study_uid, series_uid, job.new_version_id
            )

            return jsonify({
                "success": True,
                "version_id": job.new_version_id,
                "retrack_queued": True,
                "queue_position": queue_position,
            })

    @bp.get("/api/retrack/status/<study_uid>/<series_uid>")
    def retrack_status(study_uid: str, series_uid: str):
        """Get retrack status for a series."""
        try:
            series = series_manager.get_series(study_uid, series_uid)
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404

        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)

        # Get most recent job for this series
        job = retrack_queue.get_job_status(study_uid, series_uid)

        if not job:
            # No retrack job found
            if series.tracking_status == "retracking":
                return jsonify({
                    "status": "processing",
                    "error_code": "RETRACK_PROCESSING",
                })
            return jsonify({"status": "none"})

        # Map job status to error codes
        status_map = {
            "pending": {"status": "pending", "error_code": "RETRACK_PENDING"},
            "processing": {
                "status": "processing",
                "error_code": "RETRACK_PROCESSING",
            },
            "completed": {"status": "completed"},
            "failed": {
                "status": "failed",
                "error_code": "RETRACK_FAILED",
                "error_message": job.error_message,
            },
        }

        response = status_map.get(job.status, {"status": job.status})
        response["version_id"] = job.new_version_id

        if job.status == "pending":
            queue_position = retrack_queue.get_queue_position(
                study_uid, series_uid, job.new_version_id
            )
            response["queue_position"] = queue_position

        return jsonify(response)

    return bp
