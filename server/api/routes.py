'"""Flask blueprint exposing the distributed tracking API."""'

from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict
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
        completed = sum(
            1 for item in series if item.tracking_status == "completed"
        )
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

    @bp.get("/api/dataset/version")
    def dataset_version():
        """
        Report the server's current dataset version.

        For now this is derived from the MD.ai annotations export mtime so the
        client can detect when its local dataset is out of sync with server.
        """
        from track import find_annotations_file

        try:
            annotations_path = Path(
                find_annotations_file(
                    str(config.data_dir),
                    config.project_id,
                    config.dataset_id,
                )
            )
        except FileNotFoundError:
            return jsonify({
                "error": "Annotations file not found on server"
            }), 500

        stat = annotations_path.stat()
        return jsonify({
            "project_id": config.project_id,
            "dataset_id": config.dataset_id,
            "annotations_path": str(annotations_path),
            "annotations_mtime_ns": stat.st_mtime_ns,
            "annotations_size": stat.st_size,
        })

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
            return jsonify({
                "error": "Series not found",
                "study_uid": study_uid,
                "series_uid": series_uid,
            }), 404
        except Exception as exc:
            import traceback

            print(f"Error getting series {study_uid}/{series_uid}: {exc}")
            traceback.print_exc()
            return jsonify({
                "error": f"Internal error: {str(exc)}",
                "study_uid": study_uid,
                "series_uid": series_uid,
            }), 500

        # Check status FIRST - if failed, return immediately (no expensive checks)
        tracking_status = series.tracking_status
        if tracking_status == "failed":
            return (
                jsonify({
                    "status": "failed",
                    "error_code": "TRACK_FAILED",
                    "error_message": "Tracking failed for this series (likely no annotations). Check server logs for details.",
                }),
                500,
            )

        mask_dir = _mask_series_dir(
            mask_root, flow_method, study_uid, series_uid
        )

        # Check for retracked masks first (retracked take precedence)
        retrack_masks_path = mask_dir / "retrack" / "masks"
        initial_masks_path = mask_dir / "masks"

        # Use retracked masks if they exist, otherwise use initial tracking masks
        if retrack_masks_path.exists() and list(
            retrack_masks_path.glob("*.webp")
        ):
            masks_path = retrack_masks_path
        elif initial_masks_path.exists() and list(
            initial_masks_path.glob("*.webp")
        ):
            masks_path = initial_masks_path
        else:
            masks_path = (
                initial_masks_path  # Default to initial for lazy tracking check
            )

        # Fallback lazy tracking: if masks don't exist, trigger tracking
        # NOTE: Per DISTRIBUTED_ARCHITECTURE.md, masks should be generated on startup.
        # This is a fallback for edge cases (e.g., new series added after startup).
        if not masks_path.exists() or not list(masks_path.glob("*.webp")):
            # Check if series is trackable BEFORE triggering lazy tracking
            # Use shared logic to avoid retrying non-trackable series
            from lib.trackable_series import is_series_trackable

            if not is_series_trackable(study_uid, series_uid, config):
                # Series not trackable - mark as failed and return error
                series_manager.update_tracking_status(
                    study_uid, series_uid, "failed"
                )
                return (
                    jsonify({
                        "status": "failed",
                        "error_code": "TRACK_FAILED",
                        "error_message": "Series is not trackable (no free fluid annotations or video missing).",
                    }),
                    500,
                )

            if tracking_status == "never_run":
                # Trigger lazy tracking in background
                # Note: Worker will mark as "failed" if tracking fails, preventing retries
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
            else:
                # Unknown state, return pending
                return (
                    jsonify({
                        "status": "pending",
                        "error_code": "TRACK_PENDING",
                    }),
                    202,
                )

        # Check for pre-built archive (built on tracking/retracking completion).
        # Retracked archive takes precedence. Archives are built exactly once by
        # the tracking/retrack workers; this endpoint never re-tars on demand.
        retrack_archive_path = mask_dir / "retrack" / "masks.tar"
        initial_archive_path = mask_dir / "masks.tar"

        if retrack_archive_path.exists():
            archive_path = retrack_archive_path
        elif initial_archive_path.exists():
            archive_path = initial_archive_path
        else:
            return (
                jsonify({
                    "status": "failed",
                    "error_code": "TRACK_MISSING_ARCHIVE",
                    "error_message": "Mask archive not found. Tracking/retracking did not complete successfully.",
                }),
                500,
            )

        # Serve pre-built archive
        try:
            with archive_path.open("rb") as f:
                archive_bytes = f.read()
        except Exception as exc:
            import traceback

            print(f"Error reading archive {archive_path}: {exc}")
            traceback.print_exc()
            return jsonify({
                "error": f"Failed to read mask archive: {str(exc)}",
                "error_code": "ARCHIVE_READ_ERROR",
                "archive_path": str(archive_path),
            }), 500

        # Load metadata from archive for headers
        import tarfile
        import json
        import io

        # Try .tar first (no gzip), fall back to .tar.gz for backwards compatibility
        try:
            tar = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r")
        except tarfile.ReadError:
            try:
                tar = tarfile.open(
                    fileobj=io.BytesIO(archive_bytes), mode="r:gz"
                )
            except Exception as exc:
                import traceback

                print(f"Error opening tar archive: {exc}")
                traceback.print_exc()
                return jsonify({
                    "error": f"Failed to parse mask archive: {str(exc)}",
                    "error_code": "ARCHIVE_PARSE_ERROR",
                }), 500

        try:
            with tar:
                metadata_file = tar.extractfile("metadata.json")
                if metadata_file:
                    metadata = json.load(metadata_file)
                else:
                    # Fallback: synthesize minimal headers when metadata.json is missing.
                    # This should not happen for new archives; treat as degraded state.
                    metadata = {
                        "version_id": series.current_version_id,
                        "generated_at": iso_now(),
                        "mask_count": 0,
                    }
        except Exception as exc:
            import traceback

            print(f"Error extracting metadata from archive: {exc}")
            traceback.print_exc()
            # Fallback to minimal metadata
            metadata = {
                "version_id": series.current_version_id or "",
                "generated_at": iso_now(),
                "mask_count": 0,
            }

        headers = {
            "Content-Type": "application/x-tar",
            "X-Version-ID": metadata.get("version_id") or "",
            "X-Mask-Count": str(metadata.get("mask_count", 0)),
            "X-Flow-Method": flow_method,
            "X-Generated-At": metadata.get("generated_at", iso_now()),
        }
        return Response(archive_bytes, headers=headers)

    @bp.post("/api/masks/<study_uid>/<series_uid>")
    def post_masks(study_uid: str, series_uid: str):
        """
        Accept edited masks as .tar archive, validate version, and queue retrack.
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
