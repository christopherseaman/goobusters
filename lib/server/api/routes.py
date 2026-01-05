"""Flask blueprint exposing the distributed tracking API."""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

# Add paths: project root (for lib imports) and lib/server (for server package imports)
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_lib_server = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _lib_server not in sys.path:
    sys.path.insert(0, _lib_server)

import gzip
from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    send_file,
    stream_with_context,
)

from lib.mask_archive import (
    build_mask_archive,
    build_mask_metadata,
    extract_mask_archive,
    iso_now,
    mask_series_dir,
    MaskArchiveError,
)
from lib.uploaded_masks import convert_uploaded_masks_to_annotations_df
from server.storage.retrack_queue import RetrackQueue
from server.storage.series_manager import SeriesManager, SeriesMetadata
from server.tracking_worker import trigger_lazy_tracking


def _serialize_series(metadata: SeriesMetadata) -> dict:
    payload = asdict(metadata)
    return payload


def create_api_blueprint(series_manager: SeriesManager, config) -> Blueprint:
    bp = Blueprint("distributed_api", __name__)

    # Register error handler to catch all exceptions in this blueprint
    @bp.errorhandler(Exception)
    def handle_error(e):
        import traceback
        import sys
        import logging

        error_msg = f"Unhandled error in API blueprint: {e}"
        traceback_str = traceback.format_exc()

        # Log to both logger and stderr
        try:
            logger = logging.getLogger(__name__)
            logger.error(error_msg, exc_info=True)
        except Exception:
            pass

        print(f"ERROR: {error_msg}", file=sys.stderr)
        print(traceback_str, file=sys.stderr)

        from flask import jsonify

        return jsonify({
            "error": "Internal server error",
            "error_message": str(e),
        }), 500

    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method

    @bp.get("/api/status")
    def status() -> Response:
        series = series_manager.list_series()
        completed = sum(
            1
            for item in series
            if series_manager.get_tracking_status(
                item.study_uid, item.series_uid
            )
            == "completed"
        )
        failed = sum(
            1
            for item in series
            if series_manager.get_tracking_status(
                item.study_uid, item.series_uid
            )
            == "failed"
        )
        pending = sum(
            1
            for item in series
            if series_manager.get_tracking_status(
                item.study_uid, item.series_uid
            )
            in {"never_run", "pending"}
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
        """
        Get all series with completion status and activity info.
        Returns list of series with: study_uid, series_uid, exam_number, status, activity.
        """
        all_series = series_manager.list_series()
        records = []
        for item in all_series:
            # Get full activity history (user_email -> timestamp)
            activity = series_manager.activity_history(
                item.study_uid, item.series_uid
            )

            record = {
                "study_uid": item.study_uid,
                "series_uid": item.series_uid,
                "exam_number": item.exam_number,
                "series_number": item.series_number,
                "status": item.status,  # "pending" or "completed"
                "activity": activity,  # dict of user_email -> ISO timestamp
            }
            records.append(record)

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
        """
        Get the series for a user to work on.

        This is more "get-series-for-user" than "next series" - it will return
        the same series repeatedly if:
        - Series is not complete
        - User was recently active on it
        - Others have not been active since

        This allows refreshing the page to return to the same series you were working on.
        """
        import logging
        import sys

        logger = logging.getLogger(__name__)
        print("DEBUG: /api/series/next called", file=sys.stderr)
        logger.info("DEBUG: /api/series/next endpoint called")
        try:
            user_email = request.headers.get("X-User-Email")
            logger.info(f"DEBUG: user_email={user_email}")
            print(
                f"DEBUG: Calling select_next_series with user_email={user_email}",
                file=sys.stderr,
            )
            result = series_manager.select_next_series(user_email=user_email)
            logger.info(
                f"DEBUG: select_next_series returned: {result is not None}"
            )
            print(
                f"DEBUG: select_next_series returned: {result is not None}",
                file=sys.stderr,
            )
            if not result:
                return jsonify({"no_available_series": True}), 200

            # Mark activity when server selects a series for the user
            # This ensures the selected series is immediately marked as active
            series_manager.mark_activity(
                result.study_uid, result.series_uid, user_email
            )

            return jsonify(_serialize_series(result))
        except Exception as exc:
            import traceback
            import sys
            import logging

            # Log to both logger and stderr to ensure we see it
            error_msg = f"Error in /api/series/next: {exc}"
            traceback_str = traceback.format_exc()

            # Try logger first
            try:
                logger = logging.getLogger(__name__)
                logger.error(error_msg, exc_info=True)
            except Exception:
                pass

            # Always print to stderr (will show in console/logs)
            print(f"ERROR: {error_msg}", file=sys.stderr)
            print(traceback_str, file=sys.stderr)

            return jsonify({
                "error": "Failed to select next series",
                "error_message": str(exc),
            }), 500

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

        # Include activity data in response for warning updates
        activity = series_manager.activity_history(study_uid, series_uid)
        payload = _serialize_series(metadata)
        payload["activity"] = activity
        return jsonify(payload)

    @bp.post("/api/series/<study_uid>/<series_uid>/complete")
    def series_complete(study_uid: str, series_uid: str):
        user_email = request.headers.get("X-User-Email")
        try:
            metadata = series_manager.mark_complete(
                study_uid, series_uid, user_email=user_email
            )
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
        import tarfile
        import json
        import io

        try:
            _ = series_manager.get_series(
                study_uid, series_uid
            )  # Ensure series exists
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
        tracking_status = series_manager.get_tracking_status(
            study_uid, series_uid
        )
        if tracking_status == "failed":
            return (
                jsonify({
                    "status": "failed",
                    "error_code": "TRACK_FAILED",
                    "error_message": "Tracking failed for this series (likely no annotations). Check server logs for details.",
                }),
                500,
            )

        mask_dir = mask_series_dir(
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
                # Series not trackable - tracking status computed from filesystem
                return (
                    jsonify({
                        "status": "failed",
                        "error_code": "TRACK_FAILED",
                        "error_message": "Series is not trackable (no free fluid annotations or video missing).",
                    }),
                    500,
                )

            tracking_status = series_manager.get_tracking_status(
                study_uid, series_uid
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
        # Retracked archive takes precedence and is in main output directory (not retrack/).
        # Archives are built exactly once by the tracking/retrack workers; this endpoint never re-tars on demand.
        # For retrack, tarball is in main output directory (overwrites initial tracking's tarball)
        # Check for masks.tar (server format)
        archive_path = mask_dir / "masks.tar"

        # If masks.tar doesn't exist but masks/ directory exists (from track.py), build it
        if not archive_path.exists():
            masks_dir_path = mask_dir / "masks"
            if masks_dir_path.exists() and list(masks_dir_path.glob("*.webp")):
                from server.tracking_worker import (
                    build_mask_archive_from_directory,
                )
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    f"Building masks.tar from existing masks/ directory for {study_uid}/{series_uid}"
                )
                try:
                    build_mask_archive_from_directory(
                        study_uid,
                        series_uid,
                        masks_dir_path,
                        mask_dir,
                        config,
                        series_manager,
                    )
                except Exception as exc:
                    import traceback

                    logger.error(
                        f"Failed to build masks.tar from masks/ directory: {exc}"
                    )
                    traceback.print_exc()
                    # If build fails, return error
                    return jsonify({
                        "status": "failed",
                        "error_code": "ARCHIVE_BUILD_ERROR",
                        "error_message": f"Failed to build mask archive from masks/ directory: {str(exc)}",
                    }), 500

        # If retrack/ subdirectory exists, the tarball in main output directory is from retrack
        # Otherwise, it's from initial tracking. Both use the same path.
        if not archive_path.exists():
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

        def _open_tar(bytes_blob: bytes):
            try:
                return tarfile.open(fileobj=io.BytesIO(bytes_blob), mode="r")
            except tarfile.ReadError:
                return tarfile.open(fileobj=io.BytesIO(bytes_blob), mode="r:gz")

        metadata = None
        try:
            tar = _open_tar(archive_bytes)
            with tar:
                metadata_file = tar.extractfile("metadata.json")
                if metadata_file:
                    metadata = json.load(metadata_file)
        except Exception:
            metadata = None

        # If metadata is missing (old track.py archives), rebuild archive with metadata.json
        if metadata is None:
            masks_dir = (
                mask_series_dir(
                    mask_root, config.flow_method, study_uid, series_uid
                )
                / "masks"
            )
            if masks_dir.exists() and list(masks_dir.glob("*.webp")):
                try:
                    from server.tracking_worker import (
                        build_mask_archive_from_directory,
                    )

                    logger = current_app.logger
                    logger.warning(
                        f"metadata.json missing in masks archive for {study_uid}/{series_uid}; rebuilding masks.tar"
                    )
                    build_mask_archive_from_directory(
                        study_uid,
                        series_uid,
                        masks_dir,
                        mask_dir,
                        config,
                        series_manager,
                    )
                    with (mask_dir / "masks.tar").open("rb") as f:
                        archive_bytes = f.read()
                    tar = _open_tar(archive_bytes)
                    with tar:
                        metadata_file = tar.extractfile("metadata.json")
                        if metadata_file:
                            metadata = json.load(metadata_file)
                except Exception as exc:
                    return jsonify({
                        "error": f"metadata.json not found and rebuild failed: {exc}",
                        "error_code": "MISSING_METADATA",
                    }), 500

        if metadata is None:
            return jsonify({
                "error": "metadata.json not found in mask archive",
                "error_code": "MISSING_METADATA",
            }), 500

        headers = {
            "Content-Type": "application/x-tar",
            "X-Version-ID": metadata.get("version_id") or "",
            "X-Mask-Count": str(metadata.get("mask_count", 0)),
            "X-Flow-Method": flow_method,
            "X-Generated-At": metadata.get("generated_at", iso_now()),
        }
        return Response(archive_bytes, headers=headers)

    @bp.get("/api/frames_archive/<study_uid>/<series_uid>.tar")
    def frames_archive(study_uid: str, series_uid: str):
        """
        Serve frames.tar (WebP frames, no gzip) from tracking output.
        """
        base_dir = mask_series_dir(
            mask_root, config.flow_method, study_uid, series_uid
        )
        tar_path = base_dir / "frames.tar"
        gz_path = base_dir / "frames.tar.gz"

        if tar_path.exists():
            return send_file(
                tar_path,
                mimetype="application/x-tar",
                as_attachment=False,
                download_name=f"{study_uid}_{series_uid}_frames.tar",
            )

        if gz_path.exists():

            def _decompress():
                with gzip.open(gz_path, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        yield chunk

            return Response(
                stream_with_context(_decompress()),
                mimetype="application/x-tar",
                headers={
                    "Content-Disposition": f'inline; filename="{study_uid}_{series_uid}_frames.tar"',
                },
            )

        return jsonify({"error": "frames.tar not found"}), 404

    def _get_total_frames(study_uid: str, series_uid: str) -> int:
        """Get total frame count from frames directory or frametype.json."""
        base_dir = mask_series_dir(
            mask_root, config.flow_method, study_uid, series_uid
        )
        frames_dir = base_dir / "frames"

        # Count frames in frames directory (most reliable)
        if frames_dir.exists():
            frame_files = list(frames_dir.glob("frame_*.webp"))
            if frame_files:
                return len(frame_files)

        # Fallback: get from frametype.json
        frametype_path = base_dir / "frametype.json"
        if frametype_path.exists():
            try:
                with frametype_path.open() as f:
                    frametype_data = json.load(f)
                    # frametype.json keys are frame numbers (as strings)
                    # Exclude metadata keys
                    frame_keys = [
                        k
                        for k in frametype_data.keys()
                        if k != "_version_id" and k.isdigit()
                    ]
                    if frame_keys:
                        # Total frames = max frame number + 1 (0-indexed)
                        return max(int(k) for k in frame_keys) + 1
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        return 0

    @bp.get("/api/video/<method>/<study_uid>/<series_uid>")
    def video(method: str, study_uid: str, series_uid: str):
        """
        Return video metadata including total_frames and completion status for the viewer.
        """
        try:
            series = series_manager.get_series(study_uid, series_uid)
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404

        total_frames = _get_total_frames(study_uid, series_uid)

        # Construct labels from config (same format as client)
        labels = [
            {"labelId": config.label_id, "labelName": "Fluid"},
            {"labelId": config.empty_id, "labelName": "Empty"},
        ]

        return jsonify({
            "total_frames": total_frames,
            "method": method,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "exam_number": series.exam_number,
            "status": series.status,  # "pending" or "completed"
            "labels": labels,
            "masks_annotations": [],
            "modified_frames": {},
        })

    @bp.get("/api/frames/<method>/<study_uid>/<series_uid>")
    def frames(method: str, study_uid: str, series_uid: str):
        """
        Return URLs for frame and mask archives for the viewer.
        Frames and masks are generated by the server tracking pipeline.
        """
        total_frames = _get_total_frames(study_uid, series_uid)

        # Frames archive URL (server-produced frames.tar)
        frames_archive_url = f"/api/frames_archive/{study_uid}/{series_uid}.tar"
        masks_archive_url = f"/api/masks/{study_uid}/{series_uid}"
        return jsonify({
            "frames_archive_url": frames_archive_url,
            "masks_archive_url": masks_archive_url,
            "total_frames": total_frames,
        })

    @bp.post("/api/masks/<study_uid>/<series_uid>")
    def post_masks(study_uid: str, series_uid: str):
        """
        Accept edited masks as .tar archive, validate version, and queue retrack.
        """
        try:
            _ = series_manager.get_series(
                study_uid, series_uid
            )  # Ensure series exists
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
        # Normalize: treat None and empty string as equivalent (initial state)
        current_version = series_manager.get_version_id(study_uid, series_uid)
        previous_normalized = (
            previous_version_id if previous_version_id else None
        )
        current_normalized = current_version if current_version else None
        if previous_normalized != current_normalized:
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
                dest = uploaded_masks_dir / item.name
                # Skip if source and destination are the same file (prevents SameFileError)
                if item.resolve() != dest.resolve():
                    shutil.copy2(item, dest)

            # Enqueue retrack job
            job = retrack_queue.enqueue(
                study_uid=study_uid,
                series_uid=series_uid,
                editor=editor,
                previous_version_id=previous_version_id,
                uploaded_masks_path=uploaded_masks_dir,
            )

            # DO NOT update version ID here - wait until retrack completes successfully
            # This prevents orphaned version IDs if the server is killed before completion
            # The version ID will be set in retrack_worker.py when the job completes

            # Update tracking status to indicate retrack in progress
            # Tracking status is computed from filesystem (checks for retrack/masks_temp)

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
            _ = series_manager.get_series(
                study_uid, series_uid
            )  # Ensure series exists
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404

        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)

        # Get most recent job for this series
        job = retrack_queue.get_job_status(study_uid, series_uid)

        if not job:
            # No retrack job found
            if (
                series_manager.get_tracking_status(study_uid, series_uid)
                == "retracking"
            ):
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
                "error": job.error_message,
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

    @bp.post("/api/reset-retrack/<study_uid>/<series_uid>")
    def reset_retrack(study_uid: str, series_uid: str):
        """
        Reset all retrack data for a series, returning it to initial tracking state.

        This removes:
        - retrack/ subdirectory
        - retrack queue jobs for this series
        - version_id (resets to None)
        - uploaded_masks directories for this series
        - retrack tarball (if it overwrote initial tracking's tarball)
        """
        try:
            _ = series_manager.get_series(
                study_uid, series_uid
            )  # Ensure series exists
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404

        # Remove retrack output directory
        mask_root = Path(config.mask_storage_path)
        mask_dir = mask_series_dir(
            mask_root, config.flow_method, study_uid, series_uid
        )
        retrack_dir = mask_dir / "retrack"

        retrack_existed = retrack_dir.exists()
        if retrack_existed:
            shutil.rmtree(retrack_dir)
            print(f"[RESET] Removed retrack directory: {retrack_dir}")

        # Note: If retrack completed, it overwrote masks.tar in the main output directory.
        # We can't restore the original initial tracking tarball, but the masks/ directory
        # from initial tracking should still exist. The tarball can be regenerated if needed.

        # Clear retrack queue jobs for this series (including stuck processing jobs)
        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)
        jobs = retrack_queue._load_queue()
        original_count = len(jobs)
        jobs = [
            j
            for j in jobs
            if not (j.study_uid == study_uid and j.series_uid == series_uid)
        ]
        removed_count = original_count - len(jobs)
        if removed_count > 0:
            retrack_queue._save_queue(jobs)
            print(
                f"[RESET] Removed {removed_count} retrack queue job(s) for {study_uid}/{series_uid}"
            )

        # Reset version_id in frametype.json to None (if it exists)
        frametype_path = mask_dir / "frametype.json"
        if frametype_path.exists():
            try:
                with frametype_path.open("r+") as f:
                    data = json.load(f)
                    data["_version_id"] = None
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
                print(f"[RESET] Reset version_id in frametype.json to None")
            except (json.JSONDecodeError, OSError):
                pass

        # Clean up uploaded_masks directories for this series (atomic operation - only persist on retrack success)
        uploaded_masks_base = (
            config.server_state_path
            / "uploaded_masks"
            / f"{study_uid}_{series_uid}"
        )
        if uploaded_masks_base.exists():
            shutil.rmtree(uploaded_masks_base)
            print(
                f"[RESET] Removed uploaded_masks directory: {uploaded_masks_base}"
            )

        # Reset tracking status to completed (from initial tracking)
        # Get mask count from initial tracking if available
        masks_dir = mask_dir / "masks"
        mask_count = 0
        if masks_dir.exists():
            mask_count = len(list(masks_dir.glob("*.webp")))

        # ALWAYS regenerate masks.tar from initial tracking masks/ directory and frametype.json
        # This ensures we overwrite any retracked tarball that may have been written to the main output directory
        # Use initial tracking's frametype.json from main output directory (not retrack/)
        if masks_dir.exists() and mask_count > 0:
            try:
                from lib.mask_archive import (
                    build_mask_archive,
                    build_mask_metadata,
                )

                # Verify we're using initial tracking's frametype.json (not retrack's)
                # mask_dir is the base output directory, frametype.json is at mask_dir / "frametype.json"
                frametype_path = mask_dir / "frametype.json"
                if not frametype_path.exists():
                    print(
                        f"[RESET] Warning: Initial tracking frametype.json not found at {frametype_path}"
                    )
                # build_mask_metadata looks for frametype.json at masks_dir.parent (which is mask_dir)
                # This ensures we use initial tracking's frametype.json, not retrack's
                # version_id is read from frametype.json (will be None after reset above)
                series_info = series_manager.get_series(study_uid, series_uid)
                metadata = build_mask_metadata(
                    series_info,
                    masks_dir,
                    config.flow_method,
                    config,
                )
                # Write to main output directory (overwrites any retracked tarball)
                # mask_dir is the base output directory, masks_dir is mask_dir / "masks"
                archive_path = mask_dir / "masks.tar"
                archive_bytes = build_mask_archive(masks_dir, metadata)
                with archive_path.open("wb") as f:
                    f.write(archive_bytes)
                print(
                    f"[RESET] Regenerated masks.tar from initial tracking (masks: {mask_count}, frametype: {frametype_path})"
                )
            except Exception as exc:
                import traceback

                print(f"[RESET] ERROR: Failed to regenerate masks.tar: {exc}")
                traceback.print_exc()
        else:
            print(
                f"[RESET] Warning: Cannot regenerate masks.tar - masks_dir does not exist or is empty: {masks_dir}"
            )

        # Reset completion status to "pending" and clear activity tracking
        series_manager.reopen(study_uid, series_uid)
        series_manager.clear_activity(study_uid, series_uid)
        print(
            f"[RESET] Reset completion status to 'pending' and cleared activity tracking"
        )

        # Tracking status is computed from filesystem (checks for masks.tar)
        print(
            f"[RESET] Regenerated masks.tar - tracking status will be 'completed'"
        )

        return jsonify({
            "success": True,
            "message": f"Reset retrack data for {study_uid}/{series_uid}",
            "removed_jobs": removed_count,
            "mask_count": mask_count,
        })

    @bp.post("/api/reset-retrack-all")
    def reset_retrack_all():
        """
        Reset all retrack data for ALL series, returning them to initial tracking state.

        This removes for each series:
        - retrack/ subdirectory
        - retrack queue jobs
        - version_id (resets to None)
        - uploaded_masks directories
        - retrack tarball (if it overwrote initial tracking's tarball)
        """
        all_series = series_manager.list_series()
        total_series = len(all_series)
        total_jobs_removed = 0
        total_reset = 0

        # Clear all retrack queue jobs
        queue_file = config.server_state_path / "retrack_queue.json"
        retrack_queue = RetrackQueue(queue_file)
        jobs = retrack_queue._load_queue()
        original_job_count = len(jobs)
        jobs = []  # Clear all jobs
        retrack_queue._save_queue(jobs)
        total_jobs_removed = original_job_count
        print(
            f"[RESET ALL] Removed {total_jobs_removed} retrack queue job(s) for all series"
        )

        # Reset each series
        mask_root = Path(config.mask_storage_path)
        flow_method = config.flow_method
        for series in all_series:
            try:
                # Reset completion status to "pending" and clear activity tracking for all series
                series_manager.reopen(series.study_uid, series.series_uid)
                series_manager.clear_activity(
                    series.study_uid, series.series_uid
                )

                mask_dir = mask_series_dir(
                    mask_root, flow_method, series.study_uid, series.series_uid
                )
                retrack_dir = mask_dir / "retrack"

                if retrack_dir.exists():
                    shutil.rmtree(retrack_dir)
                    print(
                        f"[RESET ALL] Removed retrack directory: {retrack_dir}"
                    )

                # Reset version_id in frametype.json to None (if it exists)
                frametype_path = mask_dir / "frametype.json"
                if frametype_path.exists():
                    try:
                        import json

                        with frametype_path.open("r+") as f:
                            data = json.load(f)
                            data["_version_id"] = None
                            f.seek(0)
                            f.truncate()
                            json.dump(data, f, indent=2)
                    except (json.JSONDecodeError, OSError):
                        pass

                # Clean up uploaded_masks directories
                uploaded_masks_base = (
                    config.server_state_path
                    / "uploaded_masks"
                    / f"{series.study_uid}_{series.series_uid}"
                )
                if uploaded_masks_base.exists():
                    shutil.rmtree(uploaded_masks_base)

                # ALWAYS regenerate masks.tar from initial tracking masks/ directory and frametype.json
                # This ensures we overwrite any retracked tarball that may have been written to the main output directory
                # Use initial tracking's frametype.json from main output directory (not retrack/)
                masks_dir = mask_dir / "masks"
                mask_count = 0
                if masks_dir.exists():
                    mask_count = len(list(masks_dir.glob("*.webp")))

                if masks_dir.exists() and mask_count > 0:
                    try:
                        from lib.mask_archive import (
                            build_mask_archive,
                            build_mask_metadata,
                        )

                        # build_mask_metadata looks for frametype.json at masks_dir.parent (which is mask_dir)
                        # This ensures we use initial tracking's frametype.json, not retrack's
                        metadata = build_mask_metadata(
                            series,
                            masks_dir,
                            config.flow_method,
                            config,
                        )
                        # Write to main output directory (overwrites any retracked tarball)
                        # mask_dir is the base output directory, masks_dir is mask_dir / "masks"
                        archive_path = mask_dir / "masks.tar"
                        archive_bytes = build_mask_archive(masks_dir, metadata)
                        with archive_path.open("wb") as f:
                            f.write(archive_bytes)
                    except Exception as exc:
                        import traceback

                        print(
                            f"[RESET ALL] ERROR: Failed to regenerate masks.tar for {series.study_uid}/{series.series_uid}: {exc}"
                        )
                        traceback.print_exc()

                # Tracking status is computed from filesystem (checks for masks.tar)
                total_reset += 1
            except Exception as exc:
                print(
                    f"[RESET ALL] Error resetting {series.study_uid}/{series.series_uid}: {exc}"
                )
                import traceback

                traceback.print_exc()

        print(
            f"[RESET ALL] Reset retrack data for {total_reset}/{total_series} series"
        )

        return jsonify({
            "success": True,
            "message": f"Reset retrack data for all series",
            "total_series": total_series,
            "reset_series": total_reset,
            "removed_jobs": total_jobs_removed,
        })

    return bp
