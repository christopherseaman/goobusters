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
        return jsonify(_serialize_series(metadata))

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
        # Retracked archive takes precedence and is in main output directory (not retrack/).
        # Archives are built exactly once by the tracking/retrack workers; this endpoint never re-tars on demand.
        # For retrack, tarball is in main output directory (overwrites initial tracking's tarball)
        archive_path = mask_dir / "masks.tar"
        
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

        # Load metadata from archive for headers
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

            # DO NOT update version ID here - wait until retrack completes successfully
            # This prevents orphaned version IDs if the server is killed before completion
            # The version ID will be set in retrack_worker.py when the job completes

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
            series = series_manager.get_series(study_uid, series_uid)
        except FileNotFoundError:
            return jsonify({"error": "Series not found"}), 404

        # Remove retrack output directory
        mask_root = Path(config.mask_storage_path)
        mask_dir = _mask_series_dir(mask_root, config.flow_method, study_uid, series_uid)
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
        jobs = [j for j in jobs if not (j.study_uid == study_uid and j.series_uid == series_uid)]
        removed_count = original_count - len(jobs)
        if removed_count > 0:
            retrack_queue._save_queue(jobs)
            print(f"[RESET] Removed {removed_count} retrack queue job(s) for {study_uid}/{series_uid}")

        # Reset version_id to None
        series_manager.update_version(study_uid, series_uid, None, "system")
        print(f"[RESET] Reset version_id to None")

        # Clean up uploaded_masks directories for this series (atomic operation - only persist on retrack success)
        uploaded_masks_base = config.server_state_path / "uploaded_masks" / f"{study_uid}_{series_uid}"
        if uploaded_masks_base.exists():
            shutil.rmtree(uploaded_masks_base)
            print(f"[RESET] Removed uploaded_masks directory: {uploaded_masks_base}")

        # Reset tracking status to completed (from initial tracking)
        # Get mask count from initial tracking if available
        masks_dir = mask_dir / "masks"
        mask_count = 0
        if masks_dir.exists():
            mask_count = len(list(masks_dir.glob("*.webp")))
        
        # Regenerate masks.tar from initial tracking masks/ directory
        # (retrack may have overwritten it, so we need to rebuild it)
        if masks_dir.exists() and mask_count > 0:
            try:
                from lib.mask_archive import build_mask_archive, build_mask_metadata
                metadata = build_mask_metadata(series, masks_dir, config.flow_method)
                archive_path = mask_dir / "masks.tar"
                archive_bytes = build_mask_archive(masks_dir, metadata)
                with archive_path.open("wb") as f:
                    f.write(archive_bytes)
                print(f"[RESET] Regenerated masks.tar from initial tracking masks")
            except Exception as exc:
                print(f"[RESET] Warning: Failed to regenerate masks.tar: {exc}")
        
        series_manager.update_tracking_status(study_uid, series_uid, "completed", mask_count)
        print(f"[RESET] Reset tracking status to completed with {mask_count} masks")

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
        print(f"[RESET ALL] Removed {total_jobs_removed} retrack queue job(s) for all series")
        
        # Reset each series
        for series in all_series:
            try:
                mask_dir = _mask_series_dir(mask_root, flow_method, series.study_uid, series.series_uid)
                retrack_dir = mask_dir / "retrack"
                
                if retrack_dir.exists():
                    shutil.rmtree(retrack_dir)
                    print(f"[RESET ALL] Removed retrack directory: {retrack_dir}")
                
                # Reset version_id to None
                series_manager.update_version(series.study_uid, series.series_uid, None, "system")
                
                # Clean up uploaded_masks directories
                uploaded_masks_base = config.server_state_path / "uploaded_masks" / f"{series.study_uid}_{series.series_uid}"
                if uploaded_masks_base.exists():
                    shutil.rmtree(uploaded_masks_base)
                
                # Regenerate masks.tar from initial tracking masks/ directory
                masks_dir = mask_dir / "masks"
                mask_count = 0
                if masks_dir.exists():
                    mask_count = len(list(masks_dir.glob("*.webp")))
                
                if masks_dir.exists() and mask_count > 0:
                    try:
                        from lib.mask_archive import build_mask_archive, build_mask_metadata
                        metadata = build_mask_metadata(series, masks_dir, config.flow_method)
                        archive_path = mask_dir / "masks.tar"
                        archive_bytes = build_mask_archive(masks_dir, metadata)
                        with archive_path.open("wb") as f:
                            f.write(archive_bytes)
                    except Exception as exc:
                        print(f"[RESET ALL] Warning: Failed to regenerate masks.tar for {series.study_uid}/{series.series_uid}: {exc}")
                
                series_manager.update_tracking_status(series.study_uid, series.series_uid, "completed", mask_count)
                total_reset += 1
            except Exception as exc:
                print(f"[RESET ALL] Error resetting {series.study_uid}/{series.series_uid}: {exc}")
                import traceback
                traceback.print_exc()
        
        print(f"[RESET ALL] Reset retrack data for {total_reset}/{total_series} series")
        
        return jsonify({
            "success": True,
            "message": f"Reset retrack data for all series",
            "total_series": total_series,
            "reset_series": total_reset,
            "removed_jobs": total_jobs_removed,
        })

    return bp
