"\"\"\"Flask blueprint exposing the distributed tracking API.\"\"\""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from lib.mask_archive import build_mask_archive, MaskArchiveError
from server.storage.series_manager import SeriesManager, SeriesMetadata


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_series(metadata: SeriesMetadata) -> dict:
    payload = asdict(metadata)
    return payload


def _mask_series_dir(mask_root: Path, flow_method: str, study_uid: str, series_uid: str) -> Path:
    return mask_root / flow_method / f"{study_uid}_{series_uid}"


def _build_mask_metadata(series: SeriesMetadata, masks_path: Path, flow_method: str) -> dict:
    mask_files = sorted(masks_path.glob("*.webp"))
    frames = []
    for file_path in mask_files:
        frame_number = None
        parts = file_path.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            frame_number = int(parts[1])
        frames.append(
            {
                "frame_number": frame_number,
                "has_mask": True,
                "is_annotation": False,
                "filename": file_path.name,
            }
        )

    metadata = {
        "study_uid": series.study_uid,
        "series_uid": series.series_uid,
        "version_id": series.current_version_id,
        "flow_method": flow_method,
        "generated_at": iso_now(),
        "frame_count": len(frames),
        "mask_count": len(frames),
        "frames": frames,
    }
    return metadata


def create_api_blueprint(series_manager: SeriesManager, config) -> Blueprint:
    bp = Blueprint("distributed_api", __name__)

    mask_root = Path(config.mask_storage_path)
    flow_method = config.flow_method

    @bp.get("/api/status")
    def status() -> Response:
        series = series_manager.list_series()
        completed = sum(1 for item in series if item.status == "completed")
        failed = sum(1 for item in series if item.tracking_status == "failed")
        pending = sum(1 for item in series if item.tracking_status in {"never_run", "pending"})
        return jsonify(
            {
                "ready": True,
                "series_total": len(series),
                "series_completed": completed,
                "series_failed": failed,
                "series_pending": pending,
            }
        )

    @bp.get("/api/series")
    def list_series():
        records = [_serialize_series(item) for item in series_manager.list_series()]
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
            metadata = series_manager.mark_activity(study_uid, series_uid, user_email)
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

        mask_dir = _mask_series_dir(mask_root, flow_method, study_uid, series_uid)
        masks_path = mask_dir / "masks"
        if not masks_path.exists():
            return (
                jsonify({"status": "pending", "error_code": "TRACK_PENDING"}),
                202,
            )

        metadata = _build_mask_metadata(series, masks_path, flow_method)
        try:
            archive_bytes = build_mask_archive(masks_path, metadata)
        except MaskArchiveError as exc:
            return jsonify({"error": str(exc)}), 500

        headers = {
            "Content-Type": "application/x-tar+gzip",
            "X-Version-ID": metadata.get("version_id") or "",
            "X-Mask-Count": str(metadata.get("mask_count", 0)),
            "X-Flow-Method": flow_method,
            "X-Generated-At": metadata.get("generated_at", iso_now()),
        }
        return Response(archive_bytes, headers=headers)

    return bp
