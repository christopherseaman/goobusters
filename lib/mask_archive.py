"""
Utilities for packaging and unpacking mask archives (.tgz) shared between
server and client components.
"""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MASK_METADATA_FILENAME = "metadata.json"


class MaskArchiveError(RuntimeError):
    """Raised when a mask archive cannot be created or extracted."""


def _add_bytes_as_file(
    tar: tarfile.TarFile, filename: str, payload: bytes
) -> None:
    info = tarfile.TarInfo(name=filename)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def build_mask_archive(
    mask_dir: Path, metadata: dict, include_metadata: bool = True
) -> bytes:
    """
    Package masks/metadata into a single .tar blob for transport.
    No gzip compression - WebP images are already compressed.
    """
    mask_dir = mask_dir.resolve()
    if not mask_dir.exists():
        raise MaskArchiveError(f"Mask directory does not exist: {mask_dir}")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        if include_metadata:
            metadata_bytes = json.dumps(
                metadata, indent=2, sort_keys=False
            ).encode("utf-8")
            _add_bytes_as_file(tar, MASK_METADATA_FILENAME, metadata_bytes)

        for path in sorted(mask_dir.glob("*.webp")):
            tar.add(path, arcname=path.name)

    buffer.seek(0)
    return buffer.read()


def _safe_members(members: Iterable[tarfile.TarInfo], destination: Path):
    destination = destination.resolve()
    for member in members:
        member_path = destination / member.name
        if not str(member_path.resolve()).startswith(str(destination)):
            raise MaskArchiveError(
                f"Unsafe path detected in archive: {member.name}"
            )
        yield member


def extract_mask_archive(archive_bytes: bytes, destination: Path) -> None:
    """
    Extract a received mask archive into `destination`, validating paths.
    Supports both .tar (no gzip) and .tar.gz (legacy) formats.
    """
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO(archive_bytes)
    # Try .tar first (no gzip), fall back to .tar.gz for backwards compatibility
    try:
        tar = tarfile.open(fileobj=buffer, mode="r")
    except tarfile.ReadError:
        buffer.seek(0)
        tar = tarfile.open(fileobj=buffer, mode="r:gz")
    
    with tar:
        tar.extractall(
            path=destination,
            members=_safe_members(tar.getmembers(), destination),
        )


def iso_now() -> str:
    """Return current UTC time as ISO format string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_mask_metadata(series, masks_path: Path, flow_method: str) -> dict:
    """
    Build metadata for mask archive per DISTRIBUTED_ARCHITECTURE.md spec.

    Returns metadata with 'frames' array. Each frame entry includes:
    - frame_number: Frame index
    - has_mask: Whether mask file exists
    - is_annotation: Whether this is an annotation frame (has label_id)
    - label_id: Label ID for annotation frames
    - filename: Name of mask file (.webp)

    Retracking uses frames where is_annotation=true, loading the actual mask files.
    Includes EMPTY_ID frames with has_mask=false per spec.
    """
    from lib.config import load_config
    import json

    config = load_config("server")
    frames_by_number: dict[int, dict] = {}

    # Prefer frametype.json if present (authoritative per-frame metadata)
    output_dir = masks_path.parent
    frametype_path = output_dir / "frametype.json"

    if frametype_path.exists():
        try:
            with frametype_path.open() as f:
                summary = json.load(f)

            if isinstance(summary, dict):
                for frame_key, info in summary.items():
                    if not isinstance(info, dict):
                        continue
                    try:
                        frame_num = int(frame_key)
                    except (TypeError, ValueError):
                        continue

                    has_mask = bool(info.get("has_mask", False))
                    is_annotation = bool(info.get("is_annotation", False))
                    label_id = info.get("label_id") or config.label_id
                    type_str = info.get("type", "tracked")
                    mask_file = info.get("mask_file")
                    if has_mask and not mask_file:
                        mask_file = f"frame_{frame_num:06d}_mask.webp"

                    frames_by_number[frame_num] = {
                        "frame_number": frame_num,
                        "has_mask": has_mask,
                        "is_annotation": is_annotation,
                        "label_id": label_id,
                        "type": type_str,
                        "filename": mask_file if has_mask else None,
                    }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # If frametype.json is unreadable, fall back to mask files + input_annotations
            frames_by_number = {}

    if not frames_by_number:
        # Fallback: infer from mask files and input_annotations.json (legacy path)
        # First, process existing mask files (tracked frames by default)
        mask_files = sorted(masks_path.glob("*.webp"))
        for file_path in mask_files:
            frame_number = None
            parts = file_path.stem.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                frame_number = int(parts[1])

            if frame_number is None:
                continue

            frames_by_number[frame_number] = {
                "frame_number": frame_number,
                "has_mask": True,
                "is_annotation": False,
                "label_id": config.label_id,
                "type": "tracked",
                "filename": file_path.name,
            }

        # Then refine using input_annotations.json for LABEL_ID / EMPTY_ID
        input_annotations_path = output_dir / "input_annotations.json"

        if input_annotations_path.exists():
            try:
                with input_annotations_path.open() as f:
                    input_data = json.load(f)

                if isinstance(input_data, str):
                    input_data = json.loads(input_data)

                if isinstance(input_data, dict):
                    for annotation in input_data.get("annotations", []):
                        if not isinstance(annotation, dict):
                            continue

                        label_id = annotation.get("labelId", "")
                        frame_number = annotation.get("frameNumber")
                        if frame_number is None:
                            continue
                        frame_num = int(frame_number)

                        # Human fluid annotation (LABEL_ID)
                        if label_id == config.label_id:
                            entry = frames_by_number.get(frame_num)
                            if entry:
                                entry["is_annotation"] = True
                                entry["label_id"] = config.label_id
                                entry["type"] = "fluid"
                            else:
                                frames_by_number[frame_num] = {
                                    "frame_number": frame_num,
                                    "has_mask": False,
                                    "is_annotation": True,
                                    "label_id": config.label_id,
                                    "type": "fluid",
                                    "filename": None,
                                }

                        # Human empty annotation (EMPTY_ID)
                        if label_id == config.empty_id:
                            frames_by_number[frame_num] = {
                                "frame_number": frame_num,
                                "has_mask": False,
                                "is_annotation": True,
                                "label_id": config.empty_id,
                                "type": "empty",
                                "filename": None,
                            }
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                # If we can't read annotations, continue with mask files only
                pass

    # Convert dict to sorted list
    frames = sorted(frames_by_number.values(), key=lambda x: x["frame_number"])

    # Count masks (frames with has_mask=True)
    mask_count = sum(1 for f in frames if f.get("has_mask", False))

    metadata = {
        "study_uid": series.study_uid,
        "series_uid": series.series_uid,
        "version_id": series.current_version_id,
        "flow_method": flow_method,
        "generated_at": iso_now(),
        "frame_count": len(frames),
        "mask_count": mask_count,
        "frames": frames,  # Single array - retracking filters by is_annotation=true
    }
    return metadata
