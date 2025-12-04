"""
Utilities for packaging and unpacking mask archives (.tgz) shared between
server and client components.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Iterable

MASK_METADATA_FILENAME = "metadata.json"


class MaskArchiveError(RuntimeError):
    """Raised when a mask archive cannot be created or extracted."""


def _add_bytes_as_file(tar: tarfile.TarFile, filename: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=filename)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def build_mask_archive(mask_dir: Path, metadata: dict, include_metadata: bool = True) -> bytes:
    """
    Package masks/metadata into a single .tgz blob for transport.
    """
    mask_dir = mask_dir.resolve()
    if not mask_dir.exists():
        raise MaskArchiveError(f"Mask directory does not exist: {mask_dir}")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        if include_metadata:
            metadata_bytes = json.dumps(metadata, indent=2, sort_keys=False).encode("utf-8")
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
            raise MaskArchiveError(f"Unsafe path detected in archive: {member.name}")
        yield member


def extract_mask_archive(archive_bytes: bytes, destination: Path) -> None:
    """
    Extract a received mask archive into `destination`, validating paths.
    """
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
        tar.extractall(path=destination, members=_safe_members(tar.getmembers(), destination))
