#!/usr/bin/env python3
"""
Test for synthesize_blank_series: materializing openable artifacts for a
series that has a video on disk but ZERO annotations.

Uses a real cv2-written mp4 (no mocks) and pins the OUTPUT shape:
frames.tar, masks.tar (all-blank metadata), and frametype.json must agree
on frame count, and every frame must be empty (has_mask=false, empty label).

Run: uv run python scripts/test_synthesize_blank_series.py
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lib_server = os.path.join(_project_root, "lib", "server")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _lib_server not in sys.path:
    sys.path.insert(0, _lib_server)

import cv2
import numpy as np

from lib.config import load_config


def _write_test_video(path: Path, n_frames: int, size: int = 64) -> None:
    """Write a real mp4 with n_frames distinct frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    for i in range(n_frames):
        frame = np.full((size, size, 3), (i * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _decodable_frame_count(path: Path) -> int:
    """Count frames cv2 can actually read back (codec-truth, not metadata)."""
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def _tar_members(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, "r") as tar:
        return [m.name for m in tar.getmembers() if m.isfile()]


def _metadata_from_tar(tar_path: Path) -> dict:
    with tarfile.open(tar_path, "r") as tar:
        member = tar.extractfile("metadata.json")
        assert member is not None, "masks.tar missing metadata.json"
        return json.loads(member.read().decode("utf-8"))


def main() -> int:
    from lib.synthesize_blank_series import synthesize_blank_series

    config = load_config("server")
    study_uid = "1.2.3.test.study"
    series_uid = "1.2.3.test.series"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "video.mp4"
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        _write_test_video(video_path, n_frames=8)
        expected_n = _decodable_frame_count(video_path)
        assert expected_n >= 1, "fixture video produced no decodable frames"

        synthesize_blank_series(
            study_uid, series_uid, video_path, output_dir, config
        )

        # --- frametype.json: N entries, all blank, empty label ---
        frametype = json.loads((output_dir / "frametype.json").read_text())
        frame_keys = [k for k in frametype if k != "_version_id"]
        assert len(frame_keys) == expected_n, (
            f"frametype.json has {len(frame_keys)} frames, expected {expected_n}"
        )
        for k in frame_keys:
            info = frametype[k]
            assert info["has_mask"] is False, f"frame {k} has_mask should be False"
            assert info["is_annotation"] is False, f"frame {k} is_annotation should be False"
            assert info["label_id"] == config.empty_id, (
                f"frame {k} label_id {info['label_id']} != empty_id {config.empty_id}"
            )

        # --- frames.tar: one webp per decodable frame ---
        frame_members = _tar_members(output_dir / "frames.tar")
        webp_members = [m for m in frame_members if m.endswith(".webp")]
        assert len(webp_members) == expected_n, (
            f"frames.tar has {len(webp_members)} webp frames, expected {expected_n}"
        )
        assert all(m.startswith("frames/frame_") for m in webp_members), (
            f"frames.tar members not under frames/: {webp_members[:3]}"
        )

        # --- masks.tar: all-blank metadata, zero masks ---
        metadata = _metadata_from_tar(output_dir / "masks.tar")
        assert metadata["frame_count"] == expected_n, (
            f"metadata frame_count {metadata['frame_count']} != {expected_n}"
        )
        assert metadata["mask_count"] == 0, (
            f"metadata mask_count {metadata['mask_count']} != 0"
        )
        assert len(metadata["frames"]) == expected_n, "metadata frames[] length mismatch"
        for f in metadata["frames"]:
            assert f["has_mask"] is False, f"frame {f['frame_number']} has_mask should be False"
            assert f["filename"] is None, f"frame {f['frame_number']} filename should be None"

        # --- masks/ dir exists and holds zero mask files ---
        masks_dir = output_dir / "masks"
        assert masks_dir.is_dir(), "masks/ directory was not created"
        assert list(masks_dir.glob("*.webp")) == [], "masks/ should contain no mask files"

    print(f"PASS: synthesize_blank_series produced {expected_n} blank frames consistently")
    return 0


if __name__ == "__main__":
    sys.exit(main())
