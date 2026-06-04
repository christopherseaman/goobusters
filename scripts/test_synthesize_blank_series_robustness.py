#!/usr/bin/env python3
"""
Robustness tests for synthesize_blank_series: a frame-write failure or a frame
count that hits the runaway cap must FAIL the job (raise), never silently
produce a "complete" series with missing frames. Also verifies atomic output
(no leftover temp files).

Run: uv run python scripts/test_synthesize_blank_series_robustness.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from _blank_series_fixtures import write_test_video
from lib.config import load_config

import lib.synthesize_blank_series as sbs
from lib.synthesize_blank_series import synthesize_blank_series

STUDY = "1.2.3.study"
SERIES = "1.2.3.series"


def _check_imwrite_failure_raises(config) -> None:
    """A failed frame write must raise, not silently mark the series complete."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "v.mp4"
        out = tmp_path / "out"
        write_test_video(video, n_frames=5)

        original = cv2.imwrite
        cv2.imwrite = lambda *a, **k: False  # simulate disk-full / encode failure
        try:
            raised = False
            try:
                synthesize_blank_series(STUDY, SERIES, video, out, config)
            except (IOError, OSError):
                raised = True
            assert raised, "imwrite failure must raise, not silently succeed"
        finally:
            cv2.imwrite = original

        # No completion marker left behind -> job will be retried.
        assert not (out / "masks.tar").exists(), (
            "masks.tar must NOT exist after a failed synthesis"
        )


def _check_max_frames_cap_raises(config) -> None:
    """Hitting the runaway frame cap must raise, not ship a truncated series."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "v.mp4"
        out = tmp_path / "out"
        write_test_video(video, n_frames=8)

        original_cap = sbs._MAX_FRAMES
        sbs._MAX_FRAMES = 3  # video has more frames than this
        try:
            raised = False
            try:
                synthesize_blank_series(STUDY, SERIES, video, out, config)
            except ValueError:
                raised = True
            assert raised, "exceeding _MAX_FRAMES must raise"
        finally:
            sbs._MAX_FRAMES = original_cap

        assert not (out / "masks.tar").exists(), (
            "masks.tar must NOT exist after a capped/failed synthesis"
        )


def _check_no_temp_files_on_success(config) -> None:
    """Successful synthesis leaves valid archives and no leftover temp files."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "v.mp4"
        out = tmp_path / "out"
        write_test_video(video, n_frames=6)

        synthesize_blank_series(STUDY, SERIES, video, out, config)

        assert (out / "masks.tar").exists() and (out / "frames.tar").exists()
        leftovers = [p.name for p in out.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], f"leftover temp files: {leftovers}"


def main() -> int:
    config = load_config("server")
    _check_imwrite_failure_raises(config)
    _check_max_frames_cap_raises(config)
    _check_no_temp_files_on_success(config)
    print("PASS: synthesis fails loudly on write/cap errors; leaves no partial marker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
