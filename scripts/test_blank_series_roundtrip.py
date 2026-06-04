#!/usr/bin/env python3
"""
Round-trip test for a blank series: a human marking frames "No Fluid" (empty)
must POST cleanly to /api/masks (no 404 now that the series is indexed) and
the empty annotation must re-ingest as labelId=empty_id, data=None. A save with
ZERO marked frames must be a graceful no-op, not an error.

Run: uv run python scripts/test_blank_series_roundtrip.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _blank_series_fixtures import build_dataset_fixture, make_config

from lib.mask_archive import build_mask_archive, mask_series_dir
from lib.server.start import create_app
from lib.synthesize_blank_series import synthesize_blank_series
from lib.uploaded_masks import convert_uploaded_masks_to_annotations_df

PROJECT_ID = "TESTPROJ"
DATASET_ID = "TESTDS"
LABEL_ID = "L_FLUID"
EMPTY_ID = "L_EMPTY"

ANNOTATED = ("1.study.annotated", "1.series.annotated")
ORPHAN = ("2.study.orphan", "2.series.orphan")


def _build_upload_tar(tmp: Path, study: str, series: str, empty_frames: list[int]) -> bytes:
    """Build an upload masks.tar mimicking the client marking frames empty."""
    masks_dir = tmp / "upload_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "study_uid": study,
        "series_uid": series,
        "frames": [
            {
                "frame_number": n,
                "has_mask": False,
                "is_annotation": True,
                "label_id": EMPTY_ID,
                "filename": None,
            }
            for n in empty_frames
        ],
    }
    return build_mask_archive(masks_dir, metadata)


def _build_lossy_upload(tmp: Path, study: str, series: str) -> bytes:
    """Upload claiming a fluid annotation but with its mask file MISSING.

    Re-ingest finds no usable annotation and raises; this is real data loss and
    must surface as an error, NOT be swallowed as a no-op success.
    """
    masks_dir = tmp / "lossy_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "study_uid": study,
        "series_uid": series,
        "frames": [
            {
                "frame_number": 2,
                "has_mask": True,
                "is_annotation": True,
                "label_id": LABEL_ID,
                "filename": "frame_000002_mask.webp",  # not included in tar
            }
        ],
    }
    return build_mask_archive(masks_dir, metadata)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        state_dir = tmp_path / "state"

        images_dir = build_dataset_fixture(
            data_dir, PROJECT_ID, DATASET_ID,
            annotated=[ANNOTATED], orphans=[ORPHAN],
            label_id=LABEL_ID, empty_id=EMPTY_ID,
        )
        config = make_config(
            data_dir, output_dir, state_dir,
            project_id=PROJECT_ID, dataset_id=DATASET_ID,
            label_id=LABEL_ID, empty_id=EMPTY_ID,
        )
        series_out = mask_series_dir(
            Path(config.mask_storage_path), config.flow_method, *ORPHAN
        )
        synthesize_blank_series(
            *ORPHAN, images_dir / ORPHAN[0] / f"{ORPHAN[1]}.mp4", series_out, config
        )

        app = create_app(config, skip_startup=True)
        client = app.test_client()

        # --- Empty annotation re-ingests as labelId=empty_id, data=None ---
        upload = _build_upload_tar(tmp_path, *ORPHAN, empty_frames=[1, 3])
        with tempfile.TemporaryDirectory() as ex:
            df, _ = convert_uploaded_masks_to_annotations_df(upload, Path(ex) / "x")
        assert list(df["labelId"]) == [EMPTY_ID, EMPTY_ID], (
            f"labelIds {list(df['labelId'])} != two empty_id"
        )
        assert df["data"].isna().all(), "empty frames must have data=None"

        # --- POST with ZERO marked frames: graceful no-op, not an error ---
        # (Done before the marked save below, which enqueues a retrack that
        # would 409 a subsequent POST.)
        empty_upload = _build_upload_tar(tmp_path, *ORPHAN, empty_frames=[])
        resp_noop = client.post(
            f"/api/masks/{ORPHAN[0]}/{ORPHAN[1]}",
            data=empty_upload,
            headers={"X-Editor": "tester@test", "Content-Type": "application/x-tar"},
        )
        assert resp_noop.status_code == 200, (
            f"zero-marked save returned {resp_noop.status_code}: {resp_noop.get_data()[:300]}"
        )
        body_noop = json.loads(resp_noop.get_data())
        assert body_noop.get("retrack_queued") is False, (
            "zero-marked save should NOT queue a retrack"
        )

        # --- POST claiming annotations but with missing mask: real error (400),
        # NOT a silent 200 no-op (would lose the user's work). ---
        lossy = _build_lossy_upload(tmp_path, *ORPHAN)
        resp_lossy = client.post(
            f"/api/masks/{ORPHAN[0]}/{ORPHAN[1]}",
            data=lossy,
            headers={"X-Editor": "tester@test", "Content-Type": "application/x-tar"},
        )
        assert resp_lossy.status_code == 400, (
            f"lossy upload (missing masks) should be 400, got {resp_lossy.status_code}: "
            f"{resp_lossy.get_data()[:300]}"
        )

        # --- POST with marked frames: accepted (not 404), retrack queued ---
        resp = client.post(
            f"/api/masks/{ORPHAN[0]}/{ORPHAN[1]}",
            data=upload,
            headers={"X-Editor": "tester@test", "Content-Type": "application/x-tar"},
        )
        assert resp.status_code == 200, (
            f"POST marked frames returned {resp.status_code}: {resp.get_data()[:300]}"
        )
        body = json.loads(resp.get_data())
        assert body.get("retrack_queued") is True, "retrack not queued for marked save"

    print("PASS: blank-series round-trip accepts empty annotations; zero-marked save is a no-op")
    return 0


if __name__ == "__main__":
    sys.exit(main())
