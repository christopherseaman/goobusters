#!/usr/bin/env python3
"""
End-to-end Flask serving test for a synthesized blank series.

After synthesis, GET /api/masks must serve the blank archive (HTTP 200,
all-blank metadata) instead of TRACK_FAILED, and GET /api/frames_archive must
serve the extracted frames. Exercises the real route code via a Flask test
client with skip_startup (no MD.ai download, no worker spawn).

Run: uv run python scripts/test_blank_series_serving.py
"""

from __future__ import annotations

import io
import json
import tarfile
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

from lib.mask_archive import mask_series_dir
from lib.server.start import create_app
from lib.synthesize_blank_series import synthesize_blank_series

PROJECT_ID = "TESTPROJ"
DATASET_ID = "TESTDS"
LABEL_ID = "L_FLUID"
EMPTY_ID = "L_EMPTY"

ANNOTATED = ("1.study.annotated", "1.series.annotated")
ORPHAN = ("2.study.orphan", "2.series.orphan")


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

        # Synthesize the orphan's artifacts on disk (as the worker would).
        video_path = images_dir / ORPHAN[0] / f"{ORPHAN[1]}.mp4"
        series_out = mask_series_dir(
            Path(config.mask_storage_path), config.flow_method, *ORPHAN
        )
        synthesize_blank_series(*ORPHAN, video_path, series_out, config)

        app = create_app(config, skip_startup=True)
        client = app.test_client()

        # GET /api/masks must serve the blank archive (200), not TRACK_FAILED.
        resp = client.get(f"/api/masks/{ORPHAN[0]}/{ORPHAN[1]}")
        assert resp.status_code == 200, (
            f"/api/masks returned {resp.status_code}, body={resp.get_data()[:300]}"
        )
        assert resp.headers.get("X-Mask-Count") == "0", (
            f"X-Mask-Count {resp.headers.get('X-Mask-Count')} != 0"
        )
        with tarfile.open(fileobj=io.BytesIO(resp.get_data()), mode="r") as tar:
            meta = json.load(tar.extractfile("metadata.json"))
        assert meta["mask_count"] == 0, f"mask_count {meta['mask_count']} != 0"
        assert meta["frame_count"] >= 1, "metadata frame_count should be >= 1"
        assert all(f["has_mask"] is False for f in meta["frames"]), (
            "every served frame must be blank (has_mask=false)"
        )

        # GET /api/frames_archive must serve the extracted frames.
        resp_frames = client.get(f"/api/frames_archive/{ORPHAN[0]}/{ORPHAN[1]}.tar")
        assert resp_frames.status_code == 200, (
            f"/api/frames_archive returned {resp_frames.status_code}"
        )
        with tarfile.open(fileobj=io.BytesIO(resp_frames.get_data()), mode="r") as tar:
            webp = [m.name for m in tar.getmembers() if m.name.endswith(".webp")]
        assert len(webp) == meta["frame_count"], (
            f"frames.tar has {len(webp)} frames, metadata says {meta['frame_count']}"
        )

    print(f"PASS: blank series served via /api/masks and /api/frames_archive "
          f"({meta['frame_count']} frames, 0 masks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
