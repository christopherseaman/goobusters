#!/usr/bin/env python3
"""
End-to-end client endpoint test: GET /api/frames on the client backend must
resolve an orphan (zero-annotation) series and return server archive URLs
instead of 404. Drives list_local_series + _find_series_info via the real
Flask route.

Run: uv run python scripts/test_blank_series_client_frames.py
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

from _blank_series_fixtures import build_dataset_fixture, make_client_config

from lib.client.start import create_app

PROJECT_ID = "TESTPROJ"
DATASET_ID = "TESTDS"
LABEL_ID = "L_FLUID"
EMPTY_ID = "L_EMPTY"
METHOD = "dis"

ANNOTATED = ("1.study.annotated", "1.series.annotated")
ORPHAN = ("2.study.orphan", "2.series.orphan")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "client_cache"
        build_dataset_fixture(
            cache, PROJECT_ID, DATASET_ID,
            annotated=[ANNOTATED], orphans=[ORPHAN],
            label_id=LABEL_ID, empty_id=EMPTY_ID,
        )
        config = make_client_config(
            cache, project_id=PROJECT_ID, dataset_id=DATASET_ID,
            label_id=LABEL_ID, empty_id=EMPTY_ID,
        )

        app = create_app(config)
        client = app.test_client()

        # /api/videos lists the orphan.
        resp_videos = client.get("/api/videos")
        assert resp_videos.status_code == 200, (
            f"/api/videos returned {resp_videos.status_code}"
        )
        listed = {(v["study_uid"], v["series_uid"]) for v in resp_videos.get_json()}
        assert ORPHAN in listed, "orphan missing from /api/videos"

        # /api/frames resolves the orphan (not 404) and returns server URLs.
        resp = client.get(f"/api/frames/{METHOD}/{ORPHAN[0]}/{ORPHAN[1]}")
        assert resp.status_code == 200, (
            f"/api/frames returned {resp.status_code}: {resp.get_data()[:300]}"
        )
        body = json.loads(resp.get_data())
        assert ORPHAN[1] in body["frames_archive_url"], (
            f"frames_archive_url wrong: {body['frames_archive_url']}"
        )
        assert ORPHAN[1] in body["masks_archive_url"], (
            f"masks_archive_url wrong: {body['masks_archive_url']}"
        )

    print("PASS: client /api/videos + /api/frames resolve the orphan series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
