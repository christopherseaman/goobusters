# Task List (Prioritized)

## High Priority (Core Workflow)

### 1. Test save + retrack end-to-end ✅ (Implementation complete, ready for manual testing)
- ✅ Conflict handling UI implemented (modal with reset/reload flow)
- ✅ Save + retrack UX wired (save button, status polling, auto-reload)
- ⏳ **Manual testing needed**: See `TESTING_GUIDE.md` for test steps
- ⏳ Verify viewer → client → server → retrack worker → viewer reload
- ⏳ Test conflict handling (409 responses) in real multi-user scenario

### 2. Wire viewer to use `/api/series/next` (Task 6) ✅
- ✅ Replace local `videoSelect` with server-driven selection
- ✅ Wire checkmark button to mark complete + get next series
- ✅ Remove Prev button (server-driven navigation doesn't support going back)
- ✅ Series selection prefers same series if user was recently active (refresh returns same series)

### 3. Activity pings (Task 7) ✅
- ✅ Send `POST /api/series/{study}/{series}/activity` every 30s while viewing
- ⏳ Surface "recent activity" in info modal

## Medium Priority (UX Polish)

### 4. Consolidate save & retrack functionality ✅
- ✅ Removed redundant retrack button (save automatically triggers retrack)
- ✅ Removed `retrackVideo()` function (now just calls `saveChanges()`)
- ✅ Updated save button tooltip to indicate it triggers retrack
- ✅ Flow is now: Save → Client builds .tar → POST to server → Server enqueues retrack → Viewer polls status

### 5. Viewer UI improvements for multiplayer workflow
- ✅ Remove Prev button (server-driven navigation doesn't support going back)
- ✅ Spacebar restarts play from beginning if at end of video
- ⏳ Scrubbing improvements:
  - Larger click target (as tall as circular scrub location indicator)
  - Scrub line indicators for frame annotation type:
    - Bar behind scrub line divided into sections by frame #
    - Colored by annotation type:
      - `track_id` (transparent)
      - `label_id` (green, 50% alpha)
      - `empty_id` (red, same alpha as label_id)
- ⏳ Empty_id annotated frames visual indicator:
  - Red outline similar to current tool's green rounded square outline
  - Large red drop-shadow pulse (similar to hovering clear frame button)

### 10. Activity warnings and recent activity display
- ⏳ Add activity metadata to all API responses:
  - Add `X-Last-Activity-At` and `X-Last-Activity-By` headers to `/api/masks/{study}/{series}` responses
  - Include in `/api/frames/{study}/{series}` responses (via client proxy)
  - This ensures client gets activity updates within keepalive period (30s)
- ⏳ Warning banner (top-left third, prominent but not modal):
  - Show when `last_activity_at` is within threshold (configurable, default 60min) AND `last_activity_by` != current user
  - Update on every API response (masks/frames)
  - Similar styling to dataset warning banner
- ⏳ Info modal activity display:
  - Show recent activity history (already available via `/api/series/{study}/{series}`)
  - Display who was active and when
  - Show "Last active by X at Y" prominently

### 6. Local edit cache (Task 8)
- Persist edits in LocalStorage + filesystem
- Restore on reload with unsaved indicator

### 7. Conflict handling refinement (Task 10)
- Test 409 `VERSION_MISMATCH` in real multi-user scenario
- Add clear reset/reload flow in viewer

## Lower Priority (Testing/Ops)

### 8. Fresh startup verification (Task 5)
- Script to wipe `output/` and `server_state/`, verify clean startup

### 9. Test suite refresh (Task 13)
- Update tests to `.tar` + `metadata.json` contract

---

## Original Detailed Tasks (Reference)

### Server (S)

1. Series navigation & status
   - Implement `/api/series`, `/api/series/next`, `/api/series/{study_uid}/{series_uid}`
   - Implement `/api/series/{study_uid}/{series_uid}/complete` and `/reopen`
   - Wire to `SeriesManager` and `server_state` layout per spec

2. Activity & "next" selection
   - Implement `/api/series/{study_uid}/{series_uid}/activity`
   - Persist recent activity and implement smart "next" selection heuristics
   - Surface activity metadata in `series_detail`

3. Identification (not auth)
   - Decide and plumb a simple identity header (e.g. `X-User-Email`) from client to server
   - Store last editor / activity user in `SeriesManager` metadata and activity logs
   - Do **not** require MD.ai token on server endpoints (MD.ai auth is client→MD.ai only)

4. Mask POST + retrack hardening
   - Finish `POST /api/masks/{study_uid}/{series_uid}` behavior:
     - Validate `.tar` payload and `metadata.json` shape
     - Ensure correct error codes (VERSION_MISMATCH, RETRACK_IN_PROGRESS, TRACK_FAILED)
     - Document headers and responses
   - Confirm retrack worker consumes uploaded archives exactly like initial tracking (frametype/masks.json, masks/, masks.tar)

5. Fresh startup from clean state
   - Script + docs to wipe `output/` and `server_state/`, then run full startup
   - Verify all trackable series end with `masks.tar` + correct metadata, and non-trackable are marked failed

### Client (C)

6. Viewer integration with navigation API
   - Use `/api/series/next` and `/api/series/{...}` for selection instead of purely local `videoSelect`
   - Wire the Next/Prev video UX to the server's selection logic

7. Activity pings
   - Send `POST /api/series/{study_uid}/{series_uid}/activity` every 30s while viewing
   - Surface "recent activity" info in the info modal

8. Local edit cache
   - Persist edits in LocalStorage + filesystem
   - Restore on reload and show unsaved indicator in the viewer

9. Save + retrack UX
   - Wire viewer save button to package edits into `.tar` and POST `/api/masks/{study_uid}/{series_uid}` with `X-Previous-Version-ID`
   - Show a clear "Retracking…" state and poll `/api/retrack/status/{study_uid}/{series_uid}` until complete
   - Refresh masks when retrack completes and update local version ID

10. Conflict + completion UX
    - Handle 409 `VERSION_MISMATCH` from `POST /api/masks` with a clear reset/reload flow
    - Add "Mark done" / "Reopen" UI wired to server completion endpoints, with a confirmation modal before marking complete

11. Dataset sync UX
    - Expose an explicit "Sync dataset" action in the client wrapper that calls `/api/dataset/sync`
    - Refresh local series list and hide version-warning banner when sync completes

### Shared / Testing (SH)

12. Dataset version negotiation v2
    - Optionally auto-trigger client sync when `/api/dataset/version` changes (behind a config flag)

13. Refresh full-system tests
    - Update `test_startup_verification.py`, `test_lazy_tracking.py`, `test_server_api.py` to the `.tar` + `metadata.json` contract and new lazy-tracking behavior
    - Make `scripts/test_full_system.py` green again as a full regression gate
