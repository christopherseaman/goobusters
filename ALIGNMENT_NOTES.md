# Code Alignment Notes

## Current State: DRY/KISS Alignment

### ✅ Shared Libraries (lib/)

**`lib/mask_archive.py`**:
- `build_mask_archive()`: Creates `.tar` (no gzip) from `mask_dir` + `metadata` dict
- `extract_mask_archive()`: Extracts `.tar` or `.tar.gz` with path validation
- `build_mask_metadata()`: Builds `metadata.json` with `frames[]` array from `frametype.json` or infers from masks
- `iso_now()`: UTC ISO timestamp helper

**`lib/mask_utils.py`** (NEW):
- `decode_base64_mask_to_webp()`: Shared base64→WebP conversion (used by client, can be used by app.py)

**`lib/uploaded_masks.py`**:
- `convert_uploaded_masks_to_annotations_df()`: Converts uploaded `.tar` archives to annotations DataFrame for retracking

**`lib/multi_frame_tracker.py`**:
- Core tracking logic used by both `track.py` and server workers

### ✅ Client (`client/client.py`)

**Uses shared libraries**:
- ✅ `lib.mask_archive.build_mask_archive()` for creating `.tar` archives
- ✅ `lib.mask_archive.iso_now()` for timestamps
- ✅ `lib.mask_utils.decode_base64_mask_to_webp()` for base64→WebP conversion
- ✅ `lib.uploaded_masks.convert_uploaded_masks_to_annotations_df()` (via server)

**Format**:
- Accepts JSON with `modified_frames` dict (base64 PNG data URLs)
- Builds `metadata.json` with `frames[]` array (distributed format)
- Creates `.tar` (no gzip) using shared helper
- POSTs to server `/api/masks/{study}/{series}` with `X-Previous-Version-ID`

### ✅ Server (`server/api/routes.py`, `server/retrack_worker.py`)

**Uses shared libraries**:
- ✅ `lib.mask_archive.build_mask_archive()` for creating `.tar` archives
- ✅ `lib.mask_archive.build_mask_metadata()` for metadata from `frametype.json`
- ✅ `lib.uploaded_masks.convert_uploaded_masks_to_annotations_df()` for processing uploaded archives
- ✅ `lib.multi_frame_tracker.process_video_with_multi_frame_tracking()` for retracking

**Format**:
- Serves `.tar` (no gzip) with `metadata.json` containing `frames[]` array
- Accepts `.tar` archives with same format
- Retrack worker uses same tracking pipeline as initial tracking

### ⚠️ Legacy (`app.py`)

**Status**: Legacy code, kept for testing/reference. **Not aligned with distributed architecture.**

**Differences**:
- Uses `masks.tar.gz` (gzipped) instead of `.tar` (no gzip)
- Builds `masks.json` (MD.ai format) instead of `metadata.json` with `frames[]` array
- Does not use `lib.mask_archive.build_mask_archive()` (hand-rolls tarfile)
- Base64→WebP conversion duplicated (can be migrated to `lib.mask_utils.decode_base64_mask_to_webp()`)

**Recommendation**: Leave as-is for now. When ready to deprecate, migrate to use shared helpers.

## Next Steps Priority

Based on `TMP_TASKS.md` and current state:

### High Priority (Core Functionality)

1. **Save + retrack UX** (Task 9) - ✅ **DONE**
   - Viewer → client backend → server → retrack worker all wired
   - Uses shared `lib.mask_archive` throughout
   - Polling and reload on completion implemented

2. **Conflict + completion UX** (Task 10) - ⚠️ **PARTIAL**
   - ✅ Mark complete with confirmation modal implemented
   - ⚠️ 409 `VERSION_MISMATCH` handling needs testing
   - ⚠️ Reopen API exists but UI not wired (deferred per user request)

3. **Series navigation & status** (Task 1) - ⚠️ **PARTIAL**
   - ✅ `/api/series`, `/api/series/next`, `/api/series/{study}/{series}` exist
   - ✅ `/api/series/{study}/{series}/complete` exists
   - ⚠️ Viewer not yet using `/api/series/next` for selection (still local `videoSelect`)

### Medium Priority (Multi-user Features)

4. **Activity & "next" selection** (Task 2) - ⚠️ **PARTIAL**
   - ✅ `/api/series/{study}/{series}/activity` exists
   - ⚠️ Smart "next" selection heuristics not fully implemented
   - ⚠️ Activity pings not sent from viewer (Task 7)

5. **Activity pings** (Task 7) - ❌ **NOT DONE**
   - Need to send `POST /api/series/{study}/{series}/activity` every 30s while viewing
   - Surface "recent activity" in info modal

### Lower Priority (Polish)

6. **Local edit cache** (Task 8) - ❌ **NOT DONE**
   - Persist edits in LocalStorage + filesystem
   - Restore on reload

7. **Dataset sync UX** (Task 11) - ✅ **DONE**
   - Blocking `/api/dataset/sync` with frame extraction implemented

8. **Fresh startup from clean state** (Task 5) - ❌ **NOT DONE**
   - Script to wipe `output/` and `server_state/`, verify clean startup

## Recommendations

1. **Test save + retrack end-to-end** to ensure no regressions
2. **Wire viewer to use `/api/series/next`** instead of local selection (Task 6)
3. **Add activity pings** from viewer (Task 7) - simple 30s interval
4. **Test conflict handling** (409 responses) in real multi-user scenario
5. **Add LocalStorage edit cache** (Task 8) for better UX on reload

