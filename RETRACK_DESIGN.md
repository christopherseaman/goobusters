# Local Annotation Feedback Loop - Design Plan

## Current State Analysis

**What exists:**
- [app.py](app.py) - Flask server that saves modified annotations to JSON + WebP files
- `/api/save_changes` - Saves all `label_id` and `empty_id` annotations to:
  - `annotations/{study_uid}_{series_uid}/modified_annotations.json` (metadata)
  - `annotations/{study_uid}_{series_uid}/masks/frame_XXXXXX_mask.webp` (masks)
  - `annotations/{study_uid}_{series_uid}/masks.tar.gz` (archive)
- [track.py](track.py) - Reads annotations from MD.ai JSON, generates TRACK_ID predictions
- [lib/multi_frame_tracker.py](lib/multi_frame_tracker.py) - Core tracking logic (`MultiFrameTracker`, `process_video_with_multi_frame_tracking`)
- [lib/opticalflowprocessor.py](lib/opticalflowprocessor.py) - Optical flow algorithms

**What's missing:**
- Function to convert local annotations (JSON + WebP) to MD.ai JSON format
- Re-tracking endpoint/mechanism that uses ONLY `label_id` and `empty_id` from `annotations/` (no MD.ai data, no `output/` data)
- Direct use of `lib/` functions without calling `track.py` (which serves a different purpose)
- UI button to trigger re-tracking

## Architecture Design

### 1. Data Flow

```
Initial Run:
MD.ai JSON → track.py → output/{method}/{study_uid}_{series_uid}/ → app.py displays

Iteration Loop:
app.py (user edits) → annotations/{study_uid}_{series_uid}/ (ONLY label_id + empty_id)
                    ↓
              Convert to MD.ai JSON format:
              1. Read modified_annotations.json (ONLY label_id and empty_id frames)
              2. Load WebP masks from annotations/masks/
              3. Convert masks to polygons (for fluid) or null (for empty)
              4. Generate MD.ai-compliant JSON
                    ↓
              Use lib/ functions directly (NOT track.py):
              1. Create OpticalFlowProcessor
              2. Call process_video_with_multi_frame_tracking() with local JSON
              3. Save tracked masks (TRACK_ID) to annotations/ or output/
                    ↓
              app.py displays updated results
```

**Key Change:** Retrack uses ONLY `label_id` and `empty_id` annotations from `annotations/`, not from `output/`. This ensures retracking is based solely on human-verified annotations.

### 2. Required Components

**A. New lib module: `lib/local_annotations.py`**

- `convert_local_to_mdai_json(study_uid, series_uid)` → MD.ai JSON file path
  - Reads `annotations/{study_uid}_{series_uid}/modified_annotations.json`
  - **ONLY processes frames with `label_id` or `empty_id`** (human-verified annotations)
  - Loads WebP masks from `annotations/{study_uid}_{series_uid}/masks/`
  - For `label_id` frames: Convert mask to polygons using `cv2.findContours()`
  - For `empty_id` frames: Use `"data": null` (MD.ai format for empty/"No Fluid" frames)
  - Generates MD.ai-compliant JSON with format:
    ```json
    {
      "datasets": [{
        "annotations": [
          {
            "labelId": "...",  // LABEL_ID or EMPTY_ID
            "StudyInstanceUID": "...",
            "SeriesInstanceUID": "...",
            "frameNumber": frame_num,  // 0-based (MD.ai uses 0-based)
            "data": {
              "foreground": [[x1, y1], [x2, y2], ...]  // For fluid frames (polygon vertices)
            },
            // OR for empty frames:
            "data": null  // MD.ai uses null for verified empty frames
          }
        ]
      }]
    }
    ```
  - Saves to temporary file, returns path
  - **KISS/DRY:** Reuses existing mask→polygon conversion logic from tracking

**B. Direct use of `lib/` functions (NOT `track.py`):**

- `track.py` serves a different purpose (batch processing from MD.ai)
- Retrack should use `lib/` functions directly:
  - `lib/opticalflowprocessor.OpticalFlowProcessor` - Create flow processor
  - `lib/multi_frame_tracker.process_video_with_multi_frame_tracking()` - Main tracking function
  - No MD.ai client needed, no data directory needed
  - Input: Local MD.ai JSON (from step A), video path, study/series UIDs
  - Output: Tracked masks (TRACK_ID) - save to `annotations/` or `output/`

**C. New app.py endpoint: `/api/retrack`**
```python
@app.route('/api/retrack/<study_uid>/<series_uid>', methods=['POST'])
def retrack_video(study_uid, series_uid):
    # 1. Convert local annotations (ONLY label_id + empty_id) to MD.ai JSON format
    # 2. Get video path from output/ or annotations/
    # 3. Create OpticalFlowProcessor with method from request/env
    # 4. Call process_video_with_multi_frame_tracking() directly:
    #    - Local JSON file (not MD.ai data)
    #    - Video path
    #    - Study/series UIDs
    #    - Output to annotations/ or output/ (TBD)
    # 5. Return status/progress
    # 6. On completion, reload video data in frontend
```

**D. Future Consideration: Save tracked masks to `annotations/`**

- **Option A (current):** Save tracked masks to `output/{method}/{study_uid}_{series_uid}/`
  - Pros: Keeps annotations/ clean (only human-verified)
  - Cons: Requires output/ directory, method-specific paths

- **Option B (future):** Save tracked masks to `annotations/{study_uid}_{series_uid}/`
  - Pros: All annotations in one place, cleaner structure
  - Cons: Requires updating `app.py` and `viewer.js` to:
    - Load `masks.json` from `annotations/` (not just `output/`)
    - Display TRACK_ID annotations as orange (tracked), not green (human-verified)
    - Handle mixed annotation types (LABEL_ID=green, EMPTY_ID=empty, TRACK_ID=orange)
    - Update `masks.json` structure to include `type` field (annotation vs tracked)

**D. UI Changes:**
- Add "Re-track" button (⚡) to top-right control buttons
- Show progress/status during re-tracking
- Reload video data after completion

### 3. Implementation Strategy

**Phase 1: Data Conversion** (lib/local_annotations.py)

- Read `annotations/{study_uid}_{series_uid}/modified_annotations.json`
- **ONLY process frames with `label_id` or `empty_id`** (human-verified annotations)
- For each frame with annotation:
  - Load mask WebP from `annotations/{study_uid}_{series_uid}/masks/`
  - If `is_empty: true`: Use `"data": null` (MD.ai format for verified empty)
  - If `is_empty: false`: Convert mask to polygons using `cv2.findContours()`
- Generate MD.ai-compliant JSON (0-based frame numbers)
- Save to temp file, return path

**Phase 2: Direct lib/ Integration**

- **DO NOT modify track.py** - it serves a different purpose (batch processing from MD.ai)
- Use `lib/` functions directly in `app.py`:
  - Import `OpticalFlowProcessor` from `lib/opticalflowprocessor`
  - Import `process_video_with_multi_frame_tracking` from `lib/multi_frame_tracker`
  - Create flow processor with method from request/env
  - Call `process_video_with_multi_frame_tracking()` with:
    - Local MD.ai JSON (from Phase 1)
    - Video path (from `output/` or `annotations/`)
    - Study/series UIDs
    - Output directory (TBD: `annotations/` or `output/`)
  - No MD.ai client, no data directory needed

**Phase 3: UI/API Layer**

- Add `/api/retrack` endpoint to app.py
- Add re-track button (⚡) to viewer UI
- Handle async/progress feedback
- Reload video data after completion

### 4. Key Considerations

**Annotation Source: ONLY `annotations/`**

- **Retrack uses ONLY `label_id` and `empty_id` from `annotations/`**
- **DO NOT use `output/` annotations** - those may include TRACK_ID (machine-generated)
- This ensures retracking is based solely on human-verified annotations
- Structure: `annotations/{study_uid}_{series_uid}/` (no `{method}` subdirectory)

**Annotation Format Consistency:**

- **Local storage (app.py):**
  - `annotations/{study_uid}_{series_uid}/modified_annotations.json` - Metadata: `{frame_key: {label_id, is_empty, modified_at}}`
  - `annotations/{study_uid}_{series_uid}/masks/frame_XXXXXX_mask.webp` - WebP masks
  - `annotations/{study_uid}_{series_uid}/masks.tar.gz` - Archive
- **MD.ai format (for lib/ functions):**
  - JSON with `annotations` array
  - Each annotation: `{labelId, frameNumber, data: {foreground: polygons}}` or `{labelId, frameNumber, data: null}`
  - Empty frames: `"data": null` (MD.ai convention for verified empty)

**Solution:** Convert local annotations (JSON + WebP) to MD.ai JSON format before retracking. This allows `lib/multi_frame_tracker.py` to work without MD.ai data directory. The conversion happens once per retrack, then lib functions process it normally.

**Empty ID Compliance:**
- Empty frames must use `"data": null` (not `{"foreground": []}`) for MD.ai compatibility
- Verified in actual MD.ai annotation files: empty/"No Fluid" frames have `"data": null`
- See TODO.md item 6-8 for validation requirements
- See `EMPTY_ID_FORMAT.md` for detailed format specification

**Performance:**
- Re-tracking single video should be fast (seconds to minutes)
- Run in background thread/subprocess to avoid blocking Flask
- Return immediately with task ID, poll for status

**Data Integrity:**

- `annotations/` directory is preserved during retrack (never cleared)
- `output/{method}/{study_uid}_{series_uid}/` may be cleared before retrack (TBD)
- Clear distinction: LABEL_ID (human-verified fluid), EMPTY_ID (human-verified empty), TRACK_ID (machine-tracked)
- Version tracking: `modified_at` timestamps in JSON metadata

**Tracked Masks Storage (Future Consideration):**

- **Current approach:** Save tracked masks to `output/{method}/{study_uid}_{series_uid}/`
- **Future option:** Save tracked masks to `annotations/{study_uid}_{series_uid}/`
  - Would require updating `app.py` and `viewer.js` to:
    - Load `masks.json` from `annotations/` (not just `output/`)
    - Display TRACK_ID annotations as orange (tracked), not green (human-verified)
    - Handle mixed annotation types in `masks.json`:
      - `type: "fluid"` + `is_annotation: true` → green (LABEL_ID, human)
      - `type: "empty"` + `is_annotation: true` → empty (EMPTY_ID, human)
      - `type: "fluid_forward"` or `"fluid_backward"` + `is_annotation: false` → orange (TRACK_ID, machine)

## Questions for User:

1. ~~**Annotation format:** Should we convert stored masks back to MD.ai polygon format, or have tracking read masks directly?~~ ✅ **RESOLVED:** Convert local annotations to MD.ai JSON format for retrack

2. **Re-tracking scope:** Re-track single video only, or batch re-track all videos with local modifications? ✅ **RESOLVED:** Single video only (KISS)

3. **Progress feedback:** Simple spinner, or detailed progress (frame N of M)?

4. **Method selection:** Re-track with same method, or allow choosing different method (e.g., switch from farneback to raft)? ✅ **RESOLVED:** Use same method (simpler)

## Implementation Phases

### Phase 1: Data Conversion (lib/local_annotations.py)

- [ ] Create `lib/local_annotations.py` module
- [ ] Implement `convert_local_to_mdai_json(study_uid, series_uid)`:
  - Load `annotations/{study_uid}_{series_uid}/modified_annotations.json`
  - **ONLY process frames with `label_id` or `empty_id`** (human-verified annotations)
  - For each frame with annotation:
    - Load mask WebP from `annotations/{study_uid}_{series_uid}/masks/`
    - If `is_empty: true`: Use `"data": null` (MD.ai format for verified empty frames)
    - If `is_empty: false`: Convert mask to polygons using `cv2.findContours()`
  - Generate MD.ai-compliant JSON structure (0-based frame numbers)
  - Save to temp file, return path
- [ ] Validate empty_id format matches MD.ai requirements: `"data": null` (verified in actual MD.ai files, TODO.md item 6-8, EMPTY_ID_FORMAT.md)
- [ ] Test conversion with existing modified annotations

### Phase 2: Direct lib/ Integration

- [ ] **DO NOT modify track.py** - it serves a different purpose
- [ ] In `app.py`, import lib functions directly:
  - `from lib.opticalflowprocessor import OpticalFlowProcessor`
  - `from lib.multi_frame_tracker import process_video_with_multi_frame_tracking`
- [ ] Create flow processor with method from request/env
- [ ] Call `process_video_with_multi_frame_tracking()` with:
  - Local MD.ai JSON (from Phase 1)
  - Video path (from `output/` or `annotations/`)
  - Study/series UIDs
  - Output directory (TBD: `annotations/` or `output/`)
- [ ] Test single video re-tracking with local annotations only
- [ ] Validate output format (masks, video, metadata)

### Phase 3: UI/API Layer

- [ ] Add `/api/retrack/<study_uid>/<series_uid>` endpoint to app.py:
  - Convert local annotations (ONLY label_id + empty_id) to MD.ai JSON
  - Get video path from `output/` or `annotations/`
  - Create OpticalFlowProcessor with method from request/env
  - Call `process_video_with_multi_frame_tracking()` directly (not track.py)
  - Save tracked masks to `output/` or `annotations/` (TBD)
  - Return status/progress
- [ ] Add re-track button (⚡) to viewer UI top-right controls
- [ ] Implement progress feedback (simple spinner or status polling)
- [ ] Reload video data after retrack completion
- [ ] Test end-to-end flow: modify mask → save → re-track → view updated results

### Phase 4: Validation

- [ ] Validate empty_id annotations are MD.ai compliant (TODO.md item 6-8, EMPTY_ID_FORMAT.md)
- [ ] Ensure annotations/ directory is never cleared during retrack
- [ ] Test that retrack works without MD.ai data directory
- [ ] Verify retrack uses ONLY label_id and empty_id from annotations/ (not output/)

### Phase 5: Future Enhancement (Optional)

- [ ] Consider saving tracked masks to `annotations/` instead of `output/`
- [ ] Update `app.py` to load `masks.json` from `annotations/` (if exists)
- [ ] Update `viewer.js` to display TRACK_ID annotations as orange (tracked)
- [ ] Update `masks.json` structure to include `type` and `is_annotation` fields
- [ ] Handle mixed annotation types (LABEL_ID=green, EMPTY_ID=empty, TRACK_ID=orange)
