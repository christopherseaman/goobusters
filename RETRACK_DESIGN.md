# Local Annotation Feedback Loop - Design Plan

## Current State Analysis

**What exists:**
- [app.py](app.py) - Flask server that saves modified annotations to JSON + PNG files
- `/api/save_changes` - Saves all `label_id` and `empty_id` annotations to:
  - `annotations/{method}/{study_uid}_{series_uid}/modified_annotations.json` (metadata)
  - `annotations/{method}/{study_uid}_{series_uid}/masks/frame_XXXXXX_mask.png` (masks)
- [track.py](track.py) - Reads annotations from MD.ai JSON, generates TRACK_ID predictions

**What's missing:**
- Function to convert local annotations (JSON + PNG) to MD.ai JSON format
- Re-tracking endpoint/mechanism that uses local annotations only (no MD.ai data needed)
- UI button to trigger re-tracking

## Architecture Design

### 1. Data Flow

```
Initial Run:
MD.ai JSON → track.py → output/{method}/{study_uid}_{series_uid}/ → app.py displays

Iteration Loop:
app.py (user edits) → annotations/{method}/{study_uid}_{series_uid}/
                    ↓
              Convert to MD.ai JSON format:
              1. Read modified_annotations.json
              2. Load PNG masks from annotations/masks/
              3. Convert masks to polygons (for fluid) or empty vertices (for empty)
              4. Generate MD.ai-compliant JSON
                    ↓
              Clear output/{method}/{study_uid}_{series_uid}/ (preserve annotations/)
                    ↓
              track.py with local JSON → output/ → app.py displays
```

### 2. Required Components

**A. New lib module: `lib/local_annotations.py`**

- `convert_local_to_mdai_json(method, study_uid, series_uid)` → MD.ai JSON file path
  - Reads `annotations/{method}/{study_uid}_{series_uid}/modified_annotations.json`
  - Loads PNG masks from `annotations/{method}/{study_uid}_{series_uid}/masks/`
  - For `label_id` frames: Convert mask to polygons using `cv2.findContours()`
  - For `empty_id` frames: Use `"data": null` (MD.ai format for empty/"No Fluid" frames)
  - Generates MD.ai-compliant JSON with format:
    ```json
    {
      "datasets": [{
        "annotations": [
          {
            "labelId": "...",
            "StudyInstanceUID": "...",
            "SeriesInstanceUID": "...",
            "SOPInstanceUID": "{series_uid}_{frame_num}",
            "frameNumber": frame_num + 1,  // 1-based (MD.ai uses 1-based)
            "data": {
              "foreground": [[x1, y1, x2, y2, ...]]  // For fluid frames
            },
            // OR for empty frames:
            "data": null  // MD.ai uses null, not empty array
          }
        ]
      }]
    }
    ```
  - Saves to temporary file, returns path
  - **KISS/DRY:** Reuses existing mask→polygon conversion logic from tracking

**B. Modified [track.py](track.py) flow:**

- Add option to use local JSON file instead of MD.ai data
- If local JSON provided, skip MD.ai download and use local file
- No changes needed to [lib/multi_frame_tracker.py](lib/multi_frame_tracker.py) - it already handles MD.ai JSON format

**C. New app.py endpoint: `/api/retrack`**
```python
@app.route('/api/retrack/<method>/<study_uid>/<series_uid>', methods=['POST'])
def retrack_video(method, study_uid, series_uid):
    # 1. Convert local annotations to MD.ai JSON format
    # 2. Clear output/{method}/{study_uid}_{series_uid}/ (preserve annotations/)
    # 3. Run track.py in subprocess with:
    #    - Local JSON file (not MD.ai data)
    #    - TEST_STUDY_UID, TEST_SERIES_UID, FLOW_METHOD env vars
    # 4. Return status/progress
    # 5. On completion, reload video data in frontend
```

**D. UI Changes:**
- Add "Re-track" button (⚡) to top-right control buttons
- Show progress/status during re-tracking
- Reload video data after completion

### 3. Implementation Strategy

**Phase 1: Data Conversion** (lib/local_annotations.py)

- Read `annotations/{method}/{study_uid}_{series_uid}/modified_annotations.json`
- For each frame with annotation:
  - Load mask PNG
  - If `is_empty: true`: Use empty polygon `{'vertices': []}`
  - If `is_empty: false`: Convert mask to polygons using `cv2.findContours()`
- Generate MD.ai-compliant JSON
- Save to temp file, return path

**Phase 2: Track Integration**

- Modify [track.py](track.py) to accept optional local JSON file path
- If provided, use local JSON instead of MD.ai data
- No changes needed to [lib/multi_frame_tracker.py](lib/multi_frame_tracker.py) - it already handles MD.ai format

**Phase 3: UI/API Layer**

- Add re-track endpoint to app.py
- Add button to viewer
- Handle async/progress feedback

### 4. Key Considerations

**Annotation Format Consistency:**

- **Local storage (app.py):**
  - `annotations/{method}/{study_uid}_{series_uid}/modified_annotations.json` - Metadata: `{frame_key: {label_id, is_empty, modified_at}}`
  - `annotations/{method}/{study_uid}_{series_uid}/masks/frame_XXXXXX_mask.png` - PNG masks
- **MD.ai format (for track.py):**
  - JSON with `annotations` array
  - Each annotation: `{labelId, frameNumber, data: {foreground: polygons}, type: 'polygon'}`
  - Empty frames: `data: {foreground: []}` (empty vertices array)

**Solution:** Convert local annotations (JSON + PNG) to MD.ai JSON format before retracking. This allows track.py to work without MD.ai data directory. The conversion happens once per retrack, then track.py processes it normally.

**Empty ID Compliance:**
- Empty frames must use `"data": null` (not `{"foreground": []}`) for MD.ai compatibility
- Verified in actual MD.ai annotation files: empty/"No Fluid" frames have `"data": null`
- See TODO.md item 6-8 for validation requirements

**Performance:**
- Re-tracking single video should be fast (seconds to minutes)
- Run in background subprocess to avoid blocking Flask
- Return immediately with task ID, poll for status

**Data Integrity:**

- `annotations/` directory is preserved during retrack (never cleared)
- `output/{method}/{study_uid}_{series_uid}/` is cleared before retrack (except annotations/)
- Clear distinction: LABEL_ID (human), EMPTY_ID (human), TRACK_ID (machine)
- Version tracking: `modified_at` timestamps in JSON metadata

## Questions for User:

1. ~~**Annotation format:** Should we convert stored masks back to MD.ai polygon format, or have tracking read masks directly?~~ ✅ **RESOLVED:** Convert local annotations to MD.ai JSON format for retrack

2. **Re-tracking scope:** Re-track single video only, or batch re-track all videos with local modifications? ✅ **RESOLVED:** Single video only (KISS)

3. **Progress feedback:** Simple spinner, or detailed progress (frame N of M)?

4. **Method selection:** Re-track with same method, or allow choosing different method (e.g., switch from farneback to raft)? ✅ **RESOLVED:** Use same method (simpler)

## Implementation Phases

### Phase 1: Data Conversion (lib/local_annotations.py)

- [ ] Create `lib/local_annotations.py` module
- [ ] Implement `convert_local_to_mdai_json(method, study_uid, series_uid)`:
  - Load `annotations/{method}/{study_uid}_{series_uid}/modified_annotations.json`
  - For each frame with annotation:
    - Load mask PNG from `annotations/{method}/{study_uid}_{series_uid}/masks/`
    - If `is_empty: true`: Use `"data": null` (MD.ai format for empty frames)
    - If `is_empty: false`: Convert mask to polygons using `cv2.findContours()`
  - Generate MD.ai-compliant JSON structure
  - Save to temp file, return path
- [ ] Validate empty_id format matches MD.ai requirements: `"data": null` (verified in actual MD.ai files, TODO.md item 6-8)
- [ ] Test conversion with existing modified annotations

### Phase 2: Track Integration

- [ ] Modify [track.py](track.py) to accept optional local JSON file path
- [ ] If local JSON provided, skip MD.ai download and use local file
- [ ] Test single video re-tracking with local annotations only
- [ ] Validate output format (masks, video, metadata)

### Phase 3: UI/API Layer

- [ ] Add `/api/retrack` endpoint to app.py:
  - Convert local annotations to MD.ai JSON
  - Clear `output/{method}/{study_uid}_{series_uid}/` (preserve `annotations/`)
  - Run track.py subprocess with local JSON
  - Return status/progress
- [ ] Add re-track button (⚡) to viewer UI top-right controls
- [ ] Implement progress feedback (simple spinner or status polling)
- [ ] Reload video data after retrack completion
- [ ] Test end-to-end flow: modify mask → save → re-track → view updated results

### Phase 4: Validation

- [ ] Validate empty_id annotations are MD.ai compliant (TODO.md item 6-8)
- [ ] Ensure annotations/ directory is never cleared during retrack
- [ ] Test that retrack works without MD.ai data directory
