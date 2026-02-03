# Goobusters Tasks

## To Do

### Series Completion Workflow

- [ ] Test completion workflow end-to-end (mark complete → verify skipped in next → reopen → verify available again)

### iPad App

- [ ] Incorrect error message when sync completes (can pull from md.ai) but server unresponsive. Currently shows "load failed" when sync works but server can't be reached.
- [ ] "Connection to server lost" (changes are saved locally) should retry server connection in background. Lost connect should drop to blocking modal after N retries. (think this was done, but wasn't the flow experienced when using app then putting it down and picking it back up)
- [ ] UI to scale with device (landscape only for now), to allow larger devices (iPad Pro 12.9", etc) and iphones
- [ ] **C1: Runtime wrapper** - Package Python runtime (Pyto or Pyodide-on-iOS) + native shell that launches backend and WebView
    - [x] Set up Apple dev account if needed (Mac/Xcode)
    - [ ] Choose runtime (Pyto vs Pyodide-on-iOS)
    - [x] Create native app wrapper (Mac/Xcode)
    - [ ] Integrate shared config loader
    - [ ] Build IPA, test on physical iPad (Mac/Xcode)
    - [ ] Verify Python process starts and serves local HTTP port (Mac/Xcode)
- [ ] **C2: MD.ai sync** - Implement `client/mdai_client.py` to auth, list series, download exams, and store under `client_cache/data`
    - [ ] Verify MD.ai SDK integration works on iPad
    - [ ] Test downloading real exam (>=500 MB) over Wi-Fi
    - [ ] Hash compare with MD.ai CLI output for validation
    - [ ] Handle offline/network error scenarios
- [ ] **C3: Frame extraction** - Use ffmpeg/opencv to split MP4 to frames, maintain manifest consumed by WebView; auto-extract after dataset sync
    - [ ] Bundle ffmpeg binary for iPad
    - [ ] Test frame extraction on downloaded video
    - [ ] Verify frame count matches metadata
    - [ ] Ensure frames accessible via `file://` path or localhost
    - [ ] Verify post-sync auto-extraction
    - [ ] Fix `context.frames` inconsistency (remove unused FrameExtractor references or implement properly)
- [ ] Test end-to-end iPad workflow: download → extract → view → edit → save → retrack
- [ ] App build: should not include prepopulated data or preset mdai token
- [ ] Enable cloudflare tunnel access to server when CF-Access-Client-Id and CF-Access-Client-Secret set
- [ ] Test that cloudflare tunnel hasn't broken the front-end
- [ ] Implement Cloudflare tunnel token access for remote connectivity without VPN (alternative to VPN requirement)

### UI/UX Improvements

- [ ] (lower priority) improved series navigation modal

### Onboarding & Auth

- [ ] Persist failstate on app first start and not getting username (not email) or mdai token
- [ ] Fetch active users from server to populate username dropdown dynamically
    - Add endpoint: GET /api/users/active (returns list of users from recent activity)
    - Use server list if available, fall back to fixed list on error
- [ ] Add "Other" option to username dropdown to allow custom name entry
    - Show text input when "Other" selected
    - Save custom name and add to dynamic list

### Device QA

- [ ] (low priority) smaller device testing

### Architecture

- [ ] **Simplify client backend architecture** - Eliminate proxy layer; frontend talks directly to server for all non-PHI operations
    - See detailed implementation plan: `SIMPLIFY_CLIENT_BACKEND.md`
    - **Client backend retains only PHI-related operations:**
        - MD.ai dataset sync (download videos/annotations)
        - Frame extraction (MP4 → WebP)
        - Local frame serving to frontend
        - Static file serving (HTML/CSS/JS)
        - User settings
    - **Remove all proxy routes:**
        - `/proxy/<path>` - frontend uses `SERVER_URL` config to call server directly
        - `/api/masks/*` - frontend calls server directly
        - `/api/series/*` - frontend calls server directly
        - `/api/video/*` (metadata) - server has this from annotations
    - **Benefits:** Simpler architecture, eliminates caching issues, clearer PHI boundary, easier debugging

### Data & Annotation Handling

- [ ] Ensure we are consistently using 0-based frame counting, to be consistent with mdai json
- [ ] Handle series with no fluid annotations gracefully (serve blank masks/metadata so client can open; no tracking needed)
    - Validate "No Fluid" frame annotation compatible with mdai json syntax. Example (frameNumber 0 (`"id": "A_gp58a1"`) & 41 (`"id": "A_AYxjY2"`) of 143, 0-based counting)"
    - StudyInstanceUID = "1.2.826.0.1.3680043.8 498. 21582572478922879563110991046360588727"
    - SeriesInstranceUID = "1.2.826.0.1.3680043.8.498.88798124921994953570699988775039906436"

### Investigations

- [ ] Jumpy video in annotation editor app but not in tracked_video.mp4? Example: exam 19; 1.2.826.0.1.3680043.8.498.12762211632497404572246503032980657292_1.2.826.0.1.3680043.8.498.90262783102403545676047413537747709850

## Blocked

- [ ] (BLOCKED: REVISIT IF ANOTHER EXAMPLE ARISES) No LABEL_ID but still being included? 1.2.826.0.1.3680043.8.498.90435151582213456262290795805216481896_1.2.826.0.1.3680043.8.498.37052967828633660121479146607377040574
- [ ] (BLOCKED: NEED TO CREATE EMPTY FRAME ANNOTATIONS) Check `tracked_annotations` generated by tracking
- [ ] (BLOCKED: NEEDS PRIORITIZATION GO/NO-GO) lib/debug_visualization.py abandoned. If needed, will will have to fix debug viz and reincorporate
- [ ] (BLOCKED: NEED TEAM TO WEIGH IN ON METHODS) Send test annotations back to mdai for the Pelvic-1 dataset (MUST BE PELVIC-1 FOR TESTING, CHECK dot.env VARS TWICE)

## Done

### Series Completion Workflow

- [x] Add completion status indicator to viewer UI (show "Completed" badge/icon)
- [x] Add "Mark as Complete" button/action in viewer
- [x] Update series list/navigation to display completion status
- [x] Ensure `/api/series/next` skips completed series (verify smart selection logic)
- [x] Add "Reopen" functionality to allow re-editing completed series

### Multiplayer Testing & Bugfixing

- [x] Test concurrent editing conflicts (version ID validation)
    - [x] Two users editing same series simultaneously
    - [x] Verify HTTP 409 conflict on save when version mismatch
    - [x] Verify conflict warning displayed to user
- [x] Test activity tracking (keep-alive mechanism) - Implementation complete, testing complete
    - [x] Verify activity timestamps update on series access
    - [x] Verify recent activity warnings shown to other users
    - [x] Test activity timeout/cleanup logic (24h expiration confirmed)
- [x] Test smart selection heuristics (`RECENT_VIEW_THRESHOLD_MINUTES`)
    - [x] User A opens series (page load or manual selection) → marks activity
    - [x] User B loads page (calls `loadNextSeries()` on init) → verified warning message retained (same series selected is acceptable - warning provides visibility)
    - [x] User B marks series complete (calls `loadNextSeries()`) → verify User B doesn't get User A's active series
    - [x] All series recently active → verify fallback to longest-since-viewed
    - [x] User A was recently active on Series 1 → User A loads page → verify User A can still get Series 1 (own activity doesn't block)
- [x] Test retrack queue conflicts
    - [x] Verify temp version blocks concurrent saves during retrack
    - [x] Verify timeout cleanup of stale temp versions
    - [x] Test parallel retrack processing (multiple series simultaneously)

### UI/UX Improvements

- [x] Navigate series through current series indicator (add prev/next series navigation, show current position in series list)
- [x] Simplify/consolidate right toolbar (reduce button clutter, combine related actions, improve organization)
- [x] Scrubline frame highlight width tweaks
- [x] Brush size scales with series size not viewport
- [x] Performance check on initial page load (measure time to interactive, identify bottlenecks)

- [x] lib/optical.py cleanup - 1998 lines but only 2 exported functions used (create_identity_file, copy_annotations_to_output). Consider refactoring or removing dead code.
    - lib/optical.py - 90% dead code (1,800 of 1,998 lines)
        - Only 2 functions are actually used: create_identity_file and copy_annotations_to_output
        - Contains entire unused classes: OpticalFlowTracker (1,385 lines), MultiAnnotationProcessor (300 lines)
        - Can be reduced from 1,998 → ~200 lines
        - combine with opticalflowprocessor
    - lib/multi_frame_tracker.py - 25% vestigial code (~250 lines)
        - 13+ instance variables initialized but never used (e.g., tracking_strategy_weight, feedback_loop_mode, tracks, track_id_counter)
        - 4 unused methods: \_track_forward, \_track_backward, \_get_previous_frame, `_get_next_frame`
        - Logger bugs: self.logger used but never initialized (will cause AttributeError)
    - lib/debug_visualization.py - 100% orphaned (242 lines)
        - Entire file unused, not imported anywhere
        - Should be deleted or moved to debug_tools/
    - lib/video_capture_manager.py - 61% unused (78 of 127 lines)
        - VideoCaptureManager class and read_frame_at_position() are unused
        - Only video_capture() and get_video_properties() are actually used
    - lib/opticalflowprocessor.py - 24% unused
        - apply_optical_flow() method defined but never called
- [x] Update mdai integration to ONLY pull the PROJECT_ID and DATASET_ID from dot.env (not all data from the project) `mdai_client.project(project_id=PROJECT_ID, dataset_id=DATASET_ID, path=DATA_DIR)` from <https://docs.md.ai/annotator/python/guides/general/>
- [x] Clean code and remove unused or overly complex bits in the current fork (incl. remove deepflow as too slow to be practical)
    - [x] Removed deepflow
    - [x] Removed temporal smoothing (unproven complexity)
    - [x] Removed unused files: parallel_processor.py, video_capture_manager.py, diagnose.py
- [x] Empty frame annotations (allow annotations for frames with no fluid in bidirectional tracking with EMPTY_ID from dot.env)
- [x] Jerky tracking? Tracking seems to jump frame-to-frame (noted, no changes made)
- [x] Performance tuning? Anything to be done with cpu vs gpu accel or parallelization? (reviewed, already optimized)
- [x] dot.env.example update for new/modified options (added TEST_STUDY_UID, TEST_SERIES_UID, improved comments)
- [x] `tracked_annotations` should include TRACK_ID, EMPTY_ID, and LABEL_ID (now uses actual environment variable values instead of hardcoded strings)
- [x] Retrack on Save (save automatically triggers retrack; removed separate retrack button)
- [x] Activity tracking implementation (mark series as active on selection/view, activity pings every 30s, activity displayed in dropdown with emoji indicators, activity cleared on reset retrack)
