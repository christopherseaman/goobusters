# Testing Guide: Save + Retrack End-to-End

## Overview

This guide covers testing the complete save + retrack workflow, including conflict handling.

## Prerequisites

1. **Server running**: `uv run python server/server.py`
2. **Retrack worker running**: `uv run python server/retrack_worker.py`
3. **Client running**: `uv run python client/client.py`
4. **At least one series with completed tracking** (masks.tar exists)

## Test 1: Basic Save + Retrack Flow

### Steps

1. **Open viewer** in browser: `http://localhost:5001/viewer`
2. **Select a series** with completed tracking
3. **Make edits**:
   - Navigate to a frame with a mask
   - Draw/edit the mask using the brush tool
   - Mark a frame as empty (🚫 button)
4. **Save changes**:
   - Click 💾 (Save) button
   - Should see success state (button turns green)
   - Console should show: `✅ Saved and queued retrack: <version_id>`
5. **Wait for retrack**:
   - Viewer should automatically poll retrack status
   - Console should show polling progress
   - When complete, masks should reload automatically
6. **Verify**:
   - Your edits should be reflected in the reloaded masks
   - Version ID should be updated
   - Frame cache should be cleared and reloaded

### Expected Behavior

- Save button shows success state (green) for 2 seconds
- Retrack status polling starts automatically
- Viewer reloads masks when retrack completes
- No manual refresh needed

## Test 2: Version Mismatch Conflict (409)

### Steps

1. **Open viewer in two browser windows/tabs** (simulating two users)
2. **Window 1**: Make edits and save (this updates the version ID)
3. **Window 2**: Make different edits and try to save
4. **Expected**: Conflict modal should appear with:
   - Title: "Version Mismatch"
   - Message: "Someone else edited this series. Your changes conflict with the server version."
   - Details showing your version ID and server version ID
5. **Click "Reset & Reload"**:
   - Local edits should be cleared
   - Viewer should reload from server
   - Your unsaved edits should be lost (expected behavior)
   - Server's version should be displayed

### Expected Behavior

- Conflict modal appears immediately on 409 response
- "Reset & Reload" clears local state and reloads from server
- No data loss on server side
- User can re-apply edits after reload

## Test 3: Retrack In Progress Conflict (409)

### Steps

1. **Start a save** (triggers retrack)
2. **While retrack is processing**, try to save again from another window/tab
3. **Expected**: Alert should appear: "Retrack already in progress: Please wait for retrack to complete."
4. **Polling should start automatically** to wait for completion

### Expected Behavior

- Alert appears (not modal, since it's a temporary state)
- Polling starts automatically
- User can wait for retrack to complete

## Test 4: Error Handling

### Test Invalid Data

1. **Try to save with no edits**: Should show "No changes to save" in console
2. **Try to save with invalid series**: Should show 404 error
3. **Try to save with server down**: Should show 502 error

### Expected Behavior

- Clear error messages for all failure cases
- No crashes or unhandled exceptions
- User can continue working after errors

## Verification Checklist

- [ ] Save button shows unsaved state when edits are made
- [ ] Save button shows success state after successful save
- [ ] Version ID is updated after save
- [ ] Retrack status polling works correctly
- [ ] Masks reload automatically after retrack completes
- [ ] Conflict modal appears for version mismatches
- [ ] Reset & Reload clears local edits and reloads from server
- [ ] Retrack-in-progress conflicts are handled gracefully
- [ ] Error messages are clear and actionable

## Debugging

### Check Server Logs

```bash
# Server logs should show:
# - POST /api/masks/{study}/{series} received
# - Retrack job enqueued
# - Version ID updated
```

### Check Worker Logs

```bash
# Worker logs should show:
# - Job dequeued
# - Retrack processing started
# - Masks generated
# - Archive built
# - Job marked as completed
```

### Check Browser Console

- Network tab: Verify POST requests to `/api/save_changes`
- Console: Check for error messages or warnings
- Application tab: Check LocalStorage (if implemented)

### Common Issues

1. **"No changes to save"**: Make sure you've actually edited a mask
2. **"Retrack timeout"**: Check that worker is running
3. **"Version mismatch"**: Expected if testing quickly - wait for retrack to complete
4. **Masks not reloading**: Check browser console for errors

## Automated Testing

See `scripts/test_server_api.py` for automated tests of the save + retrack flow.

To run:
```bash
uv run python scripts/test_full_system.py
```

This will:
- Start server and worker
- Test lazy tracking
- Test save + retrack flow
- Test retrack status polling
- Verify output files

