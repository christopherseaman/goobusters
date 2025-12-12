# Test Scripts

Test scripts for the distributed architecture using **real data only** (no mocks).

## Prerequisites

1. **Server running**: `uv run python server/server.py`
2. **Retrack worker running**: `uv run python server/retrack_worker.py` (in separate terminal)
3. **Tracked masks available**: Run `uv run python track.py` first to generate masks
4. **Configuration**: Ensure `dot.env` is configured with valid MD.ai credentials

## Test Scripts

### `test_lazy_tracking.py`

Tests lazy tracking functionality:
- Automatic tracking trigger on first mask request
- Tracking status updates (pending → processing → completed)
- Mask availability after tracking completes

**Usage:**
```bash
uv run python scripts/test_lazy_tracking.py
```

**What it tests:**
1. ✅ Find series with `tracking_status == "never_run"`
2. ✅ Request masks (triggers lazy tracking)
3. ✅ Verify tracking status updates
4. ✅ Wait for completion and verify masks available

### `test_server_api.py`

Tests the full server API flow:
- Server status endpoint
- Series navigation (get next series)
- Mask download (GET /api/masks)
- Mask upload (POST /api/masks)
- Retrack status polling
- Version conflict detection

**Usage:**
```bash
uv run python scripts/test_server_api.py
```

**What it tests:**
1. ✅ Server health check
2. ✅ Get next available series
3. ✅ Download existing masks as .tgz archive
4. ✅ Upload modified masks (simulates user edits)
5. ✅ Poll retrack status until completion
6. ✅ Verify version conflict detection

### `test_retrack_worker.py`

Tests retrack queue and worker functionality:
- Queue operations (enqueue, dequeue, status)
- Job creation with real mask data
- Output verification

**Usage:**
```bash
uv run python scripts/test_retrack_worker.py
```

**What it tests:**
1. ✅ Retrack queue basic operations
2. ✅ Job creation with real mask archives
3. ✅ Queue status tracking
4. ✅ Output verification (after worker processes)

## Running Full Test Suite

1. **Start server** (terminal 1):
   ```bash
   uv run python server/server.py
   ```

2. **Start retrack worker** (terminal 2):
   ```bash
   uv run python server/retrack_worker.py
   ```

3. **Run tests** (terminal 3):
   ```bash
   # Test queue operations
   uv run python scripts/test_retrack_worker.py

   # Test full API flow
   uv run python scripts/test_server_api.py
   ```

## Expected Output

### test_server_api.py
```
============================================================
Server API Test Suite
============================================================

[1] Testing GET /api/status...
  ✓ Server ready: True
  ✓ Series total: 150
  ✓ Series completed: 10
  ✓ Series pending: 140

[2] Testing GET /api/series/next...
  ✓ Got series: 1.2.826.0.1.3680043.../1.2.826.0.1.3680043...
  ✓ Exam number: 194
  ✓ Tracking status: completed

[3] Testing GET /api/masks/{study}/{series}...
  ✓ Downloaded mask archive (245678 bytes)
  ✓ Version ID: abc123def456
  ✓ Mask count: 145
  ✓ Metadata: 145 frames

[4] Testing POST /api/masks/{study}/{series}...
  ✓ Masks uploaded successfully
  ✓ New version ID: def456ghi789
  ✓ Retrack queued: True
  ✓ Queue position: 1

[5] Testing GET /api/retrack/status/{study}/{series}...
  Waiting for retrack to complete (max 180s)...
  Status: pending
  ⏳ RETRACK_PENDING (queue position: 1)
  Status: processing
  ⏳ RETRACK_PROCESSING
  Status: completed
  ✓ Retrack completed successfully
  ✓ Version ID: def456ghi789

[6] Testing version conflict detection...
  ✓ Version conflict correctly detected
     Current: def456ghi789
     Provided: wrong_version_id
```

### test_retrack_worker.py
```
============================================================
Retrack Worker Test Suite
============================================================

[1] Testing retrack queue operations...
  ✓ Job enqueued: abc123def456
  ✓ Status: pending
  ✓ Job status retrieved: pending
  ✓ Job dequeued: processing
  ✓ Job marked as completed

[2] Testing retrack worker integration...
  Using series: 1.2.826.0.1.3680043.../1.2.826.0.1.3680043...
  ✓ Created test archive: test_masks.tgz
  ✓ Extracted archive to uploaded_masks
  ✓ Created retrack job: def456ghi789
  ⚠ Note: Worker must be running to process this job
     Run: uv run python server/retrack_worker.py
```

## Troubleshooting

### "No series available"
- Run `uv run python track.py` first to generate tracked masks
- Check that `output/dis/{study}_{series}/masks/` contains .webp files

### "Server not available"
- Ensure server is running: `uv run python server/server.py`
- Check server URL in `dot.env` (default: `http://localhost:5000`)

### "Retrack timeout"
- Ensure retrack worker is running: `uv run python server/retrack_worker.py`
- Check worker logs for errors
- Verify video file exists at path in series metadata

### "Version conflict" (unexpected)
- This is expected if testing quickly - wait for retrack to complete
- Or use a different series for testing

## Notes

- **No mocks**: All tests use real data from `output/` directory
- **Real tracking**: Retrack worker runs actual optical flow tracking
- **Real server**: Tests hit actual Flask endpoints
- **Real queue**: Tests use filesystem-backed queue storage

These tests validate the complete distributed architecture workflow end-to-end.
