# Repository Assessment - December 8, 2025

## Executive Summary

The repository is in a **functional but incomplete** state. The distributed server architecture is implemented and working, but startup takes ~1 hour due to sequential mask generation. Test infrastructure exists but needs adjustment for realistic startup times. Some cleanup is needed for outdated reference code.

## Current State

### ✅ What's Working

1. **Server Core Implementation**
   - Server startup (`server/server.py`) - functional
   - MD.ai dataset download on startup - working
   - Series index building - working
   - Mask generation for all series - working (but slow)
   - Logging to `server/log/` - working
   - API routes structure - in place

2. **Storage & State Management**
   - `SeriesManager` - implemented
   - `RetrackQueue` - implemented
   - Filesystem-based storage - working

3. **Test Infrastructure**
   - `test_full_system.py` - orchestrates tests
   - `test_startup_verification.py` - validates startup
   - `test_lazy_tracking.py` - tests lazy tracking
   - `test_server_api.py` - tests API endpoints
   - `test_retrack_worker.py` - tests retrack queue

4. **Configuration**
   - `lib/config.py` - typed config loader working
   - `dot.env` support - working
   - Server/client config separation - implemented

### ⚠️ Issues Identified

1. **Server Startup Performance**
   - **Problem**: Server generates masks for all 112 series sequentially on startup
   - **Impact**: Takes ~1 hour to start (20-70 seconds per series)
   - **Test Impact**: `test_full_system.py` times out after 10 seconds waiting for server
   - **Root Cause**: Sequential processing in `server/startup.py:generate_masks_for_all_series()`
   - **Spec Compliance**: Per `DISTRIBUTED_ARCHITECTURE.md`, masks should be generated on startup, but spec doesn't require sequential processing

2. **Test Timeout Configuration**
   - **Problem**: `test_full_system.py` waits only 10 seconds for server health check
   - **Impact**: Tests fail even though server is working correctly
   - **Fix Needed**: Increase timeout or make startup async (serve API while tracking in background)

3. **Legacy Code**
   - `app.py` - Legacy Flask app (pre-distributed architecture)
   - `references/shreyasreeram/` - Student reference implementation (should be kept for reference but not used)
   - `references/old_debug_visualization.py` - Unused debug code

### 📋 Cleanup Recommendations

#### High Priority (Keep - Do NOT Remove)

1. **Legacy Flask App** (`app.py`)
   - **Status**: Keep - Still used for local testing/review
   - **Action**: DO NOT remove - Keep as-is
   - **Note**: Still functional for local annotation review workflows

2. **Track Script** (`track.py`)
   - **Status**: Keep - Core tracking functionality
   - **Action**: DO NOT remove - It's part of core functionality
   - **Note**: Used by server startup and may be used standalone

2. **Outdated Test Scripts in References**
   - `references/shreyasreeram/random_scripts/test_*.py` - Old test scripts
   - **Action**: Keep in references (they're reference material), but document they're not current

#### Medium Priority (Document/Organize)

1. **Reference Implementation**
   - `references/shreyasreeram/` - Keep but document it's reference-only
   - **Action**: Add README note that this is reference material, not current code

2. **Test Script Organization**
   - Current test scripts in `scripts/` are good
   - `scripts/run_all_tests.sh` - May be redundant with `test_full_system.py`
   - **Action**: Review if `run_all_tests.sh` adds value or can be removed

#### Low Priority (Future Cleanup)

1. **Unused Library Code** (per TODO.md)
   - `lib/debug_visualization.py` - 100% orphaned (242 lines)
   - `lib/optical.py` - 90% dead code (only 2 functions used)
   - **Action**: Clean up in future refactoring pass

## Next Steps

### Immediate (Fix Test Infrastructure)

1. **Fix Server Startup Timeout**
   - Option A: Increase test timeout to accommodate full startup (~1 hour)
   - Option B: Make startup async - serve API immediately, track in background
   - Option C: Add `--skip-mask-generation` flag for testing
   - **Recommendation**: Option B (async startup) - aligns with spec intent (server should be available)

2. **Update Test Scripts**
   - Modify `test_full_system.py` to handle long startup times
   - Or add test mode that skips mask generation

### Short Term (Cleanup)

1. **Archive Legacy Code**
   - Move `app.py` to `references/legacy/` or document as deprecated
   - Add deprecation notice if keeping

2. **Document Reference Code**
   - Add README in `references/shreyasreeram/` noting it's reference-only
   - Document known issues (per CLAUDE.md)

3. **Review Test Scripts**
   - Decide if `scripts/run_all_tests.sh` is needed
   - Consolidate if redundant

### Medium Term (Performance)

1. **Parallelize Mask Generation**
   - Use `max_workers` parameter in `generate_masks_for_all_series()`
   - Currently sequential, could be parallelized
   - **Impact**: Reduce startup time from ~1 hour to ~10-20 minutes (with 4-8 workers)

2. **Lazy Tracking Alternative**
   - Consider making initial tracking lazy (on first request)
   - Per spec, startup tracking is required, but could be optimized
   - **Trade-off**: Faster startup vs. immediate mask availability

### Long Term (Architecture)

1. **Client Build** (per AGENT_REVIEW_PROMPT.md)
   - Once server tests pass, move to client implementation
   - iPad app foundation (C1-C3 from DISTRIBUTED_ARCHITECTURE.md)

2. **Production Hardening**
   - HTTPS/Caddy setup
   - Structured logging
   - Error handling improvements

## Test Results

### Current Test Status

- **Server Startup**: ✅ Working (but slow - ~1 hour)
- **Test Infrastructure**: ⚠️ Timeout issues (needs adjustment)
- **API Endpoints**: ❓ Not yet tested (blocked by startup timeout)

### Test Execution Plan

1. **Fix startup timeout** (increase timeout or async startup)
2. **Run `test_full_system.py`** to get baseline
3. **Fix any API/test failures** identified
4. **Document test results** in this file

## Files to Review

### Server Implementation
- `server/server.py` - Main server entrypoint
- `server/startup.py` - Startup initialization (needs parallelization)
- `server/api/routes.py` - API endpoints
- `server/tracking_worker.py` - Tracking worker
- `server/retrack_worker.py` - Retrack worker

### Test Scripts
- `scripts/test_full_system.py` - Main test orchestrator (needs timeout fix)
- `scripts/test_startup_verification.py` - Startup validation
- `scripts/test_server_api.py` - API tests
- `scripts/test_lazy_tracking.py` - Lazy tracking tests
- `scripts/test_retrack_worker.py` - Retrack queue tests

### Configuration
- `lib/config.py` - Config loader (working)
- `dot.env.example` - Config template
- `requirements.txt` - Dependencies (missing `httpx` per AGENT_REVIEW_PROMPT.md)

## Decisions Needed

1. **Startup Strategy**: Sequential (current) vs. Parallel vs. Async
2. **Test Timeout**: How long should tests wait for server?
3. **Legacy Code**: Keep `app.py` for reference or remove?
4. **Reference Code**: Document `references/shreyasreeram/` as reference-only?

## Notes

- Server logs show successful mask generation for 112/113 series
- One series may have failed or been skipped (needs investigation)
- Logs are in `server/log/` with timestamp format (working correctly)
- MD.ai dataset download is working (using cached data in test run)

