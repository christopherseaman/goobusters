# Cleanup Plan and Next Steps

## Summary

Repository assessment complete. Server is functional but startup is slow (~1 hour for 112 series). Test infrastructure needs adjustment. Some legacy code should be archived.

## Immediate Fixes Applied

1. ✅ **Added `httpx` to requirements.txt** - Required by test scripts
2. ✅ **Increased test timeout** - From 10s to 30s (still may need more for full startup)
3. ✅ **Created assessment document** - `REPO_ASSESSMENT.md` with full findings

## Cleanup Recommendations

### High Priority - Keep (Do NOT Remove)

#### 1. Legacy Flask App (`app.py`)

**Status**: Keep - Still used for local testing/review  
**Action**: DO NOT remove or archive  
**Rationale**: 
- Still functional for local annotation review
- May be needed for reference or local workflows
- Not conflicting with distributed architecture (different use case)

**Recommendation**: Keep as-is, document that it's for local use only

#### 2. Track Script (`track.py`)

**Status**: Keep - Core tracking functionality  
**Action**: DO NOT remove or modify  
**Rationale**: 
- Core tracking entrypoint
- Used by server startup for mask generation
- May be used standalone for batch processing

**Recommendation**: Keep as-is, it's part of the core functionality

#### 2. Redundant Test Script (`scripts/run_all_tests.sh`)

**Status**: May be redundant with `test_full_system.py`  
**Action**: Review and remove if redundant  
**Rationale**: 
- `test_full_system.py` already orchestrates all tests
- Shell script may be less maintainable

**Recommendation**: Remove if `test_full_system.py` covers all cases

### Medium Priority - Document/Organize

#### 1. Reference Implementation (`references/shreyasreeram/`)

**Status**: Student reference code (per CLAUDE.md)  
**Action**: Add README documenting it's reference-only  
**Rationale**: 
- Contains known issues (delayed tracking, over-complex code)
- Should not be used as current implementation
- Keep for reference but clearly mark as such

**Recommendation**: Add `references/shreyasreeram/README_REFERENCE.md`:
```markdown
# Reference Implementation Only

This directory contains a student reference implementation with known issues:
- Delayed change tracking
- Over-complex code
- Bugs in multi-frame tracking

**DO NOT use this code in production.** See main codebase for current implementation.
```

#### 2. Unused Debug Code (`references/old_debug_visualization.py`)

**Status**: Unused (per TODO.md)  
**Action**: Move to `references/legacy/` or remove  
**Rationale**: 
- Not imported anywhere
- Replaced by current implementation

**Recommendation**: Move to `references/legacy/` for reference

### Low Priority - Future Cleanup

#### 1. Unused Library Code (per TODO.md)

**Files to clean up in future refactoring:**
- `lib/debug_visualization.py` - 100% orphaned (242 lines)
- `lib/optical.py` - 90% dead code (only 2 functions used)
- `lib/multi_frame_tracker.py` - 25% vestigial code

**Action**: Defer to future refactoring pass  
**Rationale**: 
- Not blocking current work
- Requires careful testing to ensure nothing breaks
- Better to do as dedicated cleanup task

## Next Steps

### 1. Fix Server Startup Performance (Critical)

**Problem**: Server takes ~1 hour to start (sequential mask generation)  
**Options**:

**Option A: Parallelize Mask Generation** (Recommended)
- Use `max_workers` parameter in `generate_masks_for_all_series()`
- Currently sequential, could use 4-8 workers
- **Impact**: Reduce startup from ~1 hour to ~10-20 minutes
- **Risk**: Low (tracking is already thread-safe)

**Option B: Async Startup**
- Start Flask server immediately
- Generate masks in background
- Return 202/503 for series without masks yet
- **Impact**: Server available immediately, masks generated async
- **Risk**: Medium (requires API changes)

**Option C: Test Mode Flag**
- Add `--skip-mask-generation` for testing
- Use existing masks if available
- **Impact**: Fast tests, but doesn't solve production issue
- **Risk**: Low

**Recommendation**: Implement Option A (parallelize) + Option C (test flag)

### 2. Update Test Infrastructure

**Tasks**:
1. ✅ Add `httpx` to requirements (done)
2. ✅ Increase timeout (done, but may need more)
3. Add test mode that skips mask generation
4. Add progress reporting for long-running tests

### 3. Run Full Test Suite

**After fixes**:
```bash
# Install dependencies
uv venv --seed
uv pip install -r requirements.txt

# Run tests (with test mode to skip mask generation)
uv run python scripts/test_full_system.py
```

**Expected**: All tests pass (may need additional fixes)

### 4. Archive Legacy Code

**Commands**:
```bash
# Create legacy directory
mkdir -p references/legacy

# Archive old app
mv app.py references/legacy/

# Archive old debug code
mv references/old_debug_visualization.py references/legacy/

# Document reference implementation
# (manually add README_REFERENCE.md to references/shreyasreeram/)
```

### 5. Document Decisions

**Update `DECISION_REGISTER.md`** with:
- Decision on startup strategy (parallel vs async)
- Decision on legacy code handling
- Any other architectural decisions made

## Testing Strategy

### Current Test Scripts (All in `scripts/`)

1. **`test_full_system.py`** - Main orchestrator
   - Starts server + worker
   - Runs all test suites
   - Shuts down services
   - **Status**: Needs timeout adjustment ✅

2. **`test_startup_verification.py`** - Validates startup
   - Checks dataset exists
   - Verifies masks generated
   - **Status**: Working ✅

3. **`test_server_api.py`** - API endpoint tests
   - Status, series, masks, retrack
   - **Status**: Not yet run (blocked by startup)

4. **`test_lazy_tracking.py`** - Lazy tracking tests
   - Tests on-demand tracking
   - **Status**: Not yet run

5. **`test_retrack_worker.py`** - Retrack queue tests
   - Queue operations
   - **Status**: Not yet run

### Test Execution Plan

1. **Fix startup performance** (parallelize or test mode)
2. **Run `test_full_system.py`** to get baseline
3. **Fix any failures** identified
4. **Document results** in `REPO_ASSESSMENT.md`

## Files Changed

### Modified
- `requirements.txt` - Added `httpx`
- `scripts/test_full_system.py` - Increased timeout, added progress reporting

### Created
- `REPO_ASSESSMENT.md` - Full assessment document
- `CLEANUP_PLAN.md` - This file

### To Archive (Recommended)
- `app.py` → `references/legacy/app.py`
- `references/old_debug_visualization.py` → `references/legacy/old_debug_visualization.py`

### To Remove (If Redundant)
- `scripts/run_all_tests.sh` - If `test_full_system.py` covers all cases

## Questions to Resolve

1. **Startup Strategy**: Parallel (recommended) vs Async vs Sequential?
2. **Test Timeout**: How long should tests wait? (30s may still be too short)
3. **Legacy Code**: Archive vs Delete? (Recommendation: Archive)
4. **Reference Code**: Keep as-is vs Document vs Remove? (Recommendation: Document)

## Success Criteria

- [ ] Server starts in < 20 minutes (with parallelization)
- [ ] All tests pass in `test_full_system.py`
- [ ] Legacy code archived (not deleted)
- [ ] Reference code documented
- [ ] `REPO_ASSESSMENT.md` updated with test results

