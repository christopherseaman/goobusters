# File Descriptor Leak Fix Summary

## Problem
The application was experiencing "Too many open files" (OSError: Errno 24) during video processing, specifically when running multi-frame tracking on multiple videos.

## Root Causes Identified

1. **Multiple VideoCapture objects without cleanup**: In `multi_frame_tracker.py`:
   - Line 220: Created `self.cap` that was never released
   - Line 227: Created a second `cap` object (released properly)
   - Line 516: Created local `cap` in `_process_segment` without try-finally protection
   - Line 677: Created `cap` in `_save_results` without proper cleanup

2. **Missing error handling**: No try-finally blocks to ensure cleanup even when errors occur

3. **Shared state issues**: `self.cap` was being created but not consistently managed

## Solutions Implemented

### 1. Created Video Capture Manager (`lib/video_capture_manager.py`)
- Context manager wrapper for cv2.VideoCapture
- Ensures automatic cleanup even on errors
- Provides helper functions for common operations
- Prevents file descriptor leaks

### 2. Updated `multi_frame_tracker.py`
- Removed duplicate VideoCapture creation
- Added proper initialization of `self.cap = None`
- Added cleanup code before clear frame processing
- Added final cleanup at end of process_annotations
- Updated `_save_results` to use context manager
- Added error checking for video capture initialization

### 3. Key Changes Made

#### In `process_annotations`:
```python
# Before: Created two VideoCapture objects, only released one
self.cap = cv2.VideoCapture(video_path)  # Never released
cap = cv2.VideoCapture(video_path)       # Released properly

# After: Single VideoCapture, properly released
cap = cv2.VideoCapture(video_path)
# ... use cap ...
cap.release()
self.cap = None  # Initialize as None, create when needed
```

#### In `_process_segment`:
```python
# Before: No error handling
cap = cv2.VideoCapture(video_path)
# ... processing ...
cap.release()

# After: With context manager
with video_capture(video_path) as cap:
    # ... processing ...
    # Automatic cleanup on exit
```

#### In `_save_results`:
```python
# Before: Manual management
cap = cv2.VideoCapture(video_path)
# ... processing ...
cap.release()

# After: Context manager with error handling
with video_capture(video_path) as cap:
    try:
        # ... processing ...
    finally:
        out.release()  # Ensure VideoWriter is also released
```

## Testing
Created `test_file_descriptors.py` to verify:
- No file descriptor leaks after 100 operations
- Proper error handling for invalid video paths
- Context manager cleanup works correctly

## Recommendations

1. **Use context managers consistently**: Always use the `video_capture` context manager for any video file operations

2. **Monitor resource usage**: Add logging for file descriptor counts in production

3. **Set reasonable limits**: Consider setting ulimit for the process to catch leaks early

4. **Code review checklist**:
   - All cv2.VideoCapture calls use context managers
   - All cv2.VideoWriter objects are released in finally blocks
   - No duplicate video captures for the same file
   - Error paths properly clean up resources

## Impact
- Eliminates "Too many open files" errors
- Improves application stability
- Allows processing of large video batches
- Reduces system resource consumption