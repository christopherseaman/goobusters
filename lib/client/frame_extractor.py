"""
Client frame extraction - re-exports from shared lib.

This module exists for backwards compatibility. The actual implementation
is now in lib/frame_extractor.py using imageio (no opencv dependency).
"""

from lib.frame_extractor import (
    FrameExtractionError,
    FrameExtractor,
    extract_frame,
    get_video_properties,
    iter_frames,
)

__all__ = [
    "FrameExtractionError",
    "FrameExtractor",
    "extract_frame",
    "get_video_properties",
    "iter_frames",
]
