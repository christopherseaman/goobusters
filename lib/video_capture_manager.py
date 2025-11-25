#!/usr/bin/env python3
"""
Video Capture Manager - Context Manager for OpenCV VideoCapture

This module provides a context manager wrapper for cv2.VideoCapture to ensure
proper resource cleanup and prevent file descriptor leaks.
"""

import cv2
import logging
from typing import Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def video_capture(video_path: str):
    """
    Context manager function for video capture.

    Args:
        video_path: Path to the video file

    Yields:
        cv2.VideoCapture object

    Example:
        with video_capture('video.mp4') as cap:
            ret, frame = cap.read()
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        yield cap
    finally:
        if cap is not None:
            try:
                if cap.isOpened():
                    cap.release()
            except Exception as e:
                logger.warning(f"Error releasing video capture: {str(e)}")


def get_video_properties(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get video properties safely without leaving file descriptors open.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (total_frames, width, height, fps)
    """
    with video_capture(video_path) as cap:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return total_frames, width, height, fps