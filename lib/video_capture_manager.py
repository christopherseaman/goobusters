#!/usr/bin/env python3
"""
Video Capture Manager - Context Manager for OpenCV VideoCapture

This module provides a context manager wrapper for cv2.VideoCapture to ensure
proper resource cleanup and prevent file descriptor leaks.
"""

import cv2
import logging
from typing import Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class VideoCaptureManager:
    """
    Context manager for cv2.VideoCapture that ensures proper resource cleanup.

    Usage:
        with VideoCaptureManager('video.mp4') as cap:
            ret, frame = cap.read()
            # Process frames...
    """

    def __init__(self, video_path: str):
        """
        Initialize the video capture manager.

        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.cap = None
        self.is_opened = False

    def __enter__(self):
        """Open the video capture."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            self.is_opened = True
            return self.cap
        except Exception as e:
            logger.error(f"Error opening video {self.video_path}: {str(e)}")
            if self.cap is not None:
                self.cap.release()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure the video capture is properly released."""
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                    logger.debug(f"Released video capture: {self.video_path}")
            except Exception as e:
                logger.warning(f"Error releasing video capture: {str(e)}")
            finally:
                self.cap = None
                self.is_opened = False


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


def get_video_properties(video_path: str) -> Optional[Tuple[int, int, int, float]]:
    """
    Get video properties safely without leaving file descriptors open.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (total_frames, width, height, fps) or None if error
    """
    with video_capture(video_path) as cap:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return total_frames, width, height, fps


def read_frame_at_position(video_path: str, frame_number: int) -> Optional[Tuple[bool, any]]:
    """
    Read a specific frame from a video file.

    Args:
        video_path: Path to the video file
        frame_number: Frame number to read

    Returns:
        Tuple of (success, frame) or None if error
    """
    with video_capture(video_path) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return cap.read()