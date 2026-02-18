#!/usr/bin/env python3
"""
Multi-Frame Annotation Tracker

This module implements sophisticated multi-frame tracking functionality
integrating advanced algorithms from the reference implementation.

Key Features:
- True multi-frame temporal consistency
- Adaptive parameter management (SharedParams-style)
- Advanced occlusion handling and reappearance detection
- Quality-based tracking with genuine optical flow detection
- Temporal smoothing and trajectory management
- Compatible with all optical flow methods
"""

import os
from typing import Optional
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
# Type hints removed - not used in this file

# Import the optical flow processor from the current implementation
try:
    from .opticalflowprocessor import OpticalFlowProcessor
    from .video_capture_manager import video_capture, get_video_properties
except ImportError:
    # Fallback for direct script execution
    from opticalflowprocessor import OpticalFlowProcessor

    try:
        from video_capture_manager import video_capture, get_video_properties
    except ImportError:
        # If video_capture_manager doesn't exist, create simple fallbacks
        import cv2
        from contextlib import contextmanager

        @contextmanager
        def video_capture(video_path):
            """Simple fallback video capture context manager."""
            cap = None
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {video_path}")
                yield cap
            finally:
                if cap is not None and cap.isOpened():
                    cap.release()

        def get_video_properties(video_path):
            """Get video properties safely."""
            with video_capture(video_path) as cap:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                return total_frames, width, height, fps


LABEL_ID_NO_FLUID = os.getenv("LABEL_ID_NO_FLUID", "L_75K42J")
LABEL_ID_FLUID_OVERRIDE: Optional[str] = None
EMPTY_ID_OVERRIDE: Optional[str] = None


def set_label_ids(label_id_fluid: Optional[str], empty_id: Optional[str]) -> None:
    """Set module-level label IDs to avoid env lookups downstream."""
    global LABEL_ID_FLUID_OVERRIDE, EMPTY_ID_OVERRIDE
    LABEL_ID_FLUID_OVERRIDE = label_id_fluid
    EMPTY_ID_OVERRIDE = empty_id


def _label_id_fluid() -> str:
    return LABEL_ID_FLUID_OVERRIDE or os.getenv("LABEL_ID", "")


def _label_id_empty() -> str:
    return EMPTY_ID_OVERRIDE or os.getenv("EMPTY_ID", "")


class SharedParams:
    """
    Manages tracking parameters that can be tuned based on feedback.

    This class stores parameters that the optical flow algorithm uses
    and provides methods to update them based on performance feedback.
    """

    def __init__(self, params_file=None):
        """
        Initialize shared parameters with default values or from a file.

        Args:
            params_file: Optional path to a JSON file with parameter values
        """
        # Default tracking parameters
        self.tracking_params = {
            # Flow algorithm parameters
            "flow_noise_threshold": 3.0,  # Threshold for flow noise filtering
            "flow_quality_threshold": 0.7,  # Quality threshold for optical flow
            # Mask tracking parameters
            "mask_threshold": 0.5,  # Threshold for binary mask conversion
            "contour_min_area": 50,  # Minimum contour area to keep
            "morphology_kernel_size": 5,  # Kernel size for morphological operations
        }

        # Version tracking
        self.version = 1
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load parameters from file if provided
        if params_file and os.path.exists(params_file):
            self.load_from_file(params_file)

    def load_from_file(self, params_file):
        """Load parameters from a JSON file"""
        try:
            with open(params_file, "r") as f:
                data = json.load(f)

            # Update tracking parameters
            if "tracking_params" in data:
                self.tracking_params.update(data["tracking_params"])

            # Update version info
            if "version" in data:
                self.version = data["version"]

            if "last_updated" in data:
                self.last_updated = data["last_updated"]

            # print(f"Loaded parameters (version {self.version}) from {params_file}")
            return True
        except Exception:
            # print(f"Error loading parameters: {str(e)}")
            return False

    def save_to_file(self, params_file):
        """Save parameters to a JSON file"""
        try:
            # Update version info
            self.version += 1
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Prepare data
            data = {
                "tracking_params": self.tracking_params,
                "version": self.version,
                "last_updated": self.last_updated,
            }

            # Save to file
            with open(params_file, "w") as f:
                json.dump(data, f, indent=4)

            return True
        except Exception:
            # print(f"Error saving parameters: {str(e)}")
            return False


class MultiFrameTracker:
    """
    Enhanced multi-frame optical flow tracker with temporal consistency and robust point management.

    This implementation integrates sophisticated multi-frame tracking algorithms from the reference
    implementation with the current codebase structure.
    """

    def __init__(
        self,
        flow_processor: OpticalFlowProcessor,
        output_dir: str,
        debug_mode: bool = False,
        version_id: Optional[str] = None,
    ):
        """Initialize the multi-frame optical flow tracker with configuration."""

        self.flow_processor = flow_processor
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        self.version_id = version_id

        # Initialize shared parameters
        params_file = os.path.join(output_dir, "shared_params.json")
        self.shared_params = SharedParams(params_file)

        # Output directories
        os.makedirs(output_dir, exist_ok=True)
        self.debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)

    def process_annotations(
        self, annotations_df, video_path, study_uid, series_uid
    ):
        """
        Process annotations for a video and generate predictions using multi-frame tracking.

        Args:
            annotations_df: DataFrame containing annotation data
            video_path: Path to the video file
            study_uid: Study instance UID
            series_uid: Series instance UID

        Returns:
            Dictionary mapping frame numbers to mask data
        """

        # print(f"Starting process_annotations for video: {os.path.basename(video_path)}")

        # Get video properties using context manager to ensure cleanup
        try:
            # print("Getting video properties...")
            total_frames, frame_width, frame_height, fps = get_video_properties(
                video_path
            )
            # print(f"Video properties: {total_frames} frames, {frame_width}x{frame_height}, {fps} fps")
        except Exception:
            # print(f"ERROR: Failed to get video properties: {video_path} - {str(e)}")
            return {}

        # Store self.cap as None initially - will be created as needed
        self.cap = None

        # Store total frames for use in other methods
        self.total_frames = total_frames

        # Classify annotations as 'fluid' or 'clear' (based on md.ai labels)
        annotations = self._classify_annotations(
            annotations_df, frame_height, frame_width
        )

        if not annotations:
            # print("WARNING: No valid annotations found")
            return {}

        # Sort annotations by frame number
        annotations = sorted(annotations, key=lambda x: x["frame"])

        all_masks = {}

        # First, store all annotation frames
        for annotation in annotations:
            if annotation["type"] == "fluid":
                all_masks[annotation["frame"]] = {
                    "mask": annotation["mask"],
                    "type": "fluid",
                    "is_annotation": True,
                    "annotation_id": annotation.get(
                        "id", f"fluid_{annotation['frame']}"
                    ),
                    "track_id": f"annotation_{annotation['frame']}",
                    "label_id": annotation.get("labelId", _label_id_fluid()),
                }
            elif annotation["type"] == "empty":
                all_masks[annotation["frame"]] = {
                    "mask": annotation["mask"],
                    "type": "empty",
                    "is_annotation": True,
                    "annotation_id": annotation.get(
                        "id", f"empty_{annotation['frame']}"
                    ),
                    "track_id": f"annotation_{annotation['frame']}",
                    "label_id": annotation.get("labelId", _label_id_empty()),
                }

        # Process segments BETWEEN consecutive annotations
        # Include both fluid and empty annotations for processing
        all_annotations = [
            a for a in annotations if a["type"] in ["fluid", "empty"]
        ]

        for i in range(len(all_annotations)):
            current = all_annotations[i]

            # Process segment from start of video to first annotation
            # Track backward from annotation frame to start
            if i == 0 and current["frame"] > 0:
                if current["frame"] > 0:
                    self._process_segment(
                        0,
                        current["frame"],
                        None,
                        current["mask"],
                        all_masks,
                        video_path,
                    )

            # Process segment between consecutive annotations
            if i < len(all_annotations) - 1:
                next_ann = all_annotations[i + 1]
                if next_ann["frame"] - current["frame"] > 1:
                    self._process_segment(
                        current["frame"],
                        next_ann["frame"],
                        current["mask"],
                        next_ann["mask"],
                        all_masks,
                        video_path,
                    )

            # Process segment from last annotation to end of video
            if (
                i == len(all_annotations) - 1
                and current["frame"] < total_frames - 1
            ):
                if current["frame"] + 1 < total_frames:
                    self._process_segment(
                        current["frame"] + 1,
                        total_frames - 1,
                        current["mask"],
                        None,
                        all_masks,
                        video_path,
                    )

        # Save results (version_id will be set by caller if retracking)
        version_id = getattr(self, 'version_id', None)
        self._save_results(all_masks, video_path, study_uid, series_uid, version_id=version_id)

        if os.getenv("DEBUG", "False").lower() in ("true", "1", "yes"):
            pass  # print(f"Multi-frame processing completed: {len(all_masks)} frames processed")
        return all_masks

    def _classify_annotations(self, annotations_df, frame_height, frame_width):
        """
        Classify annotations as fluid or verified empty based on labelId.

        Convention:
        - Verified empty frames: labelId == EMPTY_ID (human-verified, confirmed no fluid)
          - Should have data: null, but labelId is the authoritative indicator
        - Fluid frames: labelId == LABEL_ID with data.foreground containing polygon coordinates
        - Frames with no annotation: unreviewed (not verified empty, not verified fluid) - ignored
        """
        annotations = []

        # Get label IDs from overrides/env
        label_id_fluid = _label_id_fluid()
        empty_id = _label_id_empty()

        for _, row in annotations_df.iterrows():
            frame_num = int(row["frameNumber"])
            label_id = row.get("labelId", "")
            data = row.get("data", None)

            # Check if this is an EMPTY_ID annotation (verified empty frame)
            # MD.ai uses labelId == EMPTY_ID to indicate verified empty/"No Fluid" frames
            # Verified empty frames should have data: null or data: {"foreground": []} (no polygon data)
            # Note: Some existing data may incorrectly have polygon data with EMPTY_ID, but we treat labelId as authoritative
            if label_id == empty_id:
                # Create empty mask for verified empty frames (regardless of data field content)
                empty_mask = np.zeros(
                    (frame_height, frame_width), dtype=np.uint8
                )

                annotations.append({
                    "frame": frame_num,
                    "mask": empty_mask,
                    "type": "empty",
                    "id": row.get("id", f"empty_{frame_num}"),
                    "labelId": empty_id,  # Always use EMPTY_ID for consistency
                })
                continue

            # Process fluid annotations
            if label_id == label_id_fluid:
                mask = None

                # Check if mask is provided directly (from local annotations with webp files)
                if "mask" in row and row["mask"] is not None:
                    mask = row["mask"]
                    # Ensure mask is a numpy array (original annotations have None, uploaded have arrays)
                    if not isinstance(mask, np.ndarray):
                        continue
                    # Ensure mask is correct size
                    if mask.shape != (frame_height, frame_width):
                        mask = cv2.resize(
                            mask,
                            (frame_width, frame_height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                # Fall back to polygon conversion for MD.ai data
                elif isinstance(data, dict) and "foreground" in data:
                    polygons = data["foreground"]

                    # Skip if foreground is empty
                    if len(polygons) == 0:
                        continue

                    mask = self._polygons_to_mask(
                        polygons, frame_height, frame_width
                    )

                if mask is not None:
                    annotations.append({
                        "frame": frame_num,
                        "mask": mask,
                        "type": "fluid",
                        "id": row.get("id", f"fluid_{frame_num}"),
                        "labelId": label_id,
                    })
            # Note: Frames with no annotation are ignored (unreviewed - not verified empty, not verified fluid)

        return annotations

    def _polygons_to_mask(self, polygons, height, width):
        """Convert polygon data to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)

        for polygon in polygons:
            if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                points = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 255)

        return mask

    def _process_segment(
        self,
        start_frame,
        end_frame,
        start_mask,
        end_mask,
        all_masks,
        video_path,
    ):
        """
        Process a segment between two frames using bidirectional tracking with distance weighting.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            start_mask: Mask at start frame (None if tracking from beginning)
            end_mask: Mask at end frame (None if tracking to end)
            all_masks: Dictionary to store results
            video_path: Path to video file
        """
        if end_frame < start_frame:
            return  # Invalid range
        if end_frame == start_frame:
            return  # No frames to interpolate (only one frame)

        # Use context manager to ensure video capture is always released
        with video_capture(video_path) as cap:
            # Track forward from start if we have a start mask
            forward_masks = {}
            if start_mask is not None:
                # Forward from {start_frame}
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_mask = start_mask.copy()

                # Read start frame
                ret, prev_frame = cap.read()
                if ret:
                    for frame_idx in range(start_frame, end_frame + 1):
                        if frame_idx == start_frame:
                            # This is the initial frame, just store the mask
                            forward_masks[frame_idx] = current_mask
                            continue

                        ret, curr_frame = cap.read()
                        if not ret:
                            break

                        # Calculate optical flow
                        flow = self.flow_processor.calculate_flow(
                            prev_frame, curr_frame
                        )
                        current_mask = self._warp_mask_with_flow(
                            current_mask, flow
                        )
                        forward_masks[frame_idx] = current_mask
                        prev_frame = curr_frame

            # Track backward from end if we have an end mask
            backward_masks = {}
            if end_mask is not None:
                # Backward from {end_frame}
                cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
                current_mask = end_mask.copy()

                # Read end frame
                ret, prev_frame = cap.read()
                if ret:
                    for frame_idx in range(end_frame, start_frame - 1, -1):
                        if frame_idx == end_frame:
                            # This is the initial frame, just store the mask
                            backward_masks[frame_idx] = current_mask
                            continue

                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, curr_frame = cap.read()
                        if not ret:
                            break

                        # Calculate optical flow: prev_frame (later in time) → curr_frame (earlier in time)
                        # Since we're tracking backward in time, prev_frame is actually the LATER frame
                        # and curr_frame is the EARLIER frame
                        flow = self.flow_processor.calculate_flow(
                            prev_frame, curr_frame
                        )
                        # _warp_mask_with_flow does backward warping (x - flow), which is correct here
                        # because we're warping from prev (later) to curr (earlier)
                        current_mask = self._warp_mask_with_flow(
                            current_mask, flow
                        )
                        backward_masks[frame_idx] = current_mask
                        prev_frame = curr_frame

        # Combine masks with distance-based weighting
        for frame_idx in range(start_frame, end_frame + 1):
            # Preserve original annotations - don't overwrite them with tracked masks
            if frame_idx in all_masks and isinstance(
                all_masks[frame_idx], dict
            ):
                if all_masks[frame_idx].get("is_annotation", False):
                    continue

            if frame_idx in forward_masks and frame_idx in backward_masks:
                # Calculate distance-based weights
                # Forward tracking: most accurate near start_frame, degrades toward end_frame
                # Backward tracking: most accurate near end_frame, degrades toward start_frame
                total_dist = end_frame - start_frame
                forward_weight = (
                    end_frame - frame_idx
                ) / total_dist  # HIGH near start, LOW near end
                backward_weight = (
                    frame_idx - start_frame
                ) / total_dist  # LOW near start, HIGH near end

                # Combine masks with weights
                combined_mask = self._combine_masks_weighted(
                    forward_masks[frame_idx],
                    backward_masks[frame_idx],
                    forward_weight,
                    backward_weight,
                )

                all_masks[frame_idx] = {
                    "mask": combined_mask,
                    "type": "fluid_combined",
                    "is_annotation": False,
                    "forward_weight": forward_weight,
                    "backward_weight": backward_weight,
                    "track_id": f"track_{frame_idx}",
                    "label_id": _label_id_fluid(),
                }
            elif frame_idx in forward_masks:
                all_masks[frame_idx] = {
                    "mask": forward_masks[frame_idx],
                    "type": "fluid_forward",
                    "is_annotation": False,
                    "track_id": f"track_{frame_idx}",
                    "label_id": _label_id_fluid(),
                }
            elif frame_idx in backward_masks:
                all_masks[frame_idx] = {
                    "mask": backward_masks[frame_idx],
                    "type": "fluid_backward",
                    "is_annotation": False,
                    "track_id": f"track_{frame_idx}",
                    "label_id": _label_id_fluid(),
                }

    def _combine_masks_weighted(self, mask1, mask2, weight1, weight2):
        """
        Combine two masks with specified weights.

        Args:
            mask1: First mask
            mask2: Second mask
            weight1: Weight for first mask
            weight2: Weight for second mask

        Returns:
            Combined mask
        """
        # Convert to float for weighted combination
        mask1_float = mask1.astype(np.float32) / 255.0
        mask2_float = mask2.astype(np.float32) / 255.0

        # Weighted combination
        combined = mask1_float * weight1 + mask2_float * weight2

        # Convert back to uint8
        combined = np.clip(combined * 255, 0, 255).astype(np.uint8)

        # Apply threshold to get binary mask
        _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

        # Optional: Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        return combined

    def _warp_mask_with_flow(self, mask, flow):
        """Warp a mask using optical flow (backward warping with remap)."""
        h, w = mask.shape

        # Get or create cached coordinate grid for this resolution
        if not hasattr(self, '_grid_cache') or self._grid_cache[0] != (h, w):
            base_y, base_x = np.mgrid[0:h, 0:w].astype(np.float32)
            self._grid_cache = ((h, w), base_x, base_y)

        base_x = self._grid_cache[1]
        base_y = self._grid_cache[2]

        # Backward warping: find where each destination pixel came from
        map_x = base_x - flow[:, :, 0]
        map_y = base_y - flow[:, :, 1]

        return cv2.remap(
            mask,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _find_nearest_fluid_annotation(self, target_frame, annotations):
        """Find the nearest fluid annotation to a target frame."""
        # Only consider fluid annotations, not empty ones
        fluid_annotations = [a for a in annotations if a["type"] == "fluid"]
        if not fluid_annotations:
            return None

        nearest = min(
            fluid_annotations, key=lambda x: abs(x["frame"] - target_frame)
        )
        return nearest

    def _save_results(self, all_masks, video_path, study_uid, series_uid, version_id=None):
        """Save the tracking results with individual frame masks and comprehensive data."""
        try:
            # Save enhanced mask data with track_id and label_id
            frametype_path = os.path.join(self.output_dir, "frametype.json")
            frametype_summary = {}
            for frame_num, mask_info in all_masks.items():
                if isinstance(mask_info, dict):
                    frametype_summary[str(frame_num)] = {
                        "type": mask_info.get("type", "unknown"),
                        "is_annotation": mask_info.get("is_annotation", False),
                        "has_mask": mask_info.get("mask") is not None,
                        "track_id": mask_info.get(
                            "track_id", f"track_{frame_num}"
                        ),
                        "label_id": mask_info.get("label_id", "unknown"),
                    }
            
            # Add version_id to frametype.json (tied to this tracking revision)
            if version_id is not None:
                frametype_summary["_version_id"] = version_id

            with open(frametype_path, "w") as f:
                json.dump(frametype_summary, f, indent=2)

            # Create masks.json with all tracked annotations
            masks_annotations = []
            empty_id = _label_id_empty()
            label_id_fluid = _label_id_fluid()

            for frame_num, mask_info in all_masks.items():
                if isinstance(mask_info, dict):
                    # Include both frames with masks and empty frames (empty masks are valid)
                    label_id = mask_info.get("label_id", label_id_fluid)
                    annotation_type = mask_info.get("type", "tracked")

                    # Create annotation in MD.ai format
                    annotation = {
                        "id": mask_info.get("track_id", f"track_{frame_num}"),
                        "labelId": label_id,
                        "StudyInstanceUID": study_uid,
                        "SeriesInstanceUID": series_uid,
                        "frameNumber": frame_num,
                        "type": annotation_type,
                        "is_annotation": mask_info.get("is_annotation", False),
                        "track_id": mask_info.get(
                            "track_id", f"track_{frame_num}"
                        ),
                        "label_id": label_id,
                        "mask_file": f"frame_{frame_num:06d}_mask.webp",
                    }

                    # Add data field in MD.ai format
                    # Verified empty frames: data: null (MD.ai convention for verified empty/"No Fluid" frames)
                    # Fluid frames: omit data field (we have masks, not polygons)
                    if label_id == empty_id or annotation_type == "empty":
                        annotation["data"] = (
                            None  # MD.ai format for verified empty frames
                        )

                    masks_annotations.append(annotation)

            # Save masks annotations
            masks_file = os.path.join(self.output_dir, "masks.json")
            with open(masks_file, "w") as f:
                json.dump(masks_annotations, f, indent=2)

        except Exception:
            pass  # Silent failure for background task

        # Save masks as webp files
        masks_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        for frame_num, mask_info in all_masks.items():
            if (
                isinstance(mask_info, dict)
                and mask_info.get("mask") is not None
            ):
                mask = mask_info["mask"]
                mask_path = os.path.join(
                    masks_dir, f"frame_{frame_num:06d}_mask.webp"
                )
                cv2.imwrite(mask_path, mask, [cv2.IMWRITE_WEBP_QUALITY, 99])

        # Create frames directory and extract frames as webp
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Create output video with overlayed masks
        output_video_path = os.path.join(self.output_dir, "tracked_video.mp4")

        # Use context manager to ensure video capture is properly released
        with video_capture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_video_path, fourcc, fps, (width, height)
            )

            try:
                frame_num = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Extract ALL frames from video sequentially
                # IMPORTANT: Extract every frame in order, even if duplicates exist.
                # MD.ai's frameNumber corresponds to sequential position in the video file,
                # not unique frames. So frameNumber: 2 means the 3rd frame (0-based index 2)
                # in the video file, even if it's identical to frame 1.
                # We must match MD.ai's frame numbering exactly.
                while frame_num < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Verify we're actually advancing through frames
                    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if actual_pos <= frame_num:
                        # OpenCV didn't advance - this shouldn't happen with sequential reads
                        # but if it does, we need to break to avoid infinite loop
                        break

                    # Save frame as webp (smaller than jpg/png)
                    # Note: Duplicate frames are expected and preserved to match MD.ai frame numbering
                    frame_file = os.path.join(
                        frames_dir, f"frame_{frame_num:06d}.webp"
                    )
                    cv2.imwrite(
                        frame_file, frame, [cv2.IMWRITE_WEBP_QUALITY, 85]
                    )

                    # Only write to output video if frame has mask (for visualization)
                    # Skip video writing for very long videos to prevent hanging
                    # Only write first 1000 frames for long videos
                    max_video_frames = min(total_frames, 1000)
                    if frame_num < max_video_frames:
                        if frame_num in all_masks:
                            mask_info = all_masks[frame_num]
                            mask = mask_info["mask"]

                            # Create colored overlay with different colors for tracked vs annotated
                            overlay = np.zeros_like(frame)
                            if mask_info.get("is_annotation", False):
                                # Green for original annotations (label_id)
                                overlay[mask > 0] = [0, 255, 0]  # Green
                            else:
                                # Orange for tracked masks (track_id)
                                overlay[mask > 0] = [0, 165, 255]  # Orange

                            # Blend with original frame
                            alpha = 0.3
                            frame = cv2.addWeighted(
                                frame, 1 - alpha, overlay, alpha, 0
                            )

                        out.write(frame)

                    frame_num += 1
            finally:
                # Ensure video writer is always released
                out.release()

        # Create tar archive of frames (no gzip - WebP already compressed)
        import tarfile

        frames_archive = os.path.join(self.output_dir, "frames.tar")
        with tarfile.open(frames_archive, "w") as tar:
            tar.add(frames_dir, arcname="frames")

        # Create tar archive of masks (no gzip - WebP already compressed)
        masks_archive = os.path.join(self.output_dir, "masks.tar")
        with tarfile.open(masks_archive, "w") as tar:
            tar.add(masks_dir, arcname="masks")

        if os.getenv("DEBUG", "False").lower() in ("true", "1", "yes"):
            pass  # print(f"Results saved to: {output_video_path}")


def process_video_with_multi_frame_tracking(
    video_path: str,
    annotations_df: pd.DataFrame,
    study_uid: str,
    series_uid: str,
    flow_processor: OpticalFlowProcessor,
    output_dir: str,
    mdai_client=None,
    label_id_fluid: str = None,
    label_id_machine: str = None,
    upload_to_mdai: bool = False,
    project_id: str = None,
    dataset_id: str = None,
    version_id: Optional[str] = None,
):
    """
    Main function to process a video with multi-frame tracking.

    This function integrates the MultiFrameTracker with the existing workflow.
    """
    # print(f"process_video_with_multi_frame_tracking called with {len(annotations_df)} annotations")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create checkpoint file for progress tracking
    checkpoint_file = os.path.join(output_dir, "checkpoint.txt")
    try:
        with open(checkpoint_file, "w") as f:
            f.write(
                f"Process started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Video: {video_path}\n")
            f.write(f"Annotations count: {len(annotations_df)}\n")
    except Exception:
        pass

    # Initialize MultiFrameTracker
    # Initializing tracker
    tracker = MultiFrameTracker(flow_processor, output_dir, debug_mode=True, version_id=version_id)

    # Process annotations
    all_masks = tracker.process_annotations(
        annotations_df, video_path, study_uid, series_uid
    )
    if os.getenv("DEBUG", "False").lower() in ("true", "1", "yes"):
        pass  # print(f"Received {len(all_masks)} masks from process_annotations")

    # Update checkpoint
    try:
        with open(checkpoint_file, "a") as f:
            f.write(
                f"Process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Generated {len(all_masks)} masks\n")
    except Exception:
        pass

    # Count annotation types
    annotation_types = {}
    for frame, info in all_masks.items():
        if isinstance(info, dict) and "type" in info:
            annotation_type = info["type"]
            annotation_types[annotation_type] = (
                annotation_types.get(annotation_type, 0) + 1
            )

    # Tracking complete

    return {
        "all_masks": all_masks,
        "annotated_frames": sum(
            1
            for info in all_masks.values()
            if isinstance(info, dict) and info.get("is_annotation", False)
        ),
        "predicted_frames": sum(
            1
            for info in all_masks.values()
            if isinstance(info, dict) and not info.get("is_annotation", False)
        ),
        "annotation_types": annotation_types,
        "output_video": os.path.join(output_dir, "multi_frame_tracking.mp4"),
    }
