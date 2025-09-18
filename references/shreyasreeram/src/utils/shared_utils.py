"""Shared utility functions used by both consolidated_tracking.py and ground_truth_utils.py"""

import os
import mdai
import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any

def get_annotations_for_study_series(mdai_client, project_id, dataset_id, study_uid, series_uid, label_id=None):
    """Get annotations for a specific study and series"""
    try:
        # Get all annotations for this study/series
        annotations = mdai_client.project(project_id, dataset_id).annotations()
        
        # Filter to study/series
        annotations = [
            a for a in annotations 
            if a['StudyInstanceUID'] == study_uid and 
               a['SeriesInstanceUID'] == series_uid
        ]
        
        # Filter by label if provided
        if label_id:
            annotations = [a for a in annotations if a['labelId'] == label_id]
            
        return annotations
        
    except Exception as e:
        print(f"Error getting annotations: {str(e)}")
        return []

def process_video_with_multi_frame_tracking_enhanced(
    video_path: str,
    initial_mask: np.ndarray,
    initial_frame_number: int,
    flow_processor,
    output_dir: str = None,
    save_frames: bool = False,
    save_masks: bool = False,
    save_overlays: bool = False,
    save_debug_info: bool = False,
    debug_prefix: str = "",
    start_frame: int = None,
    end_frame: int = None,
    frame_step: int = 1,
    mask_threshold: float = 0.5,
    min_mask_area: int = 100,
    max_mask_area: int = 100000,
    min_mask_height: int = 10,
    min_mask_width: int = 10,
    max_mask_height: int = 500,
    max_mask_width: int = 500,
    min_mask_solidity: float = 0.5,
    min_mask_extent: float = 0.3,
    min_mask_eccentricity: float = 0.1,
    max_mask_eccentricity: float = 0.9
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """Process a video using multi-frame tracking with enhanced mask filtering.
    
    Args:
        video_path: Path to the video file
        initial_mask: Initial binary mask for tracking
        initial_frame_number: Frame number where the initial mask is from
        flow_processor: FlowProcessor instance for optical flow
        output_dir: Directory to save outputs (if saving enabled)
        save_frames: Whether to save the raw frames
        save_masks: Whether to save the tracked masks
        save_overlays: Whether to save the overlaid results
        save_debug_info: Whether to save debug information
        debug_prefix: Prefix for debug filenames
        start_frame: Starting frame number (default: None = start of video)
        end_frame: Ending frame number (default: None = end of video)
        frame_step: Number of frames to skip between processing
        mask_threshold: Threshold for binary mask creation
        min_mask_area: Minimum allowed mask area
        max_mask_area: Maximum allowed mask area
        min_mask_height: Minimum allowed mask height
        min_mask_width: Minimum allowed mask width
        max_mask_height: Maximum allowed mask height
        max_mask_width: Maximum allowed mask width
        min_mask_solidity: Minimum allowed mask solidity
        min_mask_extent: Minimum allowed mask extent
        min_mask_eccentricity: Minimum allowed mask eccentricity
        max_mask_eccentricity: Maximum allowed mask eccentricity
        
    Returns:
        Tuple containing:
        - List of processed frames
        - List of tracked masks
        - List of overlaid results
        - Dictionary with debug information
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Validate frame numbers
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_frames - 1
    
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"Invalid start_frame: {start_frame}")
    if end_frame < 0 or end_frame >= total_frames:
        raise ValueError(f"Invalid end_frame: {end_frame}")
    if start_frame > end_frame:
        raise ValueError(f"start_frame ({start_frame}) > end_frame ({end_frame})")
    
    # Initialize output lists
    frames = []
    masks = []
    overlays = []
    
    # Initialize debug info
    debug_info = {
        "video_path": video_path,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_step": frame_step,
        "initial_frame_number": initial_frame_number,
        "mask_stats": [],
        "frame_processing_times": [],
        "errors": []
    }
    
    try:
        # Process frames
        current_frame_number = start_frame
        current_mask = initial_mask.copy()
        
        while current_frame_number <= end_frame:
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Track mask using optical flow
            if current_frame_number != initial_frame_number:
                current_mask = flow_processor.track_mask(
                    current_mask,
                    frame_step=frame_step
                )
                
                # Apply mask filtering
                current_mask = filter_mask(
                    current_mask,
                    threshold=mask_threshold,
                    min_area=min_mask_area,
                    max_area=max_mask_area,
                    min_height=min_mask_height,
                    min_width=min_mask_width,
                    max_height=max_mask_height,
                    max_width=max_mask_width,
                    min_solidity=min_mask_solidity,
                    min_extent=min_mask_extent,
                    min_eccentricity=min_mask_eccentricity,
                    max_eccentricity=max_mask_eccentricity
                )
            
            # Create overlay
            overlay = create_overlay(frame, current_mask)
            
            # Save outputs
            frames.append(frame)
            masks.append(current_mask)
            overlays.append(overlay)
            
            # Save debug info
            if save_debug_info:
                mask_stats = calculate_mask_stats(current_mask)
                debug_info["mask_stats"].append({
                    "frame_number": current_frame_number,
                    **mask_stats
                })
            
            # Save files if requested
            if save_frames:
                cv2.imwrite(
                    os.path.join(output_dir, f"{debug_prefix}frame_{current_frame_number:03d}.png"),
                    frame
                )
            if save_masks:
                cv2.imwrite(
                    os.path.join(output_dir, f"{debug_prefix}mask_{current_frame_number:03d}.png"),
                    (current_mask * 255).astype(np.uint8)
                )
            if save_overlays:
                cv2.imwrite(
                    os.path.join(output_dir, f"{debug_prefix}overlay_{current_frame_number:03d}.png"),
                    overlay
                )
            
            # Move to next frame
            current_frame_number += frame_step
            
    except Exception as e:
        debug_info["errors"].append(str(e))
        raise e
        
    finally:
        cap.release()
        
    return frames, masks, overlays, debug_info

def filter_mask(
    mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 100,
    max_area: int = 100000,
    min_height: int = 10,
    min_width: int = 10,
    max_height: int = 500,
    max_width: int = 500,
    min_solidity: float = 0.5,
    min_extent: float = 0.3,
    min_eccentricity: float = 0.1,
    max_eccentricity: float = 0.9
) -> np.ndarray:
    """Filter a mask based on various criteria."""
    # Threshold mask
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask
    filtered_mask = np.zeros_like(binary_mask)
    
    for contour in contours:
        # Calculate basic properties
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate shape properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Calculate eccentricity (ratio of minor to major axis)
        if len(contour) >= 5:  # Required for ellipse fitting
            try:
                (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
                # Swap if needed to ensure major_axis is the larger one
                if minor_axis > major_axis:
                    major_axis, minor_axis = minor_axis, major_axis
                eccentricity = minor_axis / major_axis if major_axis > 0 else 1
            except:
                # If ellipse fitting fails, use width/height ratio
                eccentricity = min(w, h) / max(w, h)
        else:
            # For very small contours, use width/height ratio
            eccentricity = min(w, h) / max(w, h)
        
        # Debug output
        print(f"Contour properties:")
        print(f"  Area: {area} (min: {min_area}, max: {max_area})")
        print(f"  Height: {h} (min: {min_height}, max: {max_height})")
        print(f"  Width: {w} (min: {min_width}, max: {max_width})")
        print(f"  Solidity: {solidity:.2f} (min: {min_solidity})")
        print(f"  Extent: {extent:.2f} (min: {min_extent})")
        print(f"  Eccentricity: {eccentricity:.2f} (min: {min_eccentricity}, max: {max_eccentricity})")
        
        # Check all criteria
        if (min_area <= area <= max_area and
            min_height <= h <= max_height and
            min_width <= w <= max_width and
            solidity >= min_solidity and
            extent >= min_extent and
            min_eccentricity <= eccentricity <= max_eccentricity):
            # Draw valid contour
            cv2.drawContours(filtered_mask, [contour], -1, 1, -1)
            print("  ✓ Contour passed all criteria")
        else:
            print("  ✗ Contour failed criteria check")
    
    return filtered_mask

def calculate_mask_stats(mask: np.ndarray) -> Dict[str, Any]:
    """Calculate various statistics for a binary mask."""
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            "num_regions": 0,
            "total_area": 0,
            "avg_area": 0,
            "min_area": 0,
            "max_area": 0
        }
    
    # Calculate areas
    areas = [cv2.contourArea(c) for c in contours]
    
    return {
        "num_regions": len(contours),
        "total_area": sum(areas),
        "avg_area": sum(areas) / len(areas),
        "min_area": min(areas),
        "max_area": max(areas)
    }

def create_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Create an overlay of the mask on the frame."""
    overlay = frame.copy()
    mask_colored = np.zeros_like(frame)
    mask_colored[mask > 0] = color
    cv2.addWeighted(mask_colored, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay 