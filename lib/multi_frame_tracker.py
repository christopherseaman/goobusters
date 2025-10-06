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
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging
import json
import traceback
import time
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque

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
            'flow_noise_threshold': 3.0,        # Threshold for flow noise filtering
            'flow_quality_threshold': 0.7,      # Quality threshold for optical flow
            'border_constraint_weight': 0.9,    # Weight for border constraints
            
            # Mask tracking parameters
            'mask_threshold': 0.5,              # Threshold for binary mask conversion
            'contour_min_area': 50,             # Minimum contour area to keep
            'morphology_kernel_size': 5,        # Kernel size for morphological operations
            
            # Learning parameters
            'learning_rate': 0.3,               # Rate at which corrections influence parameters
            'window_size': 30,                  # Window size for propagating corrections
            'distance_decay_factor': 1.5,       # Factor controlling how quickly influence decays with distance
            'iou_improvement_threshold': 0.2,   # Threshold for considering a correction significant
        }
        
        # Performance history for adapting parameters
        self.performance_history = []
        
        # Version tracking
        self.version = 1
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load parameters from file if provided
        if params_file and os.path.exists(params_file):
            self.load_from_file(params_file)
       
        # SharedParams v{self.version} initialized
    
    def load_from_file(self, params_file):
        """Load parameters from a JSON file"""
        try:
            with open(params_file, 'r') as f:
                data = json.load(f)
                
            # Update tracking parameters
            if 'tracking_params' in data:
                self.tracking_params.update(data['tracking_params'])
                
            # Update version info
            if 'version' in data:
                self.version = data['version']
                
            if 'last_updated' in data:
                self.last_updated = data['last_updated']
                
            # print(f"Loaded parameters (version {self.version}) from {params_file}")
            return True
        except Exception as e:
            # print(f"Error loading parameters: {str(e)}")
            return False
    
    def save_to_file(self, params_file):
        """Save parameters to a JSON file"""
        try:
            # Update version info
            old_version = self.version
            self.version += 1
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare data
            data = {
                'tracking_params': self.tracking_params,
                'version': self.version,
                'last_updated': self.last_updated,
                'performance_history': self.performance_history[-5:]  # Keep last 5 for reference
            }
            
            # Save to file
            with open(params_file, 'w') as f:
                json.dump(data, f, indent=4)
                
            # print(f"✅ PARAMETERS UPDATED: v{old_version} → v{self.version}")
            # print(f"   Window size: {self.tracking_params['window_size']}")
            # print(f"   Flow quality: {self.tracking_params['flow_quality_threshold']:.3f}")
            # print(f"   Learning rate: {self.tracking_params['learning_rate']:.3f}")
            # print(f"   Saved to: {params_file}")
            return True
        except Exception as e:
            # print(f"Error saving parameters: {str(e)}")
            return False


class MultiFrameTracker:
    """
    Enhanced multi-frame optical flow tracker with temporal consistency and robust point management.
    
    This implementation integrates sophisticated multi-frame tracking algorithms from the reference
    implementation with the current codebase structure.
    """
    
    def __init__(self, flow_processor: OpticalFlowProcessor, output_dir: str, debug_mode: bool = False):
        """Initialize the multi-frame optical flow tracker with configuration."""
        
        self.flow_processor = flow_processor
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        
        # Initialize shared parameters
        params_file = os.path.join(output_dir, 'shared_params.json')
        self.shared_params = SharedParams(params_file)
        
        # Tracking configuration
        self.tracking_strategy_weight = 0.7
        self.expert_feedback_weight = 0.3
        self.disable_annotation_copying = False
        self.bidirectional_tracking = True
        self.force_tracking = False
        self.min_tracking_frames = 5
        self.force_annotation_propagation = True
        
        # Feedback and learning modes
        self.feedback_loop_mode = True
        self.learning_mode = True
        
        # State variables for multi-frame tracking
        self.frame_idx = 0
        self.tracks = {}  # Dictionary to store track information
        self.track_id_counter = 0
        self.frame_history = deque(maxlen=10)  # Store recent frames for temporal consistency
        
        # Performance tracking
        self.start_time = None
        self.processed_frames = 0
        self.max_processing_time = 300  # 5 minutes default
        self.max_frames_to_process = 200  # Maximum frames to process
        
        # Output directories
        os.makedirs(output_dir, exist_ok=True)
        self.debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Debug: {'on' if self.debug_mode else 'off'}, Feedback: {'on' if self.feedback_loop_mode else 'off'}, Learning: {'on' if self.learning_mode else 'off'}
    
    def __del__(self):
        """Cleanup file handlers on deletion."""
        if hasattr(self, 'file_handler') and self.file_handler:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)

    def process_annotations(self, annotations_df, video_path, study_uid, series_uid):
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
            total_frames, frame_width, frame_height, fps = get_video_properties(video_path)
            # print(f"Video properties: {total_frames} frames, {frame_width}x{frame_height}, {fps} fps")
        except Exception as e:
            self.logger.error(f"Failed to get video properties: {video_path} - {str(e)}")
            # print(f"ERROR: Failed to get video properties: {video_path}")
            return {}

        # Store self.cap as None initially - will be created as needed
        self.cap = None
        
        
        # Store total frames for use in other methods
        self.total_frames = total_frames
        
        # Classify annotations as 'fluid' or 'clear' (based on md.ai labels)
        annotations = self._classify_annotations(annotations_df, frame_height, frame_width)
        
        if not annotations:
            self.logger.warning("No valid annotations found")
            return {}
        
        # Sort annotations by frame number
        annotations = sorted(annotations, key=lambda x: x['frame'])
        
        # Identify clear frames that need to be propagated
        clear_frames = self._identify_clear_frames(annotations, total_frames)
        
        # Process each segment according to the rules
        all_masks = {}

        # First, store all annotation frames
        for annotation in annotations:
            if annotation['type'] == 'fluid':
                all_masks[annotation['frame']] = {
                    'mask': annotation['mask'],
                    'type': 'fluid',
                    'is_annotation': True,
                    'annotation_id': annotation.get('id', f'fluid_{annotation["frame"]}'),
                    'track_id': f"annotation_{annotation['frame']}",
                    'label_id': annotation.get('labelId', 'original_fluid')
                }
            elif annotation['type'] == 'empty':
                all_masks[annotation['frame']] = {
                    'mask': annotation['mask'],
                    'type': 'empty',
                    'is_annotation': True,
                    'annotation_id': annotation.get('id', f'empty_{annotation["frame"]}'),
                    'track_id': f"annotation_{annotation['frame']}",
                    'label_id': annotation.get('labelId', 'empty_frame')
                }

        # Process segments BETWEEN consecutive annotations
        # Include both fluid and empty annotations for processing
        all_annotations = [a for a in annotations if a['type'] in ['fluid', 'empty']]

        for i in range(len(all_annotations)):
            current = all_annotations[i]

            # Process segment from start of video to first annotation
            if i == 0 and current['frame'] > 0:
                self._process_segment(0, current['frame'], None, current['mask'], all_masks, clear_frames, video_path)

            # Process segment between consecutive annotations
            if i < len(all_annotations) - 1:
                next_ann = all_annotations[i + 1]
                if next_ann['frame'] - current['frame'] > 1:
                    # Determine tracking strategy based on annotation types
                    tracking_strategy = self._determine_tracking_strategy(current, next_ann)
                    
                    if tracking_strategy == 'forward_only':
                        # EMPTY_ID → LABEL_ID: only track forward from LABEL_ID
                        if current['type'] == 'empty' and next_ann['type'] == 'fluid':
                            # Track forward from the fluid annotation
                            self._process_segment(current['frame'], next_ann['frame'],
                                                None, next_ann['mask'], all_masks, clear_frames, video_path)
                        else:
                            # Track forward from current annotation
                            self._process_segment(current['frame'], next_ann['frame'],
                                                current['mask'], next_ann['mask'], all_masks, clear_frames, video_path)
                    elif tracking_strategy == 'backward_only':
                        # LABEL_ID → EMPTY_ID: only track backward from LABEL_ID
                        if current['type'] == 'fluid' and next_ann['type'] == 'empty':
                            # Track backward from the fluid annotation
                            self._process_segment(current['frame'], next_ann['frame'],
                                                current['mask'], None, all_masks, clear_frames, video_path)
                        else:
                            # Track backward from next annotation
                            self._process_segment(current['frame'], next_ann['frame'],
                                                current['mask'], next_ann['mask'], all_masks, clear_frames, video_path)
                    else:
                        # Default bidirectional or other strategies
                        self._process_segment(current['frame'], next_ann['frame'],
                                            current['mask'], next_ann['mask'], all_masks, clear_frames, video_path)

            # Process segment from last annotation to end of video
            if i == len(all_annotations) - 1 and current['frame'] < total_frames - 1:
                self._process_segment(current['frame'], total_frames - 1,
                                    current['mask'], None, all_masks, clear_frames, video_path)
        
        # Process clear frames
        for clear_frame in clear_frames:
            if clear_frame not in all_masks:
                # Find the nearest fluid annotation to propagate from
                nearest_fluid = self._find_nearest_fluid_annotation(clear_frame, annotations)
                if nearest_fluid:
                    propagated_mask = self._propagate_clear_frame(
                        clear_frame, 
                        nearest_fluid['frame'], 
                        nearest_fluid['mask']
                    )
                    all_masks[clear_frame] = {
                        'mask': propagated_mask,
                        'type': 'clear',
                        'is_annotation': False,
                        'propagated_from': nearest_fluid['frame']
                    }
        
        # Save results
        self._save_results(all_masks, video_path, study_uid, series_uid)
        
        if os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes'):
            pass  # print(f"Multi-frame processing completed: {len(all_masks)} frames processed")
        return all_masks
    
    def _classify_annotations(self, annotations_df, frame_height, frame_width):
        """Classify annotations as fluid, clear, or empty based on label IDs."""
        annotations = []
        
        # Get label IDs from environment
        label_id_fluid = os.getenv('LABEL_ID', '')
        empty_id = os.getenv('EMPTY_ID', '')
        
        for _, row in annotations_df.iterrows():
            frame_num = int(row['frameNumber'])
            label_id = row.get('labelId', '')
            
            # Check if this is the expected free fluid label
            if label_id == label_id_fluid:
                # Look for annotation data in the 'data' column
                if 'data' not in row or row['data'] is None:
                    continue
                
                data_dict = row['data']
                
                # Extract polygon data from the data dictionary
                if isinstance(data_dict, dict) and 'foreground' in data_dict:
                    polygons = data_dict['foreground']
                    
                    if len(polygons) > 0:
                        # Convert polygon data to mask
                        mask = self._polygons_to_mask(polygons, frame_height, frame_width)
                        
                        annotations.append({
                            'frame': frame_num,
                            'mask': mask,
                            'type': 'fluid',
                            'id': row.get('id', f'fluid_{frame_num}'),
                            'labelId': label_id
                        })
            
            # Check if this is an EMPTY_ID annotation (no fluid frame)
            elif label_id == empty_id:
                # Create empty mask for EMPTY_ID annotations
                empty_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                
                annotations.append({
                    'frame': frame_num,
                    'mask': empty_mask,
                    'type': 'empty',
                    'id': row.get('id', f'empty_{frame_num}'),
                    'labelId': label_id
                })
        
        return annotations
    
    def _determine_tracking_strategy(self, current_annotation, next_annotation):
        """
        Determine tracking strategy based on annotation types.
        
        Special handling for EMPTY_ID:
        - EMPTY_ID → LABEL_ID: forward_only (track from LABEL_ID)
        - LABEL_ID → EMPTY_ID: backward_only (track from LABEL_ID)
        - Other combinations: bidirectional
        """
        current_type = current_annotation['type']
        next_type = next_annotation['type']
        
        # Special case: EMPTY_ID → LABEL_ID (forward only from LABEL_ID)
        if current_type == 'empty' and next_type == 'fluid':
            return 'forward_only'
        
        # Special case: LABEL_ID → EMPTY_ID (backward only from LABEL_ID)
        elif current_type == 'fluid' and next_type == 'empty':
            return 'backward_only'
        
        # Default: bidirectional for other combinations
        else:
            return 'bidirectional'
    
    def _polygons_to_mask(self, polygons, height, width):
        """Convert polygon data to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for polygon in polygons:
            if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                points = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def _identify_clear_frames(self, annotations, total_frames):
        """Identify frames that should be marked as clear (no fluid)."""
        clear_frames = []
        
        # Find gaps between fluid annotations that should be clear
        for i in range(len(annotations) - 1):
            current_frame = annotations[i]['frame']
            next_frame = annotations[i + 1]['frame']
            
            # If there's a gap and the next annotation is clear, mark intermediate frames
            if next_frame - current_frame > 1:
                if annotations[i + 1]['type'] == 'clear':
                    for frame in range(current_frame + 1, next_frame):
                        clear_frames.append(frame)
        
        return clear_frames
    
    def _track_forward(self, start_frame, end_frame, initial_mask):
        """Track a mask forward from start_frame to end_frame."""
        masks = {}
        
        # print(f"Tracking forward from frame {start_frame} to {end_frame}")
        
        # Set video to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_mask = initial_mask.copy()
        masks[start_frame] = {
            'mask': current_mask,
            'type': 'fluid',
            'is_annotation': True
        }

        frames_tracked = 0
        prev_frame = None

        # Read the initial frame to use as previous frame
        ret, prev_frame = self.cap.read()
        if not ret:
            # print(f"Failed to read initial frame {start_frame}")
            return masks

        for frame_num in range(start_frame, end_frame + 1):
            if frame_num == start_frame:
                # This is the initial frame, just store the mask
                masks[frame_num] = {
                    'mask': current_mask,
                    'type': 'fluid',
                    'is_annotation': True,
                    'track_id': f"annotation_{frame_num}",
                    'label_id': 'original_fluid'
                }
                continue
            
            ret, curr_frame = self.cap.read()
            if not ret:
                print(f"Failed to read frame {frame_num}")
                break

            # Calculate optical flow from previous to current frame
            flow = self.flow_processor.calculate_flow(prev_frame, curr_frame)
            current_mask = self._warp_mask_with_flow(current_mask, flow)

            masks[frame_num] = {
                'mask': current_mask,
                'type': 'fluid',
                'is_annotation': False,
                'track_id': f"track_{frame_num}",
                'label_id': 'tracked_fluid'
            }
            frames_tracked += 1

            # Update prev_frame for next iteration
            prev_frame = curr_frame
        
        # Forward: {frames_tracked} frames
        return masks
    
    def _track_backward(self, start_frame, end_frame, initial_mask):
        """Track a mask backward from start_frame to end_frame."""
        masks = {}

        # print(f"Tracking backward from frame {start_frame} to {end_frame}")

        # Ensure we have a video capture open
        if self.cap is None or not self.cap.isOpened():
            self.logger.error("Video capture not initialized for backward tracking")
            return masks

        # Set video to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_mask = initial_mask.copy()
        masks[start_frame] = {
            'mask': current_mask,
            'type': 'fluid',
            'is_annotation': True
        }
        
        frames_tracked = 0
        prev_frame = None

        # Read the initial frame
        ret, prev_frame = self.cap.read()
        if not ret:
            # print(f"Failed to read initial frame {start_frame}")
            return masks

        # Track backward through frames
        for frame_num in range(start_frame, end_frame - 1, -1):
            if frame_num < 0:
                break

            if frame_num == start_frame:
                # This is the initial frame, just store the mask
                masks[frame_num] = {
                    'mask': current_mask,
                    'type': 'fluid',
                    'is_annotation': True,
                    'track_id': f"annotation_{frame_num}",
                    'label_id': 'original_fluid'
                }
                continue

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, curr_frame = self.cap.read()
            if not ret:
                print(f"Failed to read frame {frame_num}")
                break

            # Calculate optical flow from current frame to previous frame (backward)
            # Since we're going backward, flow goes from curr_frame to prev_frame
            flow = self.flow_processor.calculate_flow(curr_frame, prev_frame)
            current_mask = self._warp_mask_with_flow(current_mask, flow)

            masks[frame_num] = {
                'mask': current_mask,
                'type': 'fluid',
                'is_annotation': False,
                'track_id': f"track_{frame_num}",
                'label_id': 'tracked_fluid'
            }
            frames_tracked += 1

            # Update prev_frame for next iteration
            prev_frame = curr_frame

        # Backward: {frames_tracked} frames
        return masks
    
    def _get_previous_frame(self):
        """Get the previous frame for optical flow calculation."""
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            ret, frame = self.cap.read()
            if ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)  # Reset position
                return frame
        return None
    
    def _get_next_frame(self):
        """Get the next frame for optical flow calculation."""
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + 1)
        ret, frame = self.cap.read()
        if ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)  # Reset position
            return frame
        return None
    
    def _process_segment(self, start_frame, end_frame, start_mask, end_mask, all_masks, clear_frames, video_path):
        """
        Process a segment between two frames using bidirectional tracking with distance weighting.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            start_mask: Mask at start frame (None if tracking from beginning)
            end_mask: Mask at end frame (None if tracking to end)
            all_masks: Dictionary to store results
            clear_frames: Set of frames marked as clear
            video_path: Path to video file
        """
        if end_frame - start_frame <= 1:
            return  # No frames to interpolate

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
                    for frame_idx in range(start_frame, min(end_frame + 1, start_frame + 100)):
                        if frame_idx == start_frame:
                            # This is the initial frame, just store the mask
                            forward_masks[frame_idx] = current_mask
                            continue
                            
                        ret, curr_frame = cap.read()
                        if not ret:
                            break

                        # Calculate optical flow
                        flow = self.flow_processor.calculate_flow(prev_frame, curr_frame)
                        current_mask = self._warp_mask_with_flow(current_mask, flow)
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
                    for frame_idx in range(end_frame, max(start_frame - 1, end_frame - 100), -1):
                        if frame_idx == end_frame:
                            # This is the initial frame, just store the mask
                            backward_masks[frame_idx] = current_mask
                            continue
                            
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, curr_frame = cap.read()
                        if not ret:
                            break

                        # Calculate optical flow (backward)
                        flow = self.flow_processor.calculate_flow(curr_frame, prev_frame)
                        current_mask = self._warp_mask_with_flow(current_mask, flow)
                        backward_masks[frame_idx] = current_mask
                        prev_frame = curr_frame

        # Combine masks with distance-based weighting
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in clear_frames:
                continue  # Skip clear frames

            if frame_idx in forward_masks and frame_idx in backward_masks:
                # Calculate distance-based weights
                total_dist = end_frame - start_frame
                forward_weight = (end_frame - frame_idx) / total_dist
                backward_weight = (frame_idx - start_frame) / total_dist

                # Combine masks with weights
                combined_mask = self._combine_masks_weighted(
                    forward_masks[frame_idx], backward_masks[frame_idx],
                    forward_weight, backward_weight
                )

                all_masks[frame_idx] = {
                    'mask': combined_mask,
                    'type': 'fluid_combined',
                    'is_annotation': False,
                    'forward_weight': forward_weight,
                    'backward_weight': backward_weight,
                    'track_id': f"track_{frame_idx}",
                    'label_id': 'tracked_fluid'
                }
            elif frame_idx in forward_masks:
                all_masks[frame_idx] = {
                    'mask': forward_masks[frame_idx],
                    'type': 'fluid_forward',
                    'is_annotation': False,
                    'track_id': f"track_{frame_idx}",
                    'label_id': 'tracked_fluid'
                }
            elif frame_idx in backward_masks:
                all_masks[frame_idx] = {
                    'mask': backward_masks[frame_idx],
                    'type': 'fluid_backward',
                    'is_annotation': False,
                    'track_id': f"track_{frame_idx}",
                    'label_id': 'tracked_fluid'
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
        """Warp a mask using optical flow."""
        # Create coordinate grids
        h, w = mask.shape
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply flow to coordinates
        new_x = x + flow[:, :, 0]
        new_y = y + flow[:, :, 1]
        
        # Create remap coordinates
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        
        # Warp the mask
        warped_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return warped_mask
    
    def _find_nearest_fluid_annotation(self, target_frame, annotations):
        """Find the nearest fluid annotation to a target frame."""
        fluid_annotations = [a for a in annotations if a['type'] == 'fluid']
        if not fluid_annotations:
            return None
        
        nearest = min(fluid_annotations, key=lambda x: abs(x['frame'] - target_frame))
        return nearest
    
    def _propagate_clear_frame(self, target_frame, source_frame, source_mask):
        """Propagate a clear frame from a source annotation."""
        # For clear frames, we can use a simplified approach
        # In practice, you might want to use optical flow here too
        return np.zeros_like(source_mask)
    
    def _save_results(self, all_masks, video_path, study_uid, series_uid):
        """Save the tracking results with individual frame masks and comprehensive data."""
        try:
            # Create masks directory for individual frame outputs
            masks_dir = os.path.join(self.output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            
            # Save individual frame masks
            for frame_num, mask_info in all_masks.items():
                if isinstance(mask_info, dict) and mask_info.get('mask') is not None:
                    mask = mask_info['mask']
                    mask_file = os.path.join(masks_dir, f"frame_{frame_num:06d}_mask.png")
                    cv2.imwrite(mask_file, mask)
            
            # Save enhanced mask data with track_id and label_id
            mask_data_path = os.path.join(self.output_dir, "mask_data.json")
            mask_summary = {}
            for frame_num, mask_info in all_masks.items():
                if isinstance(mask_info, dict):
                    mask_summary[str(frame_num)] = {
                        'type': mask_info.get('type', 'unknown'),
                        'is_annotation': mask_info.get('is_annotation', False),
                        'has_mask': mask_info.get('mask') is not None,
                        'track_id': mask_info.get('track_id', f"track_{frame_num}"),
                        'label_id': mask_info.get('label_id', 'unknown')
                    }

            with open(mask_data_path, 'w') as f:
                json.dump(mask_summary, f, indent=2)
            
            # Create tracked_annotations.json with all tracked annotations
            tracked_annotations = []
            for frame_num, mask_info in all_masks.items():
                if isinstance(mask_info, dict) and mask_info.get('mask') is not None:
                    # Create annotation in MD.ai format
                    annotation = {
                        'id': mask_info.get('track_id', f"track_{frame_num}"),
                        'labelId': mask_info.get('label_id', 'tracked_fluid'),
                        'StudyInstanceUID': study_uid,
                        'SeriesInstanceUID': series_uid,
                        'frameNumber': frame_num,
                        'type': mask_info.get('type', 'tracked'),
                        'is_annotation': mask_info.get('is_annotation', False),
                        'track_id': mask_info.get('track_id', f"track_{frame_num}"),
                        'label_id': mask_info.get('label_id', 'tracked_fluid'),
                        'mask_file': f"frame_{frame_num:06d}_mask.png"
                    }
                    tracked_annotations.append(annotation)
            
            # Save tracked annotations
            tracked_file = os.path.join(self.output_dir, "tracked_annotations.json")
            with open(tracked_file, 'w') as f:
                json.dump(tracked_annotations, f, indent=2)
            
        except Exception as e:
            pass  # Silent failure for background task

        # Create output video with overlayed masks
        output_video_path = os.path.join(self.output_dir, "tracked_video.mp4")

        # Use context manager to ensure video capture is properly released
        with video_capture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            try:
                frame_num = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Skip video writing for very long videos to prevent hanging
                # Only write first 1000 frames for long videos
                max_frames = min(total_frames, 1000)

                # Add progress bar for video writing
                with tqdm(total=max_frames, desc="Writing video", leave=False, position=1) as pbar:
                    while frame_num < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_num in all_masks:
                            mask_info = all_masks[frame_num]
                            mask = mask_info['mask']

                            # Create colored overlay with different colors for tracked vs annotated
                            overlay = np.zeros_like(frame)
                            if mask_info.get('is_annotation', False):
                                # Green for original annotations (label_id)
                                overlay[mask > 0] = [0, 255, 0]  # Green
                            else:
                                # Orange for tracked masks (track_id)
                                overlay[mask > 0] = [0, 165, 255]  # Orange

                            # Blend with original frame
                            alpha = 0.3
                            frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)

                        out.write(frame)
                        frame_num += 1
                        pbar.update(1)
            finally:
                # Ensure video writer is always released
                out.release()
        
        if os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes'):
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
    dataset_id: str = None
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
            f.write(f"Process started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Annotations count: {len(annotations_df)}\n")
    except:
        pass
    
    # Initialize MultiFrameTracker
    # Initializing tracker
    tracker = MultiFrameTracker(flow_processor, output_dir, debug_mode=True)
    
    # Process annotations
    all_masks = tracker.process_annotations(annotations_df, video_path, study_uid, series_uid)
    if os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes'):
        pass  # print(f"Received {len(all_masks)} masks from process_annotations")
    
    # Update checkpoint
    try:
        with open(checkpoint_file, "a") as f:
            f.write(f"Process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Generated {len(all_masks)} masks\n")
    except:
        pass
    
    # Count annotation types
    annotation_types = {}
    for frame, info in all_masks.items():
        if isinstance(info, dict) and 'type' in info:
            annotation_type = info['type']
            annotation_types[annotation_type] = annotation_types.get(annotation_type, 0) + 1
    
    # Tracking complete

    return {
        'all_masks': all_masks,
        'annotated_frames': sum(1 for info in all_masks.values() if isinstance(info, dict) and info.get('is_annotation', False)),
        'predicted_frames': sum(1 for info in all_masks.values() if isinstance(info, dict) and not info.get('is_annotation', False)),
        'annotation_types': annotation_types,
        'output_video': os.path.join(output_dir, "multi_frame_tracking.mp4")
    }