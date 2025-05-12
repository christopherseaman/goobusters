import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging
import json
import traceback
from dotenv import load_dotenv

from .utils import print_mask_stats, visualize_flow, delete_existing_annotations, track_frames




import sys
import traceback



LABEL_ID_NO_FLUID = os.getenv("LABEL_ID_NO_FLUID", "L_75K42J") 

# Create a recursion detection system
call_stack = []
max_depth = 0




class MultiFrameTracker:
    """
    Implements multi-frame annotation tracking for ultrasound videos.
    
    This class handles different annotation scenarios such as:
    - All fluid annotations
    - Mixed fluid and clear annotations
    - Beginning and end of video handling
    """
    
    def __init__(self, flow_processor, output_dir, debug_mode=True):
        """
        Initialise the multi-frame tracker.
        
        Args:
            flow_processor: The optical flow processor to use
            output_dir: Directory to save outputs
            debug_mode: Whether to enable debug mode
        """
        self.cap = None
        self.flow_processor = flow_processor
        self.output_dir = output_dir
        self.debug_mode = True  # Always enable debug mode

        # When initializing MultiFrameTracker
        print("\n=== MultiFrameTracker Initialization ===")
        print(f"Using flow processor method: {flow_processor.method}")
        print(f"Debug mode: {debug_mode}")
        print(f"Output directory: {output_dir}")
        
        # Output directories
        os.makedirs(output_dir, exist_ok=True)
        self.debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Test debug file creation
        try:
            test_file = os.path.join(self.debug_dir, "debug_test.txt")
            with open(test_file, "w") as f:
                f.write(f"Debug test at {datetime.now()}")
            print(f"Successfully created test file at {test_file}")
        except Exception as e:
            print(f"ERROR creating debug test file: {str(e)}")
      
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        log_file = os.path.join(output_dir, f'multi_frame_tracker_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        
        self.logger.info("Initialized MultiFrameTracker")
        print(f"Debug mode is {'enabled' if self.debug_mode else 'disabled'}")
        print(f"Debug directory: {self.debug_dir}")
    
    def process_annotations(self, annotations_df, video_path, study_uid, series_uid):
        """
        Process annotations for a video and generate predictions.
        
        Args:
            annotations_df: df containing annotations
            video_path: Path to the video file
            study_uid: Study Instance UID
            series_uid: Series Instance UID
            
        Returns:
            Dictionary mapping frame numbers to masks
        """
        print("\n=== Processing annotations with MultiFrameTracker ===")
        print(f"Annotations count: {len(annotations_df)}")
        print(f"Video path: {video_path}")
        print(f"Study/Series UIDs: {study_uid}/{series_uid}")
    
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Study UID: {study_uid}, Series UID: {series_uid}")

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
           self.logger.error(f"Failed to open video: {video_path}")
           print(f"ERROR: Failed to open video: {video_path}")
           return {}
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Classify annotations as 'fluid' or 'clear'(based on md.ai labels)
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
        
        # Add all annotated frames to the results first
        for annotation in annotations:
            all_masks[annotation['frame']] = {
                'mask': annotation['mask'],
                'type': annotation['type'],
                'is_annotation': True
            }
        
        # Process segments between annotations
        for i in range(len(annotations)):
            current = annotations[i]
            
            # Handle segment from start to first annotation
            if i == 0 and current['frame'] > 0:
                self._process_start_segment(current, clear_frames, all_masks, video_path, total_frames)
            
            # Handle segment between current and next annotation
            if i < len(annotations) - 1:
                next_annotation = annotations[i + 1]
                self._process_middle_segment(current, next_annotation, clear_frames, all_masks, video_path)
            
            # Handle segment from last annotation to end
            if i == len(annotations) - 1 and current['frame'] < total_frames - 1:
                self._process_end_segment(current, clear_frames, all_masks, video_path, total_frames)
        
        # Create visualisation
        self._create_visualization(video_path, all_masks, annotations)

        if self.cap is not None:
           self.cap.release()
           self.cap = None
        
        return all_masks
    
    def _classify_annotations(self, annotations_df, frame_height, frame_width):
        """
        Classify annotations as 'fluid' or 'clear' based on mask content.
          """
        classified_annotations = []
    
        for _, row in annotations_df.iterrows():
            frame_num = int(row['frameNumber'])
          
        # Check if this is explicitly a "no fluid" annotation
            if row.get('labelId') == LABEL_ID_NO_FLUID:
            # This is a clear annotation (no fluid)
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                annotation_type = 'clear'
            elif isinstance(row.get('free_fluid_foreground'), list) and len(row.get('free_fluid_foreground')) > 0:
            # This is a fluid annotation
                mask = self._create_mask_from_polygons(row.get('free_fluid_foreground'), frame_height, frame_width)
                annotation_type = 'fluid'
            else:
            # Default to clear annotation if no fluid polygons
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                annotation_type = 'clear'
        
            classified_annotations.append({
                'frame': frame_num,
                'mask': mask,
                'type': annotation_type,
                'id': row.get('id', 'unknown')
            })
        
            self.logger.info(f"Classified frame {frame_num} as {annotation_type}")
    
        return classified_annotations
    
    def _identify_clear_frames(self, annotations, total_frames):
        """
        Identify frames that should be marked as clear based on annotations.
        
        Args:
            annotations: List of classified annotations
            total_frames: Total number of frames in the video
            
        Returns:
            Set of frame numbers that should be clear
        """
        clear_frames = set()
        
        # Get frame numbers and types
        frame_numbers = [a['frame'] for a in annotations]
        frame_types = [a['type'] for a in annotations]
        
        # Rule 1: If first annotation is clear, all frames before it are clear
        if frame_types[0] == 'clear':
            for frame in range(frame_numbers[0]):
                clear_frames.add(frame)
        
        # Rule 2: If last annotation is clear, all frames after it are clear
        if frame_types[-1] == 'clear':
            for frame in range(frame_numbers[-1] + 1, total_frames):
                clear_frames.add(frame)
        
        # Rule 3: All frames between two clear annotations are clear
        for i in range(len(annotations) - 1):
            if frame_types[i] == 'clear' and frame_types[i+1] == 'clear':
                for frame in range(frame_numbers[i] + 1, frame_numbers[i+1]):
                    clear_frames.add(frame)
        
        self.logger.info(f"Identified {len(clear_frames)} frames as clear")
        return clear_frames
    
    def _process_start_segment(self, first_annotation, clear_frames, all_masks, video_path, total_frames):
        """
        Process segment from start to first annotation.
        
        Args:
            first_annotation: First annotation in the video
            clear_frames: Set of frames identified as clear
            all_masks: Dictionary to store results
            video_path: Path to the video
            total_frames: Total number of frames
        """
        start_frame = 0
        end_frame = first_annotation['frame']
        
        if first_annotation['type'] == 'clear':
            # If first annotation is clear, all frames before it are clear
            self.logger.info(f"Setting frames 0 to {end_frame-1} as clear (no fluid)")
            empty_mask = np.zeros_like(first_annotation['mask'])
            
            for frame_idx in range(start_frame, end_frame):
                all_masks[frame_idx] = {
                    'mask': empty_mask,
                    'type': 'predicted_clear',
                    'source': f"clear_annotation_{end_frame}"
                }
        else:
            # If first annotation has fluid, track backward
            self.logger.info(f"Tracking from frame {end_frame} backward to start")
            backward_masks = self._track_between_frames(
                end_frame, 
                start_frame, 
                first_annotation['mask'], 
                forward=False
            )
            
            for frame_idx, mask in backward_masks.items():
                if frame_idx != end_frame and frame_idx not in clear_frames:
                    all_masks[frame_idx] = {
                        'mask': mask,
                        'type': 'predicted_fluid',
                        'source': f"backward_from_{end_frame}"
                    }
    
    def _process_middle_segment(self, current, next_annotation, clear_frames, all_masks, video_path):
        """
        Process segment between two annotations.
        
        Args:
            current: Current annotation
            next_annotation: Next annotation
            clear_frames: Set of frames identified as clear
            all_masks: Dictionary to store results
            video_path: Path to the video
        """
        start_frame = current['frame']
        end_frame = next_annotation['frame']
        
        # Skip if frames are adjacent
        if end_frame - start_frame <= 1:
            return
        
        # Case 1: Fluid to Fluid - track in both directions and combine
        if current['type'] == 'fluid' and next_annotation['type'] == 'fluid':
            self.logger.info(f"Tracking between fluid frames {start_frame} and {end_frame}")

            # Log the original masks
            self.logger.info(f"Current mask (frame {start_frame}): sum={np.sum(current['mask'])}")
            self.logger.info(f"Next mask (frame {end_frame}): sum={np.sum(next_annotation['mask'])}")
    
    # Save the original masks for visual inspection
            debug_dir = os.path.join(self.debug_dir, "fluid_to_fluid_debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"original_mask_{start_frame}.png"), 
               (current['mask'] * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(debug_dir, f"original_mask_{end_frame}.png"), 
               (next_annotation['mask'] * 255).astype(np.uint8))
            
            # Track forward from current
            forward_masks = self._track_between_frames(
                start_frame, 
                end_frame, 
                current['mask'], 
                forward=True
            )
            
            # Track backward from next
            backward_masks = self._track_between_frames(
                end_frame, 
                start_frame, 
                next_annotation['mask'], 
                forward=False
            )
            
            self.logger.info(f"Forward tracking results: {len(forward_masks)} frames")
            self.logger.info(f"Backward tracking results: {len(backward_masks)} frames")
        
        # Save some sample tracked masks
            for frame_idx in range(start_frame + 1, end_frame, max(1, (end_frame - start_frame) // 5)):
                if frame_idx in forward_masks:
                   cv2.imwrite(os.path.join(debug_dir, f"forward_{frame_idx}.png"), 
                           (forward_masks[frame_idx] * 255).astype(np.uint8))
                if frame_idx in backward_masks:
                   cv2.imwrite(os.path.join(debug_dir, f"backward_{frame_idx}.png"), 
                           (backward_masks[frame_idx] * 255).astype(np.uint8))
        
            # Combine masks
            for frame_idx in range(start_frame + 1, end_frame):
                if frame_idx in forward_masks and frame_idx in backward_masks and frame_idx not in clear_frames:
                    combined_mask = self._combine_masks(
                        [forward_masks[frame_idx], backward_masks[frame_idx]],
                        weights=[0.5, 0.5]
                    )
                    
                    all_masks[frame_idx] = {
                        'mask': combined_mask,
                        'type': 'predicted_fluid_combined',
                        'source': f"combined_{start_frame}_and_{end_frame}"
                    }
        
        # Case 2: Clear to Fluid - track backward from fluid only
        elif current['type'] == 'clear' and next_annotation['type'] == 'fluid':
            self.logger.info(f"Tracking backward from fluid frame {end_frame} to clear frame {start_frame}")
            
            backward_masks = self._track_between_frames(
                end_frame, 
                start_frame, 
                next_annotation['mask'], 
                forward=False
            )
            
            for frame_idx in range(start_frame + 1, end_frame):
                if frame_idx in backward_masks and frame_idx not in clear_frames:
                    all_masks[frame_idx] = {
                        'mask': backward_masks[frame_idx],
                        'type': 'predicted_fluid',
                        'source': f"backward_from_{end_frame}"
                    }
        
        # Case 3: Fluid to Clear - track forward from fluid only
        elif current['type'] == 'fluid' and next_annotation['type'] == 'clear':
            self.logger.info(f"Tracking forward from fluid frame {start_frame} to clear frame {end_frame}")
            
            forward_masks = self._track_between_frames(
                start_frame, 
                end_frame, 
                current['mask'], 
                forward=True
            )
            
            for frame_idx in range(start_frame + 1, end_frame):
                if frame_idx in forward_masks and frame_idx not in clear_frames:
                    all_masks[frame_idx] = {
                        'mask': forward_masks[frame_idx],
                        'type': 'predicted_fluid',
                        'source': f"forward_from_{start_frame}"
                    }
        
        # Case 4: Clear to Clear - all frames between are clear
        else:
            self.logger.info(f"Setting frames between clear frames {start_frame} and {end_frame} as clear")
            empty_mask = np.zeros_like(current['mask'])
            
            for frame_idx in range(start_frame + 1, end_frame):
                all_masks[frame_idx] = {
                    'mask': empty_mask,
                    'type': 'predicted_clear',
                    'source': f"between_clear_{start_frame}_and_{end_frame}"
                }

                

    
    def _process_end_segment(self, last_annotation, clear_frames, all_masks, video_path, total_frames):
        """
        Process segment from last annotation to end of video.
        
        Args:
            last_annotation: Last annotation in the video
            clear_frames: Set of frames identified as clear
            all_masks: Dictionary to store results
            video_path: Path to the video
            total_frames: Total number of frames
        """
        start_frame = last_annotation['frame']
        end_frame = total_frames - 1
        
        if last_annotation['type'] == 'clear':
            # If last annotation is clear, all frames after it are clear
            self.logger.info(f"Setting frames {start_frame+1} to {end_frame} as clear (no fluid)")
            empty_mask = np.zeros_like(last_annotation['mask'])
            
            for frame_idx in range(start_frame + 1, total_frames):
                all_masks[frame_idx] = {
                    'mask': empty_mask,
                    'type': 'predicted_clear',
                    'source': f"clear_annotation_{start_frame}"
                }
        else:
            # If last annotation has fluid, track forward
            self.logger.info(f"Tracking from frame {start_frame} forward to end")
            forward_masks = self._track_between_frames(
                start_frame, 
                end_frame, 
                last_annotation['mask'], 
                forward=True
            )
            
            for frame_idx, mask in forward_masks.items():
                if frame_idx != start_frame and frame_idx not in clear_frames:
                    all_masks[frame_idx] = {
                        'mask': mask,
                        'type': 'predicted_fluid',
                        'source': f"forward_from_{start_frame}"
                    }
    

    def _track_between_frames(self, start_frame, end_frame, initial_mask, forward=True, recursion_depth=0):
        """
        Track a mask between two frames using optical flow.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index
            initial_mask: Initial mask to track
            forward: Whether to track forward or backward
            recursion_depth: Current recursion depth
            
        Returns:
            List of tuples containing (frame_idx, frame, mask, flow, flow_mask, adjusted_mask)
        """
        # Anti-recursion protection
        MAX_RECURSION_DEPTH = 10
        if recursion_depth >= MAX_RECURSION_DEPTH:
            print(f"WARNING: Maximum recursion depth ({MAX_RECURSION_DEPTH}) reached in _track_between_frames.")
            print(f"  Current frame range: {start_frame} to {end_frame}")
            print(f"  Forward tracking: {forward}")
            print(f"  Frame range size: {abs(end_frame - start_frame)}")
            return {}
        
        # Validate frame range
        if abs(end_frame - start_frame) > 100:  # Limit frame range size
            print(f"WARNING: Frame range too large ({abs(end_frame - start_frame)} frames).")
            print(f"  Splitting into smaller ranges to prevent recursion issues.")
            return {}
        
        print(f"\nStarting _track_between_frames (recursion_depth={recursion_depth}):")
        print(f"  start_frame: {start_frame}")
        print(f"  end_frame: {end_frame}")
        print(f"  forward: {forward}")
        print(f"  frame range size: {abs(end_frame - start_frame)}")
        
        # Create debug directory
        debug_dir = os.path.join(self.debug_dir, f'track_{start_frame}_{end_frame}')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save test mask image
        test_mask_path = os.path.join(debug_dir, 'test_mask.png')
        cv2.imwrite(test_mask_path, (initial_mask * 255).astype(np.uint8))
        
        # Track frames
        frames = track_frames(
            self.cap,
            start_frame,
            end_frame,
            initial_mask,
            debug_dir=debug_dir,
            forward=forward,
            pbar=None,
            flow_processor=self.flow_processor,
            recursion_depth=recursion_depth
        )
        
        print(f"\n_track_between_frames completed:")
        print(f"  Frames returned: {len(frames)}")
        print(f"  Processing frames {start_frame} to {end_frame}")
        
        # Convert list to dictionary
        frames_dict = {}
        for frame_data in frames:
           if len(frame_data) >= 3:  # Ensure we have at least frame_idx, frame, and mask
            frame_idx = frame_data[0]
            mask = frame_data[2]
            frames_dict[frame_idx] = mask
    
        return frames_dict
    
    def _combine_masks(self, masks, weights=None):
        """Combine multiple masks with optional weights"""
        if not masks:
            return None
            
        # Convert masks to float32 for weighted combination
        masks = [mask.astype(np.float32) for mask in masks]
        
        if weights is None:
            weights = np.ones(len(masks)) / len(masks)
        
        # Weighted combination
        combined = np.zeros_like(masks[0])
        for mask, weight in zip(masks, weights):
            combined += mask * weight
            
        # Normalize
        combined = np.clip(combined, 0, 1)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined
    
    def _create_mask_from_polygons(self, polygons, frame_height, frame_width):
        """
        Create a mask from polygon data.
        
        Args:
            polygons: List of polygon points
            frame_height: Frame height
            frame_width: Frame width
            
        Returns:
            Binary mask
        """
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        
        return mask
    
    def _create_visualization(self, video_path, all_masks, annotations):
        """
        Create a visualization video with masks overlaid.
        
        Args:
            video_path: Path to the original video
            all_masks: Dictionary of masks for all frames
            annotations: List of original annotations
        """
        output_path = os.path.join(self.output_dir, "multi_frame_tracking.mp4")
        print(f"Creating visualization video: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create a color map for different annotation types (BGR format in OpenCV)
        colour_map = {
            'fluid': (0, 255, 0),  # Green for human annotations
            'clear': None,         # No color for clear frames
            'predicted_fluid': (0, 0, 255),  # Red for predicted fluid (BGR!)
            'predicted_fluid_combined': (0, 165, 255),  # Orange for combined predictions
            'predicted_clear': None  # No color for predicted clear
        }
        
        print(f"Processing {frame_count} frames with {len(all_masks)} masks")
        frames_with_masks = 0
        
        # Process each frame
        for frame_idx in range(frame_count):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}")
                break
            
            # Check if we have a mask for this frame
            if frame_idx in all_masks:
                mask_info = all_masks[frame_idx]
                mask = mask_info['mask']
                mask_type = mask_info['type']
                frames_with_masks += 1
                
                # Apply overlay with improved visibility
                overlay_color = colour_map.get(mask_type)
                if overlay_color and np.sum(mask) > 0:
                    # Create overlay with consistent alpha blending
                    overlay_frame = frame.copy()
                    overlay_alpha = 0.3  # Match the alpha used in debug visualization
                    
                    # Create binary mask for contour detection
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    
                    # Apply overlay with consistent alpha
                    overlay_frame[binary_mask > 0] = overlay_frame[binary_mask > 0] * (1 - overlay_alpha) + \
                                                    np.array(overlay_color, dtype=np.uint8) * overlay_alpha
                    
                    # Add mask contour for better definition
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_frame, contours, -1, overlay_color, 2)
                    
                    frame = overlay_frame
                
                # Add text info
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Type: {mask_type}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                coverage = np.mean(mask) * 100
                cv2.putText(frame, f"Coverage: {coverage:.1f}%", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to video
            out.write(frame)
            
            # Periodically report progress
            if frame_idx % 50 == 0 or frame_idx == frame_count - 1:
                print(f"Processed frame {frame_idx}/{frame_count}")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Visualization completed: {frames_with_masks} frames with masks out of {frame_count} total frames")
        self.logger.info(f"Visualization saved to {output_path}")

    def _apply_border_constraints(self, mask, frame):
        """Ensure mask stays within ultrasound region"""
        # Convert frame to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find ultrasound region
        _, thresh = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the region
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply border constraint
        mask = cv2.bitwise_and(mask, thresh)
        
        return mask

# Function to prepare annotations for MD.ai upload
def prepare_mdai_annotations(all_masks, study_uid, series_uid, label_id_fluid, label_id_machine):
    """
    Prepare annotations for MD.ai upload.
    
    Args:
        all_masks: Dictionary of masks
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        label_id_fluid: Label ID for fluid annotation
        label_id_machine: Label ID for machine group
        
    Returns:
        List of annotations in MD.ai format
    """
    import mdai
    annotations = []
    
    for frame_idx, mask_info in all_masks.items():
        # Skip clear frames (no fluid)
        if isinstance(mask_info, dict) and mask_info.get('type', '').endswith('clear'):
            continue
            
        # Skip human annotations (already in MD.ai)
        if isinstance(mask_info, dict) and mask_info.get('is_annotation', False):
            continue
            
        # Get the mask
        if isinstance(mask_info, dict) and 'mask' in mask_info:
            mask = mask_info['mask']
        else:
            mask = mask_info
        
        # Convert to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Skip empty masks
        if np.sum(binary_mask) == 0:
            continue
        
        # Convert mask to MD.ai format
        try:
            mask_data = mdai.common_utils.convert_mask_data(binary_mask)
            
            # Skip if no valid mask data
            if not mask_data:
                continue
            
            # Create annotation
            annotation = {
                'labelId': label_id_fluid,
                'StudyInstanceUID': study_uid,
                'SeriesInstanceUID': series_uid,
                'frameNumber': int(frame_idx),
                'data': mask_data,
                'groupId': label_id_machine
            }
            
            if isinstance(mask_info, dict) and 'source' in mask_info:
                annotation['note'] = f"Source: {mask_info['source']}"
            
            # Add confidence score for machine predictions
            if isinstance(mask_info, dict) and 'type' in mask_info and 'predicted' in mask_info['type']:
                annotation['confidence'] = 0.9  
            
            annotations.append(annotation)
            
        except Exception as e:
            print(f"Error creating annotation for frame {frame_idx}: {str(e)}")
            continue
    
    return annotations


#main processing function 

def process_video_with_multi_frame_tracking(video_path, annotations_df, study_uid, series_uid, 
                                         flow_processor, output_dir, mdai_client=None,
                                         label_id_fluid=None,label_id_no_fluid=None, label_id_machine=None,
                                         project_id=None, dataset_id=None,  
                                         upload_to_mdai=False,debug_mode=False):
    """
    Process a video using multi-frame tracking and optionally upload to MD.ai.
    
    Args:
        video_path: Path to the video file
        annotations_df: DataFrame containing annotations
        study_uid: Study Instance UID
        series_uid: Series Instance UID
        flow_processor: Optical flow processor
        output_dir: Directory to save outputs
        mdai_client: MD.ai client for uploads
        label_id_fluid: Label ID for fluid annotation
        label_id_machine: Label ID for machine group
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID
        upload_to_mdai: Whether to upload to MD.ai
        
    Returns:
        Dictionary with results
    """

    print("\n==== STARTING MULTI-FRAME TRACKING PROCESS ====")
    print(f"Video path: {video_path}")
    print(f"Annotations count: {len(annotations_df)}")
    print(f"Output directory: {output_dir}")
    print(f"Flow processor method: {flow_processor.method}")
    print(f"Debug mode: {debug_mode}")
    
    # Create a checkpoint file to verify this function was called
    checkpoint_file = os.path.join(output_dir, "multi_frame_tracking_started.txt")
    try:
        with open(checkpoint_file, "w") as f:
            f.write(f"Process started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Study/Series: {study_uid}/{series_uid}\n")
    except Exception as e:
        print(f"Warning: Could not create checkpoint file: {str(e)}")

    # Initialise multi-frame tracker
    tracker = MultiFrameTracker(flow_processor, output_dir, debug_mode=debug_mode)

    # Process annotations
    print("Calling tracker.process_annotations...")
    all_masks = tracker.process_annotations(annotations_df, video_path, study_uid, series_uid)
    print(f"Received {len(all_masks)} masks from process_annotations")
    
    # Update checkpoint
    try:
        with open(checkpoint_file, "a") as f:
            f.write(f"Process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Generated {len(all_masks)} masks\n")
    except:
        pass
    
    # Upload to MD.ai if requested
    if upload_to_mdai and mdai_client and label_id_fluid and label_id_machine:
        annotations = prepare_mdai_annotations(
            all_masks, study_uid, series_uid, label_id_fluid, label_id_machine
        )
        
        if annotations:
            print(f"Uploading {len(annotations)} annotations to MD.ai")
            
            try:
                # Delete existing machine annotations first
                
                deleted_count = delete_existing_annotations(
                    client=mdai_client,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    label_id=label_id_fluid,
                    group_id=label_id_machine
                )
                
                print(f"Deleted {deleted_count} existing annotations")
                
                # Upload new annotations
                upload_results = mdai_client.import_annotations(
                    annotations=annotations,
                    project_id=project_id,
                    dataset_id=dataset_id
                )
                
                print(f"Uploaded {len(annotations) - len(upload_results) if upload_results else len(annotations)} annotations to MD.ai")
                
            except Exception as e:
                print(f"Error in MD.ai upload: {str(e)}")
                traceback.print_exc()
    
    # Count annotation types
    annotation_types = {}
    for frame, info in all_masks.items():
        if isinstance(info, dict) and 'type' in info:
            annotation_type = info['type']
            annotation_types[annotation_type] = annotation_types.get(annotation_type, 0) + 1
    
    
    print("==== MULTI-FRAME TRACKING PROCESS COMPLETED ====")

    
    return {
        'all_masks': all_masks,
        'annotated_frames': sum(1 for info in all_masks.values() if isinstance(info, dict) and info.get('is_annotation', False)),
        'predicted_frames': sum(1 for info in all_masks.values() if isinstance(info, dict) and not info.get('is_annotation', False)),
        'annotation_types': annotation_types,
        'output_video': os.path.join(output_dir, "multi_frame_tracking.mp4")
    }

#testing 
def test_function():
    """Simple test function to verify imports work correctly"""
    print("Test function executed successfully")
    return "Success"

