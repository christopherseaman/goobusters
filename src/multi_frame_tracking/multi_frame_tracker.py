import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging
import json
import traceback

class MultiFrameTracker:
    """
    Implements multi-frame annotation tracking for ultrasound videos.
    
    This class handles different annotation scenarios such as:
    - All fluid annotations
    - Mixed fluid and clear annotations
    - Beginning and end of video handling
    """
    
    def __init__(self, flow_processor, output_dir, debug_mode=False):
        """
        Initialise the multi-frame tracker.
        
        Args:
            flow_processor: The optical flow processor to use
            output_dir: Directory to save outputs
            debug_mode: Whether to enable debug mode
        """
        self.flow_processor = flow_processor
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        
        #output directories
        os.makedirs(output_dir, exist_ok=True)
        self.debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        
      
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
    
        log_file = os.path.join(output_dir, f'multi_frame_tracker_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        
        self.logger.info("Initialized MultiFrameTracker")
    
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
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Study UID: {study_uid}, Series UID: {series_uid}")
        
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
        
        return all_masks
    
    def _classify_annotations(self, annotations_df, frame_height, frame_width):
        """
        Classify annotations as 'fluid' or 'clear' based on mask content.
        
        Args:
            annotations_df: DataFrame with annotations
            frame_height: Height of video frame
            frame_width: Width of video frame
            
        Returns:
            List of dictionaries with classified annotations
        """
        classified_annotations = []
        
        for _, row in annotations_df.iterrows():
            frame_num = int(row['frameNumber'])
            
            # Check for fluid content (polygons)
            data_foreground = row.get('free_fluid_foreground')
            if isinstance(data_foreground, list) and len(data_foreground) > 0:
                # This is a fluid annotation
                mask = self._create_mask_from_polygons(data_foreground, frame_height, frame_width)
                annotation_type = 'fluid'
            else:
                # This is a clear annotation (no fluid)
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
                video_path, 
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
            
            # Track forward from current
            forward_masks = self._track_between_frames(
                video_path, 
                start_frame, 
                end_frame, 
                current['mask'], 
                forward=True
            )
            
            # Track backward from next
            backward_masks = self._track_between_frames(
                video_path, 
                end_frame, 
                start_frame, 
                next_annotation['mask'], 
                forward=False
            )
            
            # Combine masks
            for frame_idx in range(start_frame + 1, end_frame):
                if frame_idx in forward_masks and frame_idx in backward_masks and frame_idx not in clear_frames:
                    combined_mask = self._combine_masks(
                        forward_masks[frame_idx],
                        backward_masks[frame_idx],
                        frame_idx,
                        start_frame,
                        end_frame
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
                video_path, 
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
                video_path, 
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
                video_path, 
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
    
    def _track_between_frames(self, video_path, start_frame, end_frame, start_mask, forward=True):
        """
        Track a mask between two frames using your existing track_frames function.
        
        Args:
            video_path: Path to the video file
            start_frame: Starting frame number
            end_frame: Ending frame number
            start_mask: Mask to track
            forward: Direction of tracking
            
        Returns:
            Dictionary mapping frame numbers to tracked masks
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        # Create debug directory
        debug_dir = os.path.join(self.debug_dir, f"track_{start_frame}_to_{end_frame}")
        if self.debug_mode:
            os.makedirs(debug_dir, exist_ok=True)
        
        try:
            # Call your existing track_frames function
            from optical_multi_frame import track_frames
            frames = track_frames(
                cap=cap,
                start_frame=start_frame,
                end_frame=end_frame,
                initial_mask=start_mask,
                debug_dir=debug_dir if self.debug_mode else None,
                forward=forward,
                pbar=None,
                flow_processor=self.flow_processor
            )
            
            # Convert list of tuples to dictionary
            result = {}
            for frame_data in frames:
                if len(frame_data) >= 3:
                    frame_idx, _, mask = frame_data[0], frame_data[1], frame_data[2]
                    result[frame_idx] = mask
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in tracking: {str(e)}")
            traceback.print_exc()
            return {}
        finally:
            cap.release()
    
    def _combine_masks(self, mask1, mask2, frame_idx, frame1_idx, frame2_idx):
        """
        Combine two masks using weighted average and union strategies.
        
        Args:
            mask1: First mask
            mask2: Second mask
            frame_idx: Current frame index
            frame1_idx: First frame index
            frame2_idx: Second frame index
            
        Returns:
            Combined mask
        """
        # Calculate weights based on distance
        total_distance = frame2_idx - frame1_idx
        if total_distance == 0:
            weight1 = 0.5
        else:
            weight1 = 1.0 - ((frame_idx - frame1_idx) / total_distance)
        weight2 = 1.0 - weight1
        
        # Convert to binary first for union operation
        binary1 = (mask1 > 0.5).astype(np.float32)
        binary2 = (mask2 > 0.5).astype(np.float32)
        
        # Step 1: Apply union operation (logical OR)
        union_mask = np.logical_or(binary1, binary2).astype(np.float32)
        
        # Step 2: Apply weighted average to areas where both masks have fluid
        intersection = np.logical_and(binary1, binary2).astype(np.float32)
        weighted_avg = (weight1 * mask1) + (weight2 * mask2)
        
        # Combine: use weighted average in intersection areas, use union elsewhere
        final_mask = intersection * weighted_avg + (union_mask - intersection)
        
     
        if self.debug_mode:
            debug_img = np.zeros((mask1.shape[0], mask1.shape[1] * 3), dtype=np.uint8)
            debug_img[:, :mask1.shape[1]] = (binary1 * 255).astype(np.uint8)
            debug_img[:, mask1.shape[1]:mask1.shape[1]*2] = (binary2 * 255).astype(np.uint8)
            debug_img[:, mask1.shape[1]*2:] = (final_mask * 255).astype(np.uint8)
            
            debug_path = os.path.join(self.debug_dir, f'combine_{frame_idx}_{frame1_idx}_{frame2_idx}.png')
            cv2.imwrite(debug_path, debug_img)
        
        return final_mask
    
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
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create a colour map for different annotation types
        colour_map = {
            'fluid': (0, 255, 0),  # Green for human fluid annotations
            'clear': None,         # No color for clear frames
            'predicted_fluid': (255, 0, 0),  # Blue for predicted fluid
            'predicted_fluid_combined': (255, 165, 0),  # Orange for combined predictions
            'predicted_clear': None  # No color for predicted clear
        }
        
        # Process each frame
        for frame_idx in range(frame_count):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we have a mask for this frame
            if frame_idx in all_masks:
                mask_info = all_masks[frame_idx]
                mask = mask_info['mask']
                mask_type = mask_info['type']
                
                # Apply overlay
                overlay_color = colour_map.get(mask_type)
                if overlay_color and np.sum(mask) > 0:
                    # Create binary mask
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    
                    # Create overlay
                    overlay = frame.copy()
                    overlay[binary_mask > 0] = overlay_color
                    
                    # Blend with original
                    alpha = 0.3
                    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
                
                # Add text
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Type: {mask_type}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        self.logger.info(f"Visualization saved to {output_path}")


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
                                         label_id_fluid=None, label_id_machine=None,
                                         project_id=None, dataset_id=None,  
                                         upload_to_mdai=False):
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
    # Initialise multi-frame tracker
    tracker = MultiFrameTracker(flow_processor, output_dir)
    
    # Process annotations
    all_masks = tracker.process_annotations(annotations_df, video_path, study_uid, series_uid)
    
    # Upload to MD.ai if requested
    if upload_to_mdai and mdai_client and label_id_fluid and label_id_machine:
        annotations = prepare_mdai_annotations(
            all_masks, study_uid, series_uid, label_id_fluid, label_id_machine
        )
        
        if annotations:
            print(f"Uploading {len(annotations)} annotations to MD.ai")
            
            try:
                # Delete existing machine annotations first
                from optical_multi_frame import delete_existing_annotations
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
    
    return {
        'all_masks': all_masks,
        'annotated_frames': sum(1 for info in all_masks.values() if isinstance(info, dict) and info.get('is_annotation', False)),
        'predicted_frames': sum(1 for info in all_masks.values() if isinstance(info, dict) and not info.get('is_annotation', False)),
        'annotation_types': annotation_types,
        'output_video': os.path.join(output_dir, "multi_frame_tracking.mp4")
    }