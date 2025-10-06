#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-contrib-python",  # IMPORTANT: Only contrib, not regular opencv-python
#     "numpy>=1.21.0,<3.0.0",  # Support both NumPy 1.x and 2.x
#     "pandas",
#     "python-dotenv",
#     "mdai==0.16.0",
#     "pydicom>=3.0.0",
#     "tqdm",
#     "scikit-image",
#     "scipy",
#     "pillow",
#     "pyyaml",
#     "torch>=2.0.0",  # Use newer PyTorch that supports NumPy 2.x
#     "torchvision>=0.15.0"  # Compatible with newer PyTorch
# ]
# ///

"""
Multi-Frame Optical Flow Tracker

This module implements sophisticated multi-frame tracking functionality
integrating advanced algorithms from the multi-frame tracking fork.

Key Features:
- True multi-frame temporal consistency
- Adaptive parameter management (SharedParams-style)
- Advanced occlusion handling and reappearance detection
- Quality-based tracking with genuine optical flow detection
- Temporal smoothing and trajectory management
- Compatible with all optical flow methods
"""

import cv2
import numpy as np
import os
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import defaultdict, deque
from datetime import datetime
from .opticalflowprocessor import OpticalFlowProcessor

# Fix for pydicom deprecation warning from mdai 0.16.0
# Approach: Monkey patch the pydicom module to prevent the deprecation warning
# This bridges the old API to the new API structure for future compatibility
import sys
import importlib.util

def patch_pydicom_for_forward_compatibility():
    """
    Patches pydicom to handle the deprecated API usage in a forward-compatible way.
    This eliminates the deprecation warning while maintaining functionality.
    """
    try:
        # Import pydicom to check current state
        import pydicom
        
        # The warning comes from pydicom/pixel_data_handlers/util.py when it imports
        # pack_bits from the deprecated location. We need to intercept this.
        if hasattr(pydicom, 'pixel_data_handlers') and hasattr(pydicom, 'pixels'):
            # Ensure the util module has pack_bits from the new location
            if hasattr(pydicom.pixel_data_handlers, 'util'):
                util_module = pydicom.pixel_data_handlers.util
                
                # If pack_bits exists in the new location, make sure it's accessible
                # from the old location without triggering the warning
                if hasattr(pydicom.pixels, 'pack_bits'):
                    # Replace the import in the util module to use the new API
                    util_module.pack_bits = pydicom.pixels.pack_bits
                    
                    # Patch the module's __getattr__ to redirect deprecated calls
                    original_getattr = getattr(pydicom.pixel_data_handlers, '__getattr__', None)
                    
                    def patched_getattr(name):
                        if name == 'pack_bits':
                            return pydicom.pixels.pack_bits
                        elif original_getattr:
                            return original_getattr(name)
                        else:
                            raise AttributeError(f"module has no attribute '{name}'")
                    
                    # Apply the patch if we have a modern pydicom
                    if hasattr(pydicom.pixel_data_handlers, '__getattr__'):
                        pydicom.pixel_data_handlers.__getattr__ = patched_getattr
        
        return True
    except (ImportError, AttributeError):
        return False

# Apply the compatibility patch
patch_pydicom_for_forward_compatibility()

# Import mdai with the patched pydicom
import mdai


def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def create_identity_file(video_output_dir: str, study_uid: str, series_uid: str, video_annotations, studies_data) -> None:
    """
    Create an identity YAML file for the video output folder containing metadata.
    
    Args:
        video_output_dir: Path to the video output directory
        study_uid: Study Instance UID
        series_uid: Series Instance UID  
        video_annotations: DataFrame containing annotation data for this video
        studies_data: DataFrame containing studies data with exam numbers
    """
    # Extract metadata from the first annotation (they should all have the same metadata)
    first_annotation = video_annotations.iloc[0]
    
    # Get exam number from studies data
    study_match = studies_data[studies_data['StudyInstanceUID'] == study_uid]
    exam_number = int(study_match.iloc[0]['number']) if not study_match.empty else 'Unknown'
    
    # Get all unique labels present in this video's annotations
    unique_labels = video_annotations[['labelId', 'labelName']].drop_duplicates()
    labels_info = unique_labels.to_dict('records')
    
    # Create identity data
    identity_data = {
        'study_instance_uid': study_uid,
        'series_instance_uid': series_uid,
        'exam_number': exam_number,
        'dataset_name': first_annotation.get('dataset', 'Unknown'),
        'dataset_id': first_annotation.get('datasetId', 'Unknown'),
        'created_at': datetime.now().isoformat(),
        'annotation_count': len(video_annotations),
        'labels': labels_info
    }
    
    # Write YAML file
    identity_file_path = os.path.join(video_output_dir, 'identity.yaml')
    try:
        with open(identity_file_path, 'w') as f:
            yaml.dump(identity_data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        pass  # Silent failure for background task


def copy_annotations_to_output(video_output_dir: str, video_annotations, annotations_data) -> None:
    """
    Copy input annotations JSON data to the video output directory.
    This contains all original annotations for the video EXCEPT any with track_id.
    
    Args:
        video_output_dir: Path to the video output directory
        video_annotations: DataFrame containing annotation data for this video
        annotations_data: Full annotations data structure
    """
    try:
        # Filter out any annotations that have track_id (these are generated, not input)
        input_annotations = []
        for annotation in annotations_data.get('annotations', []):
            # Only include annotations that don't have track_id (original annotations)
            if 'track_id' not in annotation:
                input_annotations.append(annotation)
        
        # Create input annotations data structure
        input_data = {
            'annotations': input_annotations,
            'studies': annotations_data.get('studies', []),
            'labels': annotations_data.get('labels', [])
        }
        
        # Save the input annotations data for this video
        annotations_file = os.path.join(video_output_dir, 'input_annotations.json')
        with open(annotations_file, 'w') as f:
            json.dump(input_data, f, indent=2, default=str)
        
    except Exception as e:
        pass  # Silent failure for background task


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnotationType:
    """Enumeration for annotation types."""
    FLUID = "fluid"
    CLEAR = "clear"
    UNKNOWN = "unknown"


class BidirectionalTrackingError(Exception):
    """Custom exception for bidirectional tracking failures."""
    pass


class AnnotationValidationError(Exception):
    """Custom exception for annotation validation failures."""
    pass


class MultiAnnotationProcessor:
    """
    Processes multiple annotations for bidirectional tracking with conflict resolution.
    
    Handles MD.ai annotation format with fallback to generic annotation dictionaries.
    Implements the four annotation scenarios from multiple_annotation_strategy.md:
    - F→F: Bidirectional tracking with temporal weighting
    - F→C: Forward tracking only
    - C→F: Backward tracking only
    - C→C: No tracking (maintain clear state)
    """
    
    def __init__(self, shared_params: 'SharedParams'):
        """
        Initialize the multi-annotation processor.
        
        Args:
            shared_params: SharedParams instance for configuration
        """
        self.shared_params = shared_params
        self.logger = logging.getLogger(__name__ + '.MultiAnnotationProcessor')
        
        # Configuration from SharedParams
        bp = shared_params.tracking_params.get('bidirectional_tracking', {})
        self.min_annotation_gap = bp.get('min_annotation_gap', 5)
        self.temporal_weighting_enabled = bp.get('temporal_weighting', True)
        self.conflict_resolution_method = bp.get('conflict_resolution_method', 'weighted_average')
        
        # Error handling configuration
        self.max_annotation_gap = bp.get('max_gap_size', 100)
        self.validate_annotations = bp.get('validate_annotations', True)
        self.skip_invalid_annotations = bp.get('skip_invalid_annotations', True)
        
    def parse_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse and normalize annotations from various formats.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            List of normalized annotation dictionaries with keys:
            - frame_number: int
            - type: str (AnnotationType.FLUID or AnnotationType.CLEAR)
            - mask: np.ndarray or None
            - metadata: dict with original annotation data
        """
        normalized_annotations = []
        
        for annotation in annotations:
            try:
                normalized = self._normalize_annotation(annotation)
                if normalized:
                    normalized_annotations.append(normalized)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse annotation: {e}")
                continue
        
        # Sort by frame number
        normalized_annotations.sort(key=lambda x: x['frame_number'])
        
        # Validate annotations if enabled
        if self.validate_annotations:
            normalized_annotations = self._validate_annotations(normalized_annotations)
        
        self.logger.info(f"Parsed {len(normalized_annotations)} valid annotations")
        return normalized_annotations
    
    def _validate_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate annotations for common issues and filter out invalid ones.
        
        Args:
            annotations: List of normalized annotations
            
        Returns:
            List of validated annotations
            
        Raises:
            AnnotationValidationError: If critical validation fails
        """
        valid_annotations = []
        validation_errors = []
        
        for i, annotation in enumerate(annotations):
            try:
                # Check required fields
                if 'frame_number' not in annotation:
                    raise AnnotationValidationError(f"Missing frame_number in annotation {i}")
                
                if 'type' not in annotation:
                    raise AnnotationValidationError(f"Missing type in annotation {i}")
                
                # Validate frame number
                frame_num = annotation['frame_number']
                if not isinstance(frame_num, int) or frame_num < 0:
                    raise AnnotationValidationError(f"Invalid frame_number {frame_num} in annotation {i}")
                
                # Validate annotation type
                if annotation['type'] not in [AnnotationType.FLUID, AnnotationType.CLEAR]:
                    raise AnnotationValidationError(f"Invalid annotation type {annotation['type']} in annotation {i}")
                
                # Validate fluid annotations have masks
                if annotation['type'] == AnnotationType.FLUID and annotation.get('mask') is None:
                    if self.skip_invalid_annotations:
                        self.logger.warning(f"Fluid annotation at frame {frame_num} missing mask - skipping")
                        continue
                    else:
                        raise AnnotationValidationError(f"Fluid annotation at frame {frame_num} missing mask")
                
                # Check for duplicate frame numbers
                existing_frames = [a['frame_number'] for a in valid_annotations]
                if frame_num in existing_frames:
                    if self.skip_invalid_annotations:
                        self.logger.warning(f"Duplicate annotation at frame {frame_num} - skipping")
                        continue
                    else:
                        raise AnnotationValidationError(f"Duplicate annotation at frame {frame_num}")
                
                valid_annotations.append(annotation)
                
            except AnnotationValidationError as e:
                validation_errors.append(str(e))
                if not self.skip_invalid_annotations:
                    raise
                self.logger.warning(f"Validation error: {e}")
        
        # Check if we have enough valid annotations
        if len(valid_annotations) == 0:
            self.logger.warning("No valid annotations found after validation")
            return []
        
        # Log validation summary
        if validation_errors:
            self.logger.warning(f"Validation completed with {len(validation_errors)} errors, "
                              f"{len(valid_annotations)} valid annotations remaining")
        
        return valid_annotations
    
    def _normalize_annotation(self, annotation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize a single annotation to standard format.
        
        Supports both MD.ai format and generic annotation dictionaries.
        """
        # Detect MD.ai format
        if 'StudyInstanceUID' in annotation or 'labelId' in annotation:
            return self._parse_mdai_annotation(annotation)
        
        # Generic format
        elif 'frame_number' in annotation:
            return self._parse_generic_annotation(annotation)
        
        else:
            self.logger.warning(f"Unknown annotation format: {list(annotation.keys())}")
            return None
    
    def _parse_mdai_annotation(self, annotation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse MD.ai format annotation."""
        try:
            # Extract frame number
            frame_number = annotation.get('SOPInstanceUID', 0)
            if isinstance(frame_number, str):
                # Try to extract frame number from UID
                frame_number = int(frame_number.split('.')[-1]) if '.' in frame_number else 0
            
            # Determine annotation type based on label ID
            label_id = annotation.get('labelId', '')
            # Check for EMPTY_ID (no fluid frame) or LABEL_ID_NO_FLUID
            empty_id = os.getenv("EMPTY_ID", "")
            no_fluid_id = os.getenv("LABEL_ID_NO_FLUID", "L_75K42J")
            annotation_type = AnnotationType.CLEAR if label_id in [empty_id, no_fluid_id] else AnnotationType.FLUID
            
            # Extract mask from data
            mask = None
            if 'data' in annotation and annotation_type == AnnotationType.FLUID:
                # Convert MD.ai polygon data to mask
                mask = self._create_mask_from_mdai_data(annotation['data'])
            
            return {
                'frame_number': frame_number,
                'type': annotation_type,
                'mask': mask,
                'metadata': annotation
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing MD.ai annotation: {e}")
            return None
    
    def _parse_generic_annotation(self, annotation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse generic annotation format."""
        try:
            frame_number = int(annotation['frame_number'])
            
            # Determine type
            annotation_type = annotation.get('type', AnnotationType.UNKNOWN)
            if annotation_type not in [AnnotationType.FLUID, AnnotationType.CLEAR]:
                # Try to infer from presence of mask
                has_mask = annotation.get('mask') is not None
                annotation_type = AnnotationType.FLUID if has_mask else AnnotationType.CLEAR
            
            # Extract mask
            mask = annotation.get('mask')
            if isinstance(mask, list):
                mask = np.array(mask)
            
            return {
                'frame_number': frame_number,
                'type': annotation_type,
                'mask': mask,
                'metadata': annotation
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing generic annotation: {e}")
            return None
    
    def _create_mask_from_mdai_data(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Create binary mask from MD.ai polygon data.
        
        This is a placeholder implementation - actual MD.ai data structure may vary.
        """
        # TODO: Implement based on actual MD.ai data format
        self.logger.warning("MD.ai mask creation not fully implemented")
        return None
    
    def detect_annotation_gaps(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect gaps between annotations that require tracking.
        
        Returns:
            List of gap dictionaries with keys:
            - start_annotation: dict
            - end_annotation: dict
            - gap_size: int
            - tracking_strategy: str (one of: bidirectional, forward_only, backward_only, none)
        """
        gaps = []
        
        for i in range(len(annotations) - 1):
            start_annotation = annotations[i]
            end_annotation = annotations[i + 1]
            
            gap_size = end_annotation['frame_number'] - start_annotation['frame_number'] - 1
            
            # Validate gap size
            if gap_size < 0:
                self.logger.error(f"Invalid gap: end frame {end_annotation['frame_number']} before start frame {start_annotation['frame_number']}")
                continue
            
            if gap_size >= self.min_annotation_gap:
                # Check if gap is too large
                if gap_size > self.max_annotation_gap:
                    self.logger.warning(f"Gap size {gap_size} exceeds maximum {self.max_annotation_gap} - may reduce tracking quality")
                
                tracking_strategy = self._determine_tracking_strategy(
                    start_annotation['type'],
                    end_annotation['type']
                )
                
                gaps.append({
                    'start_annotation': start_annotation,
                    'end_annotation': end_annotation,
                    'gap_size': gap_size,
                    'tracking_strategy': tracking_strategy
                })
                
                self.logger.info(f"Gap detected: frames {start_annotation['frame_number']}-{end_annotation['frame_number']} "
                               f"({gap_size} frames), strategy: {tracking_strategy}")
            elif gap_size > 0:
                self.logger.debug(f"Small gap ({gap_size} frames) between annotations - below minimum threshold {self.min_annotation_gap}")
        
        return gaps
    
    def _determine_tracking_strategy(self, start_type: str, end_type: str) -> str:
        """
        Determine tracking strategy based on annotation types.
        
        Implements the four scenarios from multiple_annotation_strategy.md:
        - F→F: bidirectional
        - F→C: forward_only
        - C→F: backward_only
        - C→C: none
        """
        if start_type == AnnotationType.FLUID and end_type == AnnotationType.FLUID:
            return "bidirectional"
        elif start_type == AnnotationType.FLUID and end_type == AnnotationType.CLEAR:
            return "forward_only"
        elif start_type == AnnotationType.CLEAR and end_type == AnnotationType.FLUID:
            return "backward_only"
        elif start_type == AnnotationType.CLEAR and end_type == AnnotationType.CLEAR:
            return "none"
        else:
            self.logger.warning(f"Unknown annotation type combination: {start_type} → {end_type}")
            return "none"

class SharedParams:
    """
    Manages tracking parameters that can be tuned based on feedback.
    
    This class stores parameters that the optical flow algorithm uses
    and provides methods to update them based on performance feedback.
    """
    
    def __init__(self, params_file: Optional[str] = None):
        """
        Initialize shared parameters with default values or from a file.
        
        Args:
            params_file: Optional path to a JSON file with parameter values
        """
        # Default tracking parameters based on the fork's implementation
        self.tracking_params = {
            # Flow algorithm parameters
            'flow_noise_threshold': 3.0,        # Threshold for flow noise filtering
            'flow_quality_threshold': 0.7,      # Quality threshold for optical flow
            'border_constraint_weight': 0.9,    # Weight for border constraints
            
            # Mask tracking parameters
            'mask_threshold': 0.5,              # Threshold for binary mask conversion
            'contour_min_area': 50,             # Minimum contour area to keep
            'morphology_kernel_size': 5,        # Kernel size for morphological operations
            
            # Multi-frame parameters
            'frame_window': 5,                  # Temporal window size
            'min_track_length': 3,              # Minimum frames for valid track
            'max_displacement_per_frame': 50,   # Maximum allowed point movement
            'occlusion_threshold': 0.5,         # Confidence threshold for occlusion
            'temporal_smoothing': True,         # Enable trajectory smoothing
            'trajectory_smoothing_window': 3,   # Smoothing window size
            'reappearance_threshold': 3,        # Frames to wait for reappearance
            'max_occlusion_frames': 5,          # Max frames before marking inactive
            
            # Learning parameters
            'learning_rate': 0.3,               # Rate at which corrections influence parameters
            'window_size': 30,                  # Window size for propagating corrections
            'distance_decay_factor': 1.5,       # Factor controlling how quickly influence decays with distance
            'iou_improvement_threshold': 0.2,   # Threshold for considering a correction significant
            
            # Bidirectional tracking parameters
            'bidirectional_tracking': {
                'enabled': False,                    # Enable bidirectional tracking mode
                'temporal_weighting': True,          # Use temporal distance weighting
                'min_annotation_gap': 5,             # Minimum frames between annotations to trigger bidirectional
                'conflict_resolution_method': 'weighted_average',  # Method for combining predictions
                'quality_threshold_for_combination': 0.6,  # Min quality to combine masks
                'max_gap_size': 100,                 # Maximum gap size for bidirectional tracking
                'fallback_to_single_direction': True,  # Fall back if bidirectional fails
                'validate_annotations': True,        # Enable annotation validation
                'skip_invalid_annotations': True     # Skip rather than fail on invalid annotations
            }
        }
        
        # Performance history for adapting parameters
        self.performance_history = []
        
        # Version tracking
        self.version = 1
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load parameters from file if provided
        if params_file and os.path.exists(params_file):
            self.load_from_file(params_file)
       
        # logger.info(f"SharedParams initialized (v{self.version}): "
        #            f"window={self.tracking_params['frame_window']}, "
        #            f"quality={self.tracking_params['flow_quality_threshold']:.3f}")
    
    def load_from_file(self, params_file: str) -> bool:
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
                
            # logger.info(f"Loaded parameters (version {self.version}) from {params_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")
            return False
    
    def save_to_file(self, params_file: str) -> bool:
        """Save current parameters to a JSON file"""
        try:
            data = {
                'version': self.version,
                'last_updated': self.last_updated,
                'tracking_params': self.tracking_params,
                'performance_history': self.performance_history[-50:],  # Keep last 50 entries
            }
            
            with open(params_file, 'w') as f:
                json.dump(convert_numpy_types(data), f, indent=2)
                
            # logger.info(f"Saved parameters (version {self.version}) to {params_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving parameters: {str(e)}")
            return False


class OpticalFlowTracker:
    """
    Enhanced multi-frame optical flow tracker with temporal consistency and robust point management.
    
    This implementation integrates sophisticated multi-frame tracking algorithms from the fork
    while maintaining compatibility with the existing interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the multi-frame optical flow tracker with configuration."""
        self.config = config
        
        # Get optical flow method from config (default to farneback)
        flow_method = config.get('optical_flow', {}).get('method', 'farneback')
        self.processor = OpticalFlowProcessor(flow_method)
        
        # Initialize SharedParams for advanced parameter management
        params_file = config.get('params_file')
        self.shared_params = SharedParams(params_file)
        
        # Update SharedParams with config values (config overrides file)
        if 'tracking' in config:
            self.shared_params.tracking_params.update(config['tracking'])
        
        # Extract key parameters for easy access
        tp = self.shared_params.tracking_params
        self.frame_window = tp['frame_window']
        self.min_track_length = tp['min_track_length']
        self.max_displacement_per_frame = tp['max_displacement_per_frame']
        self.occlusion_threshold = tp['occlusion_threshold']
        self.temporal_smoothing = tp['temporal_smoothing']
        self.trajectory_smoothing_window = tp['trajectory_smoothing_window']
        self.reappearance_threshold = tp['reappearance_threshold']
        self.max_occlusion_frames = tp['max_occlusion_frames']
        
        # Quality thresholds
        self.flow_quality_threshold = tp['flow_quality_threshold']
        self.flow_noise_threshold = tp['flow_noise_threshold']
        self.mask_threshold = tp['mask_threshold']
        
        # Bidirectional tracking configuration
        bp = tp.get('bidirectional_tracking', {})
        self.bidirectional_enabled = bp.get('enabled', False)
        self.temporal_weighting_enabled = bp.get('temporal_weighting', True)
        self.min_annotation_gap = bp.get('min_annotation_gap', 5)
        self.max_gap_size = bp.get('max_gap_size', 100)
        self.quality_threshold_for_combination = bp.get('quality_threshold_for_combination', 0.6)
        self.fallback_to_single_direction = bp.get('fallback_to_single_direction', True)
        
        # Initialize multi-annotation processor
        self.multi_annotation_processor = MultiAnnotationProcessor(self.shared_params)
        
        # Feature detection parameters (legacy compatibility)
        tracking_config = config.get('tracking', {})
        self.max_corners = tracking_config.get('max_corners', 100)
        self.quality_level = tracking_config.get('quality_level', 0.01)
        self.min_distance = tracking_config.get('min_distance', 10)
        self.block_size = tracking_config.get('block_size', 3)
        
        # State variables for multi-frame tracking
        self.frame_idx = 0
        self.trajectories = defaultdict(lambda: {
            'points': deque(maxlen=self.frame_window), 
            'confidence': deque(maxlen=self.frame_window), 
            'active': True,
            'last_seen_frame': -1,
            'creation_frame': -1,
            'occlusion_count': 0,
            'total_displacement': 0.0,
            'velocities': deque(maxlen=5),  # For velocity smoothing
            'quality_scores': deque(maxlen=self.frame_window),
            'genuine_tracking_count': 0,
            'low_quality_count': 0
        })
        self.next_track_id = 0
        self.frame_buffer = deque(maxlen=self.frame_window)
        self.previous_gray = None
        self.active_tracks = set()
        
        # Tracking statistics
        self.stats = {
            'total_frames_processed': 0,
            'genuine_flow_frames': 0,
            'low_quality_frames': 0,
            'tracking_failures': 0,
            'reappearances': 0,
            'long_tracks': 0
        }
        
        # logger.info(f"Initialized Multi-Frame OpticalFlowTracker (SharedParams v{self.shared_params.version})")
        # logger.info(f"  Frame window: {self.frame_window}, Min track length: {self.min_track_length}")
        # logger.info(f"  Flow quality threshold: {self.flow_quality_threshold:.3f}")
        # logger.info(f"  Temporal smoothing: {self.temporal_smoothing}")
    
    def detect_features(self, gray_frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Detect corner features in the frame."""
        corners = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=mask
        )
        return corners if corners is not None else np.array([]).reshape(0, 1, 2)
    
    def calculate_flow_quality(self, flow: np.ndarray, mask_region: np.ndarray) -> Dict[str, float]:
        """
        Calculate flow quality metrics based on the fork's sophisticated analysis.
        
        Args:
            flow: Optical flow field
            mask_region: Binary mask of the region of interest
            
        Returns:
            Dictionary containing flow quality metrics
        """
        if np.sum(mask_region) == 0:
            return {'mean_flow': 0.0, 'max_flow': 0.0, 'flow_consistency': 0.0, 'is_genuine': False}
        
        # Calculate flow magnitude
        flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        
        # Extract flow in the mask region
        valid_flow = flow_magnitude[mask_region > self.mask_threshold]
        
        if len(valid_flow) == 0:
            return {'mean_flow': 0.0, 'max_flow': 0.0, 'flow_consistency': 0.0, 'is_genuine': False}
        
        # Filter out noise using SharedParams noise threshold
        noise_threshold = self.flow_noise_threshold * np.std(valid_flow) + np.mean(valid_flow)
        clean_flow = valid_flow[valid_flow < noise_threshold]
        
        if len(clean_flow) == 0:
            clean_flow = valid_flow
        
        mean_flow = np.mean(clean_flow)
        max_flow = np.max(clean_flow)
        
        # Calculate flow consistency (lower std relative to mean indicates more consistent flow)
        flow_consistency = 1.0 / (1.0 + np.std(clean_flow) / (mean_flow + 1e-6))
        
        # Determine if this represents genuine tracking
        is_genuine = (mean_flow >= self.flow_quality_threshold and 
                     flow_consistency > 0.3 and
                     len(clean_flow) / len(valid_flow) > 0.7)
        
        return {
            'mean_flow': mean_flow,
            'max_flow': max_flow,
            'flow_consistency': flow_consistency,
            'is_genuine': is_genuine,
            'clean_flow_ratio': len(clean_flow) / len(valid_flow)
        }
    
    def apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations based on SharedParams."""
        kernel_size = self.shared_params.tracking_params['morphology_kernel_size']
        contour_min_area = self.shared_params.tracking_params['contour_min_area']
        
        if kernel_size <= 0:
            return mask
        
        # Apply morphological operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary_mask = (mask >= self.mask_threshold).astype(np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small contours
        if contour_min_area > 0:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(binary_mask)
            for contour in contours:
                if cv2.contourArea(contour) >= contour_min_area:
                    cv2.fillPoly(filtered_mask, [contour], 1)
            binary_mask = filtered_mask
        
        return binary_mask.astype(float)
    
    def calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate IoU between two masks."""
        mask1_binary = mask1 > self.mask_threshold
        mask2_binary = mask2 > self.mask_threshold
        
        intersection = np.logical_and(mask1_binary, mask2_binary)
        union = np.logical_or(mask1_binary, mask2_binary)
        
        if np.sum(union) == 0:
            return 1.0 if np.sum(intersection) == 0 else 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def create_new_tracks(self, points: np.ndarray, frame_idx: int) -> List[int]:
        """Create new tracks for detected points."""
        new_track_ids = []
        for point in points:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            trajectory = self.trajectories[track_id]
            trajectory['points'].append(point[0])
            trajectory['confidence'].append(1.0)
            trajectory['quality_scores'].append(1.0)
            trajectory['creation_frame'] = frame_idx
            trajectory['last_seen_frame'] = frame_idx
            trajectory['active'] = True
            trajectory['occlusion_count'] = 0
            trajectory['genuine_tracking_count'] = 0
            trajectory['low_quality_count'] = 0
            
            self.active_tracks.add(track_id)
            new_track_ids.append(track_id)
        
        return new_track_ids
    
    def update_tracks_with_flow(self, flow: np.ndarray, frame_idx: int) -> Tuple[List[int], List[int], int]:
        """
        Update tracks using dense optical flow with sophisticated quality analysis.
        
        Returns:
            Tuple of (updated_tracks, lost_tracks, genuine_count)
        """
        updated_tracks = []
        lost_tracks = []
        genuine_count = 0
        
        for track_id in list(self.active_tracks):
            trajectory = self.trajectories[track_id]
            if not trajectory['points']:
                continue
            
            prev_pt = trajectory['points'][-1]
            x, y = int(prev_pt[0]), int(prev_pt[1])
            
            # Check bounds
            if not (0 <= x < flow.shape[1] and 0 <= y < flow.shape[0]):
                trajectory['active'] = False
                self.active_tracks.discard(track_id)
                lost_tracks.append(track_id)
                continue
            
            # Get flow vector at point location
            flow_vector = flow[y, x]
            displacement = np.linalg.norm(flow_vector)
            
            # Check displacement threshold
            if displacement > self.max_displacement_per_frame:
                trajectory['occlusion_count'] += 1
                if trajectory['occlusion_count'] > self.max_occlusion_frames:
                    trajectory['active'] = False
                    self.active_tracks.discard(track_id)
                    lost_tracks.append(track_id)
                continue
            
            # Calculate new position
            new_pt = prev_pt + flow_vector
            
            # Create a small mask around the point for quality analysis
            mask_size = 10
            point_mask = np.zeros(flow.shape[:2], dtype=np.uint8)
            cv2.circle(point_mask, (x, y), mask_size, 1, -1)
            
            # Calculate flow quality
            quality_metrics = self.calculate_flow_quality(flow, point_mask)
            
            # Update trajectory
            trajectory['points'].append(new_pt)
            trajectory['confidence'].append(min(1.0, 1.0 / (displacement + 1)))
            trajectory['quality_scores'].append(quality_metrics['mean_flow'])
            trajectory['last_seen_frame'] = frame_idx
            trajectory['occlusion_count'] = 0
            trajectory['total_displacement'] += displacement
            trajectory['velocities'].append(displacement)
            
            if quality_metrics['is_genuine']:
                trajectory['genuine_tracking_count'] += 1
                genuine_count += 1
            else:
                trajectory['low_quality_count'] += 1
            
            updated_tracks.append(track_id)
        
        return updated_tracks, lost_tracks, genuine_count
    
    def update_tracks_with_lk(self, old_points: np.ndarray, new_points: np.ndarray, 
                             status: np.ndarray, frame_idx: int, track_ids: List[int]) -> Tuple[List[int], List[int], int]:
        """
        Update tracks using Lucas-Kanade optical flow.
        
        Returns:
            Tuple of (updated_tracks, lost_tracks, genuine_count)
        """
        updated_tracks = []
        lost_tracks = []
        genuine_count = 0
        
        for i, (track_id, old_pt, new_pt, st) in enumerate(zip(track_ids, old_points, new_points, status)):
            trajectory = self.trajectories[track_id]
            
            if st[0] == 1:  # Successfully tracked
                # Calculate displacement
                displacement = np.linalg.norm(new_pt[0] - old_pt[0])
                
                # Check for reasonable displacement
                if displacement < self.max_displacement_per_frame:
                    # Calculate quality score based on displacement consistency
                    velocities = list(trajectory['velocities'])
                    if velocities:
                        velocity_consistency = 1.0 / (1.0 + abs(displacement - np.mean(velocities)))
                    else:
                        velocity_consistency = 1.0
                    
                    # Determine if this is genuine tracking
                    is_genuine = (displacement > 0.5 and  # Some movement
                                 displacement < self.max_displacement_per_frame * 0.5 and  # Not too much
                                 velocity_consistency > 0.3)  # Consistent with history
                    
                    # Update trajectory
                    trajectory['points'].append(new_pt[0])
                    trajectory['confidence'].append(min(1.0, velocity_consistency))
                    trajectory['quality_scores'].append(displacement if is_genuine else 0.1)
                    trajectory['last_seen_frame'] = frame_idx
                    trajectory['occlusion_count'] = 0
                    trajectory['total_displacement'] += displacement
                    trajectory['velocities'].append(displacement)
                    
                    if is_genuine:
                        trajectory['genuine_tracking_count'] += 1
                        genuine_count += 1
                    else:
                        trajectory['low_quality_count'] += 1
                    
                    updated_tracks.append(track_id)
                else:
                    # Displacement too large - mark as lost
                    trajectory['active'] = False
                    self.active_tracks.discard(track_id)
                    lost_tracks.append(track_id)
            else:
                # Track lost
                trajectory['occlusion_count'] += 1
                
                # If occluded for too long, mark as inactive
                if trajectory['occlusion_count'] > self.max_occlusion_frames:
                    trajectory['active'] = False
                    self.active_tracks.discard(track_id)
                    lost_tracks.append(track_id)
                else:
                    # Still within occlusion threshold - predict position
                    predicted_pt = self._predict_point_position(track_id, frame_idx)
                    if predicted_pt is not None:
                        trajectory['points'].append(predicted_pt)
                        trajectory['confidence'].append(self.occlusion_threshold)
                        trajectory['quality_scores'].append(0.1)  # Low quality for prediction
                        updated_tracks.append(track_id)
        
        return updated_tracks, lost_tracks, genuine_count
    
    def _predict_point_position(self, track_id: int, frame_idx: int) -> Optional[np.ndarray]:
        """Predict point position based on trajectory history."""
        trajectory = self.trajectories[track_id]
        points = list(trajectory['points'])
        
        if len(points) < 2:
            return None
        
        # Simple linear prediction based on velocity history
        velocities = list(trajectory['velocities'])
        if velocities:
            # Use average of recent velocities for prediction
            avg_velocity = np.mean(velocities[-3:])  # Last 3 velocities
            last_pt = np.array(points[-1])
            
            # Predict direction based on last movement
            if len(points) >= 2:
                direction = np.array(points[-1]) - np.array(points[-2])
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                    predicted_pt = last_pt + direction * avg_velocity
                    return predicted_pt
        
        # Fallback: simple linear extrapolation
        if len(points) >= 2:
            last_pt = np.array(points[-1])
            second_last_pt = np.array(points[-2])
            velocity = last_pt - second_last_pt
            predicted_pt = last_pt + velocity
            return predicted_pt
        
        return None
    
    def smooth_trajectories(self):
        """Apply temporal smoothing to trajectories."""
        if not self.temporal_smoothing:
            return
        
        for track_id in self.active_tracks:
            trajectory = self.trajectories[track_id]
            points = list(trajectory['points'])
            
            if len(points) >= self.trajectory_smoothing_window:
                # Apply moving average smoothing
                smoothed_points = []
                window_size = min(self.trajectory_smoothing_window, len(points))
                
                for i in range(len(points)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(points), i + window_size // 2 + 1)
                    
                    window_points = points[start_idx:end_idx]
                    avg_point = np.mean(window_points, axis=0)
                    smoothed_points.append(avg_point)
                
                # Update trajectory with smoothed points
                trajectory['points'] = deque(smoothed_points, maxlen=self.frame_window)
    
    def cleanup_inactive_tracks(self):
        """Remove tracks that have been inactive for too long."""
        tracks_to_remove = []
        
        for track_id, trajectory in self.trajectories.items():
            if not trajectory['active']:
                # Check if track has been inactive for too long
                frames_since_last_seen = self.frame_idx - trajectory['last_seen_frame']
                if frames_since_last_seen > self.max_occlusion_frames:
                    tracks_to_remove.append(track_id)
                    # Update stats for long tracks
                    if len(trajectory['points']) >= self.min_track_length:
                        self.stats['long_tracks'] += 1
        
        for track_id in tracks_to_remove:
            del self.trajectories[track_id]
            self.active_tracks.discard(track_id)
    
    def get_active_points(self) -> Tuple[np.ndarray, List[int]]:
        """Get current active tracking points and their track IDs."""
        points = []
        track_ids = []
        
        for track_id in self.active_tracks:
            trajectory = self.trajectories[track_id]
            if trajectory['points'] and trajectory['active']:
                points.append(trajectory['points'][-1])
                track_ids.append(track_id)
        
        if points:
            return np.array(points).reshape(-1, 1, 2), track_ids
        else:
            return np.array([]).reshape(0, 1, 2), []
    
    def track_frame(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Track features in a single frame using multi-frame temporal consistency.
        
        Args:
            frame: Input frame (BGR or grayscale)
            mask: Optional mask for feature detection
            
        Returns:
            Dictionary containing tracking results with multi-frame metadata
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Store frame for processing
        self.frame_buffer.append({
            'gray': gray,
            'index': self.frame_idx,
            'mask': mask
        })
        
        results = {
            'frame_index': self.frame_idx,
            'points': [],
            'tracks': [],
            'flow_vectors': [],
            'metadata': {
                'total_points': 0,
                'active_tracks': 0,
                'new_tracks': 0,
                'lost_tracks': 0,
                'reappeared_tracks': 0,
                'genuine_tracking_frames': 0,
                'tracking_quality': 'unknown',
                'temporal_consistency': 0.0,
                'shared_params_version': self.shared_params.version
            }
        }
        
        # First frame - initialize tracking
        if self.frame_idx == 0:
            features = self.detect_features(gray, mask)
            if len(features) > 0:
                new_track_ids = self.create_new_tracks(features, self.frame_idx)
                
                for track_id in new_track_ids:
                    trajectory = self.trajectories[track_id]
                    point = trajectory['points'][-1]
                    
                    results['points'].append([float(x) for x in point.tolist()])
                    results['tracks'].append({
                        'id': int(track_id),
                        'point': [float(x) for x in point.tolist()],
                        'confidence': float(trajectory['confidence'][-1]),
                        'status': 'new',
                        'genuine_tracking_ratio': 0.0
                    })
                
                results['metadata']['total_points'] = len(features)
                results['metadata']['new_tracks'] = len(features)
                results['metadata']['active_tracks'] = len(features)
                results['metadata']['tracking_quality'] = 'initialized'
                
                # logger.info(f"Frame {self.frame_idx}: Initialized {len(features)} multi-frame tracks")
        
        else:
            # Multi-frame tracking for subsequent frames
            if self.previous_gray is not None:
                genuine_count = 0
                updated_tracks = []
                lost_tracks = []
                
                # Get optical flow method
                flow_method = self.config.get('optical_flow', {}).get('method', 'farneback')
                
                if flow_method == 'lucas_kanade':
                    # Use Lucas-Kanade for sparse tracking
                    current_points, track_ids = self.get_active_points()
                    
                    if len(current_points) > 0:
                        new_points, status, error = cv2.calcOpticalFlowPyrLK(
                            self.previous_gray, gray, current_points, None
                        )
                        
                        updated_tracks, lost_tracks, genuine_count = self.update_tracks_with_lk(
                            current_points, new_points, status, self.frame_idx, track_ids
                        )
                else:
                    # Use dense optical flow methods
                    flow = self.processor.calculate_flow(
                        self.previous_gray, gray, method=flow_method
                    )
                    
                    if flow is not None:
                        updated_tracks, lost_tracks, genuine_count = self.update_tracks_with_flow(
                            flow, self.frame_idx
                        )
                
                # Apply temporal smoothing
                self.smooth_trajectories()
                
                # Build results for updated tracks
                total_tracks = len(updated_tracks) + len(lost_tracks)
                temporal_consistency = len(updated_tracks) / total_tracks if total_tracks > 0 else 0.0
                
                for track_id in updated_tracks:
                    trajectory = self.trajectories[track_id]
                    if trajectory['points']:
                        point = trajectory['points'][-1]
                        confidence = trajectory['confidence'][-1]
                        
                        # Calculate genuine tracking ratio
                        total_tracking = trajectory['genuine_tracking_count'] + trajectory['low_quality_count']
                        genuine_ratio = trajectory['genuine_tracking_count'] / total_tracking if total_tracking > 0 else 0.0
                        
                        results['points'].append([float(x) for x in point.tolist()])
                        results['tracks'].append({
                            'id': int(track_id),
                            'point': [float(x) for x in point.tolist()],
                            'confidence': float(confidence),
                            'status': 'tracked',
                            'genuine_tracking_ratio': float(genuine_ratio),
                            'track_length': int(len(trajectory['points'])),
                            'total_displacement': float(trajectory['total_displacement'])
                        })
                        
                        # Calculate flow vector if we have previous position
                        if len(trajectory['points']) >= 2:
                            prev_point = trajectory['points'][-2]
                            flow_vector = point - prev_point
                            results['flow_vectors'].append({
                                'track_id': int(track_id),
                                'from': [float(x) for x in prev_point.tolist()],
                                'to': [float(x) for x in point.tolist()],
                                'vector': [float(x) for x in flow_vector.tolist()],
                                'magnitude': float(np.linalg.norm(flow_vector)),
                                'is_genuine': bool(genuine_ratio > 0.5)
                            })
                
                # Update statistics
                self.stats['total_frames_processed'] += 1
                self.stats['genuine_flow_frames'] += 1 if genuine_count > 0 else 0
                self.stats['low_quality_frames'] += 1 if genuine_count == 0 else 0
                self.stats['tracking_failures'] += len(lost_tracks)
                
                results['metadata']['lost_tracks'] = len(lost_tracks)
                results['metadata']['genuine_tracking_frames'] = genuine_count
                results['metadata']['temporal_consistency'] = temporal_consistency
                
                # Determine tracking quality
                if genuine_count > len(updated_tracks) * 0.7:
                    results['metadata']['tracking_quality'] = 'excellent'
                elif genuine_count > len(updated_tracks) * 0.4:
                    results['metadata']['tracking_quality'] = 'good'
                elif genuine_count > 0:
                    results['metadata']['tracking_quality'] = 'fair'
                else:
                    results['metadata']['tracking_quality'] = 'poor'
                
                # Detect new features if needed
                active_count = len(self.active_tracks)
                if active_count < self.max_corners // 2:
                    # Create detection mask avoiding existing tracks
                    detection_mask = mask.copy() if mask is not None else np.ones_like(gray, dtype=np.uint8) * 255
                    
                    for track_id in self.active_tracks:
                        trajectory = self.trajectories[track_id]
                        if trajectory['points']:
                            pt = trajectory['points'][-1].astype(int)
                            cv2.circle(detection_mask, tuple(pt), self.min_distance, 0, -1)
                    
                    new_features = self.detect_features(gray, detection_mask)
                    if len(new_features) > 0:
                        new_track_ids = self.create_new_tracks(new_features, self.frame_idx)
                        
                        for track_id in new_track_ids:
                            trajectory = self.trajectories[track_id]
                            point = trajectory['points'][-1]
                            
                            results['points'].append([float(x) for x in point.tolist()])
                            results['tracks'].append({
                                'id': int(track_id),
                                'point': [float(x) for x in point.tolist()],
                                'confidence': float(trajectory['confidence'][-1]),
                                'status': 'new',
                                'genuine_tracking_ratio': 0.0
                            })
                        
                        results['metadata']['new_tracks'] = len(new_track_ids)
                        # logger.info(f"Frame {self.frame_idx}: Added {len(new_track_ids)} new tracks")
                
                # Cleanup inactive tracks
                self.cleanup_inactive_tracks()
                
                # Update metadata
                results['metadata']['active_tracks'] = len(self.active_tracks)
                results['metadata']['total_points'] = len(results['points'])
                
                # logger.info(f"Frame {self.frame_idx}: {len(self.active_tracks)} active tracks, "
                #           f"quality={results['metadata']['tracking_quality']}, "
                #           f"genuine={genuine_count}, temporal_consistency={temporal_consistency:.2f}")
        
        # Store current frame for next iteration
        self.previous_gray = gray.copy()
        self.frame_idx += 1
        return results
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive multi-frame tracking statistics."""
        active_trajectories = [t for t in self.trajectories.values() if t['active']]
        all_trajectories = list(self.trajectories.values())
        
        # Calculate trajectory statistics
        total_displacement = 0
        total_trajectory_length = 0
        occlusion_counts = []
        genuine_ratios = []
        track_lengths = []
        
        for trajectory in all_trajectories:
            points = list(trajectory['points'])
            track_lengths.append(len(points))
            total_trajectory_length += len(points)
            total_displacement += trajectory.get('total_displacement', 0)
            occlusion_counts.append(trajectory['occlusion_count'])
            
            # Calculate genuine tracking ratio
            total_tracking = trajectory['genuine_tracking_count'] + trajectory['low_quality_count']
            if total_tracking > 0:
                genuine_ratios.append(trajectory['genuine_tracking_count'] / total_tracking)
        
        stats = {
            'total_tracks': len(all_trajectories),
            'active_tracks': len(active_trajectories),
            'completed_tracks': len(all_trajectories) - len(active_trajectories),
            'frames_processed': self.frame_idx,
            'average_track_length': np.mean(track_lengths) if track_lengths else 0,
            'longest_track': max(track_lengths) if track_lengths else 0,
            'shortest_track': min(track_lengths) if track_lengths else 0,
            'average_displacement_per_frame': total_displacement / total_trajectory_length if total_trajectory_length > 0 else 0,
            'total_displacement': total_displacement,
            'average_occlusion_count': np.mean(occlusion_counts) if occlusion_counts else 0,
            'max_occlusion_count': max(occlusion_counts) if occlusion_counts else 0,
            'average_genuine_ratio': np.mean(genuine_ratios) if genuine_ratios else 0,
            'temporal_consistency_ratio': len([t for t in all_trajectories if len(t['points']) >= self.min_track_length]) / len(all_trajectories) if all_trajectories else 0,
            'tracking_quality_stats': {
                'genuine_flow_frames': self.stats['genuine_flow_frames'],
                'low_quality_frames': self.stats['low_quality_frames'],
                'tracking_failures': self.stats['tracking_failures'],
                'long_tracks': self.stats['long_tracks']
            },
            'shared_params_version': self.shared_params.version
        }
        
        return stats
    
    def export_trajectories(self, format: str = 'json') -> Dict[str, Any]:
        """Export multi-frame trajectory data in specified format."""
        trajectory_data = {
            'metadata': {
                'total_frames': self.frame_idx,
                'total_tracks': len(self.trajectories),
                'active_tracks': len(self.active_tracks),
                'frame_window': self.frame_window,
                'min_track_length': self.min_track_length,
                'temporal_smoothing': self.temporal_smoothing,
                'shared_params_version': self.shared_params.version,
                'export_timestamp': datetime.now().isoformat(),
                'tracking_statistics': self.get_tracking_statistics()
            },
            'trajectories': {}
        }
        
        for track_id, trajectory in self.trajectories.items():
            points_list = [pt.tolist() for pt in trajectory['points']]
            confidence_list = list(trajectory['confidence'])
            quality_list = list(trajectory.get('quality_scores', []))
            velocities_list = list(trajectory.get('velocities', []))
            
            # Calculate trajectory metrics
            total_displacement = trajectory.get('total_displacement', 0)
            total_tracking = trajectory['genuine_tracking_count'] + trajectory['low_quality_count']
            genuine_ratio = trajectory['genuine_tracking_count'] / total_tracking if total_tracking > 0 else 0.0
            
            trajectory_data['trajectories'][str(track_id)] = {
                'points': points_list,
                'confidence': confidence_list,
                'quality_scores': quality_list,
                'velocities': velocities_list,
                'active': trajectory['active'],
                'length': len(points_list),
                'creation_frame': trajectory['creation_frame'],
                'last_seen_frame': trajectory['last_seen_frame'],
                'occlusion_count': trajectory['occlusion_count'],
                'total_displacement': total_displacement,
                'genuine_tracking_count': trajectory['genuine_tracking_count'],
                'low_quality_count': trajectory['low_quality_count'],
                'genuine_tracking_ratio': genuine_ratio,
                'average_velocity': np.mean(velocities_list) if velocities_list else 0,
                'max_velocity': np.max(velocities_list) if velocities_list else 0,
                'trajectory_smoothness': np.std(velocities_list) if len(velocities_list) > 1 else 0
            }
        
        if format.lower() == 'numpy':
            # Convert to numpy arrays for analysis
            np_data = {}
            for track_id, traj in trajectory_data['trajectories'].items():
                np_data[track_id] = {
                    'points': np.array(traj['points']),
                    'confidence': np.array(traj['confidence']),
                    'quality_scores': np.array(traj['quality_scores']),
                    'active': traj['active'],
                    'creation_frame': traj['creation_frame'],
                    'last_seen_frame': traj['last_seen_frame'],
                    'occlusion_count': traj['occlusion_count'],
                    'genuine_tracking_ratio': traj['genuine_tracking_ratio']
                }
            np_data['metadata'] = trajectory_data['metadata']
            return np_data
        
        return trajectory_data
    
    def combine_masks_with_temporal_weighting(self, forward_mask: np.ndarray, backward_mask: np.ndarray,
                                            current_frame: int, start_frame: int, end_frame: int) -> np.ndarray:
        """
        Combine forward and backward masks using temporal distance weighting.
        
        Args:
            forward_mask: Mask from forward tracking
            backward_mask: Mask from backward tracking
            current_frame: Current frame number
            start_frame: Start annotation frame number
            end_frame: End annotation frame number
            
        Returns:
            Combined mask using temporal weighting
            
        Raises:
            BidirectionalTrackingError: If mask combination fails
        """
        try:
            # Validate inputs
            if forward_mask is None or backward_mask is None:
                raise BidirectionalTrackingError("Cannot combine None masks")
            
            if forward_mask.shape != backward_mask.shape:
                raise BidirectionalTrackingError(f"Mask shape mismatch: {forward_mask.shape} vs {backward_mask.shape}")
            
            if current_frame < start_frame or current_frame > end_frame:
                raise BidirectionalTrackingError(f"Current frame {current_frame} outside range [{start_frame}, {end_frame}]")
            
            if not self.temporal_weighting_enabled:
                # Simple average if temporal weighting is disabled
                return (forward_mask.astype(np.float32) + backward_mask.astype(np.float32)) / 2.0
            
            # Calculate temporal weights
            total_distance = end_frame - start_frame
            if total_distance <= 0:
                logger.warning(f"Invalid frame range: start={start_frame}, end={end_frame}")
                return forward_mask  # Fallback to forward mask
            
            # Temporal distance weighting algorithm from architectural spec:
            forward_weight = (end_frame - current_frame) / total_distance
            backward_weight = (current_frame - start_frame) / total_distance
            
            # Validate weights
            if forward_weight < 0 or backward_weight < 0:
                raise BidirectionalTrackingError(f"Negative weights: forward={forward_weight}, backward={backward_weight}")
            
            # Ensure weights sum to 1.0
            weight_sum = forward_weight + backward_weight
            if weight_sum > 0:
                forward_weight /= weight_sum
                backward_weight /= weight_sum
            else:
                logger.warning("Zero weight sum - using equal weights")
                forward_weight = backward_weight = 0.5
            
            # Weighted combination
            combined_mask = (forward_weight * forward_mask.astype(np.float32) +
                            backward_weight * backward_mask.astype(np.float32))
            
            # Validate result
            if np.any(np.isnan(combined_mask)) or np.any(np.isinf(combined_mask)):
                raise BidirectionalTrackingError("Invalid values (NaN/inf) in combined mask")
            
            # Apply morphological operations for cleanup
            try:
                combined_mask = self.apply_morphological_operations(combined_mask)
            except Exception as e:
                logger.warning(f"Morphological operations failed: {e}")
                # Continue with uncleaned mask
            
            logger.debug(f"Combined masks at frame {current_frame}: "
                         f"forward_weight={forward_weight:.3f}, backward_weight={backward_weight:.3f}")
            
            return combined_mask
            
        except Exception as e:
            if isinstance(e, BidirectionalTrackingError):
                raise
            else:
                raise BidirectionalTrackingError(f"Mask combination failed: {e}")
    
    def track_bidirectional_between_annotations(self, video_path: str, start_annotation: Dict[str, Any],
                                              end_annotation: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """
        Perform bidirectional tracking between two annotations.
        
        Args:
            video_path: Path to video file
            start_annotation: Starting annotation dictionary
            end_annotation: Ending annotation dictionary
            
        Returns:
            Dictionary mapping frame numbers to predicted masks
        """
        start_frame = start_annotation['frame_number']
        end_frame = end_annotation['frame_number']
        start_mask = start_annotation['mask']
        end_mask = end_annotation['mask']
        
        if start_mask is None or end_mask is None:
            logger.warning(f"Missing mask data for bidirectional tracking between frames {start_frame}-{end_frame}")
            return {}
        
        # Track forward from start annotation
        forward_masks = self._track_forward_from_annotation(video_path, start_annotation, end_frame)
        
        # Track backward from end annotation
        backward_masks = self._track_backward_from_annotation(video_path, end_annotation, start_frame)
        
        # Combine masks with temporal weighting
        combined_masks = {}
        
        for frame_idx in range(start_frame + 1, end_frame):
            forward_mask = forward_masks.get(frame_idx)
            backward_mask = backward_masks.get(frame_idx)
            
            if forward_mask is not None and backward_mask is not None:
                # Both directions successful - combine with temporal weighting
                combined_mask = self.combine_masks_with_temporal_weighting(
                    forward_mask, backward_mask, frame_idx, start_frame, end_frame
                )
                combined_masks[frame_idx] = combined_mask
                
            elif forward_mask is not None:
                # Only forward tracking successful
                combined_masks[frame_idx] = forward_mask
                logger.debug(f"Frame {frame_idx}: Using forward mask only")
                
            elif backward_mask is not None:
                # Only backward tracking successful
                combined_masks[frame_idx] = backward_mask
                logger.debug(f"Frame {frame_idx}: Using backward mask only")
                
            else:
                # Both failed - log warning
                logger.warning(f"Both forward and backward tracking failed for frame {frame_idx}")
        
        logger.info(f"Bidirectional tracking completed: {len(combined_masks)} frames between {start_frame}-{end_frame}")
        return combined_masks
    
    def _track_forward_from_annotation(self, video_path: str, annotation: Dict[str, Any],
                                     end_frame: int) -> Dict[int, np.ndarray]:
        """Track forward from an annotation to a target frame."""
        start_frame = annotation['frame_number']
        initial_mask = annotation['mask']
        
        cap = cv2.VideoCapture(video_path)
        try:
            tracked_masks = self._track_frames_between(
                cap, start_frame, end_frame, initial_mask, forward=True
            )
            return tracked_masks
        finally:
            cap.release()
    
    def _track_backward_from_annotation(self, video_path: str, annotation: Dict[str, Any],
                                      start_frame: int) -> Dict[int, np.ndarray]:
        """Track backward from an annotation to a target frame."""
        end_frame = annotation['frame_number']
        initial_mask = annotation['mask']
        
        cap = cv2.VideoCapture(video_path)
        try:
            tracked_masks = self._track_frames_between(
                cap, start_frame, end_frame, initial_mask, forward=False
            )
            return tracked_masks
        finally:
            cap.release()
    
    def _track_frames_between(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int,
                            initial_mask: np.ndarray, forward: bool = True) -> Dict[int, np.ndarray]:
        """
        Core tracking algorithm between frames using optical flow.
        
        Args:
            cap: OpenCV VideoCapture object
            start_frame: Starting frame number
            end_frame: Ending frame number
            initial_mask: Initial mask to track
            forward: True for forward tracking, False for backward
            
        Returns:
            Dictionary mapping frame numbers to tracked masks
        """
        logger = logging.getLogger(__name__)
        
        # Initialize tracking variables
        tracked_masks = {}
        current_mask = initial_mask.copy()
        
        # Determine frame sequence
        if forward:
            frame_sequence = range(start_frame, end_frame + 1)
        else:
            frame_sequence = range(start_frame, end_frame - 1, -1)
        
        # Initialize frames
        frame_sequence = list(frame_sequence)
        if len(frame_sequence) < 2:
            return {frame_sequence[0]: current_mask} if frame_sequence else {}
        
        # Get initial frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_sequence[0])
        ret, prev_frame = cap.read()
        if not ret:
            logger.error(f"Failed to read initial frame {frame_sequence[0]}")
            return {}
        
        # Store initial mask
        tracked_masks[frame_sequence[0]] = current_mask.copy()
        
        # Track through sequence
        for i in range(1, len(frame_sequence)):
            target_frame = frame_sequence[i]
            
            # Read next frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, curr_frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {target_frame}, stopping tracking")
                break
            
            # Compute optical flow
            try:
                flow = self.optical_flow_processor.compute_flow(prev_frame, curr_frame)
                if flow is None:
                    logger.warning(f"Failed to compute optical flow for frame {target_frame}")
                    break
                
                # Warp mask using optical flow
                warped_mask = self._warp_mask_with_flow(current_mask, flow)
                
                # Quality assessment
                if not self._assess_tracking_quality(current_mask, warped_mask, flow):
                    logger.warning(f"Poor tracking quality at frame {target_frame}")
                    # Still continue but log the issue
                
                # Post-process mask
                processed_mask = self._post_process_mask(warped_mask)
                
                # Store result
                tracked_masks[target_frame] = processed_mask
                current_mask = processed_mask
                prev_frame = curr_frame
                
            except Exception as e:
                logger.error(f"Error tracking frame {target_frame}: {e}")
                break
        
        logger.debug(f"Tracked {len(tracked_masks)} frames from {start_frame} to {end_frame}")
        return tracked_masks
    
    def _warp_mask_with_flow(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp mask using optical flow field."""
        h, w = mask.shape[:2]
        
        # Create coordinate grid
        y, x = np.mgrid[0:h, 0:w]
        
        # Apply flow to get new coordinates
        new_x = x + flow[:, :, 0]
        new_y = y + flow[:, :, 1]
        
        # Remap mask using flow
        warped_mask = cv2.remap(
            mask.astype(np.float32),
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Convert back to binary mask
        return (warped_mask > 0.5).astype(np.uint8) * 255
    
    def _assess_tracking_quality(self, prev_mask: np.ndarray, curr_mask: np.ndarray,
                               flow: np.ndarray) -> bool:
        """Assess quality of tracking between frames."""
        try:
            # Calculate mask areas
            prev_area = np.sum(prev_mask > 0)
            curr_area = np.sum(curr_mask > 0)
            
            if prev_area == 0:
                return False
            
            # Area ratio check
            area_ratio = curr_area / prev_area
            if area_ratio < 0.3 or area_ratio > 3.0:
                return False
            
            # Calculate IoU if both masks have area
            if curr_area > 0:
                intersection = np.sum((prev_mask > 0) & (curr_mask > 0))
                union = np.sum((prev_mask > 0) | (curr_mask > 0))
                iou = intersection / union if union > 0 else 0
                
                if iou < 0.2:
                    return False
            
            # Flow magnitude check
            mask_region = prev_mask > 0
            if np.any(mask_region):
                flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                mean_flow = np.mean(flow_magnitude[mask_region])
                
                # Flag if flow is too large (likely noise)
                if mean_flow > 50:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return False
    
    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process tracked mask to improve quality."""
        # Get morphological kernel size from SharedParams
        kernel_size = getattr(self.shared_params, 'morphology_kernel_size', 3)
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply morphological operations
        # Close small gaps
        processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Open to remove noise
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        # Find largest connected component
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Keep only the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            processed = np.zeros_like(mask)
            cv2.fillPoly(processed, [largest_contour], 255)
        
        return processed
    
    def process_multiple_annotations(self, annotations: List[Dict[str, Any]], video_path: str,
                                   output_dir: str) -> bool:
        """
        Process video with multiple annotations using bidirectional tracking.
        
        Args:
            annotations: List of annotation dictionaries
            video_path: Path to input video
            output_dir: Directory to save results
            
        Returns:
            True if processing succeeded
        """
        if not self.bidirectional_enabled:
            logger.warning("Bidirectional tracking is disabled. Use single annotation processing.")
            return False
        
        if len(annotations) == 0:
            logger.warning("No annotations provided - skipping video processing")
            return True
        
        # Parse and normalize annotations
        normalized_annotations = self.multi_annotation_processor.parse_annotations(annotations)
        
        if len(normalized_annotations) == 0:
            logger.error("No valid annotations found after parsing")
            return False
        
        if len(normalized_annotations) == 1:
            logger.info("Single annotation detected - falling back to single annotation processing")
            single_annotation = normalized_annotations[0]
            return self.process_video(video_path, output_dir, f"single_{single_annotation['frame_number']}")
        
        # Detect gaps requiring tracking
        gaps = self.multi_annotation_processor.detect_annotation_gaps(normalized_annotations)
        
        if len(gaps) == 0:
            logger.info("No gaps detected between annotations - no tracking needed")
            return True
        
        # Process each gap according to its tracking strategy
        all_predicted_masks = {}
        
        for gap in gaps:
            try:
                gap_masks = self._process_annotation_gap(video_path, gap)
                all_predicted_masks.update(gap_masks)
                
            except Exception as e:
                logger.error(f"Failed to process gap {gap['start_annotation']['frame_number']}-"
                                f"{gap['end_annotation']['frame_number']}: {e}")
                if not self.fallback_to_single_direction:
                    return False
                continue
        
        # Save results
        self._save_multi_annotation_results(all_predicted_masks, normalized_annotations,
                                          video_path, output_dir)
        
        # Log final processing statistics
        self._log_processing_statistics(gaps, all_predicted_masks, normalized_annotations)
        
        logger.info(f"Multi-annotation processing completed: {len(all_predicted_masks)} predicted frames")
        return True
    
    def _log_processing_statistics(self, gaps: List[Dict[str, Any]], predicted_masks: Dict[int, np.ndarray],
                                 annotations: List[Dict[str, Any]]):
        """Log comprehensive statistics about the processing session."""
        total_gaps = len(gaps)
        successful_predictions = len(predicted_masks)
        
        # Count gaps by strategy
        strategy_counts = {}
        for gap in gaps:
            strategy = gap['tracking_strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Count annotation types
        fluid_annotations = sum(1 for a in annotations if a['type'] == AnnotationType.FLUID)
        clear_annotations = sum(1 for a in annotations if a['type'] == AnnotationType.CLEAR)
        
        # Calculate coverage statistics
        total_frames_in_gaps = sum(gap['gap_size'] for gap in gaps)
        coverage_percentage = (successful_predictions / total_frames_in_gaps * 100) if total_frames_in_gaps > 0 else 0
        
        logger.info("=== BIDIRECTIONAL TRACKING STATISTICS ===")
        logger.info(f"Annotations processed: {len(annotations)} ({fluid_annotations} fluid, {clear_annotations} clear)")
        logger.info(f"Gaps detected: {total_gaps}")
        logger.info(f"Tracking strategies used: {dict(strategy_counts)}")
        logger.info(f"Successful predictions: {successful_predictions}/{total_frames_in_gaps} ({coverage_percentage:.1f}%)")
        logger.info(f"Bidirectional config: weighting={self.temporal_weighting_enabled}, "
                        f"min_gap={self.min_annotation_gap}, max_gap={self.max_gap_size}")
        logger.info("==========================================")
    
    def _process_annotation_gap(self, video_path: str, gap: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """Process a single annotation gap according to its tracking strategy."""
        strategy = gap['tracking_strategy']
        start_annotation = gap['start_annotation']
        end_annotation = gap['end_annotation']
        gap_size = gap['gap_size']
        
        start_frame = start_annotation['frame_number']
        end_frame = end_annotation['frame_number']
        
        logger.info(f"Processing gap {start_frame}→{end_frame} ({gap_size} frames) using strategy: {strategy}")
        
        processing_start_time = time.time()
        
        try:
            if strategy == "bidirectional":
                logger.debug(f"Starting bidirectional tracking between frames {start_frame}-{end_frame}")
                result = self.track_bidirectional_between_annotations(video_path, start_annotation, end_annotation)
                
            elif strategy == "forward_only":
                logger.debug(f"Starting forward-only tracking from frame {start_frame} to {end_frame}")
                result = self._track_forward_from_annotation(video_path, start_annotation, end_frame)
                
            elif strategy == "backward_only":
                logger.debug(f"Starting backward-only tracking from frame {end_frame} to {start_frame}")
                result = self._track_backward_from_annotation(video_path, end_annotation, start_frame)
                
            elif strategy == "none":
                logger.debug(f"Creating clear masks for frames {start_frame}-{end_frame}")
                # No tracking - maintain clear state
                result = {}
                for frame_idx in range(start_frame + 1, end_frame):
                    result[frame_idx] = np.zeros((480, 640), dtype=np.float32)  # TODO: Get actual frame dimensions
                    
            else:
                raise BidirectionalTrackingError(f"Unknown tracking strategy: {strategy}")
            
            processing_time = time.time() - processing_start_time
            success_rate = len(result) / gap_size * 100 if gap_size > 0 else 0
            
            logger.info(f"Gap processing completed: {len(result)}/{gap_size} frames ({success_rate:.1f}%) "
                       f"in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - processing_start_time
            logger.error(f"Gap processing failed after {processing_time:.2f}s: {e}")
            raise BidirectionalTrackingError(f"Failed to process gap {start_frame}-{end_frame}: {e}")
    
    def _save_multi_annotation_results(self, predicted_masks: Dict[int, np.ndarray],
                                     annotations: List[Dict[str, Any]], video_path: str,
                                     output_dir: str):
        """Save results from multi-annotation processing."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predicted masks
        results = {
            'video_path': video_path,
            'annotations': annotations,
            'predicted_frames': list(predicted_masks.keys()),
            'bidirectional_config': {
                'enabled': self.bidirectional_enabled,
                'temporal_weighting': self.temporal_weighting_enabled,
                'min_gap': self.min_annotation_gap
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        results_file = os.path.join(output_dir, 'multi_annotation_results.json')
        with open(results_file, 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        logger.info(f"Multi-annotation results saved to {results_file}")
    
    def process_video(self, video_path: str, output_dir: str, annotation_id: str) -> bool:
        """
        Process a video with multi-frame tracking and save results.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            annotation_id: Identifier for this annotation
            
        Returns:
            True if processing succeeded
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False
            
            # Create output directory
            annotation_output_dir = os.path.join(output_dir, annotation_id)
            os.makedirs(annotation_output_dir, exist_ok=True)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # logger.info(f"Processing video: {video_path}")
            # logger.info(f"  Frames: {frame_count}, FPS: {fps}")
            # logger.info(f"  Output: {annotation_output_dir}")
            
            all_results = []
            
            # Process each frame
            with tqdm(total=frame_count, desc="Tracking frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Track frame
                    result = self.track_frame(frame)
                    all_results.append(result)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Active': result['metadata']['active_tracks'],
                        'Quality': result['metadata']['tracking_quality'],
                        'Genuine': result['metadata']['genuine_tracking_frames']
                    })
            
            cap.release()
            
            # Save comprehensive results
            results_file = os.path.join(annotation_output_dir, 'tracking_results.json')
            with open(results_file, 'w') as f:
                json.dump(convert_numpy_types({
                    'video_info': {
                        'path': video_path,
                        'frame_count': frame_count,
                        'fps': fps
                    },
                    'tracking_results': all_results,
                    'final_statistics': self.get_tracking_statistics(),
                    'processing_timestamp': datetime.now().isoformat()
                }), f, indent=2)
            
            # Export trajectories
            trajectories = self.export_trajectories('json')
            trajectories_file = os.path.join(annotation_output_dir, 'trajectories.json')
            with open(trajectories_file, 'w') as f:
                json.dump(convert_numpy_types(trajectories), f, indent=2)
            
            # Save SharedParams for future use
            params_file = os.path.join(annotation_output_dir, 'tracking_params.json')
            self.shared_params.save_to_file(params_file)
            
            # logger.info(f"Multi-frame tracking completed for {annotation_id}")
            # logger.info(f"  Results saved to: {results_file}")
            # logger.info(f"  Trajectories saved to: {trajectories_file}")
            
            # Print final statistics
            stats = self.get_tracking_statistics()
            # logger.info(f"Final Statistics:")
            # logger.info(f"  Total tracks: {stats['total_tracks']}")
            # logger.info(f"  Average track length: {stats['average_track_length']:.1f}")
            # logger.info(f"  Temporal consistency: {stats['temporal_consistency_ratio']:.2f}")
            # logger.info(f"  Average genuine ratio: {stats['average_genuine_ratio']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return False


# Main execution block
if __name__ == '__main__':
    # Import libraries and set constants
    from dotenv import load_dotenv
    import mdai
    import pandas as pd
    from tqdm import tqdm
    
    load_dotenv('dot.env')

    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')

    # Load optical flow methods from environment
    FLOW_METHOD = [method.strip() for method in os.getenv('FLOW_METHOD', 'farneback').split(',')]
    
    ACCESS_TOKEN = os.getenv('MDAI_TOKEN')
    DATA_DIR = os.getenv('DATA_DIR')
    DOMAIN = os.getenv('DOMAIN')
    PROJECT_ID = os.getenv('PROJECT_ID')
    DATASET_ID = os.getenv('DATASET_ID')
    ANNOTATIONS = os.path.join(DATA_DIR, os.getenv('ANNOTATIONS'))
    LABEL_ID = os.getenv('LABEL_ID')
    
    if ACCESS_TOKEN is None:
        print("ACCESS_TOKEN is not set, please set ACCESS_TOKEN in dot.env")
        exit()
    # else:
    #     print("ACCESS_TOKEN is set")
    # print(f"Using optical flow method: {FLOW_METHOD}")
    # print(f"DATA_DIR={DATA_DIR}")
    # print(f"DOMAIN={DOMAIN}")
    # print(f"PROJECT_ID={PROJECT_ID}")
    # print(f"DATASET_ID={DATASET_ID}")
    # print(f"ANNOTATIONS={ANNOTATIONS}")
    # print(f"LABEL_ID={LABEL_ID}")
    
    # Start MD.ai client (skip connection test if we have cached data)
    try:
        mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
    except Exception as e:
        # print(f"Warning: Could not connect to MD.ai ({e})")
        # print("Attempting to use cached data...")
        mdai_client = None
    
    # Download the dataset from MD.ai (or use cached version)
    if mdai_client:
        project = mdai_client.project(project_id=PROJECT_ID, path=DATA_DIR)
        dataset = project.get_dataset_by_id(DATASET_ID)
        BASE = dataset.images_dir
    else:
        # Use cached data directly
        BASE = os.path.join(DATA_DIR, 'mdai_ucsf_project_x9N2LJBZ_images_2025-09-18-050340')
    
    # Load the annotations
    annotations_data = mdai.common_utils.json_to_dataframe(ANNOTATIONS)
    annotations_df = pd.DataFrame(annotations_data['annotations'])
    labels = annotations_df['labelId'].unique()
    
    # Create the label map, LABEL_ID => 1, others in labels => 0
    labels_dict = {LABEL_ID: 1}
    if mdai_client:
        project.set_labels_dict(labels_dict)
        dataset.classes_dict = project.classes_dict
    
    # Filter annotations for the free fluid label
    free_fluid_annotations = annotations_df[annotations_df['labelId'] == LABEL_ID].copy()
    
    # Function to construct the video path
    def construct_video_path(base_dir, study_uid, series_uid):
        return os.path.join(base_dir, study_uid, f"{series_uid}.mp4")
    
    # Add video paths to the dataframe
    free_fluid_annotations['video_path'] = free_fluid_annotations.apply(
        lambda row: construct_video_path(BASE, row['StudyInstanceUID'], row['SeriesInstanceUID']), axis=1)
    
    # Check if video files exist
    free_fluid_annotations['file_exists'] = free_fluid_annotations['video_path'].apply(os.path.exists)
    
    # Count the number of annotations with and without corresponding video files
    num_with_files = free_fluid_annotations['file_exists'].sum()
    num_without_files = len(free_fluid_annotations) - num_with_files
    
    # print(f"Annotations with corresponding video files: {num_with_files}")
    # print(f"Annotations without corresponding video files: {num_without_files}")
    
    # Select annotations for processing
    TEST_STUDY_UID = os.getenv('TEST_STUDY_UID')
    TEST_SERIES_UID = os.getenv('TEST_SERIES_UID')

    if TEST_STUDY_UID and TEST_SERIES_UID:
        # Use specific test study if provided
        matched_annotations = free_fluid_annotations[
            (free_fluid_annotations['StudyInstanceUID'] == TEST_STUDY_UID) &
            (free_fluid_annotations['SeriesInstanceUID'] == TEST_SERIES_UID) &
            (free_fluid_annotations['file_exists'])
        ]
        # print(f"Using TEST_STUDY_UID: Found {len(matched_annotations)} annotations")
    elif DEBUG:
        matched_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']].sample(n=5, random_state=42)
    else:
        matched_annotations = free_fluid_annotations[free_fluid_annotations['file_exists']]
    
    # Performance optimization removed - no measurable benefit for this workload
    
    # Import the new multi-frame tracker and optical flow processor
    from .multi_frame_tracker import process_video_with_multi_frame_tracking
    from .opticalflowprocessor import OpticalFlowProcessor
    
    # Group annotations by video for multi-frame processing
    video_groups = matched_annotations.groupby(['StudyInstanceUID', 'SeriesInstanceUID'])

    # Main processing loop using true multi-frame tracking
    total_videos = len(video_groups)
    total_methods = len(FLOW_METHOD)
    print(f"\nFound {total_videos} videos with {total_methods} optical flow method(s): {', '.join(FLOW_METHOD)}")

    if total_videos > 10:
        estimated_time = total_videos * total_methods * 60  # ~60 seconds per video per method
        print(f"Estimated total time: ~{estimated_time // 60} minutes ({estimated_time // 3600} hours)")

    for method in FLOW_METHOD:
        print(f"\n{'='*60}")
        print(f"Running {method} optical flow method")
        print(f"{'='*60}")
        output_base_dir = os.path.join('output', method)
        os.makedirs(output_base_dir, exist_ok=True)

        # Process each video with all its annotations
        video_count = 0
        total_videos = len(video_groups)
        for (study_uid, series_uid), video_annotations in tqdm(video_groups, desc=f"{method}", position=0, leave=True):
            video_count += 1
            try:
                # Get video path (should be the same for all annotations in this group)
                video_path = video_annotations.iloc[0]['video_path']
                
                # Processing {len(video_annotations)} annotations
                
                # Create output directory for this video
                video_output_dir = os.path.join(output_base_dir, f"{study_uid}_{series_uid}")
                os.makedirs(video_output_dir, exist_ok=True)
                
                # Generate identity file for this video output folder
                create_identity_file(video_output_dir, study_uid, series_uid, video_annotations, annotations_data['studies'])
                
                # Copy original annotations JSON to video output directory
                copy_annotations_to_output(video_output_dir, video_annotations, annotations_data)
                
                # Initialize optical flow processor
                # print(f"Initializing OpticalFlowProcessor with method: {method}")
                flow_processor = OpticalFlowProcessor(method)
                # print("OpticalFlowProcessor initialized successfully")

                # Process the video with multi-frame tracking
                result = process_video_with_multi_frame_tracking(
                    video_path=video_path,
                    annotations_df=video_annotations,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    flow_processor=flow_processor,
                    output_dir=video_output_dir,
                    mdai_client=mdai_client,
                    label_id_fluid=LABEL_ID,
                    label_id_machine=os.getenv('LABEL_ID_MACHINE_GROUP', 'G_RJY6Qn'),
                    upload_to_mdai=False,  # Set to True if you want to upload results
                    project_id=PROJECT_ID,
                    dataset_id=DATASET_ID
                )

                # if DEBUG:
                #     print(f"✅ Successfully processed video with {method}")
                #     print(f"   - Annotated frames: {result['annotated_frames']}")
                #     print(f"   - Predicted frames: {result['predicted_frames']}")
                #     print(f"   - Annotation types: {result['annotation_types']}")
                #     print(f"   - Output video: {result['output_video']}")
                
            except Exception as e:
                logger.error(f"Error processing video {study_uid}/{series_uid} with {method}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*60)
    print("✅ All optical flow methods completed successfully")
    print("="*60)

    # Ensure clean exit
    import sys
    sys.stdout.flush()
    sys.stderr.flush()