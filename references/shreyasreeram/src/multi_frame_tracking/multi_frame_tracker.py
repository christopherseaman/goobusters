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

from .utils import visualize_flow, delete_existing_annotations, track_frames

import sys
import traceback

LABEL_ID_NO_FLUID = os.getenv("LABEL_ID_NO_FLUID", "L_75K42J") 

# Create a recursion detection system
call_stack = []
max_depth = 0

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
       
        print(f"\n🔧 SharedParams initialized:")
        print(f"🔧   - Version: {self.version}")
        print(f"🔧   - Window size: {self.tracking_params['window_size']}")
        print(f"🔧   - Flow quality: {self.tracking_params['flow_quality_threshold']:.3f}")
        print(f"🔧   - Learning rate: {self.tracking_params['learning_rate']:.3f}")
        if params_file:
            print(f"🔧   - Loaded from file: {params_file}")
        else:
            print(f"🔧   - Using default parameters")
        print(f"🔧 ==========================================\n")
    
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
                
            print(f"Loaded parameters (version {self.version}) from {params_file}")
            return True
        except Exception as e:
            print(f"Error loading parameters: {str(e)}")
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
                
            print(f"✅ PARAMETERS UPDATED: v{old_version} → v{self.version}")
            print(f"   Window size: {self.tracking_params['window_size']}")
            print(f"   Flow quality: {self.tracking_params['flow_quality_threshold']:.3f}")
            print(f"   Learning rate: {self.tracking_params['learning_rate']:.3f}")
            print(f"   Saved to: {params_file}")
            return True
        except Exception as e:
            print(f"Error saving parameters: {str(e)}")
            return False
    
    def update_from_feedback(self, metrics):
        """
        Enhanced parameter learning that adjusts 6+ parameters based on feedback metrics.
        Now includes learning_rate, mask_threshold, flow_noise_threshold, and more.
        """
        print(f"\n🔧 ===== ENHANCED PARAMETER LEARNING DEBUG =====")
        print(f"🔧 update_from_feedback() called!")
        print(f"🔧 Current IoU: {metrics.get('mean_iou', 0):.4f}")
        print(f"🔧 Current Dice: {metrics.get('mean_dice', 0):.4f}")
        print(f"🔧 Performance history length: {len(self.performance_history)}")
        print(f"🔧 Current parameters:")
        for param_name, param_value in self.tracking_params.items():
            print(f"🔧   - {param_name}: {param_value}")
        print(f"🔧 ==========================================\n")
        
        # Store performance history
        self.performance_history.append(metrics)
        
        current_iou = metrics.get('mean_iou', 0)
        current_dice = metrics.get('mean_dice', 0)
        iou_over_threshold = metrics.get('iou_over_0.7', 0)
        
        print(f"Current Performance: IoU={current_iou:.4f}, Dice={current_dice:.4f}, >0.7={iou_over_threshold*100:.1f}%")
        
        # Track which parameters we're changing
        changes_made = []
        improved = False
        
        # AGGRESSIVE MODE: Always try to improve if performance is below thresholds
        if current_iou < 0.65 or iou_over_threshold < 0.3:  # Lowered thresholds for more aggressive adaptation
            print(f"🔧 Performance below target - applying ENHANCED aggressive parameter adjustments")
            improved = True
            
            # 1. EXISTING: Window size adjustment (more aggressive)
            old_window = self.tracking_params['window_size']
            self.tracking_params['window_size'] = min(200, old_window + 20)  # Increased from +15 to +20
            changes_made.append(f"window_size: {old_window} → {self.tracking_params['window_size']}")
            
            # 2. EXISTING: Flow quality threshold (more aggressive)
            old_quality = self.tracking_params['flow_quality_threshold']
            self.tracking_params['flow_quality_threshold'] = max(0.1, old_quality - 0.15)  # Increased from -0.1 to -0.15
            changes_made.append(f"flow_quality_threshold: {old_quality:.3f} → {self.tracking_params['flow_quality_threshold']:.3f}")
            
            # 3. EXISTING: Learning rate (more aggressive)
            old_lr = self.tracking_params['learning_rate']
            self.tracking_params['learning_rate'] = min(0.9, old_lr + 0.25)  # Increased from +0.2 to +0.25
            changes_made.append(f"learning_rate: {old_lr:.3f} → {self.tracking_params['learning_rate']:.3f}")
            
            # 4. NEW: Flow noise threshold adjustment
            old_noise = self.tracking_params['flow_noise_threshold']
            self.tracking_params['flow_noise_threshold'] = max(1.0, old_noise - 0.7)  # More aggressive noise filtering
            changes_made.append(f"flow_noise_threshold: {old_noise:.3f} → {self.tracking_params['flow_noise_threshold']:.3f}")
            
            # 5. NEW: Mask threshold adjustment
            old_mask = self.tracking_params['mask_threshold']
            self.tracking_params['mask_threshold'] = max(0.25, old_mask - 0.15)  # More relaxed mask threshold
            changes_made.append(f"mask_threshold: {old_mask:.3f} → {self.tracking_params['mask_threshold']:.3f}")
            
            # 6. NEW: Border constraint weight
            old_border = self.tracking_params['border_constraint_weight']
            self.tracking_params['border_constraint_weight'] = max(0.3, old_border - 0.1)  # Relax border constraints
            changes_made.append(f"border_constraint_weight: {old_border:.3f} → {self.tracking_params['border_constraint_weight']:.3f}")
            
            # 7. NEW: Contour minimum area (relax to catch smaller features)
            old_contour = self.tracking_params['contour_min_area']
            self.tracking_params['contour_min_area'] = max(20, old_contour - 15)  # Allow smaller contours
            changes_made.append(f"contour_min_area: {old_contour} → {self.tracking_params['contour_min_area']}")
            
            # 8. NEW: Morphology kernel size
            old_morph = self.tracking_params['morphology_kernel_size']
            self.tracking_params['morphology_kernel_size'] = min(7, old_morph + 1)  # Slightly larger morphology
            changes_made.append(f"morphology_kernel_size: {old_morph} → {self.tracking_params['morphology_kernel_size']}")
            
            # 9. NEW: Distance decay factor (for learning propagation)
            old_decay = self.tracking_params['distance_decay_factor']
            self.tracking_params['distance_decay_factor'] = max(1.0, old_decay - 0.2)  # Less aggressive decay
            changes_made.append(f"distance_decay_factor: {old_decay:.3f} → {self.tracking_params['distance_decay_factor']:.3f}")
            
            print(f"🔧 ENHANCED ADAPTATION COMPLETE - Adjusted {len(changes_made)} parameters!")
            for change in changes_made:
                print(f"🔧   {change}")
            
            return improved
        
        # If we have history, compare with previous performance
        if len(self.performance_history) >= 2:
            previous_iou = self.performance_history[-2].get('mean_iou', 0)
            previous_dice = self.performance_history[-2].get('mean_dice', 0)
            previous_threshold = self.performance_history[-2].get('iou_over_0.7', 0)
            
            iou_change = current_iou - previous_iou
            dice_change = current_dice - previous_dice
            threshold_change = iou_over_threshold - previous_threshold
            
            print(f"🔧 Performance changes:")
            print(f"🔧   IoU: {previous_iou:.4f} → {current_iou:.4f} ({iou_change:+.4f})")
            print(f"🔧   Dice: {previous_dice:.4f} → {current_dice:.4f} ({dice_change:+.4f})")
            print(f"🔧   >0.7: {previous_threshold*100:.1f}% → {iou_over_threshold*100:.1f}% ({threshold_change*100:+.1f}%)")
            
            # SIGNIFICANT PERFORMANCE DROP - Emergency corrections
            if iou_change < -0.05 or dice_change < -0.05:
                print("🔧 SIGNIFICANT PERFORMANCE DROP DETECTED - Emergency parameter corrections!")
                improved = True
                
                # Emergency adjustments - be more aggressive
                old_window = self.tracking_params['window_size']
                old_quality = self.tracking_params['flow_quality_threshold']
                old_lr = self.tracking_params['learning_rate']
                old_noise = self.tracking_params['flow_noise_threshold']
                old_mask = self.tracking_params['mask_threshold']
                
                # Make larger adjustments
                self.tracking_params['window_size'] = min(200, old_window + 30)
                self.tracking_params['flow_quality_threshold'] = max(0.1, old_quality - 0.2)
                self.tracking_params['learning_rate'] = min(0.95, old_lr + 0.3)
                self.tracking_params['flow_noise_threshold'] = max(0.8, old_noise - 1.0)
                self.tracking_params['mask_threshold'] = max(0.1, old_mask - 0.2)
                
                changes_made = [
                    f"window_size: {old_window} → {self.tracking_params['window_size']} (EMERGENCY +30)",
                    f"flow_quality_threshold: {old_quality:.3f} → {self.tracking_params['flow_quality_threshold']:.3f} (EMERGENCY -0.2)",
                    f"learning_rate: {old_lr:.3f} → {self.tracking_params['learning_rate']:.3f} (EMERGENCY +0.3)",
                    f"flow_noise_threshold: {old_noise:.3f} → {self.tracking_params['flow_noise_threshold']:.3f} (EMERGENCY -1.0)",
                    f"mask_threshold: {old_mask:.3f} → {self.tracking_params['mask_threshold']:.3f} (EMERGENCY -0.2)"
                ]
                
                print(f"🔧 EMERGENCY CORRECTIONS APPLIED:")
                for change in changes_made:
                    print(f"🔧   {change}")
                
                return improved
            
            # GOOD IMPROVEMENT - Fine-tune parameters
            elif iou_change > 0.02 and dice_change > 0.02:
                print("🔧 Good improvement detected - fine-tuning parameters")
                improved = True
                
                # Smaller, more conservative adjustments when things are working
                old_window = self.tracking_params['window_size']
                old_quality = self.tracking_params['flow_quality_threshold']
                old_lr = self.tracking_params['learning_rate']
                
                self.tracking_params['window_size'] = min(100, old_window + 10)  # Smaller increase
                self.tracking_params['flow_quality_threshold'] = max(0.3, old_quality - 0.05)  # Smaller decrease
                self.tracking_params['learning_rate'] = min(0.8, old_lr + 0.1)  # Smaller increase
                
                changes_made = [
                    f"window_size: {old_window} → {self.tracking_params['window_size']} (fine-tune +10)",
                    f"flow_quality_threshold: {old_quality:.3f} → {self.tracking_params['flow_quality_threshold']:.3f} (fine-tune -0.05)",
                    f"learning_rate: {old_lr:.3f} → {self.tracking_params['learning_rate']:.3f} (fine-tune +0.1)"
                ]
                
                print(f"🔧 FINE-TUNING APPLIED:")
                for change in changes_made:
                    print(f"🔧   {change}")
                
                return improved
            
            # STAGNATION - Try different parameter combinations
            elif abs(iou_change) < 0.005 and abs(dice_change) < 0.005:
                print("🔧 Performance stagnation detected - trying alternative parameter strategy")
                improved = True
                
                # Try adjusting different parameters when stuck
                old_morph = self.tracking_params['morphology_kernel_size']
                old_contour = self.tracking_params['contour_min_area']
                old_border = self.tracking_params['border_constraint_weight']
                old_decay = self.tracking_params['distance_decay_factor']
                
                # Focus on morphological and geometric parameters
                self.tracking_params['morphology_kernel_size'] = max(3, old_morph - 1) if old_morph > 3 else min(9, old_morph + 2)
                self.tracking_params['contour_min_area'] = max(10, old_contour - 20)
                self.tracking_params['border_constraint_weight'] = max(0.4, old_border - 0.15)
                self.tracking_params['distance_decay_factor'] = max(0.8, old_decay - 0.3)
                
                changes_made = [
                    f"morphology_kernel_size: {old_morph} → {self.tracking_params['morphology_kernel_size']} (anti-stagnation)",
                    f"contour_min_area: {old_contour} → {self.tracking_params['contour_min_area']} (anti-stagnation)",
                    f"border_constraint_weight: {old_border:.3f} → {self.tracking_params['border_constraint_weight']:.3f} (anti-stagnation)",
                    f"distance_decay_factor: {old_decay:.3f} → {self.tracking_params['distance_decay_factor']:.3f} (anti-stagnation)"
                ]
                
                print(f"🔧 ANTI-STAGNATION STRATEGY APPLIED:")
                for change in changes_made:
                    print(f"🔧   {change}")
                
                return improved
            
            else:
                print(f"🔧 Performance change ({iou_change:+.4f} IoU, {dice_change:+.4f} Dice) not significant enough for parameter changes")
        else:
            print(f"🔧 Not enough performance history yet ({len(self.performance_history)} entries)")
        
        if not changes_made:
            print(f"🔧 No parameter changes made this iteration")
        
        return improved
    
class MultiFrameTracker:
    """
    Implements multi-frame annotation tracking for ultrasound videos.
    
    This class handles different annotation scenarios such as:
    - All fluid annotations
    - Mixed fluid and clear annotations
    - Beginning and end of video handling
    """
    
    def __init__(self, flow_processor, output_dir, debug_mode=True, shared_params=None):
        """
        Initialise the multi-frame tracker.
        
        Args:
            flow_processor: The optical flow processor to use
            output_dir: Directory to save outputs
            debug_mode: Whether to enable debug mode
            shared_params: SharedParams object with tracking parameters
        """
        self.cap = None
        self.flow_processor = flow_processor
        self.output_dir = output_dir
        self.debug_mode = True  # Always enable debug mode
        
        # Add these two flags for feedback loop functionality
        self.feedback_loop_mode = True  # Always enable feedback loop mode
        self.learning_mode = True  # Always enable learning mode
        
        # Set aggressive tracking parameters by default
        self.disable_annotation_copying = os.environ.get('DISABLE_ANNOTATION_COPYING', '0') == '1'
        self.bidirectional_tracking = os.environ.get('BIDIRECTIONAL_TRACKING', '1') == '1'
        self.force_tracking = os.environ.get('FORCE_TRACKING', '1') == '1'
        self.min_tracking_frames = int(os.environ.get('MIN_TRACKING_FRAMES', '30'))  # Increased from 5 to 30
        self.tracking_strategy_weight = float(os.environ.get('TRACKING_STRATEGY_WEIGHT', '2.0'))  # Increased from 1.5 to 2.0
        self.expert_feedback_weight = float(os.environ.get('EXPERT_FEEDBACK_WEIGHT', '1.0'))  # Increased from 0.5 to 1.0
        self.force_annotation_propagation = True  # Always force annotation propagation

        # Use provided shared_params or create a new one
        self.shared_params = shared_params or SharedParams()

        print(f"\n🔍 ===== MULTIFRAME TRACKER PARAMETER DEBUG =====")
        print(f"🔍 MultiFrameTracker initialized with:")
        print(f"🔍   shared_params provided: {shared_params is not None}")
        if self.shared_params:
           print(f"🔍   SharedParams version: {self.shared_params.version}")
           print(f"🔍   Window size: {self.shared_params.tracking_params['window_size']}")
           print(f"🔍   Flow quality: {self.shared_params.tracking_params['flow_quality_threshold']:.3f}")
           print(f"🔍   Flow noise: {self.shared_params.tracking_params['flow_noise_threshold']:.3f}")
           print(f"🔍   Mask threshold: {self.shared_params.tracking_params['mask_threshold']:.3f}")
        print(f"🔍 ================================================\n")

        print(f"\n🔧 MultiFrameTracker using SharedParams v{self.shared_params.version}")
        print(f"🔧 Learning mode will be: {learning_mode if 'learning_mode' in locals() else 'unknown'}")
        print(f"🔧 ==========================================\n")
        
        # Increase window size for better propagation
        if 'window_size' in self.shared_params.tracking_params and self.learning_mode:
            self.shared_params.tracking_params['window_size'] = max(50, self.shared_params.tracking_params['window_size'])
        
        # Performance metrics for the current run
        self.metrics = {
            'iou_scores': {},
            'dice_scores': {},
            'mean_iou': 0.0,
            'mean_dice': 0.0,
            'corrected_frames': 0
        }

        # When initializing MultiFrameTracker
        print("\n=== MultiFrameTracker Initialization ===")
        print(f"Using flow processor method: {flow_processor.method}")
        print(f"Debug mode: {debug_mode}")
        print(f"Output directory: {output_dir}")
        print(f"Using shared parameters version {self.shared_params.version}")
        print(f"Window size: {self.shared_params.tracking_params['window_size']}")
        print(f"Tracking strategy weight: {self.tracking_strategy_weight}")
        print(f"Expert feedback weight: {self.expert_feedback_weight}")
        print(f"Disable annotation copying: {self.disable_annotation_copying}")
        print(f"Bidirectional tracking: {self.bidirectional_tracking}")
        print(f"Force tracking: {self.force_tracking}")
        print(f"Min tracking frames: {self.min_tracking_frames}")
        print(f"Force annotation propagation: {self.force_annotation_propagation}")
        
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
        print(f"Feedback loop mode: {'enabled' if self.feedback_loop_mode else 'disabled'}")
        print(f"Learning mode: {'enabled' if self.learning_mode else 'disabled'}")
    
    def process_annotations(self, annotations_df, video_path, study_uid, series_uid):
        """
        Process annotations for a video and generate predictions.
        """
        # Set up safety mechanisms to prevent endless loop
        self.start_time = time.time()
        self.processed_frames = 0
        self.max_processing_time = int(os.environ.get('MAX_PROCESSING_TIME', '300'))  # 5 minutes default
        self.max_frames_to_process = int(os.environ.get('MAX_FRAMES', '200'))  # Maximum frames to process
        print("\n=== Processing annotations with MultiFrameTracker ===")
        print(f"Annotations count: {len(annotations_df)}")
        print(f"Video path: {video_path}")
        print(f"Study/Series UIDs: {study_uid}/{series_uid}")
        print(f"Feedback loop mode: {'enabled' if self.feedback_loop_mode else 'disabled'}")
        print(f"Learning mode: {'enabled' if self.learning_mode else 'disabled'}")
    
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
        
        # Store total frames for use in other methods
        self.total_frames = total_frames
        
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
        
        # Safety limits info
        print(f"Safety limits: max time={self.max_processing_time}s, max frames={self.max_frames_to_process}")
        
        # CRITICAL: First, add all expert annotations to the results EXACTLY as they are
        print("\nProcessing expert annotations:")
        for annotation in annotations:
            # Check safety limits
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.max_processing_time:
                print(f"\n⚠️ WARNING: Processing time limit reached ({elapsed_time:.1f}s > {self.max_processing_time}s)")
                return all_masks
                
            if self.processed_frames > self.max_frames_to_process:
                print(f"\n⚠️ WARNING: Maximum frame limit reached ({self.processed_frames} > {self.max_frames_to_process})")
                return all_masks
                
            self.processed_frames += 1
            
            frame_num = annotation['frame']  # Already 0-based from _classify_annotations
            print(f"\nProcessing expert annotation for frame {frame_num} (MD.ai frame {frame_num + 1})")
            
            # Always preserve expert annotations exactly as they are
            all_masks[frame_num] = {
                'mask': annotation['mask'].copy(),  # Make a copy to prevent modifications
                'type': annotation['type'],
                'is_annotation': True,  # Mark as expert annotation
                'is_corrected': True,
                'source': 'expert_annotation',
                'mdai_frame': frame_num + 1  # Store MD.ai frame number for reference
            }
            print(f"✓ Preserved expert annotation for frame {frame_num}")
            
            # If in learning mode, use this annotation to learn but don't modify the original
            if self.learning_mode:
                print(f"Learning from expert annotation at frame {frame_num}")
                clear_frames = self._learn_from_annotation(annotation, all_masks, clear_frames)
        
        # Now process segments between annotations for non-annotated frames
        print("\nProcessing segments between annotations:")
        for i in range(len(annotations)):
            current = annotations[i]
            current_frame = current['frame']
            
            # Skip if we've exceeded our limits
            if time.time() - self.start_time > self.max_processing_time:
                print("\n⚠️ Reached time limit, stopping segment processing")
                break
            
            if self.processed_frames > self.max_frames_to_process:
                print("\n⚠️ Reached frame limit, stopping segment processing")
                break
            
            # Handle segment from start to first annotation
            if i == 0 and current_frame > 0:
                print(f"\nProcessing start segment (0 to {current_frame-1})")
                self._process_start_segment(current, clear_frames, all_masks, video_path, total_frames)
            
            # Handle segment between current and next annotation
            if i < len(annotations) - 1:
                next_annotation = annotations[i + 1]
                next_frame = next_annotation['frame']
                if next_frame - current_frame > 1:  # Only process if there are frames between
                    print(f"\nProcessing middle segment ({current_frame+1} to {next_frame-1})")
                    self._process_middle_segment(current, next_annotation, clear_frames, all_masks, video_path)
            
            # Handle segment from last annotation to end
            if i == len(annotations) - 1 and current_frame < total_frames - 1:
                print(f"\nProcessing end segment ({current_frame+1} to {total_frames-1})")
                self._process_end_segment(current, clear_frames, all_masks, video_path, total_frames)
        
        # Create visualization
        self._create_visualization(video_path, all_masks, annotations)

        if self.cap is not None:
           self.cap.release()
           self.cap = None
        
        # Print summary of masks by type
        mask_types = {}
        expert_annotations = 0
        for frame_idx, mask_info in all_masks.items():
            if isinstance(mask_info, dict):
                mask_type = mask_info['type']
                mask_types[mask_type] = mask_types.get(mask_type, 0) + 1
                if mask_info.get('is_annotation', False):
                    expert_annotations += 1
        
        print("\nMask types summary:")
        for mask_type, count in mask_types.items():
            print(f"  {mask_type}: {count} frames")
        print(f"Expert annotations preserved: {expert_annotations}")

        # ENFORCE no-fluid constraints as final step (ADD THIS)
        all_masks = self._enforce_no_fluid_constraints(all_masks, annotations)
            
        return all_masks
    
    def _classify_annotations(self, annotations_df, frame_height, frame_width):
        """
        Classify annotations as 'fluid' or 'clear' based on mask content.
        """
        classified_annotations = []
        print("\nClassifying annotations:")

        for _, row in annotations_df.iterrows():
            # Get frame number - already adjusted to 0-based in ground_truth_utils.py
            frame_num = int(row['frameNumber'])
            mdai_frame = frame_num + 1  # Store MD.ai frame number for reference
          
            # Check if this is explicitly marked as no-fluid
            is_no_fluid = row.get('is_no_fluid', False)
            
            # Get the annotation ID if available
            annotation_id = row.get('id', 'unknown')
            print(f"\nProcessing annotation {annotation_id} at frame {frame_num} (MD.ai frame {mdai_frame})")
            
            if is_no_fluid:
                # This is a clear annotation (no fluid)
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                annotation_type = 'clear'
                print(f"Frame {frame_num}: Classified as clear (explicit no-fluid annotation)")
            elif isinstance(row.get('free_fluid_foreground'), list) and len(row.get('free_fluid_foreground')) > 0:
                # This is a fluid annotation - create mask from polygons
                polygons = row.get('free_fluid_foreground')
                mask = self._create_mask_from_polygons(polygons, frame_height, frame_width)
                annotation_type = 'fluid'
                print(f"Frame {frame_num}: Classified as fluid (has {len(polygons)} polygons)")
                print(f"  Mask sum: {np.sum(mask)}")
            else:
                # Default to clear annotation if no fluid polygons
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                annotation_type = 'clear'
                print(f"Frame {frame_num}: Classified as clear (no polygons)")
        
            classified_annotations.append({
                'frame': frame_num,  # Using 0-based frame number
                'mdai_frame': mdai_frame,  # Store MD.ai frame number for reference
                'mask': mask.copy(),  # Make a copy to prevent modifications
                'type': annotation_type,
                'id': annotation_id,
                'is_no_fluid': is_no_fluid,  # Preserve the no-fluid flag
                'original_polygons': row.get('free_fluid_foreground', []) if annotation_type == 'fluid' else []  # Store original polygons
            })

        # Log summary
        fluid_count = sum(1 for ann in classified_annotations if ann['type'] == 'fluid')
        clear_count = sum(1 for ann in classified_annotations if ann['type'] == 'clear')
        no_fluid_count = sum(1 for ann in classified_annotations if ann.get('is_no_fluid', False))
        
        print(f"\nClassification summary:")
        print(f"- Total annotations: {len(classified_annotations)}")
        print(f"- Fluid frames: {fluid_count}")
        print(f"- Clear frames: {clear_count}")
        print(f"- No-fluid annotations: {no_fluid_count}")
        print("\nFrame number mapping examples:")
        for ann in classified_annotations[:5]:
            print(f"System frame {ann['frame']} = MD.ai frame {ann['mdai_frame']}")

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
        
        # Get frame numbers and types, and track no-fluid labels
        frame_numbers = [a['frame'] for a in annotations]
        frame_types = [a['type'] for a in annotations]
        no_fluid_frames = [a['frame'] for a in annotations if a.get('is_no_fluid', False)]
        
        self.logger.info(f"Found {len(no_fluid_frames)} no-fluid frames: {sorted(no_fluid_frames)}")
        print(f"\nProcessing no-fluid regions:")
        print(f"Found {len(no_fluid_frames)} explicit no-fluid frames: {sorted(no_fluid_frames)}")
        
        # CRITICAL: Process no-fluid frames first - this is the highest priority rule
        if len(no_fluid_frames) >= 2:
            print("\nRule 0: Processing regions between no-fluid annotations (HIGHEST PRIORITY)")
            no_fluid_frames = sorted(no_fluid_frames)
            
            # Process each pair of no-fluid frames
            for i in range(len(no_fluid_frames) - 1):
                start_frame = no_fluid_frames[i]
                end_frame = no_fluid_frames[i + 1]
                print(f"  Found no-fluid region from {start_frame} to {end_frame}")
                
                # Add ALL frames in this range to clear_frames (inclusive)
                for frame in range(start_frame, end_frame + 1):
                    clear_frames.add(frame)
                    
                print(f"  Added {end_frame - start_frame + 1} frames to clear set")
        
        # Rule 1: If last annotation is clear/no-fluid, all frames after it are clear
        if frame_types[-1] == 'clear' or annotations[-1].get('is_no_fluid', False):
            end_frame = frame_numbers[-1]
            print(f"\nRule 1: Last annotation is clear/no-fluid at frame {end_frame}")
            print(f"  Setting frames {end_frame+1} to {total_frames-1} as clear")
            for frame in range(end_frame + 1, total_frames):
                clear_frames.add(frame)
        
        # Rule 2: All frames between two clear annotations are clear
        print("\nRule 2: Processing regions between clear annotations")
        for i in range(len(annotations) - 1):
            if frame_types[i] == 'clear' and frame_types[i+1] == 'clear':
                start_frame = frame_numbers[i]
                end_frame = frame_numbers[i+1]
                print(f"  Found clear region from {start_frame} to {end_frame}")
                for frame in range(start_frame + 1, end_frame):
                    clear_frames.add(frame)
        
        # Log summary of clear frames
        clear_ranges = []
        start = None
        prev = None
        for frame in sorted(clear_frames):
            if start is None:
                start = frame
            elif frame != prev + 1:
                clear_ranges.append(f"{start}-{prev}")
                start = frame
            prev = frame
        if start is not None:
            clear_ranges.append(f"{start}-{prev}")
            
        print(f"\nClear frame ranges: {', '.join(clear_ranges)}")
        print(f"Total clear frames: {len(clear_frames)}")
        
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
                    'source': f"clear_annotation_{end_frame}",
                    'is_annotation': False
                }
        else:
            # If first annotation has fluid, be very conservative about backward tracking
            self.logger.info(f"First fluid annotation at frame {end_frame}")
            
            # Only track backward a limited number of frames
            max_backward_frames = min(5, end_frame)  # Even more conservative - only 5 frames back
            track_start = max(end_frame - max_backward_frames, 0)
            
            if track_start > 0:
                self.logger.info(f"Limited backward tracking from {end_frame} to {track_start}")
                backward_masks = self._track_between_frames(
                    end_frame, 
                    track_start, 
                    first_annotation['mask'], 
                    forward=False
                )
                
                for frame_idx, mask in backward_masks.items():
                    if frame_idx != end_frame and frame_idx not in clear_frames:
                        # Use very strict thresholds for backward tracking from first annotation
                        binary_mask = (mask > 0.8).astype(np.uint8)  # Higher threshold (0.8) for first annotation
                        if np.sum(binary_mask) > 200:  # Higher area threshold
                            # Check if the mask is similar to the first annotation
                            first_annotation_mask = (first_annotation['mask'] > 0.5).astype(np.uint8)
                            iou = self.calculate_iou(binary_mask, first_annotation_mask)
                            
                            if iou > 0.5:  # Only keep if similar to first annotation
                                all_masks[frame_idx] = {
                                    'mask': mask,
                                    'type': 'predicted_fluid',
                                    'source': f"strict_backward_from_{end_frame}",
                                    'is_annotation': False,
                                    'confidence': iou  # Store the confidence
                                }
                            else:
                                # If not similar enough, don't make any prediction
                                self.logger.info(f"Frame {frame_idx}: Insufficient IoU with first annotation ({iou:.2f})")
                        else:
                            self.logger.info(f"Frame {frame_idx}: Insufficient area in backward tracking")
    
    def _process_middle_segment(self, current, next_annotation, clear_frames, all_masks, video_path):
        """
        Process segment between two annotations.
        """
        start_frame = current['frame']
        end_frame = next_annotation['frame']
        
        # Skip if frames are adjacent
        if end_frame - start_frame <= 1:
            return
            
        # CRITICAL: If both annotations are no-fluid, enforce no fluid in between
        if current.get('is_no_fluid', False) and next_annotation.get('is_no_fluid', False):
            self.logger.info(f"Setting frames {start_frame} to {end_frame} as clear (no-fluid constraint)")
            empty_mask = np.zeros_like(current['mask'])
            
            # Add ALL frames in this range to clear_frames and all_masks (inclusive)
            for frame_idx in range(start_frame, end_frame + 1):
                # Add to clear_frames set
                clear_frames.add(frame_idx)
                
                # Add to all_masks with explicit no-fluid flag
                all_masks[frame_idx] = {
                    'mask': empty_mask,
                    'type': 'predicted_clear',
                    'source': f"no_fluid_constraint_{start_frame}_to_{end_frame}",
                    'is_annotation': False,
                    'is_no_fluid': True  # Explicitly mark as no-fluid
                }
                
            print(f"Enforced no-fluid constraint for frames {start_frame} to {end_frame}")
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
            
            # Track forward from current with relaxed quality threshold
            forward_masks = self._track_between_frames(
                start_frame, 
                end_frame, 
                current['mask'], 
                forward=True,
                #quality_threshold=0.4  # Relaxed threshold
            )
            
            # Track backward from next with relaxed quality threshold
            backward_masks = self._track_between_frames(
                end_frame, 
                start_frame, 
                next_annotation['mask'], 
                forward=False,
                #quality_threshold=0.4  # Relaxed threshold
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
        
            # Combine masks with dynamic weighting based on distance
            for frame_idx in range(start_frame + 1, end_frame):
                if frame_idx in forward_masks and frame_idx in backward_masks and frame_idx not in clear_frames:
                    # Calculate weights based on distance to each annotation
                    total_dist = end_frame - start_frame
                    forward_weight = (end_frame - frame_idx) / total_dist
                    backward_weight = (frame_idx - start_frame) / total_dist
                    
                    combined_mask = self._combine_masks(
                        [forward_masks[frame_idx], backward_masks[frame_idx]],
                        weights=[forward_weight, backward_weight]
                    )
                    
                    all_masks[frame_idx] = {
                        'mask': combined_mask,
                        'type': 'predicted_fluid_combined',
                        'source': f"combined_{start_frame}_and_{end_frame}",
                        'is_annotation': False
                    }
                elif frame_idx in forward_masks and frame_idx not in clear_frames:
                    # Use forward mask if backward not available
                    all_masks[frame_idx] = {
                        'mask': forward_masks[frame_idx],
                        'type': 'predicted_fluid_forward',
                        'source': f"forward_from_{start_frame}",
                        'is_annotation': False
                    }
                elif frame_idx in backward_masks and frame_idx not in clear_frames:
                    # Use backward mask if forward not available
                    all_masks[frame_idx] = {
                        'mask': backward_masks[frame_idx],
                        'type': 'predicted_fluid_backward',
                        'source': f"backward_from_{end_frame}",
                        'is_annotation': False
                    }
        
        # Case 2: Clear to Fluid - track backward from fluid only
        elif current['type'] == 'clear' and next_annotation['type'] == 'fluid':
            self.logger.info(f"Tracking backward from fluid frame {end_frame} to clear frame {start_frame}")
            
            backward_masks = self._track_between_frames(
                end_frame, 
                start_frame, 
                next_annotation['mask'], 
                forward=False,
            #quality_threshold=0.4  # Relaxed threshold
            )
            
            for frame_idx in range(start_frame + 1, end_frame):
                if frame_idx in backward_masks and frame_idx not in clear_frames:
                    all_masks[frame_idx] = {
                        'mask': backward_masks[frame_idx],
                        'type': 'predicted_fluid',
                        'source': f"backward_from_{end_frame}",
                        'is_annotation': False
                    }
        
        # Case 3: Fluid to Clear - track forward from fluid only
        elif current['type'] == 'fluid' and next_annotation['type'] == 'clear':
            self.logger.info(f"Tracking forward from fluid frame {start_frame} to clear frame {end_frame}")
            
            forward_masks = self._track_between_frames(
                start_frame, 
                end_frame, 
                current['mask'], 
                forward=True,
                #quality_threshold=0.4  # Relaxed threshold
            )
            
            for frame_idx in range(start_frame + 1, end_frame):
                if frame_idx in forward_masks and frame_idx not in clear_frames:
                    all_masks[frame_idx] = {
                        'mask': forward_masks[frame_idx],
                        'type': 'predicted_fluid',
                        'source': f"forward_from_{start_frame}",
                        'is_annotation': False
                    }
        
        # Case 4: Clear to Clear - all frames between are clear
        else:
            self.logger.info(f"Setting frames between clear frames {start_frame} and {end_frame} as clear")
            empty_mask = np.zeros_like(current['mask'])
            
            for frame_idx in range(start_frame + 1, end_frame):
                # Add to both clear_frames and all_masks
                clear_frames.add(frame_idx)
                all_masks[frame_idx] = {
                    'mask': empty_mask,
                    'type': 'predicted_clear',
                    'source': f"between_clear_{start_frame}_and_{end_frame}",
                    'is_annotation': False,
                    'is_no_fluid': current.get('is_no_fluid', False) and next_annotation.get('is_no_fluid', False)  # Only propagate if both are no-fluid
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
        
        # CRITICAL: If last annotation is no-fluid, all remaining frames should be clear
        if last_annotation.get('is_no_fluid', False):
            self.logger.info(f"Setting frames {start_frame} to {end_frame} as clear (no-fluid constraint)")
            empty_mask = np.zeros_like(last_annotation['mask'])
            
            # Add ALL frames in this range to clear_frames and all_masks (inclusive)
            for frame_idx in range(start_frame, total_frames):
                # Add to clear_frames set
                clear_frames.add(frame_idx)
                
                # Add to all_masks with explicit no-fluid flag
                all_masks[frame_idx] = {
                    'mask': empty_mask,
                    'type': 'predicted_clear',
                    'source': f"no_fluid_constraint_end_{start_frame}",
                    'is_annotation': False,
                    'is_no_fluid': True  # Explicitly mark as no-fluid
                }
            
            print(f"Enforced no-fluid constraint for frames {start_frame} to {end_frame}")
            return
            
        # Handle regular clear annotation
        elif last_annotation['type'] == 'clear':
            self.logger.info(f"Setting frames {start_frame+1} to {end_frame} as clear (no fluid)")
            empty_mask = np.zeros_like(last_annotation['mask'])
            
            for frame_idx in range(start_frame + 1, total_frames):
                all_masks[frame_idx] = {
                    'mask': empty_mask,
                    'type': 'predicted_clear',
                    'source': f"clear_annotation_{start_frame}",
                    'is_annotation': False
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
                        'source': f"forward_from_{start_frame}",
                        'is_annotation': False
                    }

    def _learn_from_annotation(self, annotation, all_masks, clear_frames):
        """
        Learn from an annotation by propagating its mask to nearby frames.
        
        Args:
            annotation: Dictionary containing annotation info
            all_masks: Dictionary of all masks
            clear_frames: Set of frames marked as clear
        """
        frame_idx = annotation['frame']
        mask = annotation['mask']
        is_no_fluid = annotation.get('is_no_fluid', False)
        
        print(f"\nLearning from {'no-fluid' if is_no_fluid else 'fluid'} annotation at frame {frame_idx}")
        
        # For fluid annotations (corrections), propagate to nearby frames
        if annotation['type'] == 'fluid':
            if np.sum(mask) == 0:
                print(f"Skipping empty mask at frame {frame_idx}")
                return
            
            # Use window size from shared params or environment variables - more aggressive
            # Use a larger window size to propagate annotations further
            window = max(self.min_tracking_frames, int(self.shared_params.tracking_params.get('window_size', 50)))
            print(f"Using tracking window size: {window} (AGGRESSIVE MODE)")
            
            # Get total frames from cap if not already set
            if not hasattr(self, 'total_frames'):
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # FORCEFUL MODE: If we're forcing annotation propagation, propagate masks
            # even if tracking would normally fail
            print(f"Force annotation propagation mode: {self.force_annotation_propagation}")
            
            # Debug directory for learning visualizations
            learn_debug_dir = os.path.join(self.debug_dir, f'learning_from_{frame_idx}')
            os.makedirs(learn_debug_dir, exist_ok=True)
            
            # Save the correction mask for reference
            mask_path = os.path.join(learn_debug_dir, f'correction_{frame_idx}.png')
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            print(f"Saved correction mask to {mask_path}")
            
            # Track forward
            forward_masks = self._track_between_frames(
                frame_idx, 
                min(frame_idx + window, self.total_frames - 1),
                mask,
                forward=True
            )
            
            # Track backward
            backward_masks = self._track_between_frames(
                frame_idx,
                max(frame_idx - window, 0),
                mask,
                forward=False
            )
            
            # Apply tracked masks
            for tracked_masks in [forward_masks, backward_masks]:
                for tracked_frame, tracked_mask in tracked_masks.items():
                    if tracked_frame != frame_idx:  # Don't override the original frame
                        # Only apply if frame isn't already marked as clear
                        if tracked_frame not in clear_frames:
                            all_masks[tracked_frame] = {
                                'mask': tracked_mask,
                                'type': 'learned_fluid',
                                'source': f"learned_from_{frame_idx}",
                                'is_annotation': False
                            }
                            print(f"Applied learned fluid mask to frame {tracked_frame}")
        
        # For clear/no-fluid annotations, ensure nearby frames respect the clear designation
        elif annotation['type'] == 'clear':
            # Use a window for clear annotations
            window = 15  # Smaller window for clear frames to avoid over-propagation
            
            # Add frames around this clear frame to the clear_frames set
            # Only if they're not already marked as fluid by an expert
            clear_applied = 0
            for offset in range(-window, window + 1):
                nearby_frame = frame_idx + offset
                
                # Check if in valid range
                if nearby_frame >= 0 and nearby_frame < self.total_frames:
                    # Check if this frame already has an expert annotation
                    if (nearby_frame not in all_masks or 
                        not all_masks[nearby_frame].get('is_corrected', False)):
                        
                        # Add to clear frames set
                        clear_frames.add(nearby_frame)
                        
                        # Create empty mask for this frame
                        all_masks[nearby_frame] = {
                            'mask': np.zeros_like(mask),
                            'type': 'learned_clear',
                            'source': f"learned_from_{frame_idx}",
                            'is_annotation': False,
                            'is_no_fluid': is_no_fluid  # Preserve no-fluid flag
                        }
                        clear_applied += 1
            
            print(f"Applied clear designation to {clear_applied} nearby frames")
            
            # If this is an explicit no-fluid annotation, be more aggressive
            if is_no_fluid:
                print("No-fluid annotation detected - applying more aggressive clear propagation")
                # Extend the clear zone further if confidence is high
                extended_window = window * 2
                for offset in range(-extended_window, extended_window + 1):
                    nearby_frame = frame_idx + offset
                    if nearby_frame >= 0 and nearby_frame < self.total_frames:
                        # Only override if not an expert annotation
                        if (nearby_frame not in all_masks or 
                            not all_masks[nearby_frame].get('is_corrected', False)):
                            clear_frames.add(nearby_frame)
                            print(f"Extended clear zone to frame {nearby_frame}")
        
        print(f"Finished learning from {'no-fluid' if is_no_fluid else 'fluid'} annotation at frame {frame_idx}")
        return clear_frames

    def _track_between_frames(self, start_frame, end_frame, initial_mask, forward=True, recursion_depth=0):
        """
        Track a mask between two frames using optical flow.
        Now properly uses SharedParams for dynamic parameter adjustment.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index
            initial_mask: Initial mask to track
            forward: Whether to track forward or backward
            recursion_depth: Current recursion depth
            
        Returns:
            Dictionary of tracked masks {frame_idx: mask}
        """

        print(f"\n🔍 ===== _track_between_frames PARAMETER DEBUG =====")
        print(f"🔍 About to call track_frames with SharedParams:")
        print(f"🔍   SharedParams available: {self.shared_params is not None}")
        if self.shared_params:
            current_params = self.shared_params.tracking_params
            print(f"🔍   Current SharedParams:")
            print(f"🔍     Version: {self.shared_params.version}")
            print(f"🔍     Flow quality: {current_params['flow_quality_threshold']:.3f}")
            print(f"🔍     Flow noise: {current_params['flow_noise_threshold']:.3f}")
            print(f"🔍     Mask threshold: {current_params['mask_threshold']:.3f}")
            print(f"🔍     Window size: {current_params['window_size']}")
        print(f"🔍 =================================================\n")

        # Check if we've exceeded our time limit
        elapsed_time = time.time() - self.start_time
        max_time = int(os.environ.get('MAX_PROCESSING_TIME', '300'))
        if elapsed_time > max_time:
            print(f"\n⚠️ WARNING: Processing time limit reached ({elapsed_time:.1f}s > {max_time}s)")
            print("Returning partial results to prevent endless loop")
            return {}
            
        # Count this frame in our processing
        self.processed_frames += 1
        max_frames = int(os.environ.get('MAX_FRAMES', '200'))
        if self.processed_frames > max_frames:
            print(f"\n⚠️ WARNING: Maximum frame count reached ({self.processed_frames} > {max_frames})")
            print("Returning partial results to prevent endless loop")
            return {}

        # Anti-recursion protection
        MAX_RECURSION_DEPTH = 10
        if recursion_depth >= MAX_RECURSION_DEPTH:
            print(f"WARNING: Maximum recursion depth ({MAX_RECURSION_DEPTH}) reached in _track_between_frames.")
            print(f"  Current frame range: {start_frame} to {end_frame}")
            print(f"  Forward tracking: {forward}")
            print(f"  Frame range size: {abs(end_frame - start_frame)}")
            return {}

        # Validate frame range
        max_range = int(os.environ.get('MAX_TRACKING_FRAMES', '50'))
        if abs(end_frame - start_frame) > max_range:
            print(f"WARNING: Frame range too large ({abs(end_frame - start_frame)} > {max_range} frames).")
            print(f"  Limiting range to {max_range} frames")
            if forward:
                end_frame = start_frame + max_range
            else:
                end_frame = start_frame - max_range
            print(f"  New range: {start_frame} to {end_frame}")

        print(f"\nStarting _track_between_frames (recursion_depth={recursion_depth}):")
        print(f"  start_frame: {start_frame}")
        print(f"  end_frame: {end_frame}")
        print(f"  forward: {forward}")
        print(f"  frame range size: {abs(end_frame - start_frame)}")
        print(f"  Using SharedParams v{self.shared_params.version}")

        # Create debug directory with timestamp for this tracking session
        debug_dir = os.path.join(self.debug_dir, f'track_{start_frame}_{end_frame}_{int(time.time())}')
        os.makedirs(debug_dir, exist_ok=True)

        # Save initial mask and parameters
        cv2.imwrite(os.path.join(debug_dir, 'initial_mask.png'), (initial_mask * 255).astype(np.uint8))
        with open(os.path.join(debug_dir, 'tracking_params.txt'), 'w') as f:
            f.write(f"Start frame: {start_frame}\n")
            f.write(f"End frame: {end_frame}\n")
            f.write(f"Forward: {forward}\n")
            f.write(f"SharedParams version: {self.shared_params.version}\n")
            f.write(f"Flow parameters:\n")
            for k, v in self.shared_params.tracking_params.items():
                f.write(f"  {k}: {v}\n")

        # Track frames using SharedParams (NO hardcoded quality_threshold!)
        frames = track_frames(
            self.cap,
            start_frame,
            end_frame,
            initial_mask,
            debug_dir=debug_dir,
            forward=forward,
            pbar=None,
            flow_processor=self.flow_processor,
            recursion_depth=recursion_depth,
            shared_params=self.shared_params  # Pass SharedParams instead of hardcoded values!
        )

        # Convert list to dictionary and analyze tracking quality
        frames_dict = {}
        for frame_data in frames:
            if len(frame_data) >= 3:
                frame_idx = frame_data[0]
                frame = frame_data[1]
                mask = frame_data[2]
                
                # Calculate mask statistics using SharedParams mask threshold
                mask_threshold = self.shared_params.tracking_params['mask_threshold']
                mask_area = np.sum(mask > mask_threshold)
                initial_area = np.sum(initial_mask > mask_threshold)
                area_ratio = mask_area / initial_area if initial_area > 0 else float('inf')
                
                # If mask area changes dramatically, save debug info
                if area_ratio < 0.5 or area_ratio > 2.0:
                    debug_frame_dir = os.path.join(debug_dir, f'frame_{frame_idx}_anomaly')
                    os.makedirs(debug_frame_dir, exist_ok=True)
                    
                    # Save the frame and mask
                    cv2.imwrite(os.path.join(debug_frame_dir, 'frame.png'), frame)
                    cv2.imwrite(os.path.join(debug_frame_dir, 'mask.png'), (mask * 255).astype(np.uint8))
                    
                    # Save analysis with SharedParams info
                    with open(os.path.join(debug_frame_dir, 'analysis.txt'), 'w') as f:
                        f.write(f"Frame: {frame_idx}\n")
                        f.write(f"Initial mask area: {initial_area}\n")
                        f.write(f"Current mask area: {mask_area}\n")
                        f.write(f"Area ratio: {area_ratio:.2f}\n")
                        f.write(f"SharedParams version: {self.shared_params.version}\n")
                        f.write(f"Mask threshold used: {mask_threshold}\n")
                        
                    self.logger.warning(f"Frame {frame_idx}: Unusual mask behavior detected (area ratio: {area_ratio:.2f})")
                
                frames_dict[frame_idx] = mask

        print(f"✅ _track_between_frames completed with SharedParams v{self.shared_params.version}")
        print(f"    Returned {len(frames_dict)} tracked frames")
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
            'fluid': (0, 255, 0),                # Green for human annotations
            'clear': None,                       # No color for clear frames
            'predicted_fluid': (0, 0, 255),      # Red for predicted fluid (BGR!)
            'predicted_fluid_combined': (0, 165, 255),  # Orange for combined predictions
            'predicted_clear': None,             # No color for predicted clear
            'learned_forward': (255, 0, 0),      # Blue for learned forward
            'learned_backward': (255, 0, 255)    # Magenta for learned backward
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
        """Apply border constraints to the mask based on frame characteristics"""
        # Use shared parameters instead of hardcoded values
        border_constraint_weight = self.shared_params.tracking_params['border_constraint_weight']
        
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

    # Add evaluation metric methods
    def calculate_iou(self, pred_mask, gt_mask):
        """
        Calculate Intersection over Union between predicted and ground truth masks.
        
        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            
        Returns:
            IoU score (0-1)
        """
        # Convert to binary if needed
        if pred_mask.dtype != bool and pred_mask.dtype != np.bool_:
            pred_mask = pred_mask > self.shared_params.tracking_params['mask_threshold']
        if gt_mask.dtype != bool and gt_mask.dtype != np.bool_:
            gt_mask = gt_mask > self.shared_params.tracking_params['mask_threshold']
            
        # Handle empty masks
        if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            return 1.0  # Both empty = perfect match
        if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            return 0.0  # One empty, one not = no overlap
            
        # Calculate intersection and union
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        # Return IoU
        return intersection / union if union > 0 else 0.0
    
    def calculate_dice(self, pred_mask, gt_mask):
        """
        Calculate Dice coefficient between predicted and ground truth masks.
        
        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask
            
        Returns:
            Dice coefficient (0-1)
        """
        # Convert to binary if needed
        if pred_mask.dtype != bool and pred_mask.dtype != np.bool_:
            pred_mask = pred_mask > self.shared_params.tracking_params['mask_threshold']
        if gt_mask.dtype != bool and gt_mask.dtype != np.bool_:
            gt_mask = gt_mask > self.shared_params.tracking_params['mask_threshold']
            
        # Handle empty masks
        if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            return 1.0  # Both empty = perfect match
        if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            return 0.0  # One empty, one not = no overlap
            
        # Calculate intersection and sums
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        sum_pred = pred_mask.sum()
        sum_gt = gt_mask.sum()
        
        # Return Dice coefficient
        return 2.0 * intersection / (sum_pred + sum_gt) if (sum_pred + sum_gt) > 0 else 0.0
    
    def evaluate_predictions(self, predicted_masks, ground_truth_masks):
        """
        Evaluate predictions against ground truth masks.
        
        Args:
            predicted_masks: Dictionary of predicted masks by frame
            ground_truth_masks: Dictionary of ground truth masks by frame
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Initialize metrics
        iou_scores = {}
        dice_scores = {}
        
        # Calculate metrics for each frame
        common_frames = set(predicted_masks.keys()) & set(ground_truth_masks.keys())
        
        for frame_idx in common_frames:
            # Get masks
            pred_mask = predicted_masks[frame_idx]
            gt_mask = ground_truth_masks[frame_idx]
            
            # Skip if either is not a valid mask
            if not isinstance(pred_mask, np.ndarray) or not isinstance(gt_mask, np.ndarray):
                continue
                
            # Extract mask from dict if needed
            if isinstance(pred_mask, dict) and 'mask' in pred_mask:
                pred_mask = pred_mask['mask']
            if isinstance(gt_mask, dict) and 'mask' in gt_mask:
                gt_mask = gt_mask['mask']
            
            # Calculate metrics
            iou = self.calculate_iou(pred_mask, gt_mask)
            dice = self.calculate_dice(pred_mask, gt_mask)
            
            # Store results
            iou_scores[frame_idx] = iou
            dice_scores[frame_idx] = dice
        
        # Calculate mean scores
        mean_iou = np.mean(list(iou_scores.values())) if iou_scores else 0
        mean_dice = np.mean(list(dice_scores.values())) if dice_scores else 0
        
        # Create full metrics dictionary
        metrics = {
            'iou_scores': iou_scores,
            'dice_scores': dice_scores,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'num_frames': len(common_frames)
        }
        
        # Update instance metrics
        self.metrics.update(metrics)
        
        # Log summary
        self.logger.info(f"Evaluation - Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}, Frames: {len(common_frames)}")
        print(f"\nEvaluation Results:")
        print(f"  Mean IoU: {mean_iou:.4f}")
        print(f"  Mean Dice: {mean_dice:.4f}")
        print(f"  Evaluated Frames: {len(common_frames)}")
        
        return metrics
    
    def _enforce_no_fluid_constraints(self, all_masks, annotations):
        """Enforce no-fluid constraints as immutable rules - call this LAST"""
    
    # Find all no-fluid annotations
        no_fluid_frames = [ann['frame'] for ann in annotations if ann.get('is_no_fluid', False)]
    
        if not no_fluid_frames:
           return all_masks
    
        print(f"\n🚫 ENFORCING NO-FLUID CONSTRAINTS on {len(no_fluid_frames)} frames")
    
    # Create empty mask template
        if annotations:
           empty_mask = np.zeros_like(annotations[0]['mask'])
        else:
             empty_mask = np.zeros((480, 640), dtype=np.uint8)  # fallback
    
        violations_fixed = 0
    
    # Process regions between no-fluid annotations
        if len(no_fluid_frames) >= 2:
            sorted_frames = sorted(no_fluid_frames)
            for i in range(len(sorted_frames) - 1):
              start = sorted_frames[i]
              end = sorted_frames[i + 1]
            
              print(f"🚫 Enforcing no-fluid region: frames {start} to {end}")
            
            # FORCE all frames in this range to be clear
              for frame_idx in range(start, end + 1):
                 if frame_idx in all_masks:
                    current_mask = all_masks[frame_idx]
                    if isinstance(current_mask, dict):
                        current_sum = np.sum(current_mask.get('mask', empty_mask))
                    else:
                        current_sum = np.sum(current_mask)
                    
                    if current_sum > 0:
                        print(f"🚫 VIOLATION FIXED: Frame {frame_idx} had mask sum {current_sum}, now 0")
                        violations_fixed += 1
                
                # OVERWRITE with empty mask
                 all_masks[frame_idx] = {
                    'mask': empty_mask.copy(),
                    'type': 'enforced_no_fluid',
                    'source': 'no_fluid_constraint_enforcement',
                    'is_annotation': False,
                    'is_no_fluid': True,
                    'immutable': True
                }
    
        print(f"🚫 Fixed {violations_fixed} no-fluid constraint violations")
        return all_masks

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
            
            # Create annotation - convert frame_idx to MD.ai's 1-based system
            mdai_frame = int(frame_idx) + 1  # Convert to 1-based frame number
            
            annotation = {
                'labelId': label_id_fluid,
                'StudyInstanceUID': study_uid,
                'SeriesInstanceUID': series_uid,
                'frameNumber': mdai_frame,  # Using 1-based frame number for MD.ai
                'data': mask_data,
                'groupId': label_id_machine
            }
            
            if isinstance(mask_info, dict) and 'source' in mask_info:
                annotation['note'] = f"Source: {mask_info['source']} [System frame: {frame_idx}, MD.ai frame: {mdai_frame}]"
            
            # Add confidence score for machine predictions
            if isinstance(mask_info, dict) and 'type' in mask_info and ('predicted' in mask_info['type'] or 'learned' in mask_info['type']):
                annotation['confidence'] = 0.9  
            
            annotations.append(annotation)
            
        except Exception as e:
            print(f"Error creating annotation for frame {frame_idx} (MD.ai frame {int(frame_idx)+1}): {str(e)}")
            continue
    
    return annotations




# This function has been moved to consolidated_tracking.py
# The previous version applied a -1 frame correction which caused evaluation issues

# Main processing function for multi-frame tracking
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
    
    # Set feedback loop and learning mode if needed
    tracker.feedback_loop_mode = True  # Enable feedback loop mode
    tracker.learning_mode = True  # Enable learning from corrections
    
    # Apply environment variables to affect tracking behavior
    tracker.tracking_strategy_weight = float(os.environ.get('TRACKING_STRATEGY_WEIGHT', '1.5'))
    tracker.expert_feedback_weight = float(os.environ.get('EXPERT_FEEDBACK_WEIGHT', '0.5'))
    tracker.disable_annotation_copying = os.environ.get('DISABLE_ANNOTATION_COPYING', '0') == '1'
    tracker.bidirectional_tracking = os.environ.get('BIDIRECTIONAL_TRACKING', '1') == '1'
    tracker.force_tracking = os.environ.get('FORCE_TRACKING', '1') == '1'
    tracker.min_tracking_frames = int(os.environ.get('MIN_TRACKING_FRAMES', '5'))
    
    print(f"Feedback loop mode: {'enabled' if tracker.feedback_loop_mode else 'disabled'}")
    print(f"Learning mode: {'enabled' if tracker.learning_mode else 'disabled'}")
    print(f"Tracking strategy weight: {tracker.tracking_strategy_weight}")
    print(f"Expert feedback weight: {tracker.expert_feedback_weight}")
    print(f"Disable annotation copying: {tracker.disable_annotation_copying}")
    print(f"Bidirectional tracking: {tracker.bidirectional_tracking}")
    print(f"Force tracking: {tracker.force_tracking}")
    print(f"Min tracking frames: {tracker.min_tracking_frames}")

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

# Testing function
def test_function():
    """Simple test function to verify imports work correctly"""
    print("Test function executed successfully")
    return "Success"