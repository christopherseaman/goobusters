#!/usr/bin/env python3
"""
Unit tests for bidirectional tracking components and temporal weighting algorithms.

This test suite validates the core functionality of the bidirectional multi-frame
tracking system including annotation processing, gap detection, and temporal weighting.
"""

import unittest
import numpy as np
import os
import tempfile
import json
import time
from unittest.mock import Mock, patch, MagicMock

# Import the classes we're testing
import sys
sys.path.append('lib')
from optical import (
    MultiAnnotationProcessor, OpticalFlowTracker, SharedParams, 
    AnnotationType, BidirectionalTrackingError, AnnotationValidationError
)


class TestMultiAnnotationProcessor(unittest.TestCase):
    """Test cases for MultiAnnotationProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.shared_params = SharedParams()
        self.processor = MultiAnnotationProcessor(self.shared_params)
    
    def test_parse_generic_annotations(self):
        """Test parsing of generic annotation format."""
        annotations = [
            {'frame_number': 10, 'type': AnnotationType.FLUID, 'mask': np.ones((100, 100))},
            {'frame_number': 50, 'type': AnnotationType.CLEAR, 'mask': None},
            {'frame_number': 30, 'type': AnnotationType.FLUID, 'mask': np.ones((100, 100))}
        ]
        
        parsed = self.processor.parse_annotations(annotations)
        
        # Should be sorted by frame number
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]['frame_number'], 10)
        self.assertEqual(parsed[1]['frame_number'], 30)
        self.assertEqual(parsed[2]['frame_number'], 50)
    
    def test_annotation_validation(self):
        """Test annotation validation logic."""
        # Valid annotations
        valid_annotations = [
            {'frame_number': 10, 'type': AnnotationType.FLUID, 'mask': np.ones((100, 100))},
            {'frame_number': 20, 'type': AnnotationType.CLEAR, 'mask': None}
        ]
        
        validated = self.processor._validate_annotations(valid_annotations)
        self.assertEqual(len(validated), 2)
        
        # Invalid annotations - should raise exception if skip_invalid_annotations=False
        self.processor.skip_invalid_annotations = False
        invalid_annotations = [
            {'frame_number': -1, 'type': AnnotationType.FLUID, 'mask': np.ones((100, 100))},  # Negative frame
        ]
        
        with self.assertRaises(AnnotationValidationError):
            self.processor._validate_annotations(invalid_annotations)
    
    def test_gap_detection(self):
        """Test annotation gap detection."""
        annotations = [
            {'frame_number': 10, 'type': AnnotationType.FLUID, 'mask': np.ones((100, 100))},
            {'frame_number': 50, 'type': AnnotationType.FLUID, 'mask': np.ones((100, 100))},  # Gap of 39 frames
            {'frame_number': 55, 'type': AnnotationType.CLEAR, 'mask': None}  # Gap of 4 frames (below threshold)
        ]
        
        gaps = self.processor.detect_annotation_gaps(annotations)
        
        # Should detect one gap (the large one)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]['gap_size'], 39)
        self.assertEqual(gaps[0]['tracking_strategy'], 'bidirectional')  # F→F
    
    def test_tracking_strategy_determination(self):
        """Test the four tracking strategy scenarios."""
        # F→F should be bidirectional
        strategy = self.processor._determine_tracking_strategy(AnnotationType.FLUID, AnnotationType.FLUID)
        self.assertEqual(strategy, 'bidirectional')
        
        # F→C should be forward_only
        strategy = self.processor._determine_tracking_strategy(AnnotationType.FLUID, AnnotationType.CLEAR)
        self.assertEqual(strategy, 'forward_only')
        
        # C→F should be backward_only
        strategy = self.processor._determine_tracking_strategy(AnnotationType.CLEAR, AnnotationType.FLUID)
        self.assertEqual(strategy, 'backward_only')
        
        # C→C should be none
        strategy = self.processor._determine_tracking_strategy(AnnotationType.CLEAR, AnnotationType.CLEAR)
        self.assertEqual(strategy, 'none')


class TestTemporalWeighting(unittest.TestCase):
    """Test cases for temporal weighting algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = {'optical_flow': {'method': 'farneback'}}
        self.tracker = OpticalFlowTracker(config)
        self.tracker.bidirectional_enabled = True
        self.tracker.temporal_weighting_enabled = True
    
    def test_temporal_weight_calculation(self):
        """Test temporal distance weight calculation."""
        forward_mask = np.ones((100, 100), dtype=np.float32)
        backward_mask = np.zeros((100, 100), dtype=np.float32)
        
        # Test frame in the middle - should have equal weights
        combined = self.tracker.combine_masks_with_temporal_weighting(
            forward_mask, backward_mask, 
            current_frame=25, start_frame=20, end_frame=30
        )
        
        # Should be 0.5 * ones + 0.5 * zeros = 0.5
        expected_value = 0.5
        self.assertAlmostEqual(np.mean(combined), expected_value, places=2)
        
        # Test frame closer to start - backward weight should be higher
        combined = self.tracker.combine_masks_with_temporal_weighting(
            forward_mask, backward_mask,
            current_frame=22, start_frame=20, end_frame=30
        )
        
        # Forward weight = (30-22)/(30-20) = 8/10 = 0.8
        # Backward weight = (22-20)/(30-20) = 2/10 = 0.2
        # Combined = 0.8*1 + 0.2*0 = 0.8
        expected_value = 0.8
        self.assertAlmostEqual(np.mean(combined), expected_value, places=2)
    
    def test_temporal_weighting_edge_cases(self):
        """Test edge cases in temporal weighting."""
        forward_mask = np.ones((100, 100), dtype=np.float32)
        backward_mask = np.zeros((100, 100), dtype=np.float32)
        
        # Test invalid frame range
        with self.assertRaises(BidirectionalTrackingError):
            self.tracker.combine_masks_with_temporal_weighting(
                forward_mask, backward_mask,
                current_frame=15, start_frame=20, end_frame=30  # current_frame < start_frame
            )
        
        # Test mismatched mask shapes
        small_mask = np.ones((50, 50), dtype=np.float32)
        with self.assertRaises(BidirectionalTrackingError):
            self.tracker.combine_masks_with_temporal_weighting(
                forward_mask, small_mask,
                current_frame=25, start_frame=20, end_frame=30
            )
        
        # Test None masks
        with self.assertRaises(BidirectionalTrackingError):
            self.tracker.combine_masks_with_temporal_weighting(
                None, backward_mask,
                current_frame=25, start_frame=20, end_frame=30
            )
    
    def test_temporal_weighting_disabled(self):
        """Test behavior when temporal weighting is disabled."""
        self.tracker.temporal_weighting_enabled = False
        
        forward_mask = np.ones((100, 100), dtype=np.float32)
        backward_mask = np.zeros((100, 100), dtype=np.float32)
        
        combined = self.tracker.combine_masks_with_temporal_weighting(
            forward_mask, backward_mask,
            current_frame=25, start_frame=20, end_frame=30
        )
        
        # Should be simple average: (1 + 0) / 2 = 0.5
        expected_value = 0.5
        self.assertAlmostEqual(np.mean(combined), expected_value, places=2)


class TestBidirectionalConfiguration(unittest.TestCase):
    """Test cases for bidirectional tracking configuration."""
    
    def test_shared_params_bidirectional_config(self):
        """Test SharedParams bidirectional configuration."""
        shared_params = SharedParams()
        
        # Check default values
        bp = shared_params.tracking_params['bidirectional_tracking']
        self.assertFalse(bp['enabled'])
        self.assertTrue(bp['temporal_weighting'])
        self.assertEqual(bp['min_annotation_gap'], 5)
        self.assertEqual(bp['conflict_resolution_method'], 'weighted_average')
    
    def test_tracker_configuration(self):
        """Test OpticalFlowTracker bidirectional configuration."""
        config = {
            'optical_flow': {'method': 'farneback'},
            'tracking': {
                'bidirectional_tracking': {
                    'enabled': True,
                    'temporal_weighting': False,
                    'min_annotation_gap': 10
                }
            }
        }
        
        tracker = OpticalFlowTracker(config)
        
        # Configuration should be applied
        self.assertTrue(tracker.bidirectional_enabled)
        self.assertFalse(tracker.temporal_weighting_enabled)
        self.assertEqual(tracker.min_annotation_gap, 10)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.shared_params = SharedParams()
        self.processor = MultiAnnotationProcessor(self.shared_params)
    
    def test_empty_annotations(self):
        """Test handling of empty annotation list."""
        result = self.processor.parse_annotations([])
        self.assertEqual(len(result), 0)
    
    def test_malformed_annotations(self):
        """Test handling of malformed annotations."""
        malformed_annotations = [
            {'invalid': 'annotation'},  # Missing required fields
            {'frame_number': 'not_an_int', 'type': AnnotationType.FLUID},  # Invalid frame number
            {}  # Empty annotation
        ]
        
        # Should handle gracefully and return empty list
        result = self.processor.parse_annotations(malformed_annotations)
        self.assertEqual(len(result), 0)
    
    def test_annotation_processing_failure_recovery(self):
        """Test recovery from annotation processing failures."""
        config = {'optical_flow': {'method': 'farneback'}}
        tracker = OpticalFlowTracker(config)
        tracker.bidirectional_enabled = True
        tracker.fallback_to_single_direction = True
        
        # Mock a failing gap processing
        with patch.object(tracker, '_process_annotation_gap', side_effect=Exception("Simulated failure")):
            # Should not raise exception due to fallback_to_single_direction=True
            annotations = [
                {'frame_number': 10, 'type': AnnotationType.FLUID, 'mask': np.ones((100, 100))},
                {'frame_number': 50, 'type': AnnotationType.FLUID, 'mask': np.ones((100, 100))}
            ]
            
            # Create a temporary video file path for testing
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                video_path = tmp_file.name
            
            try:
                # This should complete without raising an exception
                result = tracker.process_multiple_annotations(annotations, video_path, '/tmp/test_output')
                # The method should return True even if individual gaps fail when fallback is enabled
                self.assertTrue(isinstance(result, bool))
            finally:
                # Clean up
                if os.path.exists(video_path):
                    os.unlink(video_path)


def run_temporal_weighting_benchmarks():
    """Run performance benchmarks for temporal weighting algorithms."""
    print("\n=== TEMPORAL WEIGHTING BENCHMARKS ===")
    
    config = {'optical_flow': {'method': 'farneback'}}
    tracker = OpticalFlowTracker(config)
    tracker.bidirectional_enabled = True
    
    # Test with different mask sizes
    mask_sizes = [(100, 100), (256, 256), (512, 512), (1024, 1024)]
    
    for height, width in mask_sizes:
        forward_mask = np.random.rand(height, width).astype(np.float32)
        backward_mask = np.random.rand(height, width).astype(np.float32)
        
        # Time the temporal weighting operation
        start_time = time.time()
        
        for _ in range(100):  # Run 100 iterations for timing
            combined = tracker.combine_masks_with_temporal_weighting(
                forward_mask, backward_mask,
                current_frame=25, start_frame=20, end_frame=30
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to milliseconds
        
        print(f"Mask size {height}x{width}: {avg_time:.2f}ms per combination")
    
    print("========================================\n")


if __name__ == '__main__':
    # Run unit tests
    print("Running bidirectional tracking unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run benchmarks
    try:
        run_temporal_weighting_benchmarks()
    except Exception as e:
        print(f"Benchmark failed: {e}")