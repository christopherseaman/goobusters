#!/usr/bin/env python3
"""
Simple test to verify bidirectional tracking implementation status.
"""

import sys
import os
import numpy as np
sys.path.insert(0, 'lib')

def test_optical_flow_imports():
    """Test if we can import the optical flow components."""
    print("=== Testing Imports ===")
    try:
        # Fix the relative import by modifying the import in optical.py temporarily
        import optical
        print("✓ Successfully imported optical module")
        return True
    except Exception as e:
        print(f"✗ Failed to import optical module: {e}")
        return False

def test_class_instantiation():
    """Test if we can create instances of the classes."""
    print("\n=== Testing Class Instantiation ===")
    try:
        # Mock the OpticalFlowProcessor to avoid the import issue
        sys.modules['lib.opticalflowprocessor'] = type('MockModule', (), {
            'OpticalFlowProcessor': type('MockProcessor', (), {
                '__init__': lambda self, method: None
            })
        })()
        
        from optical import OpticalFlowTracker, SharedParams, MultiAnnotationProcessor
        
        # Test SharedParams
        shared_params = SharedParams()
        print("✓ Created SharedParams instance")
        
        # Test MultiAnnotationProcessor
        processor = MultiAnnotationProcessor(shared_params)
        print("✓ Created MultiAnnotationProcessor instance")
        
        # Test OpticalFlowTracker
        config = {'optical_flow': {'method': 'farneback'}}
        tracker = OpticalFlowTracker(config)
        print("✓ Created OpticalFlowTracker instance")
        
        return tracker, processor, shared_params
        
    except Exception as e:
        print(f"✗ Failed to create instances: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_bidirectional_methods(tracker):
    """Test if bidirectional methods exist and are callable."""
    print("\n=== Testing Bidirectional Methods ===")
    
    if tracker is None:
        print("✗ No tracker instance available")
        return False
    
    methods_to_check = [
        'track_bidirectional_between_annotations',
        '_track_forward_from_annotation', 
        '_track_backward_from_annotation',
        'combine_masks_with_temporal_weighting',
        'process_multiple_annotations'
    ]
    
    all_exist = True
    for method_name in methods_to_check:
        if hasattr(tracker, method_name):
            method = getattr(tracker, method_name)
            if callable(method):
                print(f"✓ Method {method_name} exists and is callable")
            else:
                print(f"✗ Method {method_name} exists but is not callable")
                all_exist = False
        else:
            print(f"✗ Method {method_name} does not exist")
            all_exist = False
    
    return all_exist

def test_temporal_weighting(tracker):
    """Test temporal weighting functionality."""
    print("\n=== Testing Temporal Weighting ===")
    
    if tracker is None:
        print("✗ No tracker instance available")
        return False
    
    try:
        # Test basic temporal weighting
        forward_mask = np.ones((100, 100), dtype=np.float32)
        backward_mask = np.zeros((100, 100), dtype=np.float32)
        
        # Enable temporal weighting
        tracker.temporal_weighting_enabled = True
        
        result = tracker.combine_masks_with_temporal_weighting(
            forward_mask, backward_mask,
            current_frame=25, start_frame=20, end_frame=30
        )
        
        if result is not None:
            print(f"✓ Temporal weighting works: result shape {result.shape}, mean value {np.mean(result):.3f}")
            return True
        else:
            print("✗ Temporal weighting returned None")
            return False
            
    except Exception as e:
        print(f"✗ Temporal weighting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_annotation_processing(processor):
    """Test annotation processing functionality."""
    print("\n=== Testing Annotation Processing ===")
    
    if processor is None:
        print("✗ No processor instance available")
        return False
    
    try:
        # Test annotation parsing
        test_annotations = [
            {'frame_number': 10, 'type': 'fluid', 'mask': np.ones((100, 100))},
            {'frame_number': 50, 'type': 'fluid', 'mask': np.ones((100, 100))},
        ]
        
        parsed = processor.parse_annotations(test_annotations)
        print(f"✓ Parsed {len(parsed)} annotations")
        
        # Test gap detection
        gaps = processor.detect_annotation_gaps(parsed)
        print(f"✓ Detected {len(gaps)} gaps")
        
        if len(gaps) > 0:
            print(f"  Gap strategy: {gaps[0]['tracking_strategy']}")
            print(f"  Gap size: {gaps[0]['gap_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Annotation processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bidirectional_tracking_status():
    """Test the actual bidirectional tracking implementation status."""
    print("\n=== Testing Bidirectional Implementation Status ===")
    
    # Test imports first
    if not test_optical_flow_imports():
        return False
    
    # Create instances
    tracker, processor, shared_params = test_class_instantiation()
    
    # Test method existence
    methods_exist = test_bidirectional_methods(tracker)
    
    # Test temporal weighting
    temporal_works = test_temporal_weighting(tracker)
    
    # Test annotation processing
    annotation_works = test_annotation_processing(processor)
    
    print("\n=== SUMMARY ===")
    print(f"Core imports: {'✓' if tracker else '✗'}")
    print(f"Bidirectional methods exist: {'✓' if methods_exist else '✗'}")
    print(f"Temporal weighting works: {'✓' if temporal_works else '✗'}")
    print(f"Annotation processing works: {'✓' if annotation_works else '✗'}")
    
    # Check if the actual tracking implementation is there
    if tracker and hasattr(tracker, '_track_forward_from_annotation'):
        method = getattr(tracker, '_track_forward_from_annotation')
        # Check if method is just a placeholder
        import inspect
        source = inspect.getsource(method)
        if 'TODO' in source or 'placeholder' in source.lower() or 'return {}' in source:
            print("⚠️  Forward tracking method is just a placeholder")
        else:
            print("✓ Forward tracking method has implementation")
    
    if tracker and hasattr(tracker, '_track_backward_from_annotation'):
        method = getattr(tracker, '_track_backward_from_annotation')
        import inspect
        source = inspect.getsource(method)
        if 'TODO' in source or 'placeholder' in source.lower() or 'return {}' in source:
            print("⚠️  Backward tracking method is just a placeholder")
        else:
            print("✓ Backward tracking method has implementation")
    
    return tracker is not None

if __name__ == '__main__':
    print("Testing Bidirectional Tracking Implementation Status...")
    test_bidirectional_tracking_status()