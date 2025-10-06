#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced mask output implementation.
This script verifies all the new features are working correctly.
"""

import os
import json
import yaml
import cv2
import numpy as np
from pathlib import Path

def test_implementation():
    """Test all the new features we implemented."""
    print("🧪 Testing Enhanced Mask Output Implementation")
    print("=" * 60)
    
    # Find the most recent output directory
    output_base = "output/farneback"
    if not os.path.exists(output_base):
        print("❌ No output directory found. Run tracking first.")
        return False
    
    # Get the most recent video output directory
    video_dirs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
    if not video_dirs:
        print("❌ No video output directories found.")
        return False
    
    # Use the first available directory for testing
    test_dir = os.path.join(output_base, video_dirs[0])
    print(f"📁 Testing directory: {test_dir}")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Identity file exists and has correct structure
    total_tests += 1
    print("\n1️⃣ Testing identity.yaml...")
    identity_file = os.path.join(test_dir, "identity.yaml")
    if os.path.exists(identity_file):
        try:
            with open(identity_file, 'r') as f:
                identity_data = yaml.safe_load(f)
            
            required_fields = ['study_instance_uid', 'series_instance_uid', 'exam_number', 
                             'dataset_name', 'dataset_id', 'created_at', 'annotation_count', 'labels']
            
            if all(field in identity_data for field in required_fields):
                print("   ✅ identity.yaml has correct structure")
                print(f"   📊 Exam number: {identity_data.get('exam_number')}")
                print(f"   📊 Dataset: {identity_data.get('dataset_name')}")
                print(f"   📊 Annotations: {identity_data.get('annotation_count')}")
                tests_passed += 1
            else:
                print("   ❌ identity.yaml missing required fields")
        except Exception as e:
            print(f"   ❌ Error reading identity.yaml: {e}")
    else:
        print("   ❌ identity.yaml not found")
    
    # Test 2: Input and tracked annotations JSON files exist
    total_tests += 1
    print("\n2️⃣ Testing annotation files...")
    input_ann_file = os.path.join(test_dir, "input_annotations.json")
    tracked_ann_file = os.path.join(test_dir, "tracked_annotations.json")
    original_ann_file = os.path.join(test_dir, "original_annotations.json")  # Legacy support
    
    input_exists = os.path.exists(input_ann_file) or os.path.exists(original_ann_file)
    tracked_exists = os.path.exists(tracked_ann_file)
    
    if input_exists and tracked_exists:
        print("   ✅ Both input_annotations.json and tracked_annotations.json exist")
        tests_passed += 1
    elif input_exists:
        print("   ⚠️  input_annotations.json exists but tracked_annotations.json missing")
    elif tracked_exists:
        print("   ⚠️  tracked_annotations.json exists but input_annotations.json missing")
    else:
        print("   ❌ Missing annotation files")
    
    # Test 3: Enhanced mask_data.json with track_id and label_id
    total_tests += 1
    print("\n3️⃣ Testing enhanced mask_data.json...")
    mask_data_file = os.path.join(test_dir, "mask_data.json")
    if os.path.exists(mask_data_file):
        try:
            with open(mask_data_file, 'r') as f:
                mask_data = json.load(f)
            
            # Check if any frame has the new fields
            has_track_id = any('track_id' in frame_data for frame_data in mask_data.values())
            has_label_id = any('label_id' in frame_data for frame_data in mask_data.values())
            
            if has_track_id and has_label_id:
                print("   ✅ mask_data.json has track_id and label_id fields")
                # Show sample data
                sample_frame = list(mask_data.keys())[0]
                sample_data = mask_data[sample_frame]
                print(f"   📊 Sample frame {sample_frame}: track_id={sample_data.get('track_id')}, label_id={sample_data.get('label_id')}")
                tests_passed += 1
            else:
                print("   ❌ mask_data.json missing track_id or label_id fields")
        except Exception as e:
            print(f"   ❌ Error reading mask_data.json: {e}")
    else:
        print("   ❌ mask_data.json not found")
    
    # Test 4: Individual frame masks directory
    total_tests += 1
    print("\n4️⃣ Testing individual frame masks...")
    masks_dir = os.path.join(test_dir, "masks")
    if os.path.exists(masks_dir):
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
        if mask_files:
            print(f"   ✅ Found {len(mask_files)} individual mask files")
            # Test that masks are valid images
            sample_mask = os.path.join(masks_dir, mask_files[0])
            try:
                img = cv2.imread(sample_mask, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                    print(f"   ✅ Sample mask is valid ({img.shape[1]}x{img.shape[0]})")
                    tests_passed += 1
                else:
                    print("   ❌ Sample mask is invalid")
            except Exception as e:
                print(f"   ❌ Error reading sample mask: {e}")
        else:
            print("   ❌ No mask files found in masks directory")
    else:
        print("   ❌ masks directory not found")
    
    # Test 5: Video file with color coding
    total_tests += 1
    print("\n5️⃣ Testing color-coded video...")
    video_file = os.path.join(test_dir, "multi_frame_tracking.mp4")
    if os.path.exists(video_file):
        try:
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"   ✅ Video file is valid")
                print(f"   📊 Dimensions: {width}x{height}, FPS: {fps:.1f}, Frames: {frame_count}")
                cap.release()
                tests_passed += 1
            else:
                print("   ❌ Video file cannot be opened")
        except Exception as e:
            print(f"   ❌ Error reading video file: {e}")
    else:
        print("   ❌ multi_frame_tracking.mp4 not found")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Implementation needs fixes.")
        return False

if __name__ == "__main__":
    success = test_implementation()
    exit(0 if success else 1)
