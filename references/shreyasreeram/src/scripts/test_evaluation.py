#!/usr/bin/env python3
import numpy as np
from src.utils import calculate_iou, calculate_dice, evaluate_with_iou

def test_evaluation_utils():
    """Test the evaluation utilities with simple test cases"""
    print("Testing evaluation utilities...")
    
    # Create simple test masks
    mask1 = np.zeros((10, 10), dtype=np.uint8)
    mask2 = np.zeros((10, 10), dtype=np.uint8)
    
    # Perfect overlap case
    mask1[2:5, 2:5] = 1
    mask2[2:5, 2:5] = 1
    
    iou = calculate_iou(mask1, mask2)
    dice = calculate_dice(mask1, mask2)
    
    print("\nTest Case 1 - Perfect Overlap:")
    print(f"IoU: {iou:.4f} (expected: 1.0000)")
    print(f"Dice: {dice:.4f} (expected: 1.0000)")
    assert abs(iou - 1.0) < 1e-6, "IoU should be 1.0 for perfect overlap"
    assert abs(dice - 1.0) < 1e-6, "Dice should be 1.0 for perfect overlap"
    
    # Partial overlap case
    mask2[3:6, 3:6] = 1
    
    iou = calculate_iou(mask1, mask2)
    dice = calculate_dice(mask1, mask2)
    
    print("\nTest Case 2 - Partial Overlap:")
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")
    assert 0 < iou < 1, "IoU should be between 0 and 1 for partial overlap"
    assert 0 < dice < 1, "Dice should be between 0 and 1 for partial overlap"
    
    # No overlap case
    mask1 = np.zeros((10, 10), dtype=np.uint8)
    mask2 = np.zeros((10, 10), dtype=np.uint8)
    mask1[1:4, 1:4] = 1
    mask2[6:9, 6:9] = 1
    
    iou = calculate_iou(mask1, mask2)
    dice = calculate_dice(mask1, mask2)
    
    print("\nTest Case 3 - No Overlap:")
    print(f"IoU: {iou:.4f} (expected: 0.0000)")
    print(f"Dice: {dice:.4f} (expected: 0.0000)")
    assert abs(iou - 0.0) < 1e-6, "IoU should be 0.0 for no overlap"
    assert abs(dice - 0.0) < 1e-6, "Dice should be 0.0 for no overlap"
    
    # Test evaluate_with_iou function
    masks1 = {1: mask1, 2: mask1}
    masks2 = {1: mask2, 2: mask2}
    
    results = evaluate_with_iou(masks1, masks2)
    print("\nTest Case 4 - evaluate_with_iou:")
    print(f"Results: {results}")
    assert 'mean_iou' in results, "Results should contain mean_iou"
    assert 'mean_dice' in results, "Results should contain mean_dice"
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_evaluation_utils() 