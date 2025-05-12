# evaluation_utils.py
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt

def calculate_iou(mask1, mask2):
    """Calculate IoU between two binary masks"""
    if mask1 is None or mask2 is None:
        return 0.0
        
    # Convert to binary if needed
    binary1 = (mask1 > 0.5).astype(np.uint8) if not isinstance(mask1, bool) else mask1.astype(np.uint8)
    binary2 = (mask2 > 0.5).astype(np.uint8) if not isinstance(mask2, bool) else mask2.astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    
    # Calculate IoU
    return intersection / union if union > 0 else 0.0

def calculate_dice(mask1, mask2):
    """Calculate Dice coefficient between two binary masks"""
    if mask1 is None or mask2 is None:
        return 0.0
        
    # Convert to binary if needed
    binary1 = (mask1 > 0.5).astype(np.uint8) if not isinstance(mask1, bool) else mask1.astype(np.uint8)
    binary2 = (mask2 > 0.5).astype(np.uint8) if not isinstance(mask2, bool) else mask2.astype(np.uint8)
    
    # Calculate intersection
    intersection = np.logical_and(binary1, binary2).sum()
    
    # Calculate Dice coefficient (2*|A∩B|)/(|A|+|B|)
    return (2 * intersection) / (binary1.sum() + binary2.sum()) if (binary1.sum() + binary2.sum()) > 0 else 0.0

def evaluate_with_iou(algorithm_masks, ground_truth_masks):
    """
    Evaluate algorithm masks against ground truth using IoU and Dice metrics
    
    Args:
        algorithm_masks: Dictionary of algorithm masks {frame_idx: mask}
        ground_truth_masks: Dictionary of ground truth masks {frame_idx: mask}
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Store IoU for each frame
    iou_scores = []
    dice_scores = []
    
    # Process each frame
    common_frames = set(algorithm_masks.keys()) & set(ground_truth_masks.keys())
    
    for frame_idx in common_frames:
        # Get masks
        pred_mask = algorithm_masks[frame_idx]
        gt_mask = ground_truth_masks[frame_idx]
        
        # Handle different data formats
        if isinstance(pred_mask, dict) and 'mask' in pred_mask:
            pred_mask = pred_mask['mask']
        if isinstance(gt_mask, dict) and 'mask' in gt_mask:
            gt_mask = gt_mask['mask']
        
        # Calculate IoU
        iou = calculate_iou(pred_mask, gt_mask)
        iou_scores.append(iou)
        
        # Calculate Dice coefficient
        dice = calculate_dice(pred_mask, gt_mask)
        dice_scores.append(dice)
    
    # Calculate metrics
    if iou_scores:
        metrics['mean_iou'] = np.mean(iou_scores)
        metrics['median_iou'] = np.median(iou_scores)
        metrics['min_iou'] = np.min(iou_scores)
        metrics['max_iou'] = np.max(iou_scores)
        metrics['std_iou'] = np.std(iou_scores)
        
        metrics['mean_dice'] = np.mean(dice_scores)
        metrics['median_dice'] = np.median(dice_scores)
        
        # Calculate percentage of frames with IoU > thresholds
        metrics['iou_over_0.5'] = sum(1 for s in iou_scores if s > 0.5) / len(iou_scores)
        metrics['iou_over_0.7'] = sum(1 for s in iou_scores if s > 0.7) / len(iou_scores) 
        metrics['iou_over_0.9'] = sum(1 for s in iou_scores if s > 0.9) / len(iou_scores)
    else:
        # If no frames to evaluate
        metrics['mean_iou'] = 0.0
        metrics['mean_dice'] = 0.0
        metrics['frames_evaluated'] = 0
        
    # Add frame count info
    metrics['frames_evaluated'] = len(common_frames)
    metrics['algorithm_frames'] = len(algorithm_masks)
    metrics['ground_truth_frames'] = len(ground_truth_masks)
    metrics['frame_coverage'] = len(common_frames) / max(len(algorithm_masks), len(ground_truth_masks)) if max(len(algorithm_masks), len(ground_truth_masks)) > 0 else 0
    
    return metrics

def create_evaluation_report(metrics, output_dir, study_uid, series_uid):
    """Create a Markdown report of evaluation metrics"""
    report_path = os.path.join(output_dir, f"evaluation_{study_uid}_{series_uid}.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Evaluation Report: {study_uid}/{series_uid}\n\n")
        
        f.write("## Summary Metrics\n\n")
        f.write(f"- Mean IoU: {metrics.get('mean_iou', 0):.4f}\n")
        f.write(f"- Median IoU: {metrics.get('median_iou', 0):.4f}\n")
        f.write(f"- Mean Dice: {metrics.get('mean_dice', 0):.4f}\n")
        f.write(f"- Frames with IoU > 0.7: {metrics.get('iou_over_0.7', 0)*100:.1f}%\n\n")
        
        f.write(f"- Frames Evaluated: {metrics.get('frames_evaluated', 0)}\n")
        f.write(f"- Algorithm Frames: {metrics.get('algorithm_frames', 0)}\n")
        f.write(f"- Ground Truth Frames: {metrics.get('ground_truth_frames', 0)}\n")
        f.write(f"- Frame Coverage: {metrics.get('frame_coverage', 0)*100:.1f}%\n\n")
        
    return report_path

