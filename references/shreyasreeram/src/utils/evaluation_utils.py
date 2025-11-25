# evaluation_utils.py
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union between two binary masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    
    if np.sum(union) == 0:
        # If both masks are empty, consider it perfect match
        return 1.0 if np.sum(pred_mask) == np.sum(gt_mask) else 0.0
    
    return np.sum(intersection) / np.sum(union)

def calculate_dice(pred_mask, gt_mask):
    """
    Calculate Dice coefficient between two binary masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    
    if np.sum(pred_mask) + np.sum(gt_mask) == 0:
        # If both masks are empty, consider it perfect match
        return 1.0
    
    return 2.0 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(gt_mask))

def evaluate_tracking(predicted_masks, ground_truth_masks):
    """
    Evaluate tracking performance by comparing predicted masks to ground truth.
    
    Args:
        predicted_masks: Dictionary of predicted masks {frame_idx: mask}
        ground_truth_masks: Dictionary of ground truth masks {frame_idx: mask}
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'iou_scores': [],
        'dice_scores': [],
        'frame_metrics': {}
    }
    
    # Calculate metrics for each frame
    for frame_idx in predicted_masks.keys():
        if frame_idx not in ground_truth_masks:
            continue
            
        pred_mask = predicted_masks[frame_idx]
        gt_mask = ground_truth_masks[frame_idx]
        
        # Convert to binary masks if needed
        if isinstance(pred_mask, dict):
            pred_mask = pred_mask['mask']
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        # Calculate metrics
        iou = calculate_iou(pred_mask, gt_mask)
        dice = calculate_dice(pred_mask, gt_mask)
        
        metrics['iou_scores'].append(iou)
        metrics['dice_scores'].append(dice)
        metrics['frame_metrics'][frame_idx] = {
            'iou': iou,
            'dice': dice
        }
    
    # Calculate summary metrics
    if metrics['iou_scores']:
        metrics['mean_iou'] = np.mean(metrics['iou_scores'])
        metrics['median_iou'] = np.median(metrics['iou_scores'])
        metrics['mean_dice'] = np.mean(metrics['dice_scores'])
        metrics['iou_over_0.7'] = np.mean([s >= 0.7 for s in metrics['iou_scores']])
    else:
        metrics['mean_iou'] = 0.0
        metrics['median_iou'] = 0.0
        metrics['mean_dice'] = 0.0
        metrics['iou_over_0.7'] = 0.0
    
    return metrics

def save_evaluation_visualizations(predicted_masks, ground_truth_masks, output_dir):
    """
    Save visualizations comparing predicted and ground truth masks.
    
    Args:
        predicted_masks: Dictionary of predicted masks {frame_idx: mask}
        ground_truth_masks: Dictionary of ground truth masks {frame_idx: mask}
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_idx in predicted_masks.keys():
        if frame_idx not in ground_truth_masks:
            continue
            
        pred_mask = predicted_masks[frame_idx]
        gt_mask = ground_truth_masks[frame_idx]
        
        # Convert to binary masks
        if isinstance(pred_mask, dict):
            pred_mask = pred_mask['mask']
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        # Create visualization
        height, width = pred_mask.shape
        vis_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Red channel: Ground truth
        vis_img[gt_mask > 0] = [255, 0, 0]
        
        # Green channel: Predictions
        vis_img[pred_mask > 0] = [0, 255, 0]
        
        # Yellow: Overlap
        vis_img[np.logical_and(pred_mask > 0, gt_mask > 0)] = [255, 255, 0]
        
        # Save visualization
        cv2.imwrite(
            os.path.join(output_dir, f'comparison_frame_{frame_idx}.png'),
            vis_img
        )

def create_evaluation_report(metrics, output_dir, study_uid, series_uid, predicted_masks, ground_truth_masks):
    """Create a Markdown report of evaluation metrics"""
    report_path = os.path.join(output_dir, f"evaluation_{study_uid}_{series_uid}.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Evaluation Report: {study_uid}/{series_uid}\n\n")
        
        f.write("## Summary Metrics\n\n")
        f.write(f"- Mean IoU: {metrics.get('mean_iou', 0):.4f}\n")
        f.write(f"- Median IoU: {metrics.get('median_iou', 0):.4f}\n")
        f.write(f"- Mean Dice: {metrics.get('mean_dice', 0):.4f}\n")
        f.write(f"- Frames with IoU > 0.7: {metrics.get('iou_over_0.7', 0)*100:.1f}%\n\n")
        
        f.write(f"- Frames Evaluated: {len(metrics.get('iou_scores', []))}\n")
        f.write(f"- Algorithm Frames: {len(predicted_masks)}\n")
        f.write(f"- Ground Truth Frames: {len(ground_truth_masks)}\n")
        f.write(f"- Frame Coverage: {len(metrics.get('iou_scores', [])) / len(predicted_masks) * 100:.1f}%\n\n")
        
    return report_path

def evaluate_with_iou(algorithm_masks, ground_truth_masks):
    """
    Evaluate algorithm masks against ground truth masks using IoU and Dice metrics.
    
    Args:
        algorithm_masks: Dictionary of algorithm masks {frame_idx: mask or {mask: mask}}
        ground_truth_masks: Dictionary of ground truth masks {frame_idx: mask}
        
    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = {
        'iou_scores': [],
        'dice_scores': [],
        'frame_metrics': {}
    }
    
    # Find common frames
    common_frames = set(algorithm_masks.keys()) & set(ground_truth_masks.keys())
    print(f"Evaluating {len(common_frames)} common frames")
    
    for frame_idx in common_frames:
        # Get masks
        algo_mask = algorithm_masks[frame_idx]
        gt_mask = ground_truth_masks[frame_idx]
        
        # Handle dict format for algorithm masks
        if isinstance(algo_mask, dict):
            algo_mask = algo_mask['mask']
        
        # Convert to binary
        algo_mask = (algo_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        # Calculate metrics
        iou = calculate_iou(algo_mask, gt_mask)
        dice = calculate_dice(algo_mask, gt_mask)
        
        metrics['iou_scores'].append(iou)
        metrics['dice_scores'].append(dice)
        metrics['frame_metrics'][frame_idx] = {
            'iou': iou,
            'dice': dice
        }
    
    # Calculate summary metrics
    if metrics['iou_scores']:
        metrics['mean_iou'] = np.mean(metrics['iou_scores'])
        metrics['median_iou'] = np.median(metrics['iou_scores'])
        metrics['mean_dice'] = np.mean(metrics['dice_scores'])
        metrics['iou_over_0.7'] = np.mean([s >= 0.7 for s in metrics['iou_scores']])
    else:
        metrics['mean_iou'] = 0.0
        metrics['median_iou'] = 0.0
        metrics['mean_dice'] = 0.0
        metrics['iou_over_0.7'] = 0.0
    
    return metrics

