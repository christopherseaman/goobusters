#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import hashlib
import pickle
import cv2
from datetime import datetime, timedelta
from src.utils.mdai_utils import get_mdai_client, get_annotations_for_study_series
from src.utils.mask_utils import polygons_to_mask
from src.utils import evaluate_with_iou

def get_cache_dir():
    """Get cache directory path, creating it if needed"""
    cache_dir = os.path.join("src", "cache", "md_ai_data")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_cache_file(project_id, dataset_id, study_uid, series_uid, label_id):
    """Generate a cache filename based on request parameters"""
    # Create a unique identifier for this specific query
    key = f"{project_id}_{dataset_id}_{study_uid}_{series_uid}_{label_id}"
    cache_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(get_cache_dir(), f"annotations_{cache_key}.pkl")

def get_cached_annotations(project_id, dataset_id, study_uid, series_uid, label_id, max_age_hours=24):
    """Get cached annotations if available and not expired"""
    cache_file = get_cache_file(project_id, dataset_id, study_uid, series_uid, label_id)
    
    if os.path.exists(cache_file):
        # Check cache age
        file_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_modified < timedelta(hours=max_age_hours):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    return None

def cache_annotations(project_id, dataset_id, study_uid, series_uid, label_id, annotations):
    """Save annotations to cache"""
    cache_file = get_cache_file(project_id, dataset_id, study_uid, series_uid, label_id)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(annotations, f)
        print(f"Cached annotations to {cache_file}")
    except Exception as e:
        print(f"Error caching annotations: {e}")

def evaluate_selected_frames_from_mdai(
    study_uid,
    series_uid,
    ground_truth_label_id,
    algorithm_label_id,
    project_id,
    dataset_id,
    selected_frames=None,
    force_cache_refresh=False,
    save_masks=True,
    output_dir=None
):
    """
    Evaluate selected frames from MD.ai annotations.
    
    Args:
        study_uid: DICOM Study Instance UID
        series_uid: DICOM Series Instance UID
        ground_truth_label_id: MD.ai label ID for ground truth annotations
        algorithm_label_id: MD.ai label ID for algorithm annotations
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID
        selected_frames: List of frame numbers to evaluate (if None, all frames will be evaluated)
        force_cache_refresh: If True, will ignore cached annotations
        save_masks: If True, will save mask images for visualization
        output_dir: Directory to save output files (if None, will use src/output/evaluations/<study_uid>_<series_uid>)
    
    Returns:
        dict: Evaluation results
    """
    
    # Convert selected frames to list of integers if provided as string
    if isinstance(selected_frames, str):
        selected_frames = [int(f.strip()) for f in selected_frames.split(',')]
    
    if selected_frames:
        print(f"Using user-selected frames: {selected_frames}")

    # Create output directory
    if output_dir is None:
        output_dir = f"src/output/evaluations/{study_uid[-8:]}-{series_uid[-8:]}"
    os.makedirs(output_dir, exist_ok=True)
    
    # If saving masks, create masks directory
    masks_dir = os.path.join(output_dir, "masks")
    if save_masks:
        os.makedirs(masks_dir, exist_ok=True)
    
    print(f"Evaluating {len(selected_frames) if selected_frames else 'all'} selected frames")
    
    # Get ground truth annotations
    print("Fetching ground truth annotations...")
    gt_annotations = get_annotations_for_study_series(
        study_uid, series_uid, 
        project_id=project_id, 
        dataset_id=dataset_id,
        label_id=ground_truth_label_id,
        use_cache=not force_cache_refresh
    )
    
    # Get algorithm annotations
    print("Fetching algorithm annotations...")
    algo_annotations = get_annotations_for_study_series(
        study_uid, series_uid, 
        project_id=project_id, 
        dataset_id=dataset_id,
        label_id=algorithm_label_id,
        use_cache=not force_cache_refresh
    )
    
    # Determine frame dimensions - first try to get from annotations
    frame_width = 640
    frame_height = 480
    
    for ann in gt_annotations + algo_annotations:
        if 'width' in ann and 'height' in ann:
            frame_width = ann.get('width', 640)
            frame_height = ann.get('height', 480)
            break
    
    print(f"Using frame dimensions: {frame_width}x{frame_height}")
    
    # Convert annotations to masks
    gt_masks = {}
    algo_masks = {}
    
    print("Converting ground truth annotations to masks...")
    for ann in gt_annotations:
        frame_num = int(ann.get('frameNumber', 0))
        if 'data' in ann and isinstance(ann['data'], dict) and 'foreground' in ann['data']:
            # Convert polygon to mask
            foreground = ann['data']['foreground']
            if foreground and len(foreground) > 0:
                mask = polygons_to_mask(foreground, height=frame_height, width=frame_width)
                gt_masks[frame_num] = mask
                print(f"Added ground truth mask for frame {frame_num} (pixels: {np.sum(mask)})")
    
    print("Converting algorithm annotations to masks...")
    # Group algorithm annotations by frame
    algo_annotations_by_frame = {}
    for ann in algo_annotations:
        frame_num = int(ann.get('frameNumber', 0))
        if frame_num not in algo_annotations_by_frame:
            algo_annotations_by_frame[frame_num] = []
        algo_annotations_by_frame[frame_num].append(ann)
    
    # Use the most recent annotation for each frame
    for frame_num, annotations in algo_annotations_by_frame.items():
        if len(annotations) > 1:
            print(f"Found {len(annotations)} algorithm annotations for frame {frame_num}, using most recent")
            # Sort by createdAt (newest first)
            annotations.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        
        ann = annotations[0]  # Use the most recent annotation
        if 'data' in ann and isinstance(ann['data'], dict) and 'foreground' in ann['data']:
            # Convert polygon to mask
            foreground = ann['data']['foreground']
            if foreground and len(foreground) > 0:
                mask = polygons_to_mask(foreground, height=frame_height, width=frame_width)
                algo_masks[frame_num] = mask
                print(f"Added algorithm mask for frame {frame_num} (pixels: {np.sum(mask)})")
    
    # Find common frames for evaluation
    common_frames = sorted(set(gt_masks.keys()) & set(algo_masks.keys()))
    
    # Filter by selected frames if provided
    if selected_frames:
        common_frames = [f for f in common_frames if f in selected_frames]
    
    print(f"Found {len(common_frames)} common frames for evaluation")
    
    if not common_frames:
        print("No common frames found for evaluation")
        return {
            "mean_iou": 0,
            "median_iou": 0,
            "mean_dice": 0,
            "iou_over_07": 0,
            "num_frames": 0,
            "frames_over_threshold": 0
        }
    
    # Run evaluation
    print("Running evaluation...")
    
    ious = []
    dices = []
    frame_results = {}
    
    for frame in common_frames:
        gt_mask = gt_masks[frame]
        algo_mask = algo_masks[frame]
        
        # Calculate IoU and Dice
        iou = calculate_iou(gt_mask, algo_mask)
        dice = calculate_dice(gt_mask, algo_mask)
        
        ious.append(iou)
        dices.append(dice)
        
        # Store per-frame results
        frame_results[str(frame)] = {
            "iou": float(iou),
            "dice": float(dice),
            "gt_pixels": int(np.sum(gt_mask)),
            "algo_pixels": int(np.sum(algo_mask)),
            "overlap_pixels": int(np.sum(np.logical_and(gt_mask, algo_mask)))
        }
        
        # Save masks as images if requested
        if save_masks:
            cv2.imwrite(os.path.join(masks_dir, f"gt_mask_{frame}.png"), gt_mask * 255)
            cv2.imwrite(os.path.join(masks_dir, f"algo_mask_{frame}.png"), algo_mask * 255)
    
    # Calculate summary metrics
    mean_iou = np.mean(ious)
    median_iou = np.median(ious)
    mean_dice = np.mean(dices)
    
    # Calculate percentage of frames with IoU > 0.7
    iou_over_07 = sum(1 for iou in ious if iou > 0.7) / len(ious) if ious else 0
    
    # Print results
    print("\n=== Evaluation Summary ===")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Median IoU: {median_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Frames with IoU > 0.7: {iou_over_07 * 100:.1f}%")
    
    # Save results
    results = {
        "mean_iou": float(mean_iou),
        "median_iou": float(median_iou),
        "mean_dice": float(mean_dice),
        "iou_over_07": float(iou_over_07),
        "num_frames": len(common_frames),
        "frames_over_threshold": sum(1 for iou in ious if iou > 0.7)
    }
    
    # Save frame results
    with open(os.path.join(output_dir, "frame_results.json"), "w") as f:
        json.dump(frame_results, f, indent=2)
    
    # Save summary results
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save full evaluation results
    full_results = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "ground_truth_label_id": ground_truth_label_id,
        "algorithm_label_id": algorithm_label_id,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "frame_dimensions": {"width": frame_width, "height": frame_height},
        "evaluated_frames": common_frames,
        "metrics": results,
        "frame_results": frame_results
    }
    
    with open(os.path.join(output_dir, "full_evaluation_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate masks from MD.ai annotations")
    parser.add_argument("--study-uid", required=True, help="DICOM Study Instance UID")
    parser.add_argument("--series-uid", required=True, help="DICOM Series Instance UID")
    parser.add_argument("--ground-truth-label-id", required=True, help="MD.ai label ID for ground truth annotations")
    parser.add_argument("--algorithm-label-id", required=True, help="MD.ai label ID for algorithm annotations")
    parser.add_argument("--project-id", required=True, help="MD.ai project ID")
    parser.add_argument("--dataset-id", required=True, help="MD.ai dataset ID")
    parser.add_argument("--selected-frames", help="Comma-separated list of frame numbers to evaluate")
    parser.add_argument("--no-cache", action="store_true", help="Force refresh of cached annotations")
    parser.add_argument("--no-save-masks", action="store_false", dest="save_masks", help="Don't save mask images")
    parser.add_argument("--output-dir", help="Directory to save output files")
    
    args = parser.parse_args()
    
    evaluate_selected_frames_from_mdai(
        study_uid=args.study_uid,
        series_uid=args.series_uid,
        ground_truth_label_id=args.ground_truth_label_id,
        algorithm_label_id=args.algorithm_label_id,
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        selected_frames=args.selected_frames,
        force_cache_refresh=args.no_cache,
        save_masks=args.save_masks,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 