import os
import numpy as np
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

def calculate_dice(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    dice = 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(pred_mask)) if (np.sum(gt_mask) + np.sum(pred_mask)) > 0 else 0
    return dice

def overlay_mask(image, mask, color=[0, 255, 0], alpha=0.5):
    """Overlay a mask on an image"""
    overlay = image.copy()
    if len(color) == 3:
        color = color + [255]  # Add alpha channel
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[mask > 0] = color
    cv2.addWeighted(mask_rgba[:,:,:3], alpha, overlay, 1-alpha, 0, overlay)
    return overlay

def create_sparse_validation_set(ground_truth_dir, algorithm_results_dir, num_frames=25, output_dir=None, selected_frames=None):
    """
    Create a sparse validation set by selecting frames evenly throughout the video
    
    Args:
        ground_truth_dir: Directory containing ground truth masks
        algorithm_results_dir: Directory containing algorithm results
        num_frames: Number of frames to include in validation set (used if selected_frames is None)
        output_dir: Directory to save validation results
        selected_frames: List of specific frame numbers to include (overrides num_frames)
    
    Returns:
        Dictionary with validation results
    """
    print(f"\nStarting sparse validation...")
    print(f"Ground truth directory: {ground_truth_dir}")
    print(f"Algorithm results directory: {algorithm_results_dir}")
    
    # Find all ground truth files and sort them
    gt_files = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.png')])
    total_frames = len(gt_files)
    print(f"Found {total_frames} ground truth files")
    
    # Extract frame numbers, handling both formats (frame_0047.png and 0047.png)
    def extract_frame_num(filename):
        try:
            # Remove .png extension
            base = filename.rsplit('.', 1)[0]
            # Remove 'frame_' prefix if it exists
            if base.startswith('frame_'):
                base = base[6:]
            # Convert to integer
            return int(base)
        except ValueError as e:
            print(f"Warning: Could not parse frame number from {filename}: {e}")
            return None
    
    # Create mapping of frame numbers to files
    frame_to_file = {extract_frame_num(f): f for f in gt_files if extract_frame_num(f) is not None}
    print(f"Successfully parsed {len(frame_to_file)} frame numbers")
    
    if not frame_to_file:
        error_msg = "No valid frame numbers found in files"
        print(f"Error: {error_msg}")
        return {"error": error_msg, "metrics": {"mean_iou": 0.0, "median_iou": 0.0, "mean_dice": 0.0, "iou_over_0.7": 0.0}}
    
    # If specific frames are provided, use those
    if selected_frames:
        selected_frames = [f for f in selected_frames if f in frame_to_file]
        if not selected_frames:
            error_msg = "None of the selected frames were found in the dataset"
            print(f"Error: {error_msg}")
            return {"error": error_msg, "metrics": {"mean_iou": 0.0, "median_iou": 0.0, "mean_dice": 0.0, "iou_over_0.7": 0.0}}
    else:
        # Select frames evenly distributed throughout the video
        all_frames = sorted(frame_to_file.keys())
        if num_frames >= len(all_frames):
            selected_frames = all_frames
        else:
            step = len(all_frames) / num_frames
            selected_frames = [all_frames[int(i * step)] for i in range(num_frames)]
    
    print(f"Selected {len(selected_frames)} frames for evaluation")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Collect results
    results = {
        "frame_ious": {},
        "frame_dice": {},
        "selected_frames": []
    }
    
    # Process each selected frame
    for frame_num in selected_frames:
        gt_file = frame_to_file[frame_num]
        results["selected_frames"].append(frame_num)
        
        # Load ground truth mask
        gt_path = os.path.join(ground_truth_dir, gt_file)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Load algorithm mask (try both naming formats)
        algo_file = gt_file  # First try exact match
        algo_path = os.path.join(algorithm_results_dir, algo_file)
        
        if not os.path.exists(algo_path):
            # Try alternate format
            alt_file = f"{frame_num:04d}.png"
            algo_path = os.path.join(algorithm_results_dir, alt_file)
        
        if os.path.exists(algo_path):
            algo_mask = cv2.imread(algo_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert masks to binary
            gt_binary = gt_mask > 0
            algo_binary = algo_mask > 0
            
            # Calculate metrics
            iou = calculate_iou(gt_binary, algo_binary)
            dice = calculate_dice(gt_binary, algo_binary)
            
            results["frame_ious"][frame_num] = float(iou)
            results["frame_dice"][frame_num] = float(dice)
            
            print(f"Frame {frame_num}: IoU = {iou:.4f}, Dice = {dice:.4f}")
            
            # Visualize if output directory is specified
            if output_dir:
                # Create visualization with masks only
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(gt_binary, cmap='gray')
                axes[0].set_title("Ground Truth")
                axes[1].imshow(algo_binary, cmap='gray')
                axes[1].set_title(f"Algorithm (IoU: {iou:.3f})")
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"frame_{frame_num}_comparison.png"))
                plt.close()
        else:
            print(f"Warning: No algorithm mask found for frame {frame_num}")
    
    # Calculate summary metrics
    if results["frame_ious"]:
        iou_values = list(results["frame_ious"].values())
        dice_values = list(results["frame_dice"].values())
        
        results["summary"] = {
            "mean_iou": float(np.mean(iou_values)),
            "median_iou": float(np.median(iou_values)),
            "mean_dice": float(np.mean(dice_values)),
            "iou_over_0.7": len([x for x in iou_values if x > 0.7]) / len(iou_values)
        }
        
        print("\nFinal Metrics:")
        print(f"Mean IoU: {results['summary']['mean_iou']:.4f}")
        print(f"Mean Dice: {results['summary']['mean_dice']:.4f}")
        print(f"Frames with IoU > 0.7: {results['summary']['iou_over_0.7']*100:.1f}%")
    
    # Save results if output directory is specified
    if output_dir:
        results_path = os.path.join(output_dir, "sparse_validation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed results to: {results_path}")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create sparse validation set")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground truth masks")
    parser.add_argument("--algo_dir", required=True, help="Directory containing algorithm result masks")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames to include in validation")
    parser.add_argument("--output_dir", help="Directory to save validation results")
    parser.add_argument("--frames", help="Comma-separated list of specific frame numbers to include (e.g., '30,67,84')")
    
    args = parser.parse_args()
    
    # Process selected frames if provided
    selected_frames = None
    if args.frames:
        selected_frames = [int(f.strip()) for f in args.frames.split(',')]
        print(f"Using user-selected frames: {selected_frames}")
    
    results = create_sparse_validation_set(
        args.gt_dir, 
        args.algo_dir,
        args.num_frames,
        args.output_dir,
        selected_frames
    )
    
    # Print summary metrics
    if "summary" in results:
        print("\nSummary Metrics (Sparse Validation Set):")
        print(f"Mean IoU: {results['summary']['mean_iou']:.4f}")
        print(f"Median IoU: {results['summary']['median_iou']:.4f}")
        print(f"Mean Dice: {results['summary']['mean_dice']:.4f}")
        print(f"Frames with IoU > 0.7: {results['summary']['iou_over_0.7']*100:.1f}%")

if __name__ == "__main__":
    main() 