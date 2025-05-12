# visualization_utils.py
import cv2
import numpy as np
import os
from tqdm import tqdm

def visualize_comparison(video_path, algorithm_masks, ground_truth_masks, output_path):
    """
    Create a side-by-side comparison video showing algorithm vs ground truth
    
    Args:
        video_path: Path to the original video
        algorithm_masks: Dictionary of algorithm masks {frame_idx: mask}
        ground_truth_masks: Dictionary of ground truth masks {frame_idx: mask}
        output_path: Path to save the visualization video
        
    Returns:
        Path to the output video
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    # Define colors
    algorithm_color = (0, 0, 255)  # Red for algorithm
    ground_truth_color = (0, 255, 0)  # Green for ground truth
    overlap_color = (255, 0, 0)  # Blue for overlap
    
    # Process each frame
    frame_nums = sorted(list(set(algorithm_masks.keys()) | set(ground_truth_masks.keys())))
    
    for frame_idx in tqdm(frame_nums, desc="Creating comparison video"):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Create visualization frame
        vis_frame = np.zeros((height, width*2, 3), dtype=np.uint8)
        
        # Copy original frame to both sides
        vis_frame[:, :width] = frame.copy()  # Original with algorithm mask
        vis_frame[:, width:] = frame.copy()  # Original with ground truth mask
        
        # Draw algorithm mask on left side
        if frame_idx in algorithm_masks:
            pred_mask = algorithm_masks[frame_idx]
            if isinstance(pred_mask, dict) and 'mask' in pred_mask:
                pred_mask = pred_mask['mask']
                
            binary_pred = (pred_mask > 0.5).astype(np.uint8)
            pred_contours, _ = cv2.findContours(binary_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_frame[:, :width], pred_contours, -1, algorithm_color, 2)
            
            # Add semi-transparent overlay
            overlay = vis_frame[:, :width].copy()
            overlay[binary_pred > 0] = algorithm_color
            cv2.addWeighted(overlay, 0.3, vis_frame[:, :width], 0.7, 0, vis_frame[:, :width])
        
        # Draw ground truth mask on right side
        if frame_idx in ground_truth_masks:
            gt_mask = ground_truth_masks[frame_idx]
            if isinstance(gt_mask, dict) and 'mask' in gt_mask:
                gt_mask = gt_mask['mask']
                
            binary_gt = (gt_mask > 0.5).astype(np.uint8)
            gt_contours, _ = cv2.findContours(binary_gt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_frame[:, width:], gt_contours, -1, ground_truth_color, 2)
            
            # Add semi-transparent overlay
            overlay = vis_frame[:, width:].copy()
            overlay[binary_gt > 0] = ground_truth_color
            cv2.addWeighted(overlay, 0.3, vis_frame[:, width:], 0.7, 0, vis_frame[:, width:])
        
        # Calculate and display IoU if both masks exist
        if frame_idx in algorithm_masks and frame_idx in ground_truth_masks:
            pred_mask = algorithm_masks[frame_idx]
            gt_mask = ground_truth_masks[frame_idx]
            
            if isinstance(pred_mask, dict) and 'mask' in pred_mask:
                pred_mask = pred_mask['mask']
            if isinstance(gt_mask, dict) and 'mask' in gt_mask:
                gt_mask = gt_mask['mask']
                
            binary_pred = (pred_mask > 0.5).astype(np.uint8)
            binary_gt = (gt_mask > 0.5).astype(np.uint8)
            
            # Calculate IoU and Dice
            from evaluation_utils import calculate_iou, calculate_dice
            iou = calculate_iou(binary_pred, binary_gt)
            dice = calculate_dice(binary_pred, binary_gt)
            
            # Display metrics
            cv2.putText(vis_frame, f"IoU: {iou:.4f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Dice: {dice:.4f}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, height - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(vis_frame, "Algorithm", (width//3, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, algorithm_color, 2)
        cv2.putText(vis_frame, "Ground Truth", (width + width//3, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, ground_truth_color, 2)
        
        # Write frame to output
        out.write(vis_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path