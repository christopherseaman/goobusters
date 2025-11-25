import cv2
import numpy as np
import os
from tqdm import tqdm

# Usage
base_dir = os.path.join('optical_flow', 'annotation_1022')
video_path = os.path.join(base_dir, 'masked_1.2.826.0.1.3680043.8.498.94045070278013858526642639823753659634')
mask_dir = os.path.join(base_dir, 'masks')
output_dir = base_dir

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Metrics storage
iou_scores = []
area_changes = []

def diagnose_tracking(video_path, mask_dir, output_dir):
    prev_mask = None
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in tqdm(range(total_frames), desc="Analyzing frames"):
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Read corresponding mask
        mask_path = os.path.join(mask_dir, f"mask_{frame_idx:04d}.png")
        if not os.path.exists(mask_path):
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if prev_mask is not None:
            # Calculate IoU
            intersection = np.logical_and(mask, prev_mask)
            union = np.logical_or(mask, prev_mask)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            iou_scores.append(iou)
            
            # Calculate area change
            area_change = (np.sum(mask) - np.sum(prev_mask)) / np.sum(prev_mask) if np.sum(prev_mask) > 0 else 0
            area_changes.append(area_change)
            
            # Visualize differences
            diff = cv2.absdiff(mask, prev_mask)
            heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            
            # Overlay heatmap on frame
            overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
            
            # Add text with metrics
            cv2.putText(overlay, f"IoU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(overlay, f"Area Change: {area_change:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # Save diagnostic frame
            cv2.imwrite(os.path.join(output_dir, f"diagnostic_{frame_idx:04d}.png"), overlay)
        
        prev_mask = mask

    cap.release()

    # Plot metrics
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(iou_scores, label='IoU')
    plt.plot(area_changes, label='Area Change')
    plt.xlabel('Frame')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Tracking Metrics Over Time')
    plt.savefig(os.path.join(output_dir, 'tracking_metrics.png'))
    plt.close()

    return iou_scores, area_changes

if __name__ == "__main__":
    iou_scores, area_changes = diagnose_tracking(video_path, mask_dir, output_dir)