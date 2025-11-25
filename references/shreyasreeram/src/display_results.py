import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def display_tracking_results(tracker_results):
    """
    Display comprehensive results from the multi-frame tracking system.
    
    Args:
        tracker_results: Dictionary containing tracking results
    """
    print("\n=== Multi-Frame Tracking Results ===")
    
    # 1. Basic Statistics
    all_masks = tracker_results['all_masks']
    print(f"\nFrame Statistics:")
    print(f"Total Processed Frames: {len(all_masks)}")
    print(f"Expert Annotations: {tracker_results['annotated_frames']}")
    print(f"Predicted Frames: {tracker_results['predicted_frames']}")
    
    # 2. Annotation Type Distribution
    print("\nAnnotation Type Distribution:")
    for ann_type, count in tracker_results['annotation_types'].items():
        print(f"  {ann_type}: {count} frames")
    
    # 3. Create visualization plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Annotation Type Distribution
    plt.subplot(2, 2, 1)
    types = list(tracker_results['annotation_types'].keys())
    counts = list(tracker_results['annotation_types'].values())
    plt.bar(types, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Annotation Types')
    plt.ylabel('Number of Frames')
    
    # Plot 2: Annotation vs Prediction Ratio
    plt.subplot(2, 2, 2)
    ratio_data = [
        tracker_results['annotated_frames'],
        tracker_results['predicted_frames']
    ]
    plt.pie(ratio_data, labels=['Expert Annotations', 'Predictions'], 
            autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
    plt.title('Expert Annotations vs Predictions')
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(tracker_results['output_video'])
    plt.savefig(os.path.join(output_dir, f'tracking_results_{timestamp}.png'))
    plt.close()
    
    # 4. Output Video Information
    print(f"\nVisualization Outputs:")
    print(f"1. Results plots saved to: tracking_results_{timestamp}.png")
    print(f"2. Tracking visualization video: {tracker_results['output_video']}")
    
    # 5. Additional Metrics (if available)
    if 'metrics' in tracker_results:
        metrics = tracker_results['metrics']
        print("\nPerformance Metrics:")
        if 'mean_iou' in metrics:
            print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        if 'mean_dice' in metrics:
            print(f"Mean Dice Score: {metrics['mean_dice']:.4f}")
        if 'iou_scores' in metrics:
            print(f"IoU Score Range: {min(metrics['iou_scores'].values()):.4f} - {max(metrics['iou_scores'].values()):.4f}")

def main():
    """
    Example usage of the display function.
    """
    # Example results dictionary
    example_results = {
        'all_masks': {'frame_1': {}, 'frame_2': {}},  # Replace with actual masks
        'annotated_frames': 10,
        'predicted_frames': 50,
        'annotation_types': {
            'fluid': 20,
            'predicted_fluid': 30,
            'predicted_clear': 10
        },
        'output_video': 'path/to/video.mp4'
    }
    
    display_tracking_results(example_results)

if __name__ == "__main__":
    main() 