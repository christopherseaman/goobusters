import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def create_presentation_visuals(results_dict, output_dir):
    """
    Create presentation-ready visualizations from tracking results.
    
    Args:
        results_dict: Dictionary containing tracking results
        output_dir: Directory to save visualizations
    """
    plt.style.use('seaborn')
    
    # 1. Performance Overview
    plt.figure(figsize=(20, 10))
    
    # Plot 1: IoU Distribution
    plt.subplot(2, 2, 1)
    iou_scores = results_dict['iterations'][0]['evaluation_results']['summary']['overall_mean_iou']
    plt.hist(iou_scores, bins=20, color='skyblue', alpha=0.7)
    plt.axvline(x=np.mean(iou_scores), color='r', linestyle='--', label=f'Mean IoU: {np.mean(iou_scores):.3f}')
    plt.title('IoU Score Distribution')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Performance Timeline
    plt.subplot(2, 2, 2)
    frame_metrics = results_dict['iterations'][0]['evaluation_results']['metrics']['frame_metrics']
    frames = sorted(list(map(int, frame_metrics.keys())))
    ious = [frame_metrics[str(f)]['iou'] for f in frames]
    
    plt.plot(frames, ious, 'b-', label='IoU', alpha=0.7)
    plt.title('Tracking Performance Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('IoU Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Success Rate Analysis
    plt.subplot(2, 2, 3)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    success_rates = [
        sum(1 for iou in ious if iou > t) / len(ious) * 100
        for t in thresholds
    ]
    
    plt.bar(thresholds, success_rates, color='lightgreen', alpha=0.7)
    plt.title('Success Rate at Different IoU Thresholds')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Success Rate (%)')
    
    # Plot 4: Dice vs IoU Comparison
    plt.subplot(2, 2, 4)
    dice_scores = [frame_metrics[str(f)]['dice'] for f in frames]
    plt.scatter(ious, dice_scores, alpha=0.5, c='purple')
    plt.title('Dice vs IoU Correlation')
    plt.xlabel('IoU Score')
    plt.ylabel('Dice Score')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tracking_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics
    summary_stats = {
        'mean_iou': np.mean(ious),
        'median_iou': np.median(ious),
        'std_iou': np.std(ious),
        'success_rate_70': success_rates[2],  # Success rate at 0.7 IoU threshold
        'mean_dice': np.mean(dice_scores),
        'total_frames': len(frames)
    }
    
    return summary_stats

def print_presentation_summary(summary_stats):
    """
    Print a presentation-friendly summary of tracking results.
    """
    print("\n=== Tracking System Performance Summary ===")
    print(f"\nOverall Performance Metrics:")
    print(f"- Mean IoU: {summary_stats['mean_iou']:.3f}")
    print(f"- Median IoU: {summary_stats['median_iou']:.3f}")
    print(f"- IoU Standard Deviation: {summary_stats['std_iou']:.3f}")
    print(f"- Success Rate (IoU > 0.7): {summary_stats['success_rate_70']:.1f}%")
    print(f"- Mean Dice Score: {summary_stats['mean_dice']:.3f}")
    print(f"\nTracking Coverage:")
    print(f"- Total Frames Processed: {summary_stats['total_frames']}")
    
    # Performance categorization
    if summary_stats['mean_iou'] > 0.7:
        performance_category = "Excellent"
    elif summary_stats['mean_iou'] > 0.5:
        performance_category = "Good"
    else:
        performance_category = "Needs Improvement"
        
    print(f"\nOverall Performance Category: {performance_category}")

if __name__ == "__main__":
    # Example usage
    output_dir = "presentation_outputs"
    # results_dict should be your actual results dictionary
    # summary_stats = create_presentation_visuals(results_dict, output_dir)
    # print_presentation_summary(summary_stats) 