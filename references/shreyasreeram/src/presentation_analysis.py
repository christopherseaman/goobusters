import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

def analyze_feedback_loop_results(results_files):
    """
    Analyze multiple feedback loop result files and generate presentation-ready visualizations.
    
    Args:
        results_files: List of paths to feedback loop result JSON files
    """
    all_results = []
    for file_path in results_files:
        with open(file_path, 'r') as f:
            results = json.load(f)
            all_results.append(results)
    
    # Create output directory for visualizations
    output_dir = 'presentation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot settings
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [12, 8]
    
    # 1. Overall Performance Metrics
    plt.figure()
    metrics = []
    labels = []
    for i, result in enumerate(all_results):
        for iteration in result['iterations']:
            eval_results = iteration['evaluation_results']
            if 'summary' in eval_results:
                metrics.append([
                    eval_results['summary']['overall_mean_iou'],
                    eval_results['summary']['overall_mean_dice']
                ])
                labels.append(f"Run {i+1}\nIter {iteration['iteration_number']}")
    
    if metrics:
        metrics = np.array(metrics)
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, metrics[:, 0], width, label='Mean IoU')
        plt.bar(x + width/2, metrics[:, 1], width, label='Mean Dice')
        plt.xlabel('Evaluation Runs')
        plt.ylabel('Score')
        plt.title('Overall Performance Metrics')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'overall_metrics.png'))
    
    # 2. Frame-by-Frame Analysis
    plt.figure()
    for i, result in enumerate(all_results):
        for iteration in result['iterations']:
            for video_key, video_data in iteration['evaluation_results'].items():
                if not video_key.endswith('genuine') and not video_key == 'summary':
                    if 'metrics' in video_data:
                        iou_scores = video_data['metrics']['iou_scores']
                        plt.plot(iou_scores, label=f'Run {i+1} Iter {iteration["iteration_number"]}')
                    elif 'frame_metrics' in video_data:
                        # Alternative data structure
                        frame_metrics = video_data['frame_metrics']
                        frames = sorted([int(k) for k in frame_metrics.keys()])
                        iou_scores = [frame_metrics[str(f)]['iou'] for f in frames]
                        plt.plot(frames, iou_scores, label=f'Run {i+1} Iter {iteration["iteration_number"]}')
    
    plt.xlabel('Frame Number')
    plt.ylabel('IoU Score')
    plt.title('Frame-by-Frame IoU Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'frame_analysis.png'))
    
    # 3. Generate Summary Statistics
    summary_stats = {
        'runs': []
    }
    
    for i, result in enumerate(all_results):
        run_stats = {
            'run_number': i + 1,
            'iterations': []
        }
        
        for iteration in result['iterations']:
            eval_results = iteration['evaluation_results']
            iter_stats = {
                'iteration_number': iteration['iteration_number'],
                'mean_iou': eval_results['summary']['overall_mean_iou'] if 'summary' in eval_results else None,
                'mean_dice': eval_results['summary']['overall_mean_dice'] if 'summary' in eval_results else None,
                'total_videos': eval_results['summary']['total_videos'] if 'summary' in eval_results else None,
                'successful_evaluations': eval_results['summary']['successful_evaluations'] if 'summary' in eval_results else None
            }
            
            # Get video-specific metrics
            for video_key, video_data in eval_results.items():
                if not video_key.endswith('genuine') and not video_key == 'summary':
                    if 'metrics' in video_data:
                        metrics = video_data['metrics']
                        iter_stats.update({
                            'frames_processed': len(metrics['iou_scores']),
                            'max_iou': max(metrics['iou_scores']),
                            'min_iou': min(metrics['iou_scores']),
                            'iou_over_0.7': metrics.get('iou_over_0.7', 0)
                        })
                    elif 'frame_metrics' in video_data:
                        frame_metrics = video_data['frame_metrics']
                        iou_scores = [m['iou'] for m in frame_metrics.values()]
                        iter_stats.update({
                            'frames_processed': len(iou_scores),
                            'max_iou': max(iou_scores),
                            'min_iou': min(iou_scores),
                            'iou_over_0.7': sum(1 for iou in iou_scores if iou > 0.7) / len(iou_scores)
                        })
            
            run_stats['iterations'].append(iter_stats)
        
        summary_stats['runs'].append(run_stats)
    
    # Save summary stats
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    return summary_stats

def print_presentation_summary(summary_stats):
    """
    Print a presentation-friendly summary of the results.
    """
    print("\n=== Multi-Frame Tracking System Evaluation ===\n")
    
    for run in summary_stats['runs']:
        print(f"\nRun {run['run_number']}:")
        print("-" * 40)
        
        for iteration in run['iterations']:
            print(f"\nIteration {iteration['iteration_number']}:")
            if iteration['mean_iou'] is not None:
                print(f"Mean IoU: {iteration['mean_iou']:.4f}")
            if iteration['mean_dice'] is not None:
                print(f"Mean Dice: {iteration['mean_dice']:.4f}")
            if 'frames_processed' in iteration:
                print(f"Frames Processed: {iteration['frames_processed']}")
                print(f"Proportion of High-Quality Frames (IoU > 0.7): {iteration['iou_over_0.7']*100:.1f}%")
                print(f"Best Frame IoU: {iteration['max_iou']:.4f}")
                print(f"Worst Frame IoU: {iteration['min_iou']:.4f}")
            
    print("\n=== Key Findings ===")
    
    # Calculate overall trends
    all_mean_ious = [iter['mean_iou'] for run in summary_stats['runs'] 
                    for iter in run['iterations'] if iter['mean_iou'] is not None]
    all_mean_dices = [iter['mean_dice'] for run in summary_stats['runs'] 
                     for iter in run['iterations'] if iter['mean_dice'] is not None]
    
    if all_mean_ious:
        print(f"\nOverall System Performance:")
        print(f"- Average Mean IoU across all runs: {np.mean(all_mean_ious):.4f}")
        print(f"- Average Mean Dice across all runs: {np.mean(all_mean_dices):.4f}")
        print(f"- Best Mean IoU: {max(all_mean_ious):.4f}")
        print(f"- Best Mean Dice: {max(all_mean_dices):.4f}")

def main():
    # Specify the feedback loop result files
    results_files = [
        'src/output/feedback_loop_exam113_20250524_215335/feedback_loop_results.json',
        'src/output/feedback_loop_exam91_20250524_182912/feedback_loop_results.json',
        'src/output/feedback_loop_20250526_exam126_095252/feedback_loop_results.json',
        'src/output/feedback_loop_20250527_exam132_071204/feedback_loop_results.json'
    ]
    
    # Analyze results and generate visualizations
    summary_stats = analyze_feedback_loop_results(results_files)
    
    # Print presentation summary
    print_presentation_summary(summary_stats)

if __name__ == "__main__":
    main() 