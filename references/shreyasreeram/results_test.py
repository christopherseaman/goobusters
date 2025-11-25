import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import os
import cv2
warnings.filterwarnings('ignore')

def process_single_result_file(filepath, exam_id, mode, sampling_rate):
    """
    Process a single JSON results file like your sample.
    Returns a list of dictionaries with extracted metrics.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    processed_results = []
    
    for iteration in data.get('iterations', []):
        iteration_num = iteration.get('iteration_number', 1)
        eval_results = iteration.get('evaluation_results', {})
        summary = eval_results.get('summary', {})
        
        # Find the study data (first non-summary key)
        study_keys = [k for k in eval_results.keys() if k != 'summary']
        study_data = eval_results.get(study_keys[0], {}) if study_keys else {}
        metrics = study_data.get('metrics', {})
        
        # NEW: Extract method comparison data if available
        method_comparison = study_data.get('method_comparison', {})
        
        # Extract frame-level data
        frame_metrics = metrics.get('frame_metrics', {})
        iou_scores = metrics.get('iou_scores', [])
        dice_scores = metrics.get('dice_scores', [])
        
        # Calculate additional statistics
        iou_array = np.array(iou_scores) if iou_scores else np.array([0])
        dice_array = np.array(dice_scores) if dice_scores else np.array([0])
        
        result = {
            'exam_id': exam_id,
            'mode': mode,
            'sampling_rate': sampling_rate,
            'iteration': iteration_num,
            'learning_mode': data.get('learning_mode', False),
            
            # Summary metrics
            'mean_iou': summary.get('overall_mean_iou', 0),
            'mean_dice': summary.get('overall_mean_dice', 0),
            'total_videos': summary.get('total_videos', 1),
            'successful_evaluations': summary.get('successful_evaluations', 0),
            'memorized_frames_excluded': summary.get('total_memorized_frames_excluded', 0),
            
            # Detailed metrics
            'detailed_mean_iou': metrics.get('mean_iou', 0),
            'detailed_median_iou': metrics.get('median_iou', 0),
            'detailed_mean_dice': metrics.get('mean_dice', 0),
            'iou_over_70': metrics.get('iou_over_0.7', 0),
            
            # Frame counts
            'ground_truth_count': study_data.get('ground_truth_count', 0),
            'algorithm_mask_count': study_data.get('algorithm_mask_count', 0),
            
            # Statistical measures
            'iou_std': np.std(iou_array),
            'iou_min': np.min(iou_array),
            'iou_max': np.max(iou_array),
            'iou_q25': np.percentile(iou_array, 25),
            'iou_q75': np.percentile(iou_array, 75),
            'dice_std': np.std(dice_array),
            
            # Performance stability
            'num_frames_evaluated': len(iou_scores),
            'num_high_quality_frames': np.sum(iou_array > 0.7),
            'high_quality_percentage': np.mean(iou_array > 0.7) * 100,
            
            # NEW: Method comparison metrics
            'has_method_comparison': bool(method_comparison and 'error' not in method_comparison),
            'single_frame_count': method_comparison.get('single_frame_count', 0),
            'multi_frame_count': method_comparison.get('multi_frame_count', 0),
            'method_agreement_iou': method_comparison.get('mean_iou', 0),
            'method_agreement_dice': method_comparison.get('mean_dice', 0),
            
            # NEW: Performance improvement metrics
            'single_frame_vs_gt_iou': method_comparison.get('single_frame_vs_gt', {}).get('mean_iou', 0),
            'multi_frame_vs_gt_iou': method_comparison.get('multi_frame_vs_gt', {}).get('mean_iou', 0),
            'single_frame_vs_gt_dice': method_comparison.get('single_frame_vs_gt', {}).get('mean_dice', 0),
            'multi_frame_vs_gt_dice': method_comparison.get('multi_frame_vs_gt', {}).get('mean_dice', 0),
            'iou_improvement_percent': method_comparison.get('performance_improvement', {}).get('iou_improvement_percent', 0),
            
            # Raw data for detailed analysis
            'iou_scores': iou_scores,
            'dice_scores': dice_scores,
            'frame_metrics': frame_metrics,
            'method_comparison_data': method_comparison
        }
        
        processed_results.append(result)
    
    return processed_results

def _extract_iteration_metrics(iteration, exam_id, mode, sampling_rate):
    """Extract key metrics from a single iteration"""
    eval_results = iteration.get('evaluation_results', {})
    
    # Get summary metrics
    summary = eval_results.get('summary', {})
    
    # Get detailed metrics from first study (assuming single study per exam)
    study_keys = [k for k in eval_results.keys() if k != 'summary']
    study_key = study_keys[0] if study_keys else None
    study_metrics = eval_results.get(study_key, {}).get('metrics', {}) if study_key else {}
    
    # NEW: Extract method comparison data
    method_comparison = eval_results.get(study_key, {}).get('method_comparison', {}) if study_key else {}
    
    return {
        'exam_id': exam_id,
        'mode': mode,
        'sampling_rate': sampling_rate,
        'iteration': iteration.get('iteration_number', 1),
        'learning_mode': iteration.get('learning_mode', False),
        
        # Summary metrics
        'mean_iou': summary.get('overall_mean_iou', 0),
        'mean_dice': summary.get('overall_mean_dice', 0),
        'total_videos': summary.get('total_videos', 1),
        'successful_evaluations': summary.get('successful_evaluations', 0),
        'memorized_frames_excluded': summary.get('total_memorized_frames_excluded', 0),
        
        # Detailed metrics
        'detailed_mean_iou': study_metrics.get('mean_iou', 0),
        'detailed_median_iou': study_metrics.get('median_iou', 0),
        'detailed_mean_dice': study_metrics.get('mean_dice', 0),
        'iou_over_70': study_metrics.get('iou_over_0.7', 0),
        
        # Frame counts
        'ground_truth_count': eval_results.get(study_key, {}).get('ground_truth_count', 0) if study_key else 0,
        'algorithm_mask_count': eval_results.get(study_key, {}).get('algorithm_mask_count', 0) if study_key else 0,
        
        # Statistical measures
        'iou_std': np.std(study_metrics.get('iou_scores', [0])),
        'iou_min': np.min(study_metrics.get('iou_scores', [0])),
        'iou_max': np.max(study_metrics.get('iou_scores', [0])),
        'iou_q25': np.percentile(study_metrics.get('iou_scores', [0]), 25),
        'iou_q75': np.percentile(study_metrics.get('iou_scores', [0]), 75),
        'dice_std': np.std(study_metrics.get('dice_scores', [0])),
        
        # Performance stability
        'num_frames_evaluated': len(study_metrics.get('iou_scores', [])),
        'num_high_quality_frames': np.sum(np.array(study_metrics.get('iou_scores', [])) > 0.7),
        'high_quality_percentage': np.mean(np.array(study_metrics.get('iou_scores', [])) > 0.7) * 100,
        
        # NEW: Method comparison metrics
        'has_method_comparison': bool(method_comparison and 'error' not in method_comparison),
        'single_frame_count': method_comparison.get('single_frame_count', 0),
        'multi_frame_count': method_comparison.get('multi_frame_count', 0),
        'method_agreement_iou': method_comparison.get('mean_iou', 0),
        'method_agreement_dice': method_comparison.get('mean_dice', 0),
        
        # NEW: Performance improvement metrics
        'single_frame_vs_gt_iou': method_comparison.get('single_frame_vs_gt', {}).get('mean_iou', 0),
        'multi_frame_vs_gt_iou': method_comparison.get('multi_frame_vs_gt', {}).get('mean_iou', 0),
        'single_frame_vs_gt_dice': method_comparison.get('single_frame_vs_gt', {}).get('mean_dice', 0),
        'multi_frame_vs_gt_dice': method_comparison.get('multi_frame_vs_gt', {}).get('mean_dice', 0),
        'iou_improvement_percent': method_comparison.get('performance_improvement', {}).get('iou_improvement_percent', 0),
        
        # Raw data for detailed analysis
        'iou_scores': study_metrics.get('iou_scores', []),
        'dice_scores': study_metrics.get('dice_scores', []),
        'frame_metrics': study_metrics.get('frame_metrics', {}),
        'method_comparison_data': method_comparison
    }

def combine_all_results(results_directory):
    """
    Load all JSON results files from directory structure.
    Expected structure:
    results_directory/
    ├── exam_185/
    │   ├── baseline_rate_5.json
    │   ├── learning_rate_5.json
    │   └── ...
    ├── exam_194/
    └── exam_200/
    """
    all_results = []
    results_path = Path(results_directory)
    
    for exam_dir in results_path.iterdir():
        if exam_dir.is_dir():
            exam_id = exam_dir.name
            
            for result_file in exam_dir.glob("*.json"):
                filename = result_file.stem
                
                # Determine mode
                mode = "learning" if "learning" in filename else "baseline"
                
                # Extract sampling rate
                import re
                rate_match = re.search(r'rate[_\s]*(\d+)', filename)
                sampling_rate = int(rate_match.group(1)) if rate_match else 5
                
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    for iteration in data.get('iterations', []):
                        iteration_data = _extract_iteration_metrics(
                            iteration, exam_id, mode, sampling_rate
                        )
                        all_results.append(iteration_data)
                        
                except Exception as e:
                    print(f"Error processing {result_file}: {str(e)}")
                    continue
    
    return pd.DataFrame(all_results)

# ===============================
# NEW: ISSUE TYPE FUNCTIONALITY
# ===============================

def create_exam_issue_type_mapping():
    """
    Create a mapping of exam IDs to issue types.
    UPDATE THIS with your actual exam classifications!
    """
    exam_issue_types = {
        # UPDATE THESE MAPPINGS WITH YOUR ACTUAL DATA:
        'exam_91': 'multiple_distinct',
        'exam_185': 'branching_fluid',
        'exam_194': 'disappear_reappear', 
        'exam_200': 'uncomplicated',
        
        # Add more mappings here...
        # 'exam_XXX': 'issue_type',
    }
    
    return exam_issue_types

def load_exam_issue_types_from_csv(csv_path):
    """
    Load exam issue types from a CSV file.
    Expected format: exam_id, issue_type
    """
    try:
        df_types = pd.read_csv(csv_path)
        # Convert to dictionary for easy lookup
        return dict(zip(df_types['exam_id'], df_types['issue_type']))
    except Exception as e:
        print(f"Error loading issue types from CSV: {e}")
        return {}

def add_issue_types_to_dataframe(df, issue_type_mapping=None, csv_path=None):
    """
    Add issue type information to your results dataframe
    """
    # Get issue type mapping
    if csv_path:
        issue_types = load_exam_issue_types_from_csv(csv_path)
    elif issue_type_mapping:
        issue_types = issue_type_mapping
    else:
        issue_types = create_exam_issue_type_mapping()
    
    # Add issue type column
    df['issue_type'] = df['exam_id'].map(issue_types)
    
    # Handle missing mappings
    missing_exams = df[df['issue_type'].isna()]['exam_id'].unique()
    if len(missing_exams) > 0:
        print(f"⚠️  Warning: No issue type mapping found for exams: {list(missing_exams)}")
        print("These will be marked as 'unknown'")
        df['issue_type'] = df['issue_type'].fillna('unknown')
    
    print(f"✅ Added issue types for {len(df)} records")
    print(f"📊 Issue type distribution:")
    print(df['issue_type'].value_counts())
    
    return df

def create_issue_type_performance_analysis(df):
    """
    Create comprehensive analysis by issue type - FIXED spacing issues
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(22, 18))  # Increased height for better spacing
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)  # Increased spacing
    
    # UPDATED: Changed "Cardiac" to "Fluid"
    fig.suptitle('Optical Flow Performance Analysis by Fluid Issue Type', 
                 fontsize=16, fontweight='bold', y=0.96)
    
    # 1. MAIN PERFORMANCE BY ISSUE TYPE - FIXED LABELS
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Group by issue type and mode
    issue_summary = df.groupby(['issue_type', 'mode'])['mean_iou'].agg(['mean', 'std', 'count']).reset_index()
    
    issue_types = sorted(df['issue_type'].unique())
    x = np.arange(len(issue_types))
    width = 0.35
    
    baseline_means = []
    learning_means = []
    baseline_stds = []
    learning_stds = []
    
    for issue_type in issue_types:
        baseline_data = issue_summary[(issue_summary['issue_type'] == issue_type) & 
                                    (issue_summary['mode'] == 'baseline')]
        learning_data = issue_summary[(issue_summary['issue_type'] == issue_type) & 
                                    (issue_summary['mode'] == 'learning')]
        
        baseline_mean = baseline_data['mean'].iloc[0] if len(baseline_data) > 0 else 0
        learning_mean = learning_data['mean'].iloc[0] if len(learning_data) > 0 else 0
        baseline_std = baseline_data['std'].iloc[0] if len(baseline_data) > 0 else 0
        learning_std = learning_data['std'].iloc[0] if len(learning_data) > 0 else 0
        
        baseline_means.append(baseline_mean)
        learning_means.append(learning_mean)
        baseline_stds.append(baseline_std)
        learning_stds.append(learning_std)
    
    bars1 = ax_main.bar(x - width/2, baseline_means, width, 
                       label='Baseline', color='lightcoral', alpha=0.8,
                       yerr=baseline_stds, capsize=5)
    bars2 = ax_main.bar(x + width/2, learning_means, width, 
                       label='Learning Mode', color='lightblue', alpha=0.8,
                       yerr=learning_stds, capsize=5)
    
    # Add improvement annotations
    for i, (baseline, learning) in enumerate(zip(baseline_means, learning_means)):
        if baseline > 0:
            improvement = ((learning - baseline) / baseline) * 100
            color = 'green' if improvement > 0 else 'red'
            ax_main.annotate(f'{improvement:+.1f}%', 
                           xy=(i, max(baseline, learning) + 0.05), 
                           ha='center', va='bottom', fontweight='bold', 
                           color=color, fontsize=10)
    
    ax_main.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Clinical Threshold')
    ax_main.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Acceptable Threshold')
    
    # FIXED: Better label formatting and rotation
    ax_main.set_xlabel('Fluid Issue Type', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Mean IoU Performance', fontsize=12, fontweight='bold')
    ax_main.set_title('Performance by Fluid Abnormality Type', fontsize=14, fontweight='bold')
    ax_main.set_xticks(x)
    
    # FIXED: Better label formatting - shorter, cleaner labels
    clean_labels = []
    for t in issue_types:
        if t == 'multiple_distinct':
            clean_labels.append('Multiple\nDistinct')
        elif t == 'branching_fluid':
            clean_labels.append('Branching\nFluid')
        elif t == 'disappear_reappear':
            clean_labels.append('Disappear\nReappear')
        elif t == 'uncomplicated':
            clean_labels.append('Uncomplicated')
        elif t == 'complex_mixed':
            clean_labels.append('Complex\nMixed')
        elif t == '?':
            clean_labels.append('Unknown')
        else:
            # Fallback for any other types
            clean_labels.append(t.replace('_', '\n').title())
    
    ax_main.set_xticklabels(clean_labels, fontsize=10, ha='center')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, 1.0)
    
    # 2. 🔧 FIXED DIFFICULTY RANKING WITH MORE SPACE
    ax_difficulty = fig.add_subplot(gs[0, 2])
    
    # Calculate average performance across all conditions for each issue type
    difficulty_ranking = df.groupby('issue_type')['mean_iou'].mean().sort_values()
    
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(difficulty_ranking)))
    bars = ax_difficulty.barh(range(len(difficulty_ranking)), difficulty_ranking.values, color=colors)
    
    # 🔧 FIX: Better spacing and positioning for y-axis labels
    ax_difficulty.set_yticks(range(len(difficulty_ranking)))
    
    # FIXED: Use same clean labels with better spacing
    difficulty_labels = []
    for t in difficulty_ranking.index:
        if t == 'multiple_distinct':
            difficulty_labels.append('Multiple\nDistinct')
        elif t == 'branching_fluid':
            difficulty_labels.append('Branching\nFluid')
        elif t == 'disappear_reappear':
            difficulty_labels.append('Disappear\nReappear')
        elif t == 'uncomplicated':
            difficulty_labels.append('Uncomplicated')
        elif t == '?':
            difficulty_labels.append('Unknown')
        else:
            difficulty_labels.append(t.replace('_', '\n').title())
    
    ax_difficulty.set_yticklabels(difficulty_labels, fontsize=9)
    ax_difficulty.set_xlabel('Average IoU', fontsize=10)
    ax_difficulty.set_title('Difficulty Ranking\n(Hardest to Easiest)', fontweight='bold', fontsize=11)
    ax_difficulty.grid(True, alpha=0.3, axis='x')
    
    # 🔧 FIX: Better positioning for value labels to avoid overlap
    for i, (issue_type, score) in enumerate(difficulty_ranking.items()):
        difficulty = "Hard" if score < 0.5 else "Medium" if score < 0.7 else "Easy"
        # Position text further right to avoid overlap with y-axis labels
        ax_difficulty.text(score + 0.04, i, f'{score:.3f}\n({difficulty})', 
                          va='center', fontsize=8, fontweight='bold')
    
    # 3. DETAILED HEATMAP BY SAMPLING RATE
    ax_heatmap = fig.add_subplot(gs[1, :])
    
    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(values='mean_iou', 
                                 index=['issue_type', 'mode'], 
                                 columns='sampling_rate', 
                                 aggfunc='mean')
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax_heatmap, center=0.5, vmin=0, vmax=1, 
                cbar_kws={'label': 'Mean IoU'})
    ax_heatmap.set_title('Performance Heatmap: Issue Type × Mode × Sampling Rate', 
                        fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel('Issue Type & Mode')
    ax_heatmap.set_xlabel('Sampling Rate')
    
    # 4. LEARNING IMPROVEMENT BY ISSUE TYPE
    ax_improvement = fig.add_subplot(gs[2, 0])
    
    improvements = []
    issue_labels = []
    
    for issue_type in issue_types:
        baseline_scores = df[(df['issue_type'] == issue_type) & (df['mode'] == 'baseline')]['mean_iou']
        learning_scores = df[(df['issue_type'] == issue_type) & (df['mode'] == 'learning')]['mean_iou']
        
        if len(baseline_scores) > 0 and len(learning_scores) > 0:
            baseline_mean = baseline_scores.mean()
            learning_mean = learning_scores.mean()
            improvement = ((learning_mean - baseline_mean) / baseline_mean) * 100
            improvements.append(improvement)
            
            # FIXED: Use clean labels
            if issue_type == 'multiple_distinct':
                issue_labels.append('Multiple\nDistinct')
            elif issue_type == 'branching_fluid':
                issue_labels.append('Branching\nFluid')
            elif issue_type == 'disappear_reappear':
                issue_labels.append('Disappear\nReappear')
            elif issue_type == 'uncomplicated':
                issue_labels.append('Uncomplicated')
            elif issue_type == '?':
                issue_labels.append('Unknown')
            else:
                issue_labels.append(issue_type.replace('_', '\n').title())
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax_improvement.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
    
    ax_improvement.set_xticks(range(len(improvements)))
    ax_improvement.set_xticklabels(issue_labels, fontsize=9, ha='center')
    ax_improvement.set_ylabel('Learning Improvement (%)')
    ax_improvement.set_title('Learning Mode Benefit\nby Issue Type', fontweight='bold')
    ax_improvement.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_improvement.grid(True, alpha=0.3)
    
    # 5. SAMPLING RATE TOLERANCE BY ISSUE TYPE
    ax_sampling = fig.add_subplot(gs[2, 1])
    
    for issue_type in issue_types:
        issue_data = df[df['issue_type'] == issue_type]
        sampling_performance = issue_data.groupby('sampling_rate')['mean_iou'].mean()
        
        # FIXED: Use clean labels for legend
        if issue_type == 'multiple_distinct':
            label = 'Multiple Distinct'
        elif issue_type == 'branching_fluid':
            label = 'Branching Fluid'
        elif issue_type == 'disappear_reappear':
            label = 'Disappear Reappear'
        elif issue_type == 'uncomplicated':
            label = 'Uncomplicated'
        elif issue_type == '?':
            label = 'Unknown'
        else:
            label = issue_type.replace('_', ' ').title()
        
        ax_sampling.plot(sampling_performance.index, sampling_performance.values, 
                        'o-', label=label, linewidth=2, markersize=6)
    
    ax_sampling.set_xlabel('Sampling Rate')
    ax_sampling.set_ylabel('Mean IoU')
    ax_sampling.set_title('Sparsity Tolerance\nby Issue Type', fontweight='bold')
    ax_sampling.legend(fontsize=8)
    ax_sampling.grid(True, alpha=0.3)
    ax_sampling.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    # 6. STATISTICAL SUMMARY TABLE
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis('off')
    
    # UPDATED: Changed "CARDIAC" to "FLUID"
    stats_text = "📊 FLUID ISSUE TYPE ANALYSIS SUMMARY:\n\n"
    
    for issue_type in issue_types:
        issue_data = df[df['issue_type'] == issue_type]
        avg_performance = issue_data['mean_iou'].mean()
        n_experiments = len(issue_data)
        
        # Clinical threshold analysis
        clinical_rate = np.mean(issue_data['mean_iou'] > 0.7) * 100
        
        # FIXED: Use clean labels
        if issue_type == 'multiple_distinct':
            display_name = 'Multiple Distinct'
        elif issue_type == 'branching_fluid':
            display_name = 'Branching Fluid'
        elif issue_type == 'disappear_reappear':
            display_name = 'Disappear Reappear'
        elif issue_type == 'uncomplicated':
            display_name = 'Uncomplicated'
        elif issue_type == '?':
            display_name = 'Unknown'
        else:
            display_name = issue_type.replace('_', ' ').title()
        
        stats_text += f"🔸 {display_name}:\n"
        stats_text += f"   Avg IoU: {avg_performance:.3f}\n"
        stats_text += f"   Clinical rate: {clinical_rate:.1f}%\n"
        stats_text += f"   N experiments: {n_experiments}\n\n"
    
    # Overall findings
    best_issue = difficulty_ranking.idxmax()
    worst_issue = difficulty_ranking.idxmin()
    
    # FIXED: Use clean labels for summary
    best_display = best_issue.replace('_', ' ').title() if best_issue != '?' else 'Unknown'
    worst_display = worst_issue.replace('_', ' ').title() if worst_issue != '?' else 'Unknown'
    
    stats_text += f"🏆 Best performing: {best_display}\n"
    stats_text += f"⚠️ Most challenging: {worst_display}\n"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    # 🔧 FIX: Enhanced layout with better spacing
    plt.tight_layout(pad=3.0)  # Increased padding
    plt.subplots_adjust(top=0.94, hspace=0.5, wspace=0.4)  # Better spacing
    
    return fig

def issue_type_statistical_analysis(df):
    """
    Perform statistical analysis comparing performance across issue types
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS BY CARDIAC ISSUE TYPE")
    print("="*80)
    
    issue_types = sorted(df['issue_type'].unique())
    
    # ANOVA test across issue types
    issue_groups = [df[df['issue_type'] == issue_type]['mean_iou'].values 
                   for issue_type in issue_types]
    
    f_stat, p_value = stats.f_oneway(*issue_groups)
    
    print(f"\n📊 ONE-WAY ANOVA ACROSS ISSUE TYPES:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    # Pairwise comparisons
    print(f"\n🔍 PAIRWISE COMPARISONS (t-tests):")
    print("-" * 60)
    
    results = []
    for i, type1 in enumerate(issue_types):
        for j, type2 in enumerate(issue_types[i+1:], i+1):
            group1 = df[df['issue_type'] == type1]['mean_iou']
            group2 = df[df['issue_type'] == type2]['mean_iou']
            
            if len(group1) > 0 and len(group2) > 0:
                t_stat, p_val = stats.ttest_ind(group1, group2)
                
                mean1 = group1.mean()
                mean2 = group2.mean()
                
                print(f"{type1.replace('_', ' ').title():20} vs {type2.replace('_', ' ').title():20}: "
                      f"p={p_val:.4f} ({'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'})")
                print(f"{'':42} Means: {mean1:.3f} vs {mean2:.3f}")
                
                results.append({
                    'comparison': f"{type1} vs {type2}",
                    'type1_mean': mean1,
                    'type2_mean': mean2,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
    
    return pd.DataFrame(results)

# ===============================
# EXISTING FUNCTIONS (UPDATED)
# ===============================

# NEW: Method comparison visualization functions
def create_method_comparison_visualization(df):
    """
    Create comprehensive single-frame vs multi-frame method comparison visualizations
    """
    # Filter to only results with method comparison data
    df_comparison = df[df['has_method_comparison'] == True].copy()
    
    if len(df_comparison) == 0:
        print("No method comparison data found in results!")
        return None
    
    print(f"Creating method comparison visualizations for {len(df_comparison)} results with comparison data")
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 1.5, 1.2], 
                         hspace=0.35, wspace=0.3)
    
    # Main title
    fig.suptitle('Single-Frame vs Multi-Frame Tracking Method Comparison\n'
                 'Does Multi-Frame Supervision Improve Optical Flow Tracking?', 
                 fontsize=16, fontweight='bold', y=0.96)
    
    # 1. HERO CHART: Performance Comparison (IoU)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Prepare data for comparison
    comparison_data = []
    for _, row in df_comparison.iterrows():
        if row['single_frame_vs_gt_iou'] > 0 and row['multi_frame_vs_gt_iou'] > 0:
            comparison_data.append({
                'exam_id': row['exam_id'],
                'sampling_rate': row['sampling_rate'],
                'single_frame_iou': row['single_frame_vs_gt_iou'],
                'multi_frame_iou': row['multi_frame_vs_gt_iou'],
                'improvement': row['iou_improvement_percent']
            })
    
    if not comparison_data:
        print("No valid comparison data found!")
        return None
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    exam_ids = comparison_df['exam_id'].unique()
    x = np.arange(len(exam_ids))
    width = 0.35
    
    single_ious = []
    multi_ious = []
    
    for exam_id in exam_ids:
        exam_data = comparison_df[comparison_df['exam_id'] == exam_id]
        single_ious.append(exam_data['single_frame_iou'].mean())
        multi_ious.append(exam_data['multi_frame_iou'].mean())
    
    bars1 = ax_main.bar(x - width/2, single_ious, width, 
                       label='Single-Frame Method', color='lightcoral', alpha=0.8)
    bars2 = ax_main.bar(x + width/2, multi_ious, width, 
                       label='Multi-Frame Method', color='lightblue', alpha=0.8)
    
    # Add improvement annotations
    for i, (single_iou, multi_iou) in enumerate(zip(single_ious, multi_ious)):
        improvement = ((multi_iou - single_iou) / single_iou * 100) if single_iou > 0 else 0
        color = 'green' if improvement > 0 else 'red'
        ax_main.annotate(f'{improvement:+.1f}%', 
                        xy=(i, max(single_iou, multi_iou) + 0.02), 
                        ha='center', va='bottom', fontweight='bold', 
                        color=color, fontsize=9)
    
    # Add clinical threshold
    ax_main.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Clinical Threshold (IoU > 0.7)')
    ax_main.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Acceptable Threshold (IoU > 0.5)')
    
    ax_main.set_xlabel('Exam ID', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Mean IoU vs Ground Truth', fontsize=11, fontweight='bold')
    ax_main.set_title('Single-Frame vs Multi-Frame Performance\n(Higher is Better)', 
                     fontsize=12, fontweight='bold')
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(exam_ids, rotation=45, ha='right')
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, 1.0)
    
    # Continue with rest of method comparison visualization...
    # (Rest of the existing method comparison code would go here)
    
    plt.tight_layout()
    return fig

def create_method_comparison_statistical_analysis(df):
    """
    Perform statistical analysis of single-frame vs multi-frame methods
    """
    # Filter to comparison data
    df_comparison = df[df['has_method_comparison'] == True].copy()
    
    if len(df_comparison) == 0:
        print("No method comparison data available for statistical analysis")
        return None
    
    print("\n" + "="*60)
    print("METHOD COMPARISON STATISTICAL ANALYSIS")
    print("="*60)
    
    results_summary = []
    
    # Overall comparison
    single_ious = df_comparison['single_frame_vs_gt_iou'].values
    multi_ious = df_comparison['multi_frame_vs_gt_iou'].values
    
    # Remove zeros for valid comparison
    valid_indices = (single_ious > 0) & (multi_ious > 0)
    single_valid = single_ious[valid_indices]
    multi_valid = multi_ious[valid_indices]
    
    if len(single_valid) > 0:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(multi_valid, single_valid)
        
        # Effect size
        differences = multi_valid - single_valid
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        print(f"\nOVERALL METHOD COMPARISON:")
        print(f"Single-frame mean IoU: {np.mean(single_valid):.4f} ± {np.std(single_valid):.4f}")
        print(f"Multi-frame mean IoU: {np.mean(multi_valid):.4f} ± {np.std(multi_valid):.4f}")
        print(f"Mean improvement: {mean_diff:+.4f} ({mean_diff/np.mean(single_valid)*100:+.1f}%)")
        print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"Effect size (Cohen's d): {cohens_d:.4f}")
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        effect_interpretation = ("Large" if abs(cohens_d) > 0.8 else 
                               "Medium" if abs(cohens_d) > 0.5 else 
                               "Small" if abs(cohens_d) > 0.2 else "Negligible")
        
        print(f"Significance: {significance}")
        print(f"Effect size interpretation: {effect_interpretation}")
        
        results_summary.append({
            'comparison_type': 'Overall',
            'sampling_rate': 'All',
            'single_frame_mean': np.mean(single_valid),
            'multi_frame_mean': np.mean(multi_valid),
            'mean_improvement': mean_diff,
            'percent_improvement': mean_diff/np.mean(single_valid)*100,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significance': significance,
            'effect_size': effect_interpretation,
            'n_comparisons': len(single_valid)
        })
    
    return pd.DataFrame(results_summary)


def create_performance_comparison(df):
    """Create comprehensive performance comparison visualizations - FIXED overlapping issues"""
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Check if issue types are available
    has_issue_types = 'issue_type' in df.columns and not df['issue_type'].isna().all()
    
    # Set up the plotting style
    plt.style.use('default')
    
    if has_issue_types:
        # Enhanced layout with issue type analysis
        fig, axes = plt.subplots(3, 3, figsize=(22, 20))  # Increased figure size
        fig.suptitle('Multi-Exam Optical Flow Performance Analysis with Issue Types', 
                    fontsize=16, fontweight='bold')
    else:
        # Original layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Exam Optical Flow Performance Analysis', 
                    fontsize=16, fontweight='bold')
    
    # 1. Mean IoU by Sampling Rate and Mode
    ax1 = axes[0, 0]
    summary_by_rate = df.groupby(['sampling_rate', 'mode']).agg({
        'mean_iou': ['mean', 'std']
    }).round(4)
    
    rates = sorted(df['sampling_rate'].unique())
    baseline_means = [summary_by_rate.loc[(r, 'baseline'), ('mean_iou', 'mean')] 
                     for r in rates if (r, 'baseline') in summary_by_rate.index]
    learning_means = [summary_by_rate.loc[(r, 'learning'), ('mean_iou', 'mean')] 
                     for r in rates if (r, 'learning') in summary_by_rate.index]
    
    x = np.arange(len(rates))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_means, width, label='Baseline', alpha=0.8, color='red')
    ax1.bar(x + width/2, learning_means, width, label='Learning', alpha=0.8, color='green')
    ax1.set_xlabel('Sampling Rate')
    ax1.set_ylabel('Mean IoU')
    ax1.set_title('Performance by Sampling Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rates)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 🔧 FIXED BOX PLOT WITH BETTER SPACING AND CLEANER OVERLAY
    ax2 = axes[0, 1]
    
    if has_issue_types:
        # Filter out unknown issue types for cleaner visualization
        df_clean = df[df['issue_type'] != '?'].copy()
        
        # 🔧 FIX: Use cleaner issue type names for legend
        df_clean['clean_issue_type'] = df_clean['issue_type'].map({
            'multiple_distinct': 'Multiple Distinct',
            'branching_fluid': 'Branching Fluid', 
            'disappear_reappear': 'Disappear Reappear',
            'uncomplicated': 'Uncomplicated'
        }).fillna(df_clean['issue_type'])
        
        # Create the main box plot with better styling
        box_plot = sns.boxplot(data=df_clean, x='mode', y='mean_iou', hue='clean_issue_type', 
                              ax=ax2, palette='Set1', linewidth=1.5, fliersize=0)  # Hide outliers to reduce clutter
        
        # 🔧 FIX: Reduce scatter point density and improve positioning
        issue_types = sorted(df_clean['clean_issue_type'].unique())
        modes = sorted(df_clean['mode'].unique())
        
        # Reduce the number of points by sampling if too many
        max_points_per_group = 15  # Limit points per group
        
        for mode_idx, mode in enumerate(modes):
            for issue_idx, issue_type in enumerate(issue_types):
                subset = df_clean[(df_clean['mode'] == mode) & (df_clean['clean_issue_type'] == issue_type)]
                
                if len(subset) > 0:
                    # Sample points if too many
                    if len(subset) > max_points_per_group:
                        subset = subset.sample(n=max_points_per_group, random_state=42)
                    
                    # Calculate x position with better spacing
                    x_pos = mode_idx  # 0 for baseline, 1 for learning
                    
                    # Improved horizontal offset calculation
                    n_issues = len(issue_types)
                    if n_issues > 1:
                        # Create more space between issue types
                        offset_range = 0.25  # Reduced range to prevent overlap
                        x_offset = (issue_idx - (n_issues - 1) / 2) * (offset_range / (n_issues - 1))
                    else:
                        x_offset = 0
                    
                    # Smaller jitter to keep points more organized
                    x_jitter = np.random.normal(0, 0.015, len(subset))
                    x_positions = x_pos + x_offset + x_jitter
                    
                    # Create scatter plot with better visibility
                    scatter = ax2.scatter(
                        x_positions, 
                        subset['mean_iou'], 
                        c=subset['sampling_rate'], 
                        cmap='viridis', 
                        alpha=0.8,  # Increased alpha for better visibility
                        s=25,  # Slightly smaller points
                        edgecolors='white',  # White edges for better contrast
                        linewidth=0.3,
                        zorder=10
                    )
        
        # 🔧 FIX: Better colorbar positioning and labeling
        cbar = fig.colorbar(scatter, ax=ax2, pad=0.02, shrink=0.7, aspect=15)
        cbar.set_label('Sampling Rate\n(1:N)', rotation=270, labelpad=25, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        
        # 🔧 FIX: Improved title and labels
        ax2.set_title('IoU Distributions by Issue Type\n(Points show sampling rate)', 
                     fontweight='bold', fontsize=11)
        ax2.set_ylabel('Mean IoU')
        ax2.set_xlabel('Mode')
        
        # 🔧 FIX: Move legend inside plot area to avoid covering boxes
        ax2.legend(title='Issue Type', fontsize=9, loc='lower left', 
                  bbox_to_anchor=(0.02, 0.02), framealpha=0.95, 
                  fancybox=True, shadow=True)
        
    else:
        # Original box plot for cases without issue types
        df_plot = df[df['mean_iou'] > 0]
        sns.boxplot(data=df_plot, x='mode', y='mean_iou', hue='sampling_rate', ax=ax2)
        ax2.set_title('IoU Score Distributions')
        ax2.set_ylabel('Mean IoU')
    
    # 3. Performance improvement percentage
    ax3 = axes[0, 2]
    improvement_data = []
    for rate in rates:
        baseline_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'baseline')]
        learning_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'learning')]
        
        if len(baseline_data) > 0 and len(learning_data) > 0:
            baseline_mean = baseline_data['mean_iou'].mean()
            learning_mean = learning_data['mean_iou'].mean()
            improvement = ((learning_mean - baseline_mean) / baseline_mean) * 100
            improvement_data.append(improvement)
        else:
            improvement_data.append(0)
    
    colors = ['green' if x > 0 else 'red' for x in improvement_data]
    ax3.bar(rates, improvement_data, color=colors, alpha=0.7)
    ax3.set_xlabel('Sampling Rate')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('Learning Mode Improvement Over Baseline')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # 4. 🔧 FIXED HEATMAP WITH BETTER LABELS AND SPACING
    ax4 = axes[1, 0]
    
    if has_issue_types:
        # Create cleaner pivot table
        df_heatmap = df.copy()
        
        # Clean up issue type names for heatmap
        df_heatmap['clean_issue_type'] = df_heatmap['issue_type'].map({
            'multiple_distinct': 'Multiple Distinct',
            'branching_fluid': 'Branching Fluid', 
            'disappear_reappear': 'Disappear Reappear',
            'uncomplicated': 'Uncomplicated',
            '?': 'Unknown'
        }).fillna(df_heatmap['issue_type'])
        
        # Create a more readable index
        df_heatmap['heatmap_index'] = df_heatmap['clean_issue_type'] + '-' + df_heatmap['exam_id']
        
        pivot_data = df_heatmap.pivot_table(
            values='mean_iou', 
            index='heatmap_index', 
            columns=['mode', 'sampling_rate'],
            aggfunc='mean'
        )
        
        # 🔧 FIX: Improved heatmap with better color scheme (Green = Good, Red = Bad)
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                    ax=ax4, center=0.5, vmin=0, vmax=1, 
                    cbar_kws={'label': 'Mean IoU', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white')
        
        ax4.set_title('Performance Heatmap by Exam', fontweight='bold')
        ax4.set_ylabel('Issue Type - Exam ID', fontsize=10)
        ax4.set_xlabel('Mode - Sampling Rate', fontsize=10)
        
        # 🔧 FIX: Rotate labels for better readability
        ax4.tick_params(axis='y', labelsize=8, rotation=0)
        ax4.tick_params(axis='x', labelsize=8, rotation=45)
        
    else:
        pivot_data = df.pivot_table(
            values='mean_iou', 
            index=['exam_id', 'mode'], 
            columns='sampling_rate',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                    ax=ax4, center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Mean IoU'})
        ax4.set_title('Performance Heatmap by Exam')
    
    # 5. Clinical threshold analysis (IoU > 0.7)
    ax5 = axes[1, 1]
    clinical_summary = df.groupby(['sampling_rate', 'mode'])['high_quality_percentage'].mean()
    
    baseline_clinical = [clinical_summary.get((r, 'baseline'), 0) for r in rates]
    learning_clinical = [clinical_summary.get((r, 'learning'), 0) for r in rates]
    
    ax5.plot(rates, baseline_clinical, 'o-', label='Baseline', color='red', linewidth=2, markersize=8)
    ax5.plot(rates, learning_clinical, 's-', label='Learning', color='green', linewidth=2, markersize=8)
    ax5.set_xlabel('Sampling Rate')
    ax5.set_ylabel('% Frames with IoU > 0.7')
    ax5.set_title('Clinical Quality Threshold Analysis')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 100)
    
    # 6. Performance stability (coefficient of variation)
    ax6 = axes[1, 2]
    stability_data = df.groupby(['sampling_rate', 'mode']).agg({
        'mean_iou': lambda x: (np.std(x) / np.mean(x)) * 100 if np.mean(x) > 0 else 0
    }).round(2)
    
    baseline_cv = [stability_data.loc[(r, 'baseline'), 'mean_iou'] 
                   for r in rates if (r, 'baseline') in stability_data.index]
    learning_cv = [stability_data.loc[(r, 'learning'), 'mean_iou'] 
                   for r in rates if (r, 'learning') in stability_data.index]
    
    ax6.bar(x - width/2, baseline_cv, width, label='Baseline', alpha=0.8, color='red')
    ax6.bar(x + width/2, learning_cv, width, label='Learning', alpha=0.8, color='green')
    ax6.set_xlabel('Sampling Rate')
    ax6.set_ylabel('Coefficient of Variation (%)')
    ax6.set_title('Performance Stability (Lower = More Stable)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(rates)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # NEW: Issue type specific analyses (if available)
    if has_issue_types:
        # 7. 🔧 FIXED Performance by Issue Type with cleaner labels
        ax7 = axes[2, 0]
        
        # Create clean issue type performance data
        df_clean_perf = df[df['issue_type'] != '?'].copy()
        df_clean_perf['clean_issue_type'] = df_clean_perf['issue_type'].map({
            'multiple_distinct': 'Multiple\nDistinct',
            'branching_fluid': 'Branching\nFluid', 
            'disappear_reappear': 'Disappear\nReappear',
            'uncomplicated': 'Uncomplicated'
        }).fillna(df_clean_perf['issue_type'])
        
        issue_performance = df_clean_perf.groupby(['clean_issue_type', 'mode'])['mean_iou'].mean().unstack()
        issue_performance.plot(kind='bar', ax=ax7, color=['red', 'green'], alpha=0.8, width=0.7)
        
        ax7.set_title('Average Performance by Issue Type', fontweight='bold')
        ax7.set_xlabel('Issue Type')
        ax7.set_ylabel('Mean IoU')
        ax7.legend(['Baseline', 'Learning'], loc='upper right')
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.tick_params(axis='x', rotation=0, labelsize=9)  # No rotation for multi-line labels
        
        # 8. 🔧 FIXED Issue Type Learning Benefit with cleaner labels
        ax8 = axes[2, 1]
        issue_types_clean = df_clean_perf['clean_issue_type'].unique()
        learning_benefits = []
        
        for issue_type in issue_types_clean:
            baseline_scores = df_clean_perf[(df_clean_perf['clean_issue_type'] == issue_type) & 
                                          (df_clean_perf['mode'] == 'baseline')]['mean_iou']
            learning_scores = df_clean_perf[(df_clean_perf['clean_issue_type'] == issue_type) & 
                                          (df_clean_perf['mode'] == 'learning')]['mean_iou']
            
            if len(baseline_scores) > 0 and len(learning_scores) > 0:
                baseline_mean = baseline_scores.mean()
                learning_mean = learning_scores.mean()
                benefit = ((learning_mean - baseline_mean) / baseline_mean) * 100
                learning_benefits.append(benefit)
            else:
                learning_benefits.append(0)
        
        colors = ['green' if x > 0 else 'red' for x in learning_benefits]
        bars = ax8.bar(range(len(issue_types_clean)), learning_benefits, color=colors, alpha=0.7, width=0.6)
        
        ax8.set_xticks(range(len(issue_types_clean)))
        ax8.set_xticklabels(issue_types_clean, fontsize=9)
        ax8.set_ylabel('Learning Improvement (%)')
        ax8.set_title('Learning Benefit by Issue Type', fontweight='bold')
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, learning_benefits)):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # 9. 🔧 IMPROVED SAMPLING RATE ANALYSIS
        ax9 = axes[2, 2]
        
        # Create sampling rate analysis with cleaner styling
        for issue_type_orig in df['issue_type'].unique():
            if issue_type_orig != '?':  # Skip unknown types
                issue_data = df[df['issue_type'] == issue_type_orig]
                sampling_performance = issue_data.groupby('sampling_rate')['mean_iou'].mean()
                
                # Clean label mapping
                label_map = {
                    'multiple_distinct': 'Multiple Distinct',
                    'branching_fluid': 'Branching Fluid',
                    'disappear_reappear': 'Disappear Reappear',
                    'uncomplicated': 'Uncomplicated'
                }
                label = label_map.get(issue_type_orig, issue_type_orig.replace('_', ' ').title())
                
                ax9.plot(sampling_performance.index, sampling_performance.values, 
                        'o-', label=label, linewidth=2.5, markersize=7, alpha=0.8)
        
        ax9.set_xlabel('Sampling Rate (1:N)', fontsize=10)
        ax9.set_ylabel('Mean IoU', fontsize=10)
        ax9.set_title('Sparsity Tolerance by Issue Type', fontweight='bold')
        ax9.legend(fontsize=9, loc='best', framealpha=0.9)
        ax9.grid(True, alpha=0.3)
        ax9.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Acceptable')
        ax9.axhline(y=0.7, color='orange', linestyle='--', alpha=0.6, linewidth=1.5, label='Clinical')
        
        # Improve x-axis labels
        ax9.set_xticks(sorted(df['sampling_rate'].unique()))
        ax9.set_xticklabels([f'1:{rate}' for rate in sorted(df['sampling_rate'].unique())])
    
    # 🔧 FIX: Improved overall layout
    plt.tight_layout(pad=2.0)  # More padding between subplots
    plt.subplots_adjust(top=0.93)  # Leave space for main title
    
    return fig

def create_enhanced_boxplot_option1(df):
    """
    Box plot with sampling rate as nested variable
    Shows: Mode > Issue Type > Sampling Rate
    """
    plt.figure(figsize=(16, 8))
    
    # Create a combined variable for grouping
    df['mode_issue'] = df['mode'] + '_' + df['issue_type']
    
    sns.boxplot(data=df, x='mode_issue', y='mean_iou', hue='sampling_rate', 
                palette='Set2')
    
    plt.title('IoU Performance: Mode × Issue Type × Sampling Rate', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Mode and Issue Type', fontsize=12)
    plt.ylabel('Mean IoU', fontsize=12)
    
    # Clean up x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sampling Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()


def statistical_significance_analysis(df):
    """Perform comprehensive statistical analysis - ENHANCED for issue types"""
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    results_summary = []
    
    # Original analysis by sampling rate
    for rate in sorted(df['sampling_rate'].unique()):
        print(f"\nSampling Rate: {rate}")
        print("-" * 40)
        
        # Get data for this sampling rate
        rate_data = df[df['sampling_rate'] == rate]
        baseline_data = rate_data[rate_data['mode'] == 'baseline']['mean_iou']
        learning_data = rate_data[rate_data['mode'] == 'learning']['mean_iou']
        
        if len(baseline_data) > 0 and len(learning_data) > 0:
            # Descriptive statistics
            baseline_mean = baseline_data.mean()
            learning_mean = learning_data.mean()
            baseline_std = baseline_data.std()
            learning_std = learning_data.std()
            
            print(f"Baseline:  μ = {baseline_mean:.4f}, σ = {baseline_std:.4f}, n = {len(baseline_data)}")
            print(f"Learning:  μ = {learning_mean:.4f}, σ = {learning_std:.4f}, n = {len(learning_data)}")
            
            # Statistical tests
            if len(baseline_data) == len(learning_data):
                # Paired t-test (same exams)
                t_stat, p_value = stats.ttest_rel(learning_data, baseline_data)
                test_type = "Paired t-test"
            else:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(learning_data, baseline_data)
                test_type = "Independent t-test"
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_data) - 1) * baseline_std**2 + 
                                 (len(learning_data) - 1) * learning_std**2) / 
                                (len(baseline_data) + len(learning_data) - 2))
            cohens_d = (learning_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            # Improvement metrics
            absolute_improvement = learning_mean - baseline_mean
            relative_improvement = (absolute_improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
            
            print(f"{test_type}: t = {t_stat:.4f}, p = {p_value:.4f}")
            print(f"Effect size (Cohen's d): {cohens_d:.4f}")
            print(f"Absolute improvement: {absolute_improvement:+.4f}")
            print(f"Relative improvement: {relative_improvement:+.1f}%")
            
            # Interpretation
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            effect_interpretation = ("Large" if abs(cohens_d) > 0.8 else 
                                   "Medium" if abs(cohens_d) > 0.5 else 
                                   "Small" if abs(cohens_d) > 0.2 else "Negligible")
            
            print(f"Significance: {significance}")
            print(f"Effect size: {effect_interpretation}")
            
            # Store results
            results_summary.append({
                'analysis_type': 'by_sampling_rate',
                'sampling_rate': rate,
                'baseline_mean': baseline_mean,
                'learning_mean': learning_mean,
                'absolute_improvement': absolute_improvement,
                'relative_improvement': relative_improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': significance,
                'effect_size': effect_interpretation,
                'baseline_n': len(baseline_data),
                'learning_n': len(learning_data)
            })
        else:
            print("Insufficient data for statistical comparison")
    
    # NEW: Issue type analysis (if available)
    if 'issue_type' in df.columns and not df['issue_type'].isna().all():
        print(f"\n" + "="*80)
        print("ISSUE TYPE ANALYSIS")
        print("="*80)
        
        issue_stats = issue_type_statistical_analysis(df)
        
        # Add issue type results to summary
        for _, row in issue_stats.iterrows():
            results_summary.append({
                'analysis_type': 'by_issue_type',
                'comparison': row['comparison'],
                'type1_mean': row['type1_mean'],
                'type2_mean': row['type2_mean'],
                't_statistic': row['t_statistic'],
                'p_value': row['p_value'],
                'significant': row['significant']
            })
    
    return pd.DataFrame(results_summary)

def generate_executive_summary(df, stats_df):
    """Generate executive summary with key findings - ENHANCED for issue types"""
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    
    # Dataset overview
    print(f"\nDataset Overview:")
    print(f"• Total experiments: {len(df)}")
    print(f"• Number of exams: {df['exam_id'].nunique()}")
    print(f"• Sampling rates tested: {sorted(df['sampling_rate'].unique())}")
    print(f"• Total iterations analyzed: {df['iteration'].sum()}")
    
    # Issue type overview (if available)
    if 'issue_type' in df.columns and not df['issue_type'].isna().all():
        print(f"• Issue types analyzed: {sorted(df['issue_type'].unique())}")
        print(f"• Issue type distribution:")
        for issue_type, count in df['issue_type'].value_counts().items():
            print(f"  - {issue_type.replace('_', ' ').title()}: {count} experiments")
    
    # Method comparison overview
    method_comparison_count = df['has_method_comparison'].sum() if 'has_method_comparison' in df.columns else 0
    if method_comparison_count > 0:
        print(f"• Experiments with method comparison: {method_comparison_count}")
    
    # Overall performance
    overall_baseline = df[df['mode'] == 'baseline']['mean_iou'].mean()
    overall_learning = df[df['mode'] == 'learning']['mean_iou'].mean()
    overall_improvement = ((overall_learning - overall_baseline) / overall_baseline) * 100
    
    print(f"\nOverall Performance:")
    print(f"• Baseline average IoU: {overall_baseline:.4f}")
    print(f"• Learning mode average IoU: {overall_learning:.4f}")
    print(f"• Overall improvement: {overall_improvement:+.1f}%")
    
    # Issue type performance summary (if available)
    if 'issue_type' in df.columns and not df['issue_type'].isna().all():
        print(f"\nPerformance by Issue Type:")
        issue_performance = df.groupby('issue_type')['mean_iou'].mean().sort_values(ascending=False)
        for issue_type, performance in issue_performance.items():
            print(f"• {issue_type.replace('_', ' ').title()}: {performance:.4f}")
        
        # Clinical success rates by issue type
        print(f"\nClinical Success Rates (IoU > 0.7) by Issue Type:")
        for issue_type in df['issue_type'].unique():
            issue_data = df[df['issue_type'] == issue_type]
            clinical_rate = np.mean(issue_data['mean_iou'] > 0.7) * 100
            print(f"• {issue_type.replace('_', ' ').title()}: {clinical_rate:.1f}%")
    
    # Method comparison summary
    if method_comparison_count > 0:
        df_comp = df[df['has_method_comparison'] == True]
        avg_single_iou = df_comp['single_frame_vs_gt_iou'].mean()
        avg_multi_iou = df_comp['multi_frame_vs_gt_iou'].mean()
        avg_method_improvement = df_comp['iou_improvement_percent'].mean()
        
        print(f"\nMethod Comparison Results:")
        print(f"• Single-frame average IoU: {avg_single_iou:.4f}")
        print(f"• Multi-frame average IoU: {avg_multi_iou:.4f}")
        print(f"• Multi-frame advantage: {avg_method_improvement:+.1f}%")
    
    # Best and worst conditions
    best_result = df.loc[df['mean_iou'].idxmax()]
    worst_result = df.loc[df['mean_iou'].idxmin()]
    
    print(f"\nBest Performance:")
    print(f"• Exam: {best_result['exam_id']}, Mode: {best_result['mode']}, Rate: {best_result['sampling_rate']}")
    if 'issue_type' in df.columns:
        print(f"• Issue Type: {best_result.get('issue_type', 'N/A')}")
    print(f"• IoU: {best_result['mean_iou']:.4f}")
    
    print(f"\nWorst Performance:")
    print(f"• Exam: {worst_result['exam_id']}, Mode: {worst_result['mode']}, Rate: {worst_result['sampling_rate']}")
    if 'issue_type' in df.columns:
        print(f"• Issue Type: {worst_result.get('issue_type', 'N/A')}")
    print(f"• IoU: {worst_result['mean_iou']:.4f}")

def create_publication_ready_table(stats_df):
    """Create a publication-ready results table"""
    if len(stats_df) == 0:
        print("No statistical results to display")
        return
    
    print("\n" + "=" * 100)
    print("TABLE 1: STATISTICAL COMPARISON OF LEARNING MODE VS BASELINE")
    print("=" * 100)
    
    # Filter for sampling rate analysis
    sampling_rate_stats = stats_df[stats_df['analysis_type'] == 'by_sampling_rate']
    
    print(f"{'Sampling Rate':<15} {'Baseline':<12} {'Learning':<12} {'Improvement':<12} "
          f"{'p-value':<10} {'Effect Size':<12} {'Significance':<12}")
    print("-" * 100)
    
    for _, row in sampling_rate_stats.iterrows():
        print(f"{row['sampling_rate']:<15} "
              f"{row['baseline_mean']:<12.4f} "
              f"{row['learning_mean']:<12.4f} "
              f"{row['relative_improvement']:+<12.1f}% "
              f"{row['p_value']:<10.4f} "
              f"{row['effect_size']:<12} "
              f"{row['significance']:<12}")
    
    print("-" * 100)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Effect sizes: Large (|d|>0.8), Medium (|d|>0.5), Small (|d|>0.2), Negligible (|d|≤0.2)")

# Continue with remaining functions...
def find_video_for_exam_id(exam_id, annotations_json, video_base_path):
    """
    Use the SAME logic as your main script to find videos for exam IDs
    """
    # Find study UIDs for this exam ID using your existing logic
    datasets = annotations_json.get('datasets', [])
    study_uids_for_exam = []
    
    for dataset in datasets:
        studies = dataset.get('studies', [])
        for study in studies:
            if study.get('number') == exam_id and 'StudyInstanceUID' in study:
                study_uids_for_exam.append(study['StudyInstanceUID'])
    
    # If not found in studies, try annotations
    if not study_uids_for_exam:
        for dataset in datasets:
            for annotation in dataset.get('annotations', []):
                if annotation.get('examNumber') == exam_id and annotation.get('StudyInstanceUID'):
                    study_uids_for_exam.append(annotation['StudyInstanceUID'])
    
    # Find video files for these study UIDs
    video_paths = []
    for study_uid in study_uids_for_exam:
        # Look for video files using the same pattern as your main script
        study_dir = os.path.join(video_base_path, study_uid)
        if os.path.exists(study_dir):
            # Find .mp4 files in the study directory
            for file in os.listdir(study_dir):
                if file.endswith('.mp4'):
                    video_path = os.path.join(study_dir, file)
                    if os.path.exists(video_path):
                        video_paths.append(video_path)
                        break  # Take first video found
    
    return video_paths

def extract_frame_counts_for_exams(df, annotations_json, video_base_path):
    """
    Extract frame counts using the SAME exam ID logic as your main script
    """
    print("\nExtracting frame counts for exam analysis...")
    
    # Add new columns
    df['total_frames'] = None
    df['annotations_needed'] = None
    df['annotation_density_percent'] = None
    df['clinical_burden_minutes'] = None
    
    # Process each unique exam
    unique_exams = df['exam_id'].unique()
    frame_count_cache = {}
    
    for exam_id in unique_exams:
        # Extract numeric exam ID (remove 'exam_' prefix if present)
        if isinstance(exam_id, str) and exam_id.startswith('exam_'):
            numeric_exam_id = int(exam_id.replace('exam_', ''))
        else:
            try:
                numeric_exam_id = int(exam_id)
            except:
                print(f"Could not parse exam ID: {exam_id}")
                continue
        
        print(f"Processing exam {exam_id} (numeric: {numeric_exam_id})")
        
        # Find videos for this exam using your logic
        video_paths = find_video_for_exam_id(numeric_exam_id, annotations_json, video_base_path)
        
        if video_paths:
            # Use the first video found (assuming one video per exam for frame count)
            video_path = video_paths[0]
            
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                frame_count_cache[exam_id] = total_frames
                print(f"  Found {total_frames} frames in {os.path.basename(video_path)}")
                
            except Exception as e:
                print(f"  Error reading video {video_path}: {e}")
                frame_count_cache[exam_id] = None
        else:
            print(f"  No video found for exam {exam_id}")
            frame_count_cache[exam_id] = None
    
    # Apply frame counts to all rows
    for idx, row in df.iterrows():
        exam_id = row['exam_id']
        total_frames = frame_count_cache.get(exam_id)
        
        if total_frames:
            sampling_rate = row['sampling_rate']
            
            # Calculate metrics using YOUR current sampling logic
            annotations_needed = max(1, total_frames // sampling_rate)
            annotation_density = (annotations_needed / total_frames) * 100
            clinical_burden = annotations_needed * 0.5  # 30 seconds per annotation
            
            df.at[idx, 'total_frames'] = total_frames
            df.at[idx, 'annotations_needed'] = annotations_needed
            df.at[idx, 'annotation_density_percent'] = annotation_density
            df.at[idx, 'clinical_burden_minutes'] = clinical_burden
    
    return df

def create_data_sparsity_tolerance_visualization(df):
    """
    Create visualization focused on: "Can minimal expert input produce acceptable dense annotations?"
    """
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with strategic layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Optical Flow Sparse-to-Dense Annotation: Sparsity Tolerance Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # 1. SPARSITY TOLERANCE CHART
    ax_main = fig.add_subplot(gs[0, :])
    
    # Calculate sparsity tolerance metrics
    sparsity_data = []
    rates = sorted(df['sampling_rate'].unique())
    
    for rate in rates:
        learning_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'learning')]
        
        if len(learning_data) > 0:
            learning_mean = learning_data['mean_iou'].mean()
            learning_std = learning_data['mean_iou'].std()
            data_usage = 100 / rate  # Percentage of frames used as input
            
            sparsity_data.append({
                'sampling_rate': rate,
                'input_density_percent': data_usage,
                'output_quality_iou': learning_mean,
                'output_quality_std': learning_std
            })
    
    sparsity_df = pd.DataFrame(sparsity_data)
    
    # Plot the sparsity tolerance chart
    x_pos = np.arange(len(sparsity_df))
    
    # Color code based on clinical acceptability
    colors = []
    for quality in sparsity_df['output_quality_iou']:
        if quality >= 0.7:  # High quality
            colors.append('#2ca02c')  # Green
        elif quality >= 0.5:  # Moderate quality
            colors.append('#ff7f0e')  # Orange
        else:  # Low quality
            colors.append('#d62728')  # Red
    
    bars = ax_main.bar(x_pos, sparsity_df['output_quality_iou'], 
                      color=colors, alpha=0.8, 
                      yerr=sparsity_df['output_quality_std'], capsize=5)
    
    # Add clinical threshold lines
    ax_main.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Clinical Threshold (IoU > 0.7)')
    ax_main.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Acceptable Threshold (IoU > 0.5)')
    
    # Customize main chart
    ax_main.set_xlabel('Expert Input Density (% of frames manually annotated)', 
                      fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Dense Output Quality (Mean IoU)', fontsize=12, fontweight='bold')
    ax_main.set_title('Can Minimal Expert Input Produce Good Dense Annotations?', 
                     fontsize=14, fontweight='bold', pad=20)
    
    # Custom x-axis labels
    x_labels = [f'{row["input_density_percent"]:.1f}%' for _, row in sparsity_df.iterrows()]
    ax_main.set_xticks(x_pos)
    ax_main.set_xticklabels(x_labels, fontsize=10)
    ax_main.legend(loc='upper right', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, 1.0)
    
    # 2. COMPARISON BASELINE VS LEARNING
    ax_comparison = fig.add_subplot(gs[1, 0])
    
    rates_sorted = sorted(rates)
    baseline_qualities = []
    learning_qualities = []
    
    for rate in rates_sorted:
        baseline_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'baseline')]
        learning_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'learning')]
        
        baseline_mean = baseline_data['mean_iou'].mean() if len(baseline_data) > 0 else 0
        learning_mean = learning_data['mean_iou'].mean() if len(learning_data) > 0 else 0
        
        baseline_qualities.append(baseline_mean)
        learning_qualities.append(learning_mean)
    
    x_comparison = np.arange(len(rates_sorted))
    width = 0.35
    
    ax_comparison.bar(x_comparison - width/2, baseline_qualities, width, 
                     label='Baseline Parameters', color='lightcoral', alpha=0.8)
    ax_comparison.bar(x_comparison + width/2, learning_qualities, width, 
                     label='Learning Mode', color='lightblue', alpha=0.8)
    
    ax_comparison.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                         label='Acceptable Quality')
    
    ax_comparison.set_xlabel('Sampling Rate (1:N)', fontsize=11)
    ax_comparison.set_ylabel('Mean IoU', fontsize=11)
    ax_comparison.set_title('Parameter Optimization Effect', fontsize=12, fontweight='bold')
    
    x_labels = [f'1:{rate}' for rate in rates_sorted]
    ax_comparison.set_xticks(x_comparison)
    ax_comparison.set_xticklabels(x_labels, fontsize=10)
    ax_comparison.legend(fontsize=9)
    ax_comparison.grid(True, alpha=0.3)
    
    # 3. KEY FINDINGS
    ax_findings = fig.add_subplot(gs[1, 1])
    ax_findings.axis('off')
    
    # Calculate key findings
    viable_conditions = [i for i, q in enumerate(learning_qualities) if q >= 0.5]
    if viable_conditions:
        min_viable_input = min([100/rates_sorted[i] for i in viable_conditions])
        sparsest_acceptable = min([(100/rates_sorted[i], learning_qualities[i]) 
                                 for i in range(len(rates_sorted)) if learning_qualities[i] >= 0.5], 
                                key=lambda x: x[0])
    else:
        min_viable_input = 100
        sparsest_acceptable = (100, 0)
    
    findings_text = f"""
🎯 KEY FINDINGS:

SPARSE-TO-DENSE CAPABILITY:
✅ Minimum viable input: {sparsest_acceptable[0]:.1f}% 
✅ Quality achieved: IoU = {sparsest_acceptable[1]:.3f}
✅ Input reduction: {100-sparsest_acceptable[0]:.1f}% savings

AI-FAST IMPACT:
🚀 Enables dense annotation with minimal effort
💡 {100-sparsest_acceptable[0]:.1f}% reduction in annotation burden
🏥 Maintains acceptable quality for AI training
    """
    
    ax_findings.text(0.05, 0.95, findings_text, transform=ax_findings.transAxes, 
                    fontsize=11, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    return fig

# ===============================
# ENHANCED MAIN FUNCTIONS
# ===============================

def main(results_directory="results"):
    """Main function to run complete analysis"""
    print(f"Loading and processing all results from: {results_directory}")
    
    # Check if results directory exists
    if not Path(results_directory).exists():
        print(f"ERROR: Results directory '{results_directory}' not found!")
        print("Please ensure your results folder is in the correct location.")
        return None, None, None
    
    # Load all results
    df = combine_all_results(results_directory)
    
    if df.empty:
        print("No results found! Check your directory structure and file naming.")
        print("\nExpected structure:")
        print("results/")
        print("├── exam_001/")
        print("│   ├── baseline_rate_5.json")
        print("│   ├── learning_rate_5.json")
        print("│   └── ...")
        print("└── ...")
        return None, None, None
    
    print(f"Successfully loaded {len(df)} result records")
    print(f"Found {df['exam_id'].nunique()} exams, {len(df['sampling_rate'].unique())} sampling rates")
    
    # Create visualizations
    print("\nGenerating performance comparison plots...")
    fig = create_performance_comparison(df)
    plt.show()
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    stats_df = statistical_significance_analysis(df)
    
    # Generate summary reports
    generate_executive_summary(df, stats_df)
    create_publication_ready_table(stats_df)
    
    # Save results
    output_dir = Path(results_directory) / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "combined_results.csv", index=False)
    if not stats_df.empty:
        stats_df.to_csv(output_dir / "statistical_analysis.csv", index=False)
    fig.savefig(output_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Main outputs:")
    print(f"  - Combined data: {output_dir}/combined_results.csv")
    print(f"  - Statistics: {output_dir}/statistical_analysis.csv")
    print(f"  - Visualization: {output_dir}/performance_analysis.png")
    
    return df, stats_df, fig

def enhanced_main_with_method_comparison(results_directory="results", annotations_json_path=None, video_base_path=None):
    """
    Enhanced main function that includes method comparison analysis
    """
    print("=" * 60)
    print("ENHANCED OPTICAL FLOW ANALYSIS WITH METHOD COMPARISON")
    print("=" * 60)
    
    # Run original analysis
    print("Running original analysis...")
    df, stats_df, fig = main(results_directory)
    
    if df is None or len(df) == 0:
        print("No data available for analysis!")
        return None, None, None, None, None
    
    # Check for method comparison data
    method_comparison_available = df['has_method_comparison'].sum() > 0
    
    if method_comparison_available:
        print(f"\n✅ Found method comparison data in {df['has_method_comparison'].sum()} experiments")
        
        # Create method comparison visualizations
        print("\nGenerating method comparison visualizations...")
        method_comparison_fig = create_method_comparison_visualization(df)
        
        # Perform method comparison statistical analysis
        print("\nPerforming method comparison statistical analysis...")
        method_stats_df = create_method_comparison_statistical_analysis(df)
        
        if method_comparison_fig:
            plt.show()
    else:
        print("\n⚠️  No method comparison data found in results")
        print("To get method comparison data, run your evaluation with:")
        print("  --compare-methods flag")
        method_comparison_fig = None
        method_stats_df = None
    
    # Add frame analysis if paths provided
    if annotations_json_path and video_base_path:
        try:
            print(f"\nLoading annotations from: {annotations_json_path}")
            with open(annotations_json_path, 'r') as f:
                annotations_json = json.load(f)
            
            print(f"Using video base path: {video_base_path}")
            
            # Extract frame counts
            df_enhanced = extract_frame_counts_for_exams(df, annotations_json, video_base_path)
            
            # Create sparsity visualization
            print("\nGenerating sparse-to-dense annotation capability visualization...")
            sparsity_fig = create_data_sparsity_tolerance_visualization(df_enhanced)
            plt.show()
            
            # Save enhanced results
            output_dir = Path(results_directory) / "analysis_output"
            output_dir.mkdir(exist_ok=True)
            
            enhanced_output = output_dir / "frame_enhanced_results.csv"
            df_enhanced.to_csv(enhanced_output, index=False)
            
            sparsity_output = output_dir / "sparse_to_dense_analysis.png"
            sparsity_fig.savefig(sparsity_output, dpi=300, bbox_inches='tight')
            
            if method_comparison_fig:
                method_output = output_dir / "method_comparison_analysis.png"
                method_comparison_fig.savefig(method_output, dpi=300, bbox_inches='tight')
                print(f"Saved method comparison visualization to: {method_output}")
            
            if method_stats_df is not None:
                method_stats_output = output_dir / "method_comparison_statistics.csv"
                method_stats_df.to_csv(method_stats_output, index=False)
                print(f"Saved method comparison statistics to: {method_stats_output}")
            
            print(f"\nSaved enhanced results to: {enhanced_output}")
            print(f"Saved sparsity visualization to: {sparsity_output}")
            
            return df_enhanced, stats_df, fig, sparsity_fig, method_comparison_fig, method_stats_df
            
        except Exception as e:
            print(f"\nFrame analysis failed: {e}")
            print("Continuing with original analysis...")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping frame analysis (no annotations_json_path or video_base_path provided)")
        
        sparsity_fig = create_data_sparsity_tolerance_visualization(df)
        plt.show()
        
        # Save outputs
        output_dir = Path(results_directory) / "analysis_output"
        output_dir.mkdir(exist_ok=True)
        
        sparsity_output = output_dir / "sparse_to_dense_analysis.png"
        sparsity_fig.savefig(sparsity_output, dpi=300, bbox_inches='tight')
        
        if method_comparison_fig:
            method_output = output_dir / "method_comparison_analysis.png"
            method_comparison_fig.savefig(method_output, dpi=300, bbox_inches='tight')
            print(f"Saved method comparison visualization to: {method_output}")
        
        if method_stats_df is not None:
            method_stats_output = output_dir / "method_comparison_statistics.csv"
            method_stats_df.to_csv(method_stats_output, index=False)
            print(f"Saved method comparison statistics to: {method_stats_output}")
        
        print(f"Saved sparsity visualization to: {sparsity_output}")
        
        return df, stats_df, fig, sparsity_fig, method_comparison_fig, method_stats_df

# ===============================
# NEW: ENHANCED MAIN WITH ISSUE TYPES
# ===============================

def enhanced_main_with_issue_types(results_directory="results", issue_type_csv=None, issue_type_mapping=None, 
                                  annotations_json_path=None, video_base_path=None):
    """
    🆕 MAIN FUNCTION WITH ISSUE TYPE ANALYSIS
    This is your new go-to function for comprehensive analysis!
    """
    print("🚀 ENHANCED OPTICAL FLOW ANALYSIS WITH ISSUE TYPES")
    print("="*60)
    
    # Load results
    df = combine_all_results(results_directory)
    
    if df.empty:
        print("❌ No results found!")
        return None, None, None, None, None, None
    
    # 🆕 ADD ISSUE TYPES
    print("\n📋 Adding issue type classifications...")
    df = add_issue_types_to_dataframe(df, issue_type_mapping, issue_type_csv)
    
    # Create issue type visualizations
    print("\n🎨 Creating issue type performance analysis...")
    fig_issue_types = create_issue_type_performance_analysis(df)
    plt.show()
    
    # Statistical analysis by issue type
    print("\n📊 Performing statistical analysis by issue type...")
    issue_stats = issue_type_statistical_analysis(df)
    
    # Original analysis (now with issue types included)
    print("\n📈 Running enhanced performance analysis...")
    fig_original = create_performance_comparison(df)
    plt.show()
    
    stats_df = statistical_significance_analysis(df)
    generate_executive_summary(df, stats_df)
    create_publication_ready_table(stats_df)
    
    # Method comparison (if available)
    method_comparison_available = df['has_method_comparison'].sum() > 0
    method_comparison_fig = None
    method_stats_df = None
    
    if method_comparison_available:
        print(f"\n✅ Found method comparison data in {df['has_method_comparison'].sum()} experiments")
        print("\nGenerating method comparison visualizations...")
        method_comparison_fig = create_method_comparison_visualization(df)
        method_stats_df = create_method_comparison_statistical_analysis(df)
        if method_comparison_fig:
            plt.show()
    
    # Frame analysis (if paths provided)
    if annotations_json_path and video_base_path:
        try:
            print(f"\nLoading annotations from: {annotations_json_path}")
            with open(annotations_json_path, 'r') as f:
                annotations_json = json.load(f)
            
            df = extract_frame_counts_for_exams(df, annotations_json, video_base_path)
            print("✅ Enhanced with frame count analysis")
        except Exception as e:
            print(f"⚠️ Frame analysis failed: {e}")
    
    # Create sparsity visualization
    print("\n🎯 Generating sparse-to-dense annotation analysis...")
    sparsity_fig = create_data_sparsity_tolerance_visualization(df)
    plt.show()
    
    # Save enhanced results
    output_dir = Path(results_directory) / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "results_with_issue_types.csv", index=False)
    issue_stats.to_csv(output_dir / "issue_type_statistical_analysis.csv", index=False)
    stats_df.to_csv(output_dir / "enhanced_statistical_analysis.csv", index=False)
    
    fig_issue_types.savefig(output_dir / "issue_type_performance_analysis.png", 
                           dpi=300, bbox_inches='tight')
    fig_original.savefig(output_dir / "enhanced_performance_analysis.png", 
                        dpi=300, bbox_inches='tight')
    sparsity_fig.savefig(output_dir / "sparse_to_dense_analysis.png", 
                        dpi=300, bbox_inches='tight')
    
    if method_comparison_fig:
        method_comparison_fig.savefig(output_dir / "method_comparison_analysis.png", 
                                     dpi=300, bbox_inches='tight')
    if method_stats_df is not None:
        method_stats_df.to_csv(output_dir / "method_comparison_statistics.csv", index=False)
    
    print(f"\n💾 Enhanced results saved to: {output_dir}")
    print(f"📊 Key outputs:")
    print(f"   - Enhanced data: results_with_issue_types.csv")
    print(f"   - Issue type analysis: issue_type_performance_analysis.png")
    print(f"   - Enhanced performance: enhanced_performance_analysis.png")
    print(f"   - Sparsity analysis: sparse_to_dense_analysis.png")
    
    return df, stats_df, issue_stats, fig_original, fig_issue_types, sparsity_fig

# ===============================
# CONVENIENCE FUNCTIONS
# ===============================

def quick_analysis():
    """Run analysis with default settings for results in root/results/"""
    return main("results")

def quick_analysis_with_method_comparison(annotations_json_path=None, video_base_path=None):
    """Enhanced quick analysis that includes method comparison"""
    return enhanced_main_with_method_comparison("results", annotations_json_path, video_base_path)

def quick_analysis_with_issue_types(issue_type_csv=None, issue_type_mapping=None):
    """🆕 Quick analysis with issue types - RECOMMENDED!"""
    return enhanced_main_with_issue_types("results", issue_type_csv, issue_type_mapping)

def test_single_file(filepath):
    """Test processing of a single results file"""
    try:
        # Try to infer metadata from filename
        filename = Path(filepath).stem
        print(f"Testing file: {filename}")
        
        # Basic parsing
        exam_id = "test_exam"
        mode = "learning" if "learning" in filename else "baseline"
        
        # Try to extract sampling rate
        import re
        rate_match = re.search(r'rate[_\s]*(\d+)', filename)
        sampling_rate = int(rate_match.group(1)) if rate_match else 5
        
        print(f"Detected: exam_id={exam_id}, mode={mode}, sampling_rate={sampling_rate}")
        
        results = process_single_result_file(filepath, exam_id, mode, sampling_rate)
        print(f"Successfully processed {len(results)} iterations")
        
        # Show sample data
        if results:
            sample = results[0]
            print(f"Sample metrics: IoU={sample['mean_iou']:.4f}, Dice={sample['mean_dice']:.4f}")
            if sample['has_method_comparison']:
                print(f"Method comparison: Single={sample['single_frame_vs_gt_iou']:.4f}, Multi={sample['multi_frame_vs_gt_iou']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def create_exam_issue_type_template_csv():
    """Create a template CSV file for easy issue type mapping - FIXED VERSION"""
    
    # Define each list separately for easier debugging
    exam_ids = [
        'exam_68', 'exam_91', 'exam_97', 'exam_113', 'exam_123', 
        'exam_126', 'exam_132', 'exam_137', 'exam_184', 'exam_185', 
        'exam_194', 'exam_200', 'exam_160', 'exam_227'
    ]
    
    issue_types = [
        'uncomplicated',      # exam_68
        'multiple_distinct',  # exam_91
        'uncomplicated',      # exam_97
        'uncomplicated',      # exam_113
        'multiple_distinct',  # exam_123
        'branching_fluid',    # exam_126
        'uncomplicated',      # exam_132
        'multiple_distinct',  # exam_137
        'branching_fluid',    # exam_184
        'branching_fluid',    # exam_185
        '?',                  # exam_194
        '?',                  # exam_200
        'disappear_reappear', # exam_160
        'complex_mixed'       # exam_227
    ]
    
    descriptions = [
        'Uncomplicated single region',              # exam_68
        'Multiple distinct fluid regions',          # exam_91
        'Uncomplicated single region',              # exam_97
        'Uncomplicated single region',              # exam_113
        'Multiple distinct fluid regions',          # exam_123
        'Branching fluid patterns',                 # exam_126
        'Uncomplicated single region',              # exam_132
        'Multiple distinct fluid regions',          # exam_137
        'Branching fluid patterns',                 # exam_184
        'Branching fluid patterns',                 # exam_185
        'TO BE CLASSIFIED',                         # exam_194
        'TO BE CLASSIFIED',                         # exam_200
        'Fluid appears and disappears',             # exam_160
        'Complex: Branching + Disappear/Reappear'   # exam_227
    ]
    
    # Verify lengths match
    print(f"📊 Array lengths:")
    print(f"   exam_ids: {len(exam_ids)}")
    print(f"   issue_types: {len(issue_types)}")
    print(f"   descriptions: {len(descriptions)}")
    
    # Check if lengths match
    if not (len(exam_ids) == len(issue_types) == len(descriptions)):
        print("❌ ERROR: Array lengths don't match!")
        return None
    
    # Create the template data
    template_data = {
        'exam_id': exam_ids,
        'issue_type': issue_types,
        'description': descriptions
    }
    
    import pandas as pd
    template_df = pd.DataFrame(template_data)
    template_df.to_csv('exam_issue_types_template.csv', index=False)
    
    print("✅ Created template file: exam_issue_types_template.csv")
    print("📋 Your classifications:")
    for exam, issue_type in zip(exam_ids, issue_types):
        print(f"   {exam}: {issue_type}")
    
    print("\n🎯 Available issue types:")
    print("   - uncomplicated: Simple, single fluid region")
    print("   - multiple_distinct: Multiple separate fluid regions")
    print("   - branching_fluid: Tree-like branching patterns")
    print("   - disappear_reappear: Fluid appears/disappears over time")
    print("   - complex_mixed: Multiple challenging characteristics")
    
    return template_df

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    print("=" * 60)
    print("🩺 ENHANCED OPTICAL FLOW ANALYSIS WITH ISSUE TYPES")
    print("=" * 60)
    
    # Create template for issue types
    print("Creating issue type template...")
    create_exam_issue_type_template_csv()
    
    # UPDATE THESE PATHS:
    annotations_json_path = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_2025-06-03-194700.json"
    video_base_path = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_images_dataset_D_V688LQ_2025-06-03-194012"
    
    # 🆕 RECOMMENDED: Run with issue types (update the CSV file first!)
    print(f"\n🚀 Running enhanced analysis with issue types...")
    df, stats_df, issue_stats, fig_perf, fig_issues, fig_sparsity = enhanced_main_with_issue_types(
        results_directory="results",
        issue_type_csv="exam_issue_types_template.csv", 
        annotations_json_path=annotations_json_path,
        video_base_path=video_base_path
    )
    
    # Print final summary
    if df is not None:
        print(f"\n🎉 ENHANCED ANALYSIS COMPLETE!")
        print(f"   📊 {len(df)} total results processed")
        print(f"   🩺 {df['issue_type'].nunique()} issue types analyzed")
        print(f"   💾 Results saved to: results/analysis_output/")
        
        if 'issue_type' in df.columns:
            print(f"\n🩺 ISSUE TYPE SUMMARY:")
            for issue_type, performance in df.groupby('issue_type')['mean_iou'].mean().sort_values(ascending=False).items():
                clinical_rate = np.mean(df[df['issue_type'] == issue_type]['mean_iou'] > 0.7) * 100
                print(f"   {issue_type.replace('_', ' ').title()}: IoU={performance:.3f}, Clinical={clinical_rate:.1f}%")
    else:
        print("❌ Analysis failed - check your results directory and data files")