import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

def parse_markdown_summary(filepath):
    """
    Parse method comparison data from markdown summary files
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract exam ID from filepath
        exam_id = filepath.parent.name
        
        # Initialise result dictionary
        result = {
            'exam_id': exam_id,
            'source_file': str(filepath),
            'videos_with_comparison': 0,
            'method_agreement_iou': 0,
            'single_frame_predictions_avg': 0,
            'multi_frame_predictions_avg': 0,
            'multi_frame_iou_vs_gt': 0,
            'single_frame_iou_vs_gt': 0,
            'multi_frame_improvement_percent': 0,
            'has_valid_data': False
        }
        
        # Parse Method Comparison Summary section
        method_section = re.search(r'## Method Comparison Summary(.*?)## Performance Improvement Summary', content, re.DOTALL)
        if method_section:
            method_text = method_section.group(1)
            
            # Extract videos with successful comparison
            videos_match = re.search(r'Videos with successful comparison:\s*(\d+)', method_text)
            if videos_match:
                result['videos_with_comparison'] = int(videos_match.group(1))
            
            # Extract average method agreement IoU
            agreement_match = re.search(r'Average method agreement IoU:\s*([\d.]+)', method_text)
            if agreement_match:
                result['method_agreement_iou'] = float(agreement_match.group(1))
            
            # Extract average predictions per video
            single_pred_match = re.search(r'Average single-frame predictions per video:\s*([\d.]+)', method_text)
            if single_pred_match:
                result['single_frame_predictions_avg'] = float(single_pred_match.group(1))
            
            multi_pred_match = re.search(r'Average multi-frame predictions per video:\s*([\d.]+)', method_text)
            if multi_pred_match:
                result['multi_frame_predictions_avg'] = float(multi_pred_match.group(1))
        
        # Parse Performance Improvement Summary section
        perf_section = re.search(r'## Performance Improvement Summary(.*?)(?:##|$)', content, re.DOTALL)
        if perf_section:
            perf_text = perf_section.group(1)
            
            # Extract Multi-frame IoU vs GT
            multi_iou_match = re.search(r'Average Multi-frame IoU vs GT:\s*([\d.]+)', perf_text)
            if multi_iou_match:
                result['multi_frame_iou_vs_gt'] = float(multi_iou_match.group(1))
            
            # Extract Single-frame IoU vs GT
            single_iou_match = re.search(r'Average Single-frame IoU vs GT:\s*([\d.]+)', perf_text)
            if single_iou_match:
                result['single_frame_iou_vs_gt'] = float(single_iou_match.group(1))
            
            # Extract improvement percentage
            improvement_match = re.search(r'Average Multi-frame Improvement:\s*\+?([\d.]+)%', perf_text)
            if improvement_match:
                result['multi_frame_improvement_percent'] = float(improvement_match.group(1))
        
        # Check if we have valid data
        if (result['videos_with_comparison'] > 0 and 
            result['multi_frame_iou_vs_gt'] > 0 and 
            result['single_frame_iou_vs_gt'] > 0):
            result['has_valid_data'] = True
        
        return result
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def collect_all_markdown_results(results_directory):
    """
    Collect all method comparison results from markdown files
    """
    all_results = []
    results_path = Path(results_directory)
    
    print(f"Searching for markdown files in: {results_directory}")
    
    # Look for .md files in exam directories
    for exam_dir in results_path.iterdir():
        if exam_dir.is_dir():
            # Look for markdown files
            md_files = list(exam_dir.glob("*.md"))
            
            if md_files:
                print(f"Found {len(md_files)} markdown files in {exam_dir.name}")
                
                for md_file in md_files:
                    result = parse_markdown_summary(md_file)
                    if result and result['has_valid_data']:
                        all_results.append(result)
                        print(f"  ✅ Parsed data from {md_file.name}")
                    else:
                        print(f"  ⚠️  No valid data in {md_file.name}")
            else:
                print(f"No markdown files found in {exam_dir.name}")
    
    return pd.DataFrame(all_results)

def create_method_comparison_analysis(df):
    """
    Create comprehensive method comparison analysis from markdown data
    """
    if df.empty:
        print("No valid markdown data found!")
        return None
    
    print(f"\nCreating analysis for {len(df)} exams with method comparison data")
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    
    # Main title
    fig.suptitle('Single-Frame vs Multi-Frame Method Comparison Analysis\n'
                 'Results from Markdown Summary Files', 
                 fontsize=16, fontweight='bold', y=0.97)
    
    # 1. MAIN COMPARISON CHART
    ax_main = fig.add_subplot(gs[0, :2])
    
    exam_ids = df['exam_id'].values
    x = np.arange(len(exam_ids))
    width = 0.35
    
    single_ious = df['single_frame_iou_vs_gt'].values
    multi_ious = df['multi_frame_iou_vs_gt'].values
    
    bars1 = ax_main.bar(x - width/2, single_ious, width, 
                       label='Single-Frame Method', color='lightcoral', alpha=0.8)
    bars2 = ax_main.bar(x + width/2, multi_ious, width, 
                       label='Multi-Frame Method', color='lightblue', alpha=0.8)
    
    # Add improvement annotations with better positioning
    for i, (single_iou, multi_iou, improvement) in enumerate(zip(single_ious, multi_ious, df['multi_frame_improvement_percent'])):
        color = 'green' if improvement > 0 else 'red'
        # Position annotation above the higher bar with more space
        y_pos = max(single_iou, multi_iou) + 0.05
        ax_main.annotate(f'+{improvement:.1f}%', 
                        xy=(i, y_pos), 
                        ha='center', va='bottom', fontweight='bold', 
                        color=color, fontsize=8, rotation=0)
    
    # Add clinical thresholds
    ax_main.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Clinical Threshold (IoU > 0.7)')
    ax_main.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Acceptable Threshold (IoU > 0.5)')
    
    ax_main.set_xlabel('Exam ID', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Mean IoU vs Ground Truth', fontsize=11, fontweight='bold')
    ax_main.set_title('Single-Frame vs Multi-Frame Performance Comparison', 
                     fontsize=12, fontweight='bold')
    ax_main.set_xticks(x)
    
    # Fix x-axis labels with better spacing and rotation
    ax_main.set_xticklabels(exam_ids, rotation=45, ha='right', fontsize=9)
    
    # Move legend outside the plot area
    ax_main.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, 1.1)  # Give more space for annotations
    
    # 2. IMPROVEMENT DISTRIBUTION
    ax_improvement = fig.add_subplot(gs[0, 2])
    
    improvements = df['multi_frame_improvement_percent'].values
    
    ax_improvement.hist(improvements, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax_improvement.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_improvement.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2, 
                          alpha=0.8)
    
    # Add mean label without covering the chart
    ax_improvement.text(0.98, 0.85, f'Mean: {np.mean(improvements):.1f}%', 
                       transform=ax_improvement.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                       fontsize=9, fontweight='bold')
    
    ax_improvement.set_xlabel('Multi-Frame Improvement (%)', fontsize=10)
    ax_improvement.set_ylabel('Count', fontsize=10)
    ax_improvement.set_title('Distribution of\nImprovement', fontsize=11, fontweight='bold')
    ax_improvement.grid(True, alpha=0.3)
    
    # Add statistics in bottom corner
    positive_improvements = np.sum(improvements > 0)
    total_comparisons = len(improvements)
    ax_improvement.text(0.02, 0.98, f'{positive_improvements}/{total_comparisons}\n({positive_improvements/total_comparisons*100:.1f}%)\nImproved', 
                       transform=ax_improvement.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
                       fontsize=8)
    
    # 3. METHOD AGREEMENT ANALYSIS
    ax_agreement = fig.add_subplot(gs[1, 0])
    
    agreement_scores = df['method_agreement_iou'].values
    exam_indices = range(len(exam_ids))
    
    bars = ax_agreement.bar(exam_indices, agreement_scores, alpha=0.7, color='purple')
    ax_agreement.set_xlabel('Exam Index', fontsize=10)
    ax_agreement.set_ylabel('Method Agreement IoU', fontsize=10)
    ax_agreement.set_title('Inter-Method Agreement\n(How Similar Are Predictions?)', fontsize=10, fontweight='bold')
    ax_agreement.grid(True, alpha=0.3)
    
    # Set better x-tick spacing
    if len(exam_indices) > 10:
        step = max(1, len(exam_indices) // 8)
        ax_agreement.set_xticks(exam_indices[::step])
    else:
        ax_agreement.set_xticks(exam_indices)
    
    # 4. PREDICTION COUNTS COMPARISON
    ax_counts = fig.add_subplot(gs[1, 1])
    
    single_counts = df['single_frame_predictions_avg'].values
    multi_counts = df['multi_frame_predictions_avg'].values
    
    ax_counts.scatter(single_counts, multi_counts, alpha=0.7, s=100, color='orange', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line for reference
    max_count = max(np.max(single_counts), np.max(multi_counts))
    min_count = min(np.min(single_counts), np.min(multi_counts))
    ax_counts.plot([min_count, max_count], [min_count, max_count], 'r--', alpha=0.7, linewidth=2)
    
    # Add legend in corner
    ax_counts.text(0.02, 0.98, 'Red line:\nEqual Predictions', 
                  transform=ax_counts.transAxes, va='top', ha='left',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                  fontsize=8)
    
    ax_counts.set_xlabel('Single-Frame Predictions per Video', fontsize=10)
    ax_counts.set_ylabel('Multi-Frame Predictions per Video', fontsize=10)
    ax_counts.set_title('Prediction Count Comparison', fontsize=10, fontweight='bold')
    ax_counts.grid(True, alpha=0.3)
    
    # 5. PERFORMANCE vs AGREEMENT
    ax_perf_agreement = fig.add_subplot(gs[1, 2])
    
    improvements = df['multi_frame_improvement_percent'].values
    agreements = df['method_agreement_iou'].values
    
    scatter = ax_perf_agreement.scatter(agreements, improvements, alpha=0.8, s=80, 
                                       c=improvements, cmap='RdYlGn', vmin=0, vmax=np.max(improvements),
                                       edgecolors='black', linewidth=0.5)
    
    # Position colorbar better
    cbar = plt.colorbar(scatter, ax=ax_perf_agreement, shrink=0.8, pad=0.02)
    cbar.set_label('Improvement %', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    ax_perf_agreement.set_xlabel('Method Agreement IoU', fontsize=10)
    ax_perf_agreement.set_ylabel('Multi-Frame Improvement (%)', fontsize=10)
    ax_perf_agreement.set_title('Agreement vs Improvement\n(Do Similar Methods Improve More?)', fontsize=10, fontweight='bold')
    ax_perf_agreement.grid(True, alpha=0.3)
    
    # 6. SUMMARY STATISTICS
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    # Calculate comprehensive statistics
    total_exams = len(df)
    avg_single_iou = np.mean(single_ious)
    avg_multi_iou = np.mean(multi_ious)
    avg_improvement = np.mean(improvements)
    avg_agreement = np.mean(agreements)
    
    # Clinical quality analysis
    single_clinical = np.mean(single_ious > 0.7) * 100
    multi_clinical = np.mean(multi_ious > 0.7) * 100
    
    # Acceptable quality analysis  
    single_acceptable = np.mean(single_ious > 0.5) * 100
    multi_acceptable = np.mean(multi_ious > 0.5) * 100
    
    # Create a more organized summary with better formatting
    summary_text = f"""
🔬 METHOD COMPARISON FINDINGS FROM {total_exams} EXAMS:

📊 OVERALL PERFORMANCE:
• Single-Frame Average IoU: {avg_single_iou:.3f}
• Multi-Frame Average IoU: {avg_multi_iou:.3f}  
• Average Improvement: +{avg_improvement:.1f}%
• Method Agreement: {avg_agreement:.3f}

✅ QUALITY THRESHOLDS:
• Single-frame clinical quality (IoU>0.7): {single_clinical:.1f}% of exams
• Multi-frame clinical quality (IoU>0.7): {multi_clinical:.1f}% of exams
• Single-frame acceptable quality (IoU>0.5): {single_acceptable:.1f}% of exams  
• Multi-frame acceptable quality (IoU>0.5): {multi_acceptable:.1f}% of exams

🏥 CLINICAL IMPACT:
• Clinical improvement: {multi_clinical - single_clinical:+.1f} percentage points
• Acceptable improvement: {multi_acceptable - single_acceptable:+.1f} percentage points
• Best improvement: {np.max(improvements):.1f}%
• Worst case: {np.min(improvements):.1f}%

💡 CONCLUSION: {"Multi-frame supervision significantly improves tracking performance" if avg_improvement > 10 else "Multi-frame shows meaningful improvements" if avg_improvement > 5 else "Multi-frame shows modest improvements"}
    """
    
    # Split the summary into two columns for better layout
    lines = summary_text.strip().split('\n')
    mid_point = len(lines) // 2
    
    left_text = '\n'.join(lines[:mid_point])
    right_text = '\n'.join(lines[mid_point:])
    
    # Left column
    ax_summary.text(0.02, 0.95, left_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    # Right column  
    ax_summary.text(0.52, 0.95, right_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig

def create_summary_table(df):
    """Create a publication-ready summary table"""
    if df.empty:
        print("No data available for summary table")
        return
    
    print("\n" + "=" * 120)
    print("METHOD COMPARISON SUMMARY TABLE")
    print("=" * 120)
    
    print(f"{'Exam ID':<12} {'Single IoU':<12} {'Multi IoU':<12} {'Improvement':<12} "
          f"{'Agreement':<12} {'Videos':<8} {'Single Preds':<12} {'Multi Preds':<12}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['exam_id']:<12} "
              f"{row['single_frame_iou_vs_gt']:<12.4f} "
              f"{row['multi_frame_iou_vs_gt']:<12.4f} "
              f"+{row['multi_frame_improvement_percent']:<11.1f}% "
              f"{row['method_agreement_iou']:<12.4f} "
              f"{row['videos_with_comparison']:<8.0f} "
              f"{row['single_frame_predictions_avg']:<12.1f} "
              f"{row['multi_frame_predictions_avg']:<12.1f}")
    
    print("-" * 120)
    
    # Overall statistics
    print(f"{'OVERALL':<12} "
          f"{df['single_frame_iou_vs_gt'].mean():<12.4f} "
          f"{df['multi_frame_iou_vs_gt'].mean():<12.4f} "
          f"+{df['multi_frame_improvement_percent'].mean():<11.1f}% "
          f"{df['method_agreement_iou'].mean():<12.4f} "
          f"{df['videos_with_comparison'].mean():<8.1f} "
          f"{df['single_frame_predictions_avg'].mean():<12.1f} "
          f"{df['multi_frame_predictions_avg'].mean():<12.1f}")
    
    print("=" * 120)

def generate_executive_summary_md(df):
    """Generate executive summary from markdown data"""
    if df.empty:
        print("No data available for executive summary")
        return
    
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY - METHOD COMPARISON ANALYSIS")
    print("=" * 80)
    
    total_exams = len(df)
    total_videos = df['videos_with_comparison'].sum()
    
    print(f"\nDataset Overview:")
    print(f"• Total exams analyzed: {total_exams}")
    print(f"• Total videos with successful comparisons: {total_videos}")
    print(f"• Average videos per exam: {total_videos/total_exams:.1f}")
    
    # Performance metrics
    avg_single = df['single_frame_iou_vs_gt'].mean()
    avg_multi = df['multi_frame_iou_vs_gt'].mean()
    avg_improvement = df['multi_frame_improvement_percent'].mean()
    
    print(f"\nMethod Performance:")
    print(f"• Single-frame average IoU: {avg_single:.4f}")
    print(f"• Multi-frame average IoU: {avg_multi:.4f}")
    print(f"• Average improvement: +{avg_improvement:.1f}%")
    print(f"• Range of improvement: {df['multi_frame_improvement_percent'].min():.1f}% to {df['multi_frame_improvement_percent'].max():.1f}%")
    
    # Clinical impact
    single_clinical = (df['single_frame_iou_vs_gt'] > 0.7).sum()
    multi_clinical = (df['multi_frame_iou_vs_gt'] > 0.7).sum()
    
    print(f"\nClinical Quality (IoU > 0.7):")
    print(f"• Single-frame clinical quality: {single_clinical}/{total_exams} exams ({single_clinical/total_exams*100:.1f}%)")
    print(f"• Multi-frame clinical quality: {multi_clinical}/{total_exams} exams ({multi_clinical/total_exams*100:.1f}%)")
    print(f"• Clinical improvement: +{multi_clinical - single_clinical} exams")
    
    # Best and worst cases
    best_idx = df['multi_frame_improvement_percent'].idxmax()
    worst_idx = df['multi_frame_improvement_percent'].idxmin()
    
    print(f"\nBest Performance:")
    print(f"• Exam: {df.loc[best_idx, 'exam_id']}")
    print(f"• Improvement: +{df.loc[best_idx, 'multi_frame_improvement_percent']:.1f}%")
    print(f"• Multi-frame IoU: {df.loc[best_idx, 'multi_frame_iou_vs_gt']:.4f}")
    
    print(f"\nWorst Performance:")
    print(f"• Exam: {df.loc[worst_idx, 'exam_id']}")
    print(f"• Improvement: +{df.loc[worst_idx, 'multi_frame_improvement_percent']:.1f}%")
    print(f"• Multi-frame IoU: {df.loc[worst_idx, 'multi_frame_iou_vs_gt']:.4f}")

def main_markdown_analysis(results_directory="results"):
    """Main function for analyzing markdown summary files"""
    print("=" * 60)
    print("MARKDOWN METHOD COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Collect all markdown results
    df = collect_all_markdown_results(results_directory)
    
    if df.empty:
        print("\n❌ No valid markdown data found!")
        print("Make sure you have .md files with method comparison summaries in your exam directories.")
        return None, None
    
    print(f"\n✅ Successfully loaded {len(df)} exams with method comparison data")
    
    # Create visualizations
    print("\nGenerating method comparison visualizations...")
    fig = create_method_comparison_analysis(df)
    
    if fig:
        plt.show()
    
    # Generate summary reports
    generate_executive_summary_md(df)
    create_summary_table(df)
    
    # Save results
    output_dir = Path(results_directory) / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save data and visualization
    df.to_csv(output_dir / "method_comparison_results.csv", index=False)
    
    if fig:
        fig.savefig(output_dir / "method_comparison_analysis.png", dpi=300, bbox_inches='tight')
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"• Data: {output_dir}/method_comparison_results.csv")
    print(f"• Visualization: {output_dir}/method_comparison_analysis.png")
    
    return df, fig


if __name__ == "__main__":
    df, fig = main_markdown_analysis("results")
    
    if df is not None:
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"   📊 {len(df)} exams analyzed")
        print(f"   📈 Average improvement: +{df['multi_frame_improvement_percent'].mean():.1f}%")
        print(f"   🏥 Clinical quality improvement: {((df['multi_frame_iou_vs_gt'] > 0.7).sum() - (df['single_frame_iou_vs_gt'] > 0.7).sum())} exams")
    else:
        print("❌ Analysis failed - check your results directory structure")