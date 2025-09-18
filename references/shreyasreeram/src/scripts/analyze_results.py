#!/usr/bin/env python3
import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.append(str(project_root))

def find_evaluation_results(base_dir="src/output/evaluations", exam_pattern="*"):
    """Find all evaluation result directories matching the pattern"""
    exam_dirs = glob.glob(os.path.join(base_dir, exam_pattern))
    return [d for d in exam_dirs if os.path.isdir(d)]

def extract_evaluation_metrics(eval_dir):
    """Extract evaluation metrics from a single evaluation directory"""
    # Try to load summary.json
    summary_path = os.path.join(eval_dir, "evaluation_summary.json")
    if not os.path.exists(summary_path):
        # Try full_evaluation_results.json
        summary_path = os.path.join(eval_dir, "full_evaluation_results.json")
        if not os.path.exists(summary_path):
            print(f"Warning: No evaluation results found in {eval_dir}")
            return None
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Extract exam name from directory
        exam_name = os.path.basename(eval_dir)
        
        # Load frame results for more detailed analysis
        frame_results_path = os.path.join(eval_dir, "frame_results.json")
        frame_results = {}
        if os.path.exists(frame_results_path):
            with open(frame_results_path, 'r') as f:
                frame_results = json.load(f)
        
        # Determine category based on exam name (can be customized)
        category = "unknown"
        if "issue" in exam_name.lower():
            category = "issue_type"
        elif "uncomplicated" in exam_name.lower():
            category = "uncomplicated"
        
        # Return structured results
        result = {
            "exam_id": exam_name,
            "category": category,
            "mean_iou": summary.get("mean_iou", 0),
            "median_iou": summary.get("median_iou", 0),
            "mean_dice": summary.get("mean_dice", 0),
            "iou_over_07": summary.get("iou_over_07", 0),
            "num_frames": summary.get("num_frames", 0),
            "frames_over_threshold": summary.get("frames_over_threshold", 0),
            "frame_results": frame_results
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing {eval_dir}: {str(e)}")
        return None

def compile_results(eval_dirs, output_file="results.csv"):
    """Compile evaluation results from multiple directories into a single DataFrame"""
    results = []
    
    for eval_dir in eval_dirs:
        result = extract_evaluation_metrics(eval_dir)
        if result:
            # Remove frame_results before adding to dataframe
            frame_results = result.pop("frame_results", {})
            results.append(result)
    
    if not results:
        print("No valid results found")
        return None
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Also save as JSON for easier consumption
    json_output = output_file.replace(".csv", ".json")
    df.to_json(json_output, orient="records", indent=2)
    
    print(f"Compiled results saved to {output_file} and {json_output}")
    
    return df

def create_bar_chart(df, metric="mean_iou", output_file="iou_comparison.png"):
    """Create a bar chart comparing performance across exams"""
    plt.figure(figsize=(12, 6))
    
    # Set a color palette based on category if available
    if 'category' in df.columns:
        palette = {"issue_type": "crimson", "uncomplicated": "forestgreen", "unknown": "gray"}
        ax = sns.barplot(x="exam_id", y=metric, hue="category", palette=palette, data=df)
    else:
        ax = sns.barplot(x="exam_id", y=metric, data=df)
    
    # Customize the plot
    metric_name = metric.replace("_", " ").title()
    plt.title(f"{metric_name} by Exam", fontsize=16)
    plt.xlabel("Exam", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Add value labels on the bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha='center', fontsize=8
        )
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved bar chart to {output_file}")
    
    return output_file

def create_metric_comparison(df, output_file="metrics_comparison.png"):
    """Create a comparison of different metrics across exams"""
    # Melt the dataframe to long format for faceted plotting
    metrics_to_plot = ["mean_iou", "median_iou", "mean_dice", "iou_over_07"]
    df_melt = pd.melt(
        df, 
        id_vars=["exam_id", "category"], 
        value_vars=metrics_to_plot,
        var_name="metric", 
        value_name="value"
    )
    
    # Map metric names to more readable versions
    metric_names = {
        "mean_iou": "Mean IoU",
        "median_iou": "Median IoU",
        "mean_dice": "Mean Dice",
        "iou_over_07": "IoU > 0.7 (%)"
    }
    df_melt["metric"] = df_melt["metric"].map(metric_names)
    
    # Convert IoU > 0.7 to percentage for better visualization
    df_melt.loc[df_melt["metric"] == "IoU > 0.7 (%)", "value"] *= 100
    
    # Create a faceted bar chart
    plt.figure(figsize=(14, 10))
    g = sns.catplot(
        data=df_melt,
        x="exam_id",
        y="value",
        hue="category",
        col="metric",
        kind="bar",
        height=4,
        aspect=1.2,
        palette={"issue_type": "crimson", "uncomplicated": "forestgreen", "unknown": "gray"},
        col_wrap=2
    )
    
    # Customize the plot
    g.set_xticklabels(rotation=45, ha="right")
    g.set_titles("{col_name}")
    g.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved metrics comparison to {output_file}")
    
    return output_file

def visualize_mask_samples(eval_dirs, output_dir="mask_visualizations"):
    """Create visualizations of algorithm vs ground truth masks for a few representative frames"""
    os.makedirs(output_dir, exist_ok=True)
    
    for eval_dir in eval_dirs:
        try:
            # Check if masks directory exists
            masks_dir = os.path.join(eval_dir, "masks")
            if not os.path.exists(masks_dir):
                print(f"No masks found in {eval_dir}")
                continue
            
            # Get the exam name
            exam_name = os.path.basename(eval_dir)
            
            # Find ground truth and algorithm masks
            gt_masks = sorted(glob.glob(os.path.join(masks_dir, "gt_mask_*.png")))
            algo_masks = sorted(glob.glob(os.path.join(masks_dir, "algo_mask_*.png")))
            
            # Map masks by frame number
            mask_pairs = {}
            for gt_mask in gt_masks:
                frame_num = os.path.basename(gt_mask).replace("gt_mask_", "").replace(".png", "")
                for algo_mask in algo_masks:
                    if f"algo_mask_{frame_num}.png" in algo_mask:
                        mask_pairs[frame_num] = (gt_mask, algo_mask)
                        break
            
            if not mask_pairs:
                print(f"No matching mask pairs found in {eval_dir}")
                continue
                
            # Load frame results to find best, worst, and median frames
            frame_results_path = os.path.join(eval_dir, "frame_results.json")
            frame_metrics = {}
            if os.path.exists(frame_results_path):
                with open(frame_results_path, 'r') as f:
                    frame_data = json.load(f)
                    for frame, data in frame_data.items():
                        if frame in mask_pairs:
                            frame_metrics[frame] = data.get("iou", 0)
            
            # Select representative frames
            selected_frames = {}
            if frame_metrics:
                # Find best, worst, and median frames
                best_frame = max(frame_metrics, key=frame_metrics.get)
                worst_frame = min(frame_metrics, key=frame_metrics.get)
                
                # Sort by IoU and find median
                sorted_frames = sorted(frame_metrics.items(), key=lambda x: x[1])
                median_frame = sorted_frames[len(sorted_frames)//2][0]
                
                selected_frames = {
                    f"{exam_name}_best_frame_{best_frame}": mask_pairs[best_frame],
                    f"{exam_name}_worst_frame_{worst_frame}": mask_pairs[worst_frame],
                    f"{exam_name}_median_frame_{median_frame}": mask_pairs[median_frame]
                }
            else:
                # Just take a few random samples
                sample_frames = list(mask_pairs.keys())[:3]
                for i, frame in enumerate(sample_frames):
                    selected_frames[f"{exam_name}_sample_{i+1}_frame_{frame}"] = mask_pairs[frame]
            
            # Create visualizations for selected frames
            for name, (gt_path, algo_path) in selected_frames.items():
                # Load masks
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                algo_mask = cv2.imread(algo_path, cv2.IMREAD_GRAYSCALE)
                
                # Create RGB visualization
                height, width = gt_mask.shape
                vis_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Ground truth in green
                vis_img[gt_mask > 0] = [0, 255, 0]
                
                # Algorithm in red
                vis_img[algo_mask > 0] = [0, 0, 255]
                
                # Overlap in yellow
                overlap = (gt_mask > 0) & (algo_mask > 0)
                vis_img[overlap] = [0, 255, 255]
                
                # Create a side-by-side comparison
                comparison = np.zeros((height, width*3, 3), dtype=np.uint8)
                
                # Ground truth
                gt_color = np.zeros((height, width, 3), dtype=np.uint8)
                gt_color[gt_mask > 0] = [0, 255, 0]
                comparison[:, 0:width] = gt_color
                
                # Algorithm mask
                algo_color = np.zeros((height, width, 3), dtype=np.uint8)
                algo_color[algo_mask > 0] = [0, 0, 255]
                comparison[:, width:width*2] = algo_color
                
                # Overlap visualization
                comparison[:, width*2:width*3] = vis_img
                
                # Add labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(comparison, "Ground Truth", (10, 30), font, 0.7, (255, 255, 255), 2)
                cv2.putText(comparison, "Algorithm", (width + 10, 30), font, 0.7, (255, 255, 255), 2)
                cv2.putText(comparison, "Overlap", (width*2 + 10, 30), font, 0.7, (255, 255, 255), 2)
                
                # Extract IoU from filename if possible
                iou = frame_metrics.get(name.split("_frame_")[-1], None)
                if iou is not None:
                    cv2.putText(comparison, f"IoU: {iou:.4f}", (width*2 + 10, 60), font, 0.7, (255, 255, 255), 2)
                
                # Save the visualization
                output_path = os.path.join(output_dir, f"{name}_comparison.png")
                cv2.imwrite(output_path, comparison)
                print(f"Saved visualization to {output_path}")
            
        except Exception as e:
            print(f"Error creating visualizations for {eval_dir}: {str(e)}")
            import traceback
            traceback.print_exc()

def create_frame_iou_plots(eval_dirs, output_dir="frame_plots"):
    """Create plots showing IoU across frames for each exam"""
    os.makedirs(output_dir, exist_ok=True)
    
    for eval_dir in eval_dirs:
        try:
            # Load frame results
            frame_results_path = os.path.join(eval_dir, "frame_results.json")
            if not os.path.exists(frame_results_path):
                print(f"No frame results found in {eval_dir}")
                continue
                
            with open(frame_results_path, 'r') as f:
                frame_results = json.load(f)
            
            # Extract exam name
            exam_name = os.path.basename(eval_dir)
            
            # Convert to DataFrame for easier plotting
            frame_data = []
            for frame, metrics in frame_results.items():
                frame_data.append({
                    "frame": int(frame),
                    "iou": metrics.get("iou", 0),
                    "dice": metrics.get("dice", 0)
                })
            
            if not frame_data:
                print(f"No frame data found in {eval_dir}")
                continue
                
            df = pd.DataFrame(frame_data)
            df = df.sort_values("frame")
            
            # Plot IoU across frames
            plt.figure(figsize=(12, 6))
            plt.plot(df["frame"], df["iou"], marker='o', linestyle='-', markersize=4)
            plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label="IoU = 0.7 threshold")
            
            # Add a trend line
            z = np.polyfit(df["frame"], df["iou"], 1)
            p = np.poly1d(z)
            plt.plot(df["frame"], p(df["frame"]), "k--", alpha=0.5, label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
            
            plt.title(f"IoU across Frames - {exam_name}", fontsize=14)
            plt.xlabel("Frame Number", fontsize=12)
            plt.ylabel("IoU", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the figure
            output_path = os.path.join(output_dir, f"{exam_name}_frame_iou.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved frame IoU plot to {output_path}")
            
            # Close the figure to free memory
            plt.close()
            
        except Exception as e:
            print(f"Error creating frame IoU plot for {eval_dir}: {str(e)}")
            import traceback
            traceback.print_exc()

def create_html_report(df, visualizations_dir, output_file="report.html"):
    """Create an HTML report with all the visualizations and metrics"""
    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Calculate summary statistics
    overall_mean_iou = df["mean_iou"].mean()
    overall_median_iou = df["median_iou"].mean()
    overall_mean_dice = df["mean_dice"].mean()
    overall_iou_over_07 = df["iou_over_07"].mean() * 100
    
    # Create category statistics if available
    category_stats = ""
    if "category" in df.columns and df["category"].nunique() > 1:
        category_stats += "<h3>Performance by Category</h3>\n<table>\n"
        category_stats += "<tr><th>Category</th><th>Mean IoU</th><th>Success Rate (%)</th><th>Count</th></tr>\n"
        
        for category, group in df.groupby("category"):
            category_stats += f"<tr><td>{category}</td><td>{group['mean_iou'].mean():.4f}</td>"
            category_stats += f"<td>{group['iou_over_07'].mean()*100:.1f}%</td>"
            category_stats += f"<td>{len(group)}</td></tr>\n"
        
        category_stats += "</table>\n"
    
    # Find all visualizations
    bar_charts = sorted(glob.glob(os.path.join(visualizations_dir, "*_comparison.png")))
    frame_plots = sorted(glob.glob(os.path.join(visualizations_dir, "*_frame_iou.png")))
    mask_comparisons = sorted(glob.glob(os.path.join("mask_visualizations", "*_comparison.png")))
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fluid Detection Evaluation Report - {today}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-value {{ font-weight: bold; color: #007bff; }}
            .visualization {{ margin: 20px 0; text-align: center; }}
            .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
            .exam-row {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Fluid Detection Evaluation Report</h1>
        <p>Generated on {today}</p>
        
        <h2>Executive Summary</h2>
        <p>This report presents the evaluation results for the fluid detection algorithm.</p>
        
        <h3>Overall Performance</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Mean IoU</td>
                <td class="metric-value">{overall_mean_iou:.4f}</td>
                <td>Average overlap between algorithm and ground truth masks</td>
            </tr>
            <tr>
                <td>Mean Dice</td>
                <td class="metric-value">{overall_mean_dice:.4f}</td>
                <td>Alternative overlap measure, less sensitive to small areas</td>
            </tr>
            <tr>
                <td>Success Rate</td>
                <td class="metric-value">{overall_iou_over_07:.1f}%</td>
                <td>Percentage of frames with IoU > 0.7</td>
            </tr>
        </table>
        
        {category_stats}
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Exam ID</th>
                <th>Category</th>
                <th>Mean IoU</th>
                <th>Median IoU</th>
                <th>Mean Dice</th>
                <th>Success Rate (%)</th>
                <th>Frames Evaluated</th>
            </tr>
    """
    
    # Add rows for each exam
    for _, row in df.iterrows():
        category = row.get("category", "unknown")
        html_content += f"""
            <tr class="exam-row">
                <td>{row["exam_id"]}</td>
                <td>{category}</td>
                <td>{row["mean_iou"]:.4f}</td>
                <td>{row["median_iou"]:.4f}</td>
                <td>{row["mean_dice"]:.4f}</td>
                <td>{row["iou_over_07"]*100:.1f}%</td>
                <td>{int(row["num_frames"])}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
    """
    
    # Add bar charts
    if bar_charts:
        html_content += """
        <h3>Performance Comparison</h3>
        <div class="visualization">
        """
        
        for chart in bar_charts:
            chart_name = os.path.basename(chart).replace("_", " ").replace(".png", "")
            html_content += f"""
            <h4>{chart_name}</h4>
            <img src="{chart}" alt="{chart_name}">
            """
        
        html_content += "</div>"
    
    # Add frame IoU plots
    if frame_plots:
        html_content += """
        <h3>Frame-by-Frame Analysis</h3>
        <div class="visualization">
        """
        
        for plot in frame_plots:
            plot_name = os.path.basename(plot).replace("_", " ").replace(".png", "")
            html_content += f"""
            <h4>{plot_name}</h4>
            <img src="{plot}" alt="{plot_name}">
            """
        
        html_content += "</div>"
    
    # Add mask comparisons
    if mask_comparisons:
        html_content += """
        <h3>Mask Comparison Samples</h3>
        <div class="visualization">
        """
        
        for comp in mask_comparisons:
            comp_name = os.path.basename(comp).replace("_", " ").replace(".png", "")
            html_content += f"""
            <h4>{comp_name}</h4>
            <img src="{comp}" alt="{comp_name}">
            """
        
        html_content += "</div>"
    
    # Close HTML
    html_content += """
        <h2>Conclusion</h2>
        <p>
            Based on the evaluation results, the fluid detection algorithm demonstrates 
            satisfactory performance across the test set. Further improvements could focus on 
            enhancing performance on the lower-performing exams identified in this report.
        </p>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Created HTML report at {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize evaluation results")
    parser.add_argument("--input-dir", default="src/output/evaluations", help="Base directory containing evaluation results")
    parser.add_argument("--exam-pattern", default="*", help="Pattern to match exam directories")
    parser.add_argument("--output-dir", default="src/output/analysis", help="Directory to save analysis results")
    parser.add_argument("--create-report", action="store_true", help="Create an HTML report")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find evaluation directories
    eval_dirs = find_evaluation_results(args.input_dir, args.exam_pattern)
    if not eval_dirs:
        print(f"No evaluation directories found matching pattern '{args.exam_pattern}' in {args.input_dir}")
        return
    
    print(f"Found {len(eval_dirs)} evaluation directories")
    
    # Compile results
    results_file = os.path.join(args.output_dir, "combined_results.csv")
    df = compile_results(eval_dirs, results_file)
    
    if df is None or len(df) == 0:
        print("No valid results found for analysis")
        return
    
    # Create visualizations
    # Bar chart of mean IoU
    create_bar_chart(df, "mean_iou", os.path.join(args.output_dir, "mean_iou_comparison.png"))
    
    # Bar chart of success rate
    create_bar_chart(df, "iou_over_07", os.path.join(args.output_dir, "success_rate_comparison.png"))
    
    # Create metrics comparison
    create_metric_comparison(df, os.path.join(args.output_dir, "metrics_comparison.png"))
    
    # Create frame IoU plots
    create_frame_iou_plots(eval_dirs, args.output_dir)
    
    # Create mask visualizations
    visualize_mask_samples(eval_dirs, os.path.join(args.output_dir, "mask_visualizations"))
    
    # Create report if requested
    if args.create_report:
        report_path = os.path.join(args.output_dir, "evaluation_report.html")
        create_html_report(df, args.output_dir, report_path)
        print(f"\nAnalysis complete! View the full report at: {report_path}")
    else:
        print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 