# ground_truth_utils.py
import os
import json
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import os
from multi_frame_tracking.multi_frame_tracker import MultiFrameTracker
import mdai
import cv2


DATA_DIR = os.getenv('DATA_DIR')

def get_annotations_for_study_series(mdai_client, project_id, dataset_id, study_uid, series_uid, label_id=None):
    """
    Get annotations for a specific study/series combination
    
    Args:
        mdai_client: MD.ai client instance
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID
        study_uid: Study instance UID
        series_uid: Series instance UID
        label_id: Optional label ID to filter annotations
        
    Returns:
        List of annotations for the study/series
    """
    try:
        print(f"Fetching annotations for {study_uid}/{series_uid}")
        
        # First, get all annotations for the dataset
        result = mdai_client.project(
            project_id=project_id,
            dataset_id=dataset_id,
            path=DATA_DIR,
            annotations_only=True
        )
        
        # Find the latest annotations file
        annotation_files = [f for f in os.listdir(DATA_DIR) if f.startswith('mdai_ucsf_project') and f.endswith('.json')]
        
        if not annotation_files:
            print("No annotation files found")
            return []
        
        # Sort by modification time (newest first)
        annotation_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
        annotations_file = os.path.join(DATA_DIR, annotation_files[0])
        
        # Load the JSON
        with open(annotations_file, 'r') as f:
            annotations_json = json.load(f)
        
        # Extract all annotations
        all_annotations = []
        for dataset in annotations_json.get('datasets', []):
            if 'annotations' in dataset:
                all_annotations.extend(dataset['annotations'])
        
        # Filter for the specific study/series
        filtered_annotations = [
            ann for ann in all_annotations
            if ann.get('StudyInstanceUID') == study_uid and ann.get('SeriesInstanceUID') == series_uid
        ]
        
        # Further filter by label_id if provided
        if label_id:
            filtered_annotations = [
                ann for ann in filtered_annotations
                if ann.get('labelId') == label_id
            ]
        
        print(f"Found {len(filtered_annotations)} annotations for {study_uid}/{series_uid}")
        return filtered_annotations
        
    except Exception as e:
        print(f"Error fetching annotations: {str(e)}")
        traceback.print_exc()
        return []
    
def delete_existing_annotations(client, study_uid, series_uid, label_id, group_id=None):
    """
    Delete existing annotations for a study/series with a specific label ID
    
    This can be moved from your existing utils.py file if you already have it
    """
    # Your existing implementation...
    pass

#Function for creating ground truth dataset (will run if flag is specified)
LABEL_ID_FREE_FLUID = os.environ["LABEL_ID_FREE_FLUID"] = "L_13yPql"
LABEL_ID_NO_FLUID = os.getenv("LABEL_ID_NO_FLUID") 
LABEL_ID_MACHINE_GROUP = os.getenv("LABEL_ID_MACHINE_GROUP")


def create_ground_truth_dataset(video_paths, study_series_pairs, flow_processor, output_dir, 
                              mdai_client, project_id, dataset_id, ground_truth_label_id,
                              matched_annotations=None, free_fluid_annotations=None,  # Add this parameter
                              label_id_fluid=None, label_id_no_fluid=None, 
                              label_id_machine=None, annotations_json=None):
    """
    Creates a ground truth dataset following the exact same annotation processing 
    pattern as the normal workflow
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if matched_annotations is None or free_fluid_annotations is None:
        raise ValueError("Both matched_annotations and free_fluid_annotations must be provided")
    
    # Import the find_exam_number function
    from multi_frame_tracking.utils import find_exam_number
    
    results = {
        'processed_videos': [],
        'failed_videos': [],
        'total_annotations_created': 0,
        'total_upload_success': 0,
        'total_upload_failures': 0
    }
    
    print(f"Creating ground truth dataset for {len(video_paths)} videos...")
    
    for i, (video_path, (study_uid, series_uid)) in enumerate(zip(video_paths, study_series_pairs)):
        print(f"\nProcessing video {i+1}/{len(video_paths)}")
        print(f"Study UID: {study_uid}")
        print(f"Series UID: {series_uid}")

        # Find exam number for this study
        exam_number = "unknown"
        if annotations_json:
            try:
                exam_number = find_exam_number(study_uid, annotations_json)
                print(f"Exam Number: {exam_number}")
            except Exception as e:
                print(f"Could not determine exam number: {e}")
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_dir, f"exam_{exam_number}_{study_uid}_{series_uid}")
        os.makedirs(video_output_dir, exist_ok=True)

        try:
            # FOLLOW THE EXACT SAME PATTERN AS NORMAL WORKFLOW
            # Get the matched annotations for this video (includes issue type)
            video_matched_annotations = matched_annotations[
                (matched_annotations['StudyInstanceUID'] == study_uid) &
                (matched_annotations['SeriesInstanceUID'] == series_uid)
            ].copy()
            
            if len(video_matched_annotations) == 0:
                print(f"WARNING: No matched annotations found for {study_uid}/{series_uid}")
                results['failed_videos'].append({
                    'video': video_path,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'exam_number': exam_number,  # Add exam number
                    'error': 'No matched annotations found'
                })
                continue
            
           
            frame_number = int(video_matched_annotations.iloc[0]['frameNumber_free_fluid'])
            
            
            free_fluid_polygons = video_matched_annotations.iloc[0]['free_fluid_foreground']
            
            print(f"Processing matched annotation for frame {frame_number}")
            print(f"Issue type: {video_matched_annotations.iloc[0]['issue_type']}")
            
            # Continue only if we have valid polygons
            if not isinstance(free_fluid_polygons, list) or len(free_fluid_polygons) == 0:
                print("No valid polygons found, skipping")
                results['failed_videos'].append({
                    'video': video_path,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'exam_number': exam_number,  # Add exam number
                    'error': 'No valid polygons found'
                })
                continue
            
        
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open video: {video_path}")
                    results['failed_videos'].append({
                        'video': video_path,
                        'study_uid': study_uid,
                        'series_uid': series_uid,
                        'exam_number': exam_number,  # Add exam number
                        'error': 'Could not open video'
                    })
                    continue
                
                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                print(f"Video dimensions: {frame_width}x{frame_height}")
                print(f"Total frames: {total_frames}")
                
                # EXACTLY LIKE NORMAL WORKFLOW: Create initial mask from polygons
                from multi_frame_tracking.utils import polygons_to_mask
                initial_mask = polygons_to_mask(free_fluid_polygons, frame_height, frame_width)
                
                # EXACTLY LIKE NORMAL WORKFLOW: Get all annotations for this video
                video_annotations = free_fluid_annotations[
                    (free_fluid_annotations['StudyInstanceUID'] == study_uid) &
                    (free_fluid_annotations['SeriesInstanceUID'] == series_uid)
                ].copy()
                
                print(f"Found {len(video_annotations)} annotations for this video")
                
                # EXACTLY LIKE NORMAL WORKFLOW: Process using the enhanced function
                from consolidated_tracking import process_video_with_multi_frame_tracking_enhanced
                
                print("\n##############################################")
                print("# CALLING MULTI-FRAME TRACKING FUNCTION #")
                print("##############################################")
                print(f"Flow processor method: {flow_processor.method}")
                print(f"Study/Series: {study_uid}/{series_uid}")
                
                # Call the multi-frame processing function EXACTLY LIKE NORMAL WORKFLOW
                result = process_video_with_multi_frame_tracking_enhanced(
                    video_path=video_path,
                    annotations_df=video_annotations,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    flow_processor=flow_processor,
                    output_dir=video_output_dir,
                    annotations_json=annotations_json,  # Pass annotations_json for exam number
                    mdai_client=mdai_client,
                    label_id_fluid=label_id_fluid,
                    label_id_machine=label_id_machine,
                    label_id_no_fluid=label_id_no_fluid,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    upload_to_mdai=False,  # We'll upload manually with ground truth label
                    debug_mode=True
                )
                
                if not result or 'all_masks' not in result:
                    print("WARNING: No masks generated by the algorithm")
                    results['failed_videos'].append({
                        'video': video_path,
                        'study_uid': study_uid,
                        'series_uid': series_uid,
                        'exam_number': exam_number,  # Add exam number
                        'error': 'No masks generated by algorithm'
                    })
                    continue
                
                algorithm_masks = result['all_masks']
                print(f"Generated {len(algorithm_masks)} masks")
                
                # Debug: Print mask types
                mask_types = {}
                for frame_idx, mask_info in algorithm_masks.items():
                    if isinstance(mask_info, dict) and 'type' in mask_info:
                        mask_type = mask_info['type']
                        mask_types[mask_type] = mask_types.get(mask_type, 0) + 1
                
                print("Mask types generated:")
                for mask_type, count in mask_types.items():
                    print(f"  {mask_type}: {count}")
                
                # Prepare masks for upload with the ground truth label
                annotations_to_upload = []
                
                for frame_idx, mask_info in algorithm_masks.items():
                    # Skip human annotations (already in MD.ai)
                    if isinstance(mask_info, dict) and mask_info.get('is_annotation', False):
                        continue
                    
                    # Get the mask and type
                    if isinstance(mask_info, dict):
                        mask = mask_info['mask']
                        mask_type = mask_info.get('type', '')
                    else:
                        mask = mask_info
                        mask_type = 'unknown'
                    
                    # Handle clear frames differently
                    is_clear = 'clear' in mask_type.lower() if mask_type else False
                    
                    # Create annotation based on type
                    try:
                        if is_clear:
                            # For clear frames, create a no-fluid annotation
                            annotation = {
                                'labelId': ground_truth_label_id,  # Use ground truth label
                                'StudyInstanceUID': study_uid,
                                'SeriesInstanceUID': series_uid,
                                'frameNumber': int(frame_idx),
                                'groupId': label_id_machine,
                                'note': f'Ground truth: No fluid frame (predicted by algorithm) - Exam #{exam_number}'
                            }
                            annotations_to_upload.append(annotation)
                        else:
                            # For fluid frames, check if mask has content
                            binary_mask = (mask > 0.5).astype(np.uint8)
                            if np.sum(binary_mask) > 0:
                                # Convert mask to MD.ai format
                                import mdai
                                mask_data = mdai.common_utils.convert_mask_data(binary_mask)
                                if mask_data:
                                    annotation = {
                                        'labelId': ground_truth_label_id,
                                        'StudyInstanceUID': study_uid,
                                        'SeriesInstanceUID': series_uid,
                                        'frameNumber': int(frame_idx),
                                        'data': mask_data,
                                        'groupId': label_id_machine,
                                        'note': f'Ground truth: Fluid annotation (predicted by algorithm, type: {mask_type}) - Exam #{exam_number}'
                                    }
                                    annotations_to_upload.append(annotation)
                        
                    except Exception as e:
                        print(f"Error preparing annotation for frame {frame_idx}: {str(e)}")
                        continue
                
                # Upload annotations
                if annotations_to_upload:
                    print(f"Uploading {len(annotations_to_upload)} annotations to MD.ai...")
                    
                    # Import here to avoid circular import
                    from multi_frame_tracking.utils import delete_existing_annotations
                    
                    # Delete existing ground truth annotations first
                    deleted_count = delete_existing_annotations(
                        client=mdai_client,
                        study_uid=study_uid,
                        series_uid=series_uid,
                        label_id=ground_truth_label_id,
                        group_id=label_id_machine
                    )
                    print(f"Deleted {deleted_count} existing ground truth annotations")
                    
                    try:
                        failed_annotations = mdai_client.import_annotations(
                            annotations=annotations_to_upload,
                            project_id=project_id,
                            dataset_id=dataset_id
                        )
                        
                        if failed_annotations:
                            successful_count = len(annotations_to_upload) - len(failed_annotations)
                            print(f"Uploaded {successful_count} out of {len(annotations_to_upload)} annotations")
                            results['total_upload_success'] += successful_count
                            results['total_upload_failures'] += len(failed_annotations)
                        else:
                            print(f"Successfully uploaded all {len(annotations_to_upload)} annotations")
                            results['total_upload_success'] += len(annotations_to_upload)
                            print(f"Annotations uploaded for Exam #{exam_number}")
                        
                        results['total_annotations_created'] += len(annotations_to_upload)
                        
                    except Exception as e:
                        print(f"Error during upload: {str(e)}")
                        results['total_upload_failures'] += len(annotations_to_upload)
                        results['failed_videos'].append({
                            'video': video_path,
                            'study_uid': study_uid,
                            'series_uid': series_uid,
                            'exam_number': exam_number,  
                            'error': f'Upload error: {str(e)}'
                        })
                        continue
                else:
                    print("No annotations to upload")
                    results['failed_videos'].append({
                        'video': video_path,
                        'study_uid': study_uid,
                        'series_uid': series_uid,
                        'exam_number': exam_number, 
                        'error': 'No annotations to upload'
                    })
                    continue
                
                cap.release()
                
                # Record success
                results['processed_videos'].append({
                    'video': video_path,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'exam_number': exam_number,  
                    'annotations_created': len(annotations_to_upload),
                    'upload_successful': True,
                    'issue_type': video_matched_annotations.iloc[0]['issue_type'] if len(video_matched_annotations) > 0 else 'unknown'
                })
            
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                traceback.print_exc()
                results['failed_videos'].append({
                    'video': video_path,
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'exam_number': exam_number,  
                    'error': f'Processing error: {str(e)}'
                })
                continue
        
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            traceback.print_exc()
            results['failed_videos'].append({
                'video': video_path,
                'study_uid': study_uid,
                'series_uid': series_uid,
                'exam_number': exam_number,  # Add exam number
                'error': str(e)
            })
    
    # Generate summary
    results['summary'] = {
        'total_videos': len(video_paths),
        'successful_videos': len(results['processed_videos']),
        'failed_videos': len(results['failed_videos']),
        'total_annotations_created': results['total_annotations_created'],
        'total_upload_success': results['total_upload_success'],
        'total_upload_failures': results['total_upload_failures'],
        'exam_numbers_processed': sorted(list(set([v['exam_number'] for v in results['processed_videos'] + results['failed_videos']])))
    }
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'ground_truth_creation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGround truth dataset creation complete!")
    print(f"Results saved to: {results_path}")
    print(f"Exam numbers processed: {', '.join(map(str, results['summary']['exam_numbers_processed']))}")
    
    return results


def create_feedback_loop_report(results, base_output_dir):
    """
    Create a comprehensive report comparing multiple feedback loop iterations
    
    Args:
        results: Dictionary with results from all iterations
        base_output_dir: Base output directory
    """
    report_path = os.path.join(base_output_dir, "feedback_loop_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Ground Truth Feedback Loop Report\n\n")
        
        # Create summary table
        f.write("## Summary of Iterations\n\n")
        f.write("| Iteration | Videos Processed | Annotations Created | Mean IoU | Mean Dice |\n")
        f.write("|-----------|-----------------|---------------------|----------|----------|\n")
        
        for iteration in results.get('iterations', []):
            iter_num = iteration.get('iteration_number', 0)
            gt_results = iteration.get('ground_truth_results', {})
            eval_results = iteration.get('evaluation_results', {})
            
            videos_processed = gt_results.get('summary', {}).get('total_videos', 0)
            annotations_created = gt_results.get('summary', {}).get('total_annotations_created', 0)
            
            mean_iou = eval_results.get('summary', {}).get('overall_mean_iou', 0)
            mean_dice = eval_results.get('summary', {}).get('overall_mean_dice', 0)
            
            f.write(f"| {iter_num} | {videos_processed} | {annotations_created} | {mean_iou:.4f} | {mean_dice:.4f} |\n")
        
        # Create detailed sections for each iteration
        for iteration in results.get('iterations', []):
            iter_num = iteration.get('iteration_number', 0)
            eval_results = iteration.get('evaluation_results', {})
            
            f.write(f"\n## Iteration {iter_num} Details\n\n")
            
            # Add overall metrics
            if 'summary' in eval_results:
                summary = eval_results['summary']
                f.write("### Overall Metrics\n\n")
                f.write(f"- Mean IoU: {summary.get('overall_mean_iou', 0):.4f}\n")
                f.write(f"- Mean Dice: {summary.get('overall_mean_dice', 0):.4f}\n")
                f.write(f"- Best Video: {summary.get('best_video', 'N/A')}\n")
                f.write(f"- Worst Video: {summary.get('worst_video', 'N/A')}\n\n")
            
            # List per-video metrics
            f.write("### Per-Video Metrics\n\n")
            f.write("| Video | Mean IoU | Mean Dice |\n")
            f.write("|-------|----------|----------|\n")
            
            for video_id, metrics in eval_results.items():
                if video_id != 'summary' and isinstance(metrics, dict) and 'metrics' in metrics:
                    video_metrics = metrics['metrics']
                    f.write(f"| {video_id} | {video_metrics.get('mean_iou', 0):.4f} | {video_metrics.get('mean_dice', 0):.4f} |\n")
    
    # Create visualization
    if results.get('iterations', []):
        # Plot IoU progression across iterations
        iterations = [i.get('iteration_number', 0) for i in results['iterations']]
        ious = [i.get('evaluation_results', {}).get('summary', {}).get('overall_mean_iou', 0) for i in results['iterations']]
        dices = [i.get('evaluation_results', {}).get('summary', {}).get('overall_mean_dice', 0) for i in results['iterations']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, ious, 'b-', label='Mean IoU')
        plt.plot(iterations, dices, 'g-', label='Mean Dice')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Performance Metrics Across Feedback Loop Iterations')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        chart_path = os.path.join(base_output_dir, "feedback_loop_metrics.png")
        plt.savefig(chart_path)
    
    return report_path