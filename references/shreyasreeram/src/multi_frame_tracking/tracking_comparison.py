import cv2
import os
import numpy as np





class SingleFrameTracker:
    """Simple single-frame tracker for comparison purposes."""
    
    def __init__(self, flow_processor, output_dir, shared_params=None):
        self.flow_processor = flow_processor
        self.output_dir = output_dir
        self.shared_params = shared_params
    
    def process_single_annotation(self, annotations_df, video_path, study_uid, series_uid):
        """Take ONE annotation and track across entire video."""
        
        if len(annotations_df) == 0:
            return {}
        
        # Select first annotation
        selected_annotation = annotations_df.iloc[0]
        frame_number = int(selected_annotation['frameNumber'])
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Create initial mask
        if 'free_fluid_foreground' in selected_annotation:
            polygons = selected_annotation['free_fluid_foreground']
            initial_mask = self._create_mask_from_polygons(polygons, frame_height, frame_width)
        else:
            cap.release()
            return {}
        
        all_masks = {}
        
        # Add original annotation
        all_masks[frame_number] = {
            'mask': initial_mask.copy(),
            'type': 'single_frame_annotation',
            'is_annotation': True,
            'source': 'original_annotation'
        }
        
        # Track backward and forward using your existing track_frames function
        from .utils import track_frames
        
        debug_dir = os.path.join(self.output_dir, 'single_frame_debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Backward tracking
        if frame_number > 0:
            backward_frames = track_frames(
                cap, frame_number, 0, initial_mask, debug_dir,
                forward=False, pbar=None, flow_processor=self.flow_processor,
                shared_params=self.shared_params
            )
            
            for frame_data in backward_frames:
                if len(frame_data) >= 3:
                    frame_idx, frame, mask = frame_data[0], frame_data[1], frame_data[2]
                    if frame_idx != frame_number:
                        all_masks[frame_idx] = {
                            'mask': mask,
                            'type': 'single_frame_tracked',
                            'is_annotation': False,
                            'source': f'backward_from_{frame_number}'
                        }
        
        # Forward tracking
        if frame_number < total_frames - 1:
            forward_frames = track_frames(
                cap, frame_number, total_frames - 1, initial_mask, debug_dir,
                forward=True, pbar=None, flow_processor=self.flow_processor,
                shared_params=self.shared_params
            )
            
            for frame_data in forward_frames:
                if len(frame_data) >= 3:
                    frame_idx, frame, mask = frame_data[0], frame_data[1], frame_data[2]
                    if frame_idx != frame_number:
                        all_masks[frame_idx] = {
                            'mask': mask,
                            'type': 'single_frame_tracked',
                            'is_annotation': False,
                            'source': f'forward_from_{frame_number}'
                        }
        
        cap.release()
        return all_masks
    
    def _create_mask_from_polygons(self, polygons, height, width):
        """Create mask from polygons."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask


def compare_single_vs_multi_frame(single_results, multi_results):
    """Simple comparison function that returns IoU metrics."""
    
    single_frames = set(single_results.keys())
    multi_frames = set(multi_results.keys())
    common_frames = single_frames & multi_frames
    
    if not common_frames:
        return {'error': 'No common frames to compare'}
    
    ious = []
    for frame_idx in common_frames:
        # Extract masks
        single_mask = single_results[frame_idx]['mask'] if isinstance(single_results[frame_idx], dict) else single_results[frame_idx]
        multi_mask = multi_results[frame_idx]['mask'] if isinstance(multi_results[frame_idx], dict) else multi_results[frame_idx]
        
        # Calculate IoU
        single_binary = (single_mask > 0.5).astype(np.uint8)
        multi_binary = (multi_mask > 0.5).astype(np.uint8)
        
        intersection = np.sum(np.logical_and(single_binary, multi_binary))
        union = np.sum(np.logical_or(single_binary, multi_binary))
        
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
    
    return {
        'mean_iou': float(np.mean(ious)) if ious else 0.0,
        'median_iou': float(np.median(ious)) if ious else 0.0,
        'common_frames': len(common_frames),
        'single_frame_count': len(single_frames),
        'multi_frame_count': len(multi_frames)
    }
