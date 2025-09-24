"""
Enhanced multi-frame tracker with distance-weighted mask combining.
"""

def combine_masks_with_distance_weighting(mask_sources, current_frame):
    """
    Combine multiple tracked masks at a frame using distance-based weighting.

    Args:
        mask_sources: List of dicts with 'mask', 'source_frame', and optionally 'type'
        current_frame: The frame number we're combining masks for

    Returns:
        Combined mask using weighted average based on distance from source annotations
    """
    import numpy as np

    if not mask_sources:
        return None

    if len(mask_sources) == 1:
        return mask_sources[0]['mask']

    # Get dimensions from first mask
    h, w = mask_sources[0]['mask'].shape
    combined_mask = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for source in mask_sources:
        mask = source['mask'].astype(np.float32) / 255.0
        source_frame = source['source_frame']

        # Calculate weight based on distance from source annotation
        distance = abs(current_frame - source_frame)

        # Use exponential decay for weight (closer = higher weight)
        # Weight = exp(-alpha * distance) where alpha controls decay rate
        alpha = 0.05  # Tune this: smaller = slower decay, larger = faster decay
        weight = np.exp(-alpha * distance)

        # Alternative: inverse distance weighting
        # weight = 1.0 / (1.0 + distance)

        # Apply weight to mask
        combined_mask += mask * weight
        weight_sum += weight * np.ones_like(mask)

    # Normalize by weight sum to get weighted average
    # Avoid division by zero
    weight_sum = np.maximum(weight_sum, 1e-8)
    combined_mask = combined_mask / weight_sum

    # Convert back to uint8
    combined_mask = (combined_mask * 255).astype(np.uint8)

    # Apply threshold to get binary mask if needed
    threshold = 127  # Can be tuned
    combined_mask = np.where(combined_mask > threshold, 255, 0).astype(np.uint8)

    return combined_mask


def process_annotations_with_weighted_combining(self, annotations, video_path, study_uid, series_uid):
    """
    Process annotations with proper distance-weighted combining of overlapping tracks.

    This replaces the simple all_masks.update() approach with intelligent merging.
    """
    import numpy as np

    # ... initialization code ...

    # Dictionary to store all mask sources for each frame
    # Each frame will have a list of masks from different source annotations
    frame_mask_sources = {}  # frame_num -> [{'mask': mask, 'source_frame': annotation_frame}, ...]

    # Process each annotation
    for i, annotation in enumerate(annotations):
        if annotation['type'] == 'fluid':
            print(f"Processing annotation {i+1}/{len(annotations)} at frame {annotation['frame']}")

            mask = annotation['mask']
            ann_frame = annotation['frame']

            # Track forward
            if ann_frame < total_frames - 1:
                forward_masks = self._track_forward(ann_frame, total_frames - 1, mask)
                for frame_num, mask_info in forward_masks.items():
                    if frame_num not in frame_mask_sources:
                        frame_mask_sources[frame_num] = []
                    frame_mask_sources[frame_num].append({
                        'mask': mask_info['mask'],
                        'source_frame': ann_frame,
                        'type': 'forward_track'
                    })

            # Track backward
            if ann_frame > 0:
                backward_masks = self._track_backward(ann_frame, 0, mask)
                for frame_num, mask_info in backward_masks.items():
                    if frame_num not in frame_mask_sources:
                        frame_mask_sources[frame_num] = []
                    frame_mask_sources[frame_num].append({
                        'mask': mask_info['mask'],
                        'source_frame': ann_frame,
                        'type': 'backward_track'
                    })

            # Add the original annotation
            if ann_frame not in frame_mask_sources:
                frame_mask_sources[ann_frame] = []
            frame_mask_sources[ann_frame].append({
                'mask': mask,
                'source_frame': ann_frame,
                'type': 'annotation'
            })

    # Now combine all masks using distance weighting
    print("Combining overlapping masks with distance weighting...")
    all_masks = {}

    for frame_num, mask_sources in frame_mask_sources.items():
        if len(mask_sources) == 1:
            # Only one source, use it directly
            all_masks[frame_num] = {
                'mask': mask_sources[0]['mask'],
                'type': mask_sources[0]['type'],
                'source_frame': mask_sources[0]['source_frame']
            }
        else:
            # Multiple sources, combine with weighting
            combined_mask = combine_masks_with_distance_weighting(mask_sources, frame_num)
            all_masks[frame_num] = {
                'mask': combined_mask,
                'type': 'combined',
                'num_sources': len(mask_sources),
                'source_frames': [s['source_frame'] for s in mask_sources]
            }

    return all_masks