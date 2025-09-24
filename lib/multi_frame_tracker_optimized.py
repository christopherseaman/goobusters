"""
Optimized multi-frame tracker for handling many annotations efficiently.
"""
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv('dot.env')
LABEL_ID = os.getenv('LABEL_ID', 'L_13yPql')  # Human-annotated free fluid
TRACK_ID = os.getenv('TRACK_ID', 'L_JykNe7')   # Machine-annotated free fluid

def process_annotations_optimized(annotations, video_path, flow_processor, output_dir):
    """
    Optimized processing of annotations with efficient segment handling.

    Key optimizations:
    1. Only process small segments between nearby annotations
    2. Skip large gaps to prevent timeout
    3. Better progress tracking
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Write initial checkpoint
    checkpoint_file = os.path.join(output_dir, "checkpoint.txt")
    with open(checkpoint_file, "w") as f:
        f.write(f"Process started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Annotations count: {len(annotations)}\n\n")

    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video: {total_frames} frames, {frame_width}x{frame_height}")
    print(f"Processing {len(annotations)} annotations")

    # Sort annotations by frame
    fluid_annotations = sorted([a for a in annotations if a['type'] == 'fluid'],
                               key=lambda x: x['frame'])

    if not fluid_annotations:
        print("No fluid annotations found")
        return {}

    print(f"Found {len(fluid_annotations)} fluid annotations")

    # Results dictionary
    all_masks = {}

    # Add original annotations to results with human-annotated label
    for ann in fluid_annotations:
        all_masks[ann['frame']] = {
            'mask': ann['mask'],
            'type': 'fluid',
            'is_annotation': True,
            'label_id': LABEL_ID  # Human-annotated label
        }

    # Process segments between consecutive annotations
    MAX_SEGMENT_SIZE = 30  # Only process segments up to 30 frames
    MAX_TRACKING_DISTANCE = 15  # Track maximum 15 frames in each direction

    segments_processed = 0
    segments_skipped = 0

    for i in range(len(fluid_annotations) - 1):
        current_ann = fluid_annotations[i]
        next_ann = fluid_annotations[i + 1]

        gap = next_ann['frame'] - current_ann['frame']

        # Update checkpoint periodically
        if i % 5 == 0:
            with open(checkpoint_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Processing segment {i+1}/{len(fluid_annotations)-1}\n")

        if gap <= 1:
            continue  # No gap to fill

        if gap > MAX_SEGMENT_SIZE:
            print(f"  Skipping large segment {current_ann['frame']}->{next_ann['frame']} ({gap} frames)")
            segments_skipped += 1
            continue

        print(f"  Processing segment {current_ann['frame']}->{next_ann['frame']} ({gap} frames)")

        # Process this segment with bidirectional tracking
        process_segment_bidirectional(
            video_path, flow_processor,
            current_ann['frame'], next_ann['frame'],
            current_ann['mask'], next_ann['mask'],
            all_masks, MAX_TRACKING_DISTANCE
        )

        segments_processed += 1

    # Summary
    print(f"\nProcessing complete:")
    print(f"  Segments processed: {segments_processed}")
    print(f"  Segments skipped: {segments_skipped}")
    print(f"  Total frames with masks: {len(all_masks)}")
    print(f"  Coverage: {len(all_masks)/total_frames*100:.1f}%")

    # Write final checkpoint
    with open(checkpoint_file, "a") as f:
        f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing complete\n")
        f.write(f"Segments processed: {segments_processed}\n")
        f.write(f"Segments skipped: {segments_skipped}\n")
        f.write(f"Total masks: {len(all_masks)}\n")

    return all_masks


def process_segment_bidirectional(video_path, flow_processor,
                                 start_frame, end_frame,
                                 start_mask, end_mask,
                                 all_masks, max_distance):
    """
    Process a segment using bidirectional tracking with distance weighting.
    """
    gap = end_frame - start_frame
    if gap <= 1:
        return

    # Open video once for this segment
    cap = cv2.VideoCapture(video_path)

    # Forward tracking from start
    forward_masks = {}
    if start_mask is not None:
        track_distance = min(max_distance, gap - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_mask = start_mask.copy()
        ret, prev_frame = cap.read()

        if ret:
            for offset in range(1, track_distance + 1):
                ret, curr_frame = cap.read()
                if not ret:
                    break

                # Calculate optical flow
                flow = flow_processor.calculate_flow(prev_frame, curr_frame)
                current_mask = warp_mask_with_flow(current_mask, flow)
                forward_masks[start_frame + offset] = current_mask
                prev_frame = curr_frame

    # Backward tracking from end
    backward_masks = {}
    if end_mask is not None:
        track_distance = min(max_distance, gap - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)

        current_mask = end_mask.copy()
        ret, prev_frame = cap.read()

        if ret:
            for offset in range(1, track_distance + 1):
                frame_idx = end_frame - offset
                if frame_idx <= start_frame:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, curr_frame = cap.read()
                if not ret:
                    break

                # Calculate backward flow
                flow = flow_processor.calculate_flow(curr_frame, prev_frame)
                current_mask = warp_mask_with_flow(current_mask, flow)
                backward_masks[frame_idx] = current_mask
                prev_frame = curr_frame

    cap.release()

    # Combine masks with distance weighting
    for frame_idx in range(start_frame + 1, end_frame):
        if frame_idx in forward_masks and frame_idx in backward_masks:
            # Calculate weights based on distance from source
            total_dist = end_frame - start_frame
            forward_weight = (end_frame - frame_idx) / total_dist
            backward_weight = (frame_idx - start_frame) / total_dist

            # Combine masks
            combined = combine_masks_weighted(
                forward_masks[frame_idx],
                backward_masks[frame_idx],
                forward_weight,
                backward_weight
            )

            all_masks[frame_idx] = {
                'mask': combined,
                'type': 'combined',
                'is_annotation': False,
                'label_id': TRACK_ID  # Machine-annotated label
            }
        elif frame_idx in forward_masks:
            all_masks[frame_idx] = {
                'mask': forward_masks[frame_idx],
                'type': 'forward',
                'is_annotation': False,
                'label_id': TRACK_ID  # Machine-annotated label
            }
        elif frame_idx in backward_masks:
            all_masks[frame_idx] = {
                'mask': backward_masks[frame_idx],
                'type': 'backward',
                'is_annotation': False,
                'label_id': TRACK_ID  # Machine-annotated label
            }


def warp_mask_with_flow(mask, flow):
    """Warp a mask using optical flow."""
    h, w = mask.shape
    flow_map = np.column_stack((
        np.repeat(np.arange(h), w),
        np.tile(np.arange(w), h)
    )).reshape(h, w, 2).astype(np.float32)

    flow_map += flow
    warped = cv2.remap(mask, flow_map[:,:,1], flow_map[:,:,0],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Threshold to maintain binary mask
    _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
    return warped.astype(np.uint8)


def combine_masks_weighted(mask1, mask2, weight1, weight2):
    """Combine two masks with weighted average."""
    # Normalize weights
    total = weight1 + weight2
    if total > 0:
        weight1 /= total
        weight2 /= total
    else:
        weight1 = weight2 = 0.5

    # Weighted combination
    combined = (mask1.astype(np.float32) * weight1 +
                mask2.astype(np.float32) * weight2)

    # Threshold to binary
    _, result = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
    return result.astype(np.uint8)