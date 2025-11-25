"""
Debug visualization utilities for optical flow tracking.

Creates multipane grid visualizations showing:
- Original frame
- Initial mask (annotation)
- Flow mask (warped)
- Mask difference
- Adjusted mask
- Final mask
"""

import numpy as np
import cv2


def create_debug_visualization(frame, initial_mask, flow_mask, final_mask, frame_number):
    """
    Create 2x3 grid debug visualization showing tracking evolution.

    Args:
        frame: Original BGR frame
        initial_mask: Initial mask from annotation (float 0-1)
        flow_mask: Mask warped by optical flow (float 0-1)
        final_mask: Final refined mask (float 0-1)
        frame_number: Frame number for labeling

    Returns:
        grid: 2x3 grid visualization (height*2, width*3, 3)
    """
    h, w = frame.shape[:2]

    # Create 2x3 grid
    grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)

    # Convert masks to binary for contour detection
    binary_initial = (initial_mask > 0.5).astype(np.uint8) if initial_mask is not None else None
    binary_flow = (flow_mask > 0.5).astype(np.uint8) if flow_mask is not None else None
    binary_final = (final_mask > 0.5).astype(np.uint8) if final_mask is not None else None

    # Row 1, Col 1: Original Frame
    grid[:h, :w] = frame
    cv2.putText(grid, f"Frame {frame_number}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Row 1, Col 2: Initial Mask (annotation)
    if initial_mask is not None:
        initial_viz = frame.copy()

        # Red overlay for initial mask
        initial_viz[initial_mask > 0.5] = (
            initial_viz[initial_mask > 0.5] * 0.7 +
            np.array([0, 0, 255], dtype=np.uint8) * 0.3
        )

        # Add contours
        if binary_initial is not None:
            contours, _ = cv2.findContours(binary_initial, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(initial_viz, contours, -1, (0, 255, 255), 2)

            # Add metrics
            initial_area = np.sum(binary_initial)
            cv2.putText(initial_viz, f"Area: {initial_area}", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        grid[:h, w:w*2] = initial_viz

    cv2.putText(grid, "Initial Mask", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Row 1, Col 3: Flow Mask (warped)
    if flow_mask is not None:
        flow_viz = frame.copy()

        # Blue overlay for flow mask
        flow_viz[flow_mask > 0.5] = (
            flow_viz[flow_mask > 0.5] * 0.7 +
            np.array([255, 0, 0], dtype=np.uint8) * 0.3
        )

        # Add contours
        if binary_flow is not None:
            contours, _ = cv2.findContours(binary_flow, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(flow_viz, contours, -1, (0, 255, 255), 2)

            # Add metrics
            flow_area = np.sum(binary_flow)
            cv2.putText(flow_viz, f"Area: {flow_area}", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add IoU if both masks exist
            if binary_initial is not None:
                intersection = np.sum(np.logical_and(binary_initial, binary_flow))
                union = np.sum(np.logical_or(binary_initial, binary_flow))
                iou = intersection / union if union > 0 else 0
                cv2.putText(flow_viz, f"IoU: {iou:.3f}", (10, h-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        grid[:h, w*2:w*3] = flow_viz

    cv2.putText(grid, "Flow Mask", (w*2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Row 2, Col 1: Mask Difference
    if initial_mask is not None and flow_mask is not None:
        diff_viz = frame.copy()

        # Show where masks differ
        diff_mask = np.zeros_like(binary_initial)
        diff_mask[(binary_initial > 0) & (binary_flow == 0)] = 1  # Initial only - red
        diff_mask[(binary_initial == 0) & (binary_flow > 0)] = 2  # Flow only - blue

        # Apply difference visualization
        diff_viz[diff_mask == 1] = (
            diff_viz[diff_mask == 1] * 0.7 +
            np.array([0, 0, 255], dtype=np.uint8) * 0.3
        )
        diff_viz[diff_mask == 2] = (
            diff_viz[diff_mask == 2] * 0.7 +
            np.array([255, 0, 0], dtype=np.uint8) * 0.3
        )

        # Add metrics
        diff_count = np.sum(diff_mask > 0)
        cv2.putText(diff_viz, f"Diff: {diff_count} px", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(diff_viz, "Red: Init only", (10, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(diff_viz, "Blue: Flow only", (10, h-80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        grid[h:h*2, :w] = diff_viz

    cv2.putText(grid, "Difference", (10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Row 2, Col 2: Combined Overlay
    combined_viz = frame.copy()
    if initial_mask is not None and flow_mask is not None:
        # Show both masks with different colors
        combined_viz[initial_mask > 0.5] = (
            combined_viz[initial_mask > 0.5] * 0.7 +
            np.array([0, 0, 255], dtype=np.uint8) * 0.3  # Red for initial
        )
        combined_viz[flow_mask > 0.5] = (
            combined_viz[flow_mask > 0.5] * 0.5 +
            np.array([255, 0, 0], dtype=np.uint8) * 0.5  # Blue for flow
        )
        # Overlap will appear purple

    grid[h:h*2, w:w*2] = combined_viz
    cv2.putText(grid, "Combined", (w + 10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Row 2, Col 3: Final Mask
    if final_mask is not None:
        final_viz = frame.copy()

        # Green overlay for final mask
        final_viz[final_mask > 0.5] = (
            final_viz[final_mask > 0.5] * 0.7 +
            np.array([0, 255, 0], dtype=np.uint8) * 0.3
        )

        # Add contours
        if binary_final is not None:
            contours, _ = cv2.findContours(binary_final, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(final_viz, contours, -1, (0, 255, 255), 2)

            # Add metrics
            final_area = np.sum(binary_final)
            cv2.putText(final_viz, f"Area: {final_area}", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add comparison to initial if exists
            if binary_initial is not None:
                initial_area = np.sum(binary_initial)
                area_ratio = final_area / initial_area if initial_area > 0 else 0
                cv2.putText(final_viz, f"Ratio: {area_ratio:.3f}", (10, h-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        grid[h:h*2, w*2:w*3] = final_viz

    cv2.putText(grid, "Final Mask", (w*2 + 10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return grid


def create_flow_visualization(frame, flow, skip=10):
    """
    Create HSV color-coded flow visualization with vector arrows.

    Args:
        frame: Original BGR frame
        flow: Optical flow field (H, W, 2)
        skip: Step size for drawing flow vectors

    Returns:
        flow_vis: Flow visualization (same size as frame)
    """
    h, w = frame.shape[:2]

    # Extract flow components
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    # Create HSV image for flow visualization
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
    hsv[..., 1] = 255                      # Saturation = full
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Blend with original frame
    flow_vis = cv2.addWeighted(frame, 0.5, flow_vis, 0.5, 0)

    # Draw flow vectors
    for y in range(0, h, skip):
        for x in range(0, w, skip):
            fx, fy = flow[y, x]

            # Only draw significant flow
            if np.sqrt(fx*fx + fy*fy) > 1:
                cv2.arrowedLine(flow_vis, (x, y),
                               (int(x + fx), int(y + fy)),
                               (0, 255, 0), 1, tipLength=0.3)

    # Add legend
    cv2.putText(flow_vis, "Flow Visualization", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(flow_vis, "Hue=Direction, Brightness=Magnitude", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return flow_vis
