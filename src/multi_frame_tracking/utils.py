import cv2
import numpy as np
import os
from datetime import datetime
import traceback



DEBUG_MODE = True

def print_mask_stats(mask, frame_num):
    """Print coverage statistics for a mask"""
    coverage = np.mean(mask) * 100
    print(f"Frame {frame_num} - Mask coverage: {coverage:.2f}%")

def visualize_flow(frame, flow, skip=8):
    """
    Visualize optical flow for debugging
    
    Args:
        frame: Input frame
        flow: Flow field (x and y displacements)
        skip: Spacing between displayed vectors
        
    Returns:
        Visualization image
    """
    h, w = frame.shape[:2]
    
    # Create empty visualization image
    vis = np.zeros((h * 2, w, 3), dtype=np.uint8)
    
    # Copy original frame to top half
    if len(frame.shape) == 2:  # Convert grayscale to color if needed
        vis[:h, :] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis[:h, :] = frame.copy()
    
    # Create flow visualization
    # Calculate flow magnitude and angle
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    
    # Create HSV image for flow visualization
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255                       # Saturation: max
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
    
    # Convert HSV to BGR
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Copy flow visualization to bottom half
    vis[h:, :] = flow_vis
    
    # Draw flow vectors on top half
    for y in range(0, h, skip):
        for x in range(0, w, skip):
            fx, fy = flow[y, x]
            
            # Only draw significant flow
            if np.sqrt(fx*fx + fy*fy) > 1:
                cv2.arrowedLine(vis[:h], (x, y), (int(x+fx), int(y+fy)), (0, 255, 0), 1, tipLength=0.3)
    
    # Add labels
    cv2.putText(vis, "Original frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(vis, "Flow visualization", (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return vis

def debug_visualize(frame, initial_mask, flow_mask, adjusted_mask, final_mask, frame_number, flow=None):
    """
    Enhanced debug visualization with improved visualization to match numerical data.
    """
    h, w = frame.shape[:2]
    
    # Create a larger grid for visualization including flow
    if flow is not None:
        grid = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)  # 3x3 grid
    else:
        grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)  # 2x3 grid (original size)
    
    # Convert masks to binary for contour detection and area calculation
    binary_initial = (initial_mask > 0.5).astype(np.uint8) if initial_mask is not None else None
    binary_flow = (flow_mask > 0.5).astype(np.uint8) if flow_mask is not None else None
    binary_adjusted = (adjusted_mask > 0.5).astype(np.uint8) if adjusted_mask is not None else None
    binary_final = (final_mask > 0.5).astype(np.uint8) if final_mask is not None else None
    
    # Original frame (Top Left)
    grid[:h, :w] = frame
    cv2.putText(grid, "Original Frame", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Initial Mask (Top Middle)
    if initial_mask is not None:
        # Create visualization using thresholding
        initial_viz = frame.copy()
        
        # Only color pixels above threshold
        initial_viz[initial_mask > 0.5] = initial_viz[initial_mask > 0.5] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
        
        # Add contours to initial mask
        if binary_initial is not None:
            contours, _ = cv2.findContours(binary_initial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(initial_viz, contours, -1, (0, 255, 255), 1)  # Yellow contour
            
            # Add area metrics
            initial_area = np.sum(binary_initial)
            cv2.putText(initial_viz, f"Area: {initial_area}", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(initial_viz, f"Sum: {np.sum(initial_mask):.1f}", (10, h-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        grid[:h, w:w*2] = initial_viz
    cv2.putText(grid, "Initial Mask", (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Flow Mask (Top Right)
    if flow_mask is not None:
        flow_viz = frame.copy()
        
        # Only color pixels above threshold
        flow_viz[flow_mask > 0.5] = flow_viz[flow_mask > 0.5] * 0.7 + np.array([255, 0, 0], dtype=np.uint8) * 0.3
        
        # Add contours to flow mask
        if binary_flow is not None:
            contours, _ = cv2.findContours(binary_flow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(flow_viz, contours, -1, (0, 255, 255), 1)  # Yellow contour
            
            # Add area metrics
            flow_area = np.sum(binary_flow)
            cv2.putText(flow_viz, f"Area: {flow_area}", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(flow_viz, f"Sum: {np.sum(flow_mask):.1f}", (10, h-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Add IoU if both masks exist
            if binary_initial is not None:
                intersection = np.sum(np.logical_and(binary_initial, binary_flow))
                union = np.sum(np.logical_or(binary_initial, binary_flow))
                iou = intersection / union if union > 0 else 0
                cv2.putText(flow_viz, f"IoU: {iou:.4f}", (10, h-80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Add area ratio
                area_ratio = flow_area / initial_area if initial_area > 0 else 0
                cv2.putText(flow_viz, f"Ratio: {area_ratio:.4f}", (10, h-110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        grid[:h, w*2:w*3] = flow_viz
    cv2.putText(grid, "Flow Mask", (w*2 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create Difference Visualization (Middle Left)
    # Shows where masks differ by highlighting differences
    if initial_mask is not None and flow_mask is not None:
        diff_viz = frame.copy()
        
        # Create a mask that shows where the masks differ
        diff_mask = np.zeros_like(binary_initial)
        diff_mask[(binary_initial > 0) & (binary_flow == 0)] = 1  # Initial only - show in red
        diff_mask[(binary_initial == 0) & (binary_flow > 0)] = 2  # Flow only - show in blue
        
        # Apply the difference visualization
        diff_viz[diff_mask == 1] = diff_viz[diff_mask == 1] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3  # Red
        diff_viz[diff_mask == 2] = diff_viz[diff_mask == 2] * 0.7 + np.array([255, 0, 0], dtype=np.uint8) * 0.3  # Blue
        
        # Add metrics
        diff_count = np.sum(diff_mask > 0)
        cv2.putText(diff_viz, f"Diff pixels: {diff_count}", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(diff_viz, f"Red: Initial only", (10, h-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(diff_viz, f"Blue: Flow only", (10, h-80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        
        grid[h:h*2, :w] = diff_viz
    cv2.putText(grid, "Mask Difference", (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Adjusted Mask (Middle Middle)
    if adjusted_mask is not None:
        adjusted_viz = frame.copy()
        
        # Only colour pixels above threshold (0.5)
        adjusted_viz[adjusted_mask > 0.5] = adjusted_viz[adjusted_mask > 0.5] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
        
        # Add contours
        if binary_adjusted is not None:
            contours, _ = cv2.findContours(binary_adjusted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(adjusted_viz, contours, -1, (0, 255, 255), 1)  # Yellow contour
        
        grid[h:h*2, w:w*2] = adjusted_viz
    cv2.putText(grid, "Adjusted Mask", (w + 10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Final Mask (Middle Right)
    if final_mask is not None:
        final_viz = frame.copy()
        
        # Only color pixels above threshold
        final_viz[final_mask > 0.5] = final_viz[final_mask > 0.5] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
        
        # Add contours to final mask
        if binary_final is not None:
            contours, _ = cv2.findContours(binary_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(final_viz, contours, -1, (0, 255, 255), 1)  # Yellow contour
            
            # Add area metrics
            final_area = np.sum(binary_final)
            cv2.putText(final_viz, f"Area: {final_area}", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(final_viz, f"Sum: {np.sum(final_mask):.1f}", (10, h-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Add comparison to initial mask if it exists
            if binary_initial is not None:
                initial_area = np.sum(binary_initial)
                area_ratio = final_area / initial_area if initial_area > 0 else 0
                cv2.putText(final_viz, f"Area ratio: {area_ratio:.4f}", (10, h-80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        grid[h:h*2, w*2:w*3] = final_viz
    cv2.putText(grid, "Final Mask", (w*2 + 10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add flow visualization (Bottom row) if flow is provided
    if flow is not None:
        try:
            # Flow Vectors (Bottom Left)
            flow_vis = visualize_flow(frame, flow, skip=8)
            
            # If flow_vis has expected two-part format (vectors+heatmap)
            if flow_vis.shape[0] > h:
                # Top part: Flow Vectors
                grid[h*2:h*3, :w] = flow_vis[:h]
                cv2.putText(grid, "Flow Vectors", (10, h*2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Bottom part: Flow Heatmap (will be cropped if too large)
                heatmap_h = min(h, flow_vis.shape[0] - h)
                grid[h*2:h*2+heatmap_h, w:w*2] = flow_vis[h:h+heatmap_h]
                cv2.putText(grid, "Flow Heatmap", (w + 10, h*2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # If visualization format is different, just use what we have
                grid[h*2:h*3, :w] = flow_vis
                cv2.putText(grid, "Flow Visualization", (10, h*2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Mask with Flow Vectors (Bottom Right)
            mask_with_vectors = frame.copy()
            # Add mask overlay
            mask_with_vectors[final_mask > 0.5] = mask_with_vectors[final_mask > 0.5] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
            
            # Draw vectors only in mask area
            if final_mask is not None:
                # Calculate magnitude for thresholding
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
                
                # Draw arrows in mask region
                for y in range(0, h, 8):  # Smaller skip for more arrows
                    for x in range(0, w, 8):
                        if final_mask[y, x] > 0.5 and mag_norm[y, x] > 0.05:  # Only in mask and with sufficient magnitude
                            fx = flow[y, x, 0]
                            fy = flow[y, x, 1]
                            # Use yellow for better visibility on green mask
                            cv2.arrowedLine(mask_with_vectors, (x, y), 
                                         (int(x + fx), int(y + fy)),
                                         (0, 255, 255), 2, tipLength=0.3)
            
            grid[h*2:h*3, w*2:w*3] = mask_with_vectors
            cv2.putText(grid, "Mask + Flow Vectors", (w*2 + 10, h*2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
        except Exception as e:
            print(f"Error creating flow visualization: {str(e)}")
            
    # Add frame number and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(grid, f"Frame: {frame_number} | {timestamp}", 
                (10, grid.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return grid

def polygons_to_mask(polygons, frame_height, frame_width):
    """Convert polygon points to a binary mask"""
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for polygon in polygons:
        clipped_polygon = [
            [max(0, min(point[0], frame_width - 1)), max(0, min(point[1], frame_height - 1))]
            for point in polygon
        ]
        points = np.array(clipped_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

def verify_directory(directory_path):
    """Create directory if it doesn't exist and verify it's accessible"""
    os.makedirs(directory_path, exist_ok=True)
    if not os.path.exists(directory_path):
        print(f"ERROR: Could not create directory: {directory_path}")
        return False
    else:
        print(f"Using output directory: {directory_path}")
        return True

def verify_video_output(output_path):
    """Verify that an output video exists and can be opened"""
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Confirmed: Output video created at {output_path} ({file_size} bytes)")
        
        # Try to open the video to make sure it's valid
        try:
            cap = cv2.VideoCapture(output_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Video can be opened, contains {frame_count} frames")
                cap.release()
                return True
            else:
                print(f"WARNING: Video file created but cannot be opened with OpenCV")
                return False
        except Exception as e:
            print(f"Error checking video: {e}")
            return False
    else:
        print(f"WARNING: Output video not found at expected path: {output_path}")
        return False

def track_frames(cap, start_frame, end_frame, initial_mask, debug_dir=None, forward=True, pbar=None, flow_processor=None):
    """
    Track frames using optical flow with enhanced error handling and diagnostics.
    
    Args:
        cap: Video capture object
        start_frame: Starting frame number
        end_frame: Ending frame number
        initial_mask: Initial mask to track
        debug_dir: Directory for debug outputs (optional)
        forward: If True, track forward; if False, track backward
        pbar: Progress bar object
        flow_processor: Optical flow processor object
    
    Returns:
        List of tuples (frame_idx, frame, mask, flow, flow_mask, adjusted_mask)
    """
    # Define target frames for detailed analysis
    TARGET_FRAMES = [137, 138, 139]  # Add specific frame numbers you want to analyze
    VERBOSE_DEBUGGING = True  # Set to True for verbose output on all frames
    DEBUG_MODE = True  # Set debug mode locally to ensure it's available
    
    # Set a default debug directory if none provided
    if debug_dir is None:
        debug_dir = os.path.join(os.getcwd(), "debug_frames")
    
    # Ensure the debug directory exists
    os.makedirs(debug_dir, exist_ok=True)
    
    frames = []
    step = 1 if forward else -1
    frame_idx = start_frame
    direction = "forward" if forward else "backward"
    consecutive_errors = 0
    max_consecutive_errors = 3
    total_frames_processed = 0
    total_frames_skipped = 0
    
    print(f"\nStarting {direction} tracking from frame {start_frame} to {end_frame}")
    print(f"Initial frame shape: {initial_mask.shape}")
    
    # Set the video capture to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Failed to read starting frame {start_frame}.")
        return frames
        
    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        mask = initial_mask.astype(float)
        
        # Save initial frame and mask for debugging - only if debug_dir is valid
        try:
            if os.path.exists(debug_dir):
                debug_path = os.path.join(debug_dir, f'initial_frame_{frame_idx:04d}.png')
                cv2.imwrite(debug_path, prev_frame)
                debug_path = os.path.join(debug_dir, f'initial_mask_{frame_idx:04d}.png')
                cv2.imwrite(debug_path, (mask * 255).astype(np.uint8))
        except Exception as e:
            print(f"Warning: Could not save debug images: {str(e)}")
    except Exception as e:
        print(f"Error initializing first frame: {str(e)}")
        return frames

    while (forward and frame_idx <= end_frame) or (not forward and frame_idx >= end_frame):
        try:
            # Print progress every 10 frames
            if total_frames_processed % 10 == 0:
                print(f"\nProcessing frame {frame_idx}")
                print(f"Frames processed: {total_frames_processed}")
                print(f"Frames skipped: {total_frames_skipped}")
                print(f"Consecutive errors: {consecutive_errors}")
            
            # Read the current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}. Ending tracking.")
                break

            # Convert to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Initialize for first frame
            if frame_idx == start_frame:
                flow = None
                flow_mask = np.zeros_like(mask)
                adjusted_mask = np.zeros_like(mask)
                new_mask = mask.copy()
                print_mask_stats(mask, frame_idx)  # Add coverage stats
                frames.append((frame_idx, frame, new_mask, flow, flow_mask, adjusted_mask))  # Modified: added flow_mask and adjusted_mask
                total_frames_processed += 1
            else:
                # Apply optical flow
                try:
                    # Get flow field
                    flow = flow_processor.apply_optical_flow(prev_gray, frame_gray, mask)
                    if flow is None:
                        print(f"Flow computation returned None for frame {frame_idx}")
                        consecutive_errors += 1
                        total_frames_skipped += 1
                        frame_idx += step
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"Too many consecutive errors, stopping at frame {frame_idx}")
                            break
                        continue
                        
                    # Calculate flow mask
                    flow_mask = flow_processor.warp_mask(mask, flow)
                    
                    # Verify flow mask is valid
                    if flow_mask is None or np.isnan(flow_mask).any():
                        print(f"Invalid flow mask at frame {frame_idx}")
                        consecutive_errors += 1
                        total_frames_skipped += 1
                        frame_idx += step
                        continue
                        
                    # Calculate flow metrics for debugging
                    mean_flow = np.mean(np.abs(flow))
                    if mean_flow < 0.01:
                        print(f"Warning: Very small flow detected at frame {frame_idx}: {mean_flow}")
                    
                    # Calculate binary mask areas and IoU
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    binary_flow_mask = (flow_mask > 0.5).astype(np.uint8)
                    mask_area = np.sum(binary_mask)
                    flow_mask_area = np.sum(binary_flow_mask)
                    area_ratio = flow_mask_area / mask_area if mask_area > 0 else 0
                    
                    # Calculate IoU
                    intersection = np.sum(np.logical_and(binary_mask, binary_flow_mask))
                    union = np.sum(np.logical_or(binary_mask, binary_flow_mask))
                    iou = intersection / union if union > 0 else 0
                    
                    # Enhanced debugging for target frames
                    if frame_idx in TARGET_FRAMES or VERBOSE_DEBUGGING:
                        print(f"\n===== DETAILED ANALYSIS FOR FRAME {frame_idx} =====")
                        print(f"Before adjustment: mask sum = {np.sum(mask)}, flow_mask sum = {np.sum(flow_mask)}")
                        print(f"Mask ratio: flow_mask/mask = {np.sum(flow_mask)/np.sum(mask):.4f}")
                        print(f"Binary mask areas - Original: {mask_area}, Flow: {flow_mask_area}, Ratio: {area_ratio:.4f}")
                        print(f"Mask IoU: {iou:.4f}")
                        
                        # Detailed analysis of the flow
                        if np.any(mask > 0):
                            masked_flow = flow.copy()
                            masked_flow[mask <= 0] = 0
                            mean_x = np.mean(masked_flow[..., 0][mask > 0])
                            mean_y = np.mean(masked_flow[..., 1][mask > 0])
                            print(f"Average flow vector in mask: x={mean_x:.2f}, y={mean_y:.2f}")
                            
                            # Calculate flow magnitude statistics
                            flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                            masked_mag = flow_mag * (mask > 0)
                            if np.any(masked_mag > 0):
                                max_mag = np.max(masked_mag[masked_mag > 0])
                                mean_mag = np.mean(masked_mag[masked_mag > 0])
                                print(f"Flow magnitude in mask: max={max_mag:.2f}, mean={mean_mag:.2f}")
                    else:
                        # Basic diagnostic output for non-target frames
                        print(f"Before adjustment: mask sum = {np.sum(mask)}, flow_mask sum = {np.sum(flow_mask)}")
                        print(f"Binary mask areas - Original: {mask_area}, Flow: {flow_mask_area}, Ratio: {area_ratio:.4f}")
                        print(f"Mask IoU: {iou:.4f}")
                        
                    # Blend the masks with modified weights
                    adjusted_mask = flow_mask
                    blended_mask = (0.3 * mask + 0.7 * adjusted_mask).astype(float)
                   
                    # Enhanced debugging for target frames
                    if frame_idx in TARGET_FRAMES or VERBOSE_DEBUGGING:
                        print(f"After blending (0.3*mask + 0.7*flow_mask): new_mask sum = {np.sum(blended_mask)}")
                        print(f"Blend ratio: blended/original = {np.sum(blended_mask)/np.sum(mask):.4f}")
                        
                        # Save a visualization of the masks before and after blending
                        try:
                            if DEBUG_MODE and os.path.exists(debug_dir):
                                debug_compare = np.zeros((mask.shape[0], mask.shape[1]*3), dtype=np.uint8)
                                debug_compare[:, :mask.shape[1]] = (mask * 255).astype(np.uint8)
                                debug_compare[:, mask.shape[1]:mask.shape[1]*2] = (flow_mask * 255).astype(np.uint8)
                                debug_compare[:, mask.shape[1]*2:] = (blended_mask * 255).astype(np.uint8)
                                cv2.imwrite(os.path.join(debug_dir, f'mask_blending_{frame_idx:04d}.png'), debug_compare)
                        except Exception as e:
                            print(f"Warning: Could not save mask blending debug image: {str(e)}")
                    else:
                        # Basic diagnostic output for non-target frames
                        print(f"After blending: new_mask sum = {np.sum(blended_mask)}")
                   
                    new_mask = np.clip(blended_mask, 0, 1)

                    if np.sum(new_mask) > np.sum(mask) * 1.05:  # 5% growth limit
                        # Enhanced debugging for target frames
                        if frame_idx in TARGET_FRAMES or VERBOSE_DEBUGGING:
                            growth_factor = np.sum(new_mask)/np.sum(mask)
                            print(f"GROWTH CONSTRAINT APPLIED - Ratio before constraint: {growth_factor:.4f}")
                            print(f"Threshold used: {1.0 - (np.sum(mask) / np.sum(new_mask)):.4f}")
                        else:
                            # Basic diagnostic output for non-target frames
                            print(f"Applying growth constraint. Ratio: {np.sum(new_mask)/np.sum(mask):.2f}")
                        
                        # Scale back to the original size
                        scale_factor = np.sum(mask) / np.sum(new_mask)
                        # Applying a threshold that gets stricter as the mask grows
                        threshold = 1.0 - scale_factor
                        new_mask = (new_mask > threshold).astype(float)
                        print(f"Frame {frame_idx} - Mask constrained: growth factor {1/scale_factor:.2f}x, threshold {threshold:.2f}")

                    # Enhanced debugging for target frames
                    if frame_idx in TARGET_FRAMES or VERBOSE_DEBUGGING:
                        print(f"Final mask sum = {np.sum(new_mask)}")
                        print(f"Final ratio: {np.sum(new_mask)/np.sum(mask):.4f}")
                        
                        # Calculate binary area of final mask
                        binary_new_mask = (new_mask > 0.5).astype(np.uint8)
                        final_mask_area = np.sum(binary_new_mask)
                        final_area_ratio = final_mask_area / mask_area if mask_area > 0 else 0
                        print(f"Final binary mask area: {final_mask_area}, Ratio: {final_area_ratio:.4f}")
                        
                        print(f"Mask coverage: {np.mean(new_mask) * 100:.2f}%")
                        print(f"===== END DETAILED ANALYSIS FOR FRAME {frame_idx} =====\n")
                    else:
                        # Basic diagnostic output for non-target frames
                        print(f"Final mask sum = {np.sum(new_mask)}")
                        binary_new_mask = (new_mask > 0.5).astype(np.uint8)
                        final_mask_area = np.sum(binary_new_mask)
                        final_area_ratio = final_mask_area / mask_area if mask_area > 0 else 0
                        print(f"Final binary mask area: {final_mask_area}, Ratio: {final_area_ratio:.4f}")
                    
                    # Print coverage stats
                    print_mask_stats(mask, frame_idx)
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    total_frames_processed += 1
                    
                    # Store frame and flow with intermediate masks
                    frames.append((frame_idx, frame, new_mask, flow, flow_mask, adjusted_mask))
                    
                except Exception as e:
                    print(f"Error computing flow at frame {frame_idx}: {str(e)}")
                    traceback.print_exc()
                    consecutive_errors += 1
                    total_frames_skipped += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many consecutive errors, stopping at frame {frame_idx}")
                        break
                    frame_idx += step
                    continue

            # Save debug visualization with flow
            try:
                if DEBUG_MODE and os.path.exists(debug_dir):
                    print(f"DEBUG_MODE is True, creating debug frame for frame {frame_idx}")
                    debug_frame = debug_visualize(
                        frame=frame, 
                        initial_mask=initial_mask, 
                        flow_mask=flow_mask if frame_idx != start_frame else np.zeros_like(mask), 
                        adjusted_mask=adjusted_mask if frame_idx != start_frame else np.zeros_like(mask), 
                        final_mask=new_mask, 
                        frame_number=frame_idx,
                        flow=flow
                    )
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_filename = f'debug_frame_{frame_idx:04d}_{timestamp}.png'
                    cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_frame)
            except Exception as e:
                print(f"Error saving debug frame {frame_idx}: {str(e)}")

            # Update for next iteration
            prev_gray = frame_gray.copy()
            mask = new_mask.copy()
            frame_idx += step

            # Update progress bar
            if pbar:
                pbar.update(1)

        except Exception as e:
            print(f"Unexpected error processing frame {frame_idx}: {str(e)}")
            traceback.print_exc()
            consecutive_errors += 1
            total_frames_skipped += 1
            if consecutive_errors >= max_consecutive_errors:
                break
            frame_idx += step
            continue

    # Print final statistics
    print(f"\n{direction.capitalize()} tracking completed:")
    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total frames skipped: {total_frames_skipped}")
    print(f"Final frame count: {len(frames)}")
    
    return frames
