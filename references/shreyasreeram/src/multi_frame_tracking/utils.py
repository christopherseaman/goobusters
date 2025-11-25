import os
import cv2
import numpy as np
from datetime import datetime
import traceback

# Debug settings
DEBUG_MODE = True
TARGET_FRAMES = []  # Frames to analyze in detail
VERBOSE_DEBUGGING = False  # Enable verbose debugging output

def print_mask_stats(mask, frame_num):
    """Print coverage statistics for a mask"""
    coverage = np.mean(mask) * 100
    print(f"Frame {frame_num} - Mask coverage: {coverage:.2f}%")
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Mask min: {np.min(mask)}")
    print(f"Mask max: {np.max(mask)}")
    print(f"Mask mean: {np.mean(mask)}")
    print(f"Unique values: {np.unique(mask)}")

def track_frames(cap, start_frame, end_frame, initial_mask, debug_dir=None, forward=True, pbar=None, flow_processor=None, recursion_depth=0, shared_params=None, quality_threshold=None):
    """
    Track a mask between frames using GENUINE optical flow tracking.
    
    This version fixes the blending issues and enforces pure optical flow tracking.
    Now uses SharedParams for dynamic parameter adjustment.
    """
    frames = []
    frame_idx = start_frame
    step = 1 if forward else -1
    
    # Use SharedParams if available, otherwise fall back to provided threshold or default
    if shared_params is not None:
        actual_quality_threshold = shared_params.tracking_params['flow_quality_threshold']
        flow_noise_threshold = shared_params.tracking_params['flow_noise_threshold']
        mask_threshold = shared_params.tracking_params['mask_threshold']
        border_constraint_weight = shared_params.tracking_params['border_constraint_weight']
        contour_min_area = shared_params.tracking_params['contour_min_area']
        morphology_kernel_size = shared_params.tracking_params['morphology_kernel_size']
        
        print(f"🔧 Using SharedParams v{shared_params.version}:")
        print(f"    Flow quality threshold: {actual_quality_threshold}")
        print(f"    Flow noise threshold: {flow_noise_threshold}")
        print(f"    Mask threshold: {mask_threshold}")
        print(f"    Border constraint weight: {border_constraint_weight}")
    else:
        actual_quality_threshold = quality_threshold if quality_threshold is not None else 0.7
        flow_noise_threshold = 3.0  # Default
        mask_threshold = 0.5  # Default
        border_constraint_weight = 0.9  # Default
        contour_min_area = 50  # Default
        morphology_kernel_size = 5  # Default
        
        print(f"🔧 Using fallback parameters:")
        print(f"    Flow quality threshold: {actual_quality_threshold}")
        print(f"    Flow noise threshold: {flow_noise_threshold}")
        print(f"    Mask threshold: {mask_threshold}")
    
    # Tracking statistics
    total_frames_processed = 0
    total_frames_skipped = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    genuine_flow_frames = 0  # Count frames with genuine optical flow
    
    print(f"\n🔬 GENUINE OPTICAL FLOW TRACKING:")
    print(f"  Start frame: {start_frame}")
    print(f"  End frame: {end_frame}")
    print(f"  Forward: {forward}")
    print(f"  Using quality threshold: {actual_quality_threshold}")
    
    # Set video to starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"❌ Failed to read starting frame {start_frame}")
        return frames
        
    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_mask = initial_mask.astype(float)
        
        # Save initial state
        if debug_dir and os.path.exists(debug_dir):
            cv2.imwrite(os.path.join(debug_dir, f'initial_frame_{frame_idx:04d}.png'), prev_frame)
            cv2.imwrite(os.path.join(debug_dir, f'initial_mask_{frame_idx:04d}.png'), (current_mask * 255).astype(np.uint8))
            
    except Exception as e:
        print(f"❌ Error initializing: {str(e)}")
        return frames
    
    # Track through frames
    while (forward and frame_idx <= end_frame) or (not forward and frame_idx >= end_frame):
        if pbar:
            pbar.update(1)
            
        # Read next frame
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame {frame_idx}")
            break
            
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # First frame - just store it
            if frame_idx == start_frame:
                print(f"📌 Frame {frame_idx}: Initial frame (mask area: {np.sum(current_mask > mask_threshold)})")
                
                # Store with flow metadata
                frames.append((
                    frame_idx, 
                    frame, 
                    current_mask.copy(),
                    None,  # No flow for first frame
                    np.zeros_like(current_mask),  # No flow mask
                    current_mask.copy(),  # Adjusted mask = original
                    {  # Metadata
                        'optical_flow_used': False,
                        'flow_quality': 0.0,
                        'is_initial_frame': True,
                        'tracking_method': 'initial',
                        'shared_params_version': shared_params.version if shared_params else 'none'
                    }
                ))
                total_frames_processed += 1
                
            else:
                # GENUINE OPTICAL FLOW TRACKING
                print(f"🔄 Frame {frame_idx}: Computing optical flow...")
                
                try:
                    # Compute optical flow
                    flow = flow_processor.apply_optical_flow(prev_gray, frame_gray, current_mask)
                    
                    if flow is None:
                        print(f"❌ Frame {frame_idx}: Flow computation failed")
                        consecutive_errors += 1
                        total_frames_skipped += 1
                    else:
                        # Warp mask using flow
                        flow_mask = flow_processor.warp_mask(current_mask, flow)
                        
                        if flow_mask is None or np.isnan(flow_mask).any():
                            print(f"❌ Frame {frame_idx}: Invalid flow mask")
                            consecutive_errors += 1
                            total_frames_skipped += 1
                        else:
                            # Calculate flow quality using SharedParams thresholds
                            flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
                            
                            # Filter out noise using SharedParams noise threshold
                            mask_region = current_mask > mask_threshold
                            if np.sum(mask_region) > 0:
                                valid_flow = flow_magnitude[mask_region]
                                # Filter out noise
                                valid_flow = valid_flow[valid_flow < flow_noise_threshold * np.std(valid_flow) + np.mean(valid_flow)]
                                mean_flow = np.mean(valid_flow) if len(valid_flow) > 0 else 0
                                max_flow = np.max(valid_flow) if len(valid_flow) > 0 else 0
                            else:
                                mean_flow = 0
                                max_flow = 0
                            
                            # Calculate mask consistency metrics using SharedParams mask threshold
                            original_area = np.sum(current_mask > mask_threshold)
                            flow_area = np.sum(flow_mask > mask_threshold)
                            area_ratio = flow_area / original_area if original_area > 0 else 0
                            
                            # Calculate IoU between consecutive masks
                            iou = calculate_mask_iou(current_mask, flow_mask)
                            
                            print(f"📊 Frame {frame_idx} Flow Analysis:")
                            print(f"    Mean flow magnitude: {mean_flow:.3f}")
                            print(f"    Max flow magnitude: {max_flow:.3f}")
                            print(f"    Area ratio: {area_ratio:.3f}")
                            print(f"    IoU with previous: {iou:.3f}")
                            
                            # CRITICAL: Use PURE optical flow result (no blending!)
                            # Apply SharedParams-based post-processing
                            new_mask = flow_mask.copy()
                            
                            # Apply morphological operations using SharedParams kernel size
                            if morphology_kernel_size > 0:
                                kernel = np.ones((morphology_kernel_size, morphology_kernel_size), np.uint8)
                                binary_mask = (new_mask > mask_threshold).astype(np.uint8)
                                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                                new_mask = binary_mask.astype(float)
                            
                            # Remove small contours using SharedParams min area
                            if contour_min_area > 0:
                                binary_mask = (new_mask > mask_threshold).astype(np.uint8)
                                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                filtered_mask = np.zeros_like(binary_mask)
                                for contour in contours:
                                    if cv2.contourArea(contour) >= contour_min_area:
                                        cv2.fillPoly(filtered_mask, [contour], 1)
                                new_mask = filtered_mask.astype(float)
                            
                            # Apply quality checks using SharedParams thresholds
                            flow_quality_ok = mean_flow >= actual_quality_threshold
                            area_ok = 0.3 <= area_ratio <= 3.0  # Reasonable area change
                            iou_ok = iou >= 0.2  # Minimum overlap with previous frame
                            
                            # Determine if this is genuine tracking
                            is_genuine = flow_quality_ok and area_ok and iou_ok
                            
                            if is_genuine:
                                print(f"✅ Frame {frame_idx}: GENUINE optical flow tracking")
                                genuine_flow_frames += 1
                                tracking_method = 'genuine_optical_flow'
                            else:
                                print(f"⚠️  Frame {frame_idx}: Low quality optical flow")
                                print(f"    Quality OK: {flow_quality_ok} (threshold: {actual_quality_threshold})")
                                print(f"    Area OK: {area_ok} (ratio: {area_ratio:.3f})")
                                print(f"    IoU OK: {iou_ok} (IoU: {iou:.3f})")
                                tracking_method = 'low_quality_optical_flow'
                            
                            # Store result with comprehensive metadata
                            frames.append((
                                frame_idx,
                                frame,
                                new_mask,
                                flow,
                                flow_mask,
                                new_mask,  # Adjusted mask = flow mask (no blending)
                                {  # Enhanced metadata
                                    'optical_flow_used': True,
                                    'flow_quality': mean_flow,
                                    'max_flow': max_flow,
                                    'area_ratio': area_ratio,
                                    'iou_with_previous': iou,
                                    'is_genuine_tracking': is_genuine,
                                    'tracking_method': tracking_method,
                                    'shared_params_version': shared_params.version if shared_params else 'none',
                                    'quality_checks': {
                                        'flow_quality_ok': flow_quality_ok,
                                        'area_ok': area_ok,
                                        'iou_ok': iou_ok,
                                        'actual_quality_threshold': actual_quality_threshold
                                    }
                                }
                            ))
                            
                            # Update current mask for next iteration
                            current_mask = new_mask
                            consecutive_errors = 0
                            total_frames_processed += 1
                            
                            # Save debug visualization for genuine tracking frames
                            if debug_dir and is_genuine and frame_idx % 5 == 0:
                                debug_viz = debug_visualize(
                                    frame, initial_mask, flow_mask, new_mask, new_mask, 
                                    frame_idx, flow
                                )
                                cv2.imwrite(
                                    os.path.join(debug_dir, f'genuine_tracking_{frame_idx:04d}.png'), 
                                    debug_viz
                                )
                    
                except Exception as e:
                    print(f"❌ Frame {frame_idx}: Flow processing error: {str(e)}")
                    consecutive_errors += 1
                    total_frames_skipped += 1
            
            # Update for next iteration
            prev_gray = frame_gray
            
            # Check for too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                print(f"🛑 Stopping due to {consecutive_errors} consecutive errors")
                break
                
        except Exception as e:
            print(f"❌ Frame {frame_idx}: Processing error: {str(e)}")
            consecutive_errors += 1
            total_frames_skipped += 1
            
        frame_idx += step
    
    # Final statistics
    total_frames = total_frames_processed + total_frames_skipped
    print(f"\n📈 TRACKING STATISTICS:")
    print(f"    Total frames processed: {total_frames_processed}")
    print(f"    Frames with genuine optical flow: {genuine_flow_frames}")
    print(f"    Frames skipped: {total_frames_skipped}")
    print(f"    Used SharedParams version: {shared_params.version if shared_params else 'none'}")
    print(f"    Final quality threshold: {actual_quality_threshold}")
    
    if total_frames_processed > 0:
        genuine_rate = (genuine_flow_frames / total_frames_processed) * 100
        success_rate = (total_frames_processed / total_frames) * 100 if total_frames > 0 else 0
        
        print(f"    Genuine tracking rate: {genuine_rate:.1f}%")
        print(f"    Overall success rate: {success_rate:.1f}%")
        
        # Warning if genuine tracking rate is low
        if genuine_rate < 70:
            print(f"    ⚠️  WARNING: Low genuine tracking rate!")
            print(f"    ⚠️  Consider adjusting quality_threshold or checking flow processor")
    
    return frames

def calculate_mask_iou(mask1, mask2, threshold=0.5):
    """Calculate IoU between two masks."""
    binary1 = (mask1 > threshold).astype(np.uint8)
    binary2 = (mask2 > threshold).astype(np.uint8)
    
    intersection = np.sum(np.logical_and(binary1, binary2))
    union = np.sum(np.logical_or(binary1, binary2))
    
    return intersection / union if union > 0 else 0.0

def visualize_flow(frame, flow, skip=8):
    """
    Visualize optical flow using arrows.
    
    Args:
        frame (np.ndarray): Original frame
        flow (np.ndarray): Optical flow field
        skip (int): Skip factor for visualization (show every nth arrow)
        
    Returns:
        np.ndarray: Frame with flow visualization
    """
    if flow is None:
        return frame.copy()
        
    h, w = frame.shape[:2]
    y, x = np.mgrid[skip//2:h:skip, skip//2:w:skip].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    
    # Create mask of valid flow vectors
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # Create visualization
    vis = frame.copy()
    
    # Draw flow vectors
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    
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

def verify_directory(directory):
    """
    Verify that a directory exists, create it if it doesn't.
    
    Args:
        directory (str): Path to directory
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {str(e)}")
        return False

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
    
def find_exam_number(study_uid, annotations_json):
    """
    Find exam number for a given StudyInstanceUID using the correct JSON structure
    
    Args:
        study_uid: The StudyInstanceUID to search for
        annotations_json: The annotations JSON object
        
    Returns:
        Exam number as string, or "unknown" if not found
    """
    if not annotations_json or not study_uid:
        return "unknown"
    
    # Look in the datasets -> studies array structure
    datasets = annotations_json.get('datasets', [])
    for dataset in datasets:
        studies = dataset.get('studies', [])
        for study in studies:
            if study.get('StudyInstanceUID') == study_uid and 'number' in study:
                return str(study['number'])
    
    # If not found in the main structure, try the alternative location
    for dataset in datasets:
        for annotation in dataset.get('annotations', []):
            if annotation.get('StudyInstanceUID') == study_uid and annotation.get('examNumber'):
                return str(annotation['examNumber'])
    
    # For debugging purposes, let's print the study UIDs we found
    print(f"\nDEBUG: Study UID '{study_uid}' not found. Available study UIDs:")
    found_uids = set()
    for dataset in datasets:
        for study in dataset.get('studies', []):
            found_uids.add(study.get('StudyInstanceUID', ''))
    
    # Print a few examples of the UIDs we did find
    for i, uid in enumerate(list(found_uids)[:5]):
        print(f"  {uid}")
    
    if len(found_uids) > 5:
        print(f"  ... and {len(found_uids) - 5} more")
    
    return "unknown"
    
def delete_existing_annotations(client, study_uid, series_uid, label_id, group_id=None):
    """
    Delete existing annotations with the same criteria before uploading new ones.
    """
    try:
        print(f"\nChecking for existing annotations for series {series_uid}...")
        
        # Get existing annotations
        filter_criteria = {
            'StudyInstanceUID': study_uid,
            'SeriesInstanceUID': series_uid,
            'labelId': label_id
        }
        if group_id:
            filter_criteria['groupId'] = group_id
            
        # Try different approaches based on available methods
        existing_annotations = []
        try:
            existing_annotations = client.search_annotations(
                project_id=client.project_id,
                dataset_id=None,
                filter_criteria=filter_criteria
            )
        except Exception:
            # Fall back to get_annotations if search_annotations fails
            pass
        
        if not existing_annotations:
            print("No existing annotations found.")
            return 0
            
        print(f"Found {len(existing_annotations)} existing annotations to delete.")
        
        # Delete annotations
        deleted_count = 0
        for ann in existing_annotations:
            try:
                annotation_id = ann.get('id')
                if annotation_id:
                    client.delete_annotation(annotation_id)
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting annotation {ann.get('id')}: {str(e)}")
                continue
        
        print(f"Successfully deleted {deleted_count} existing annotations.")
        return deleted_count
        
    except Exception as e:
        print(f"Error during deletion: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0
    
# general_utils.py
import numpy as np

def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Any Python or numpy object
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj