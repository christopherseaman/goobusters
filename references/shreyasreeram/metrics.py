import numpy as np
import cv2

def calculate_jaccard_index(pred_mask, gt_mask):
    """Calculate the Jaccard Index (IoU) between the predicted(binary predicted mask) and ground-truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum() #returns binary mask where each pixel = 1 only if corresponding pixels in both masks are 1 
    union = np.logical_or(pred_mask, gt_mask).sum() 
    return intersection / union if union != 0 else 0 #calculations proportion of intersection / union 
 
def apply_binary_threshold(flow, threshold=0.5): #tentative threshold set to 0.5 
    """Apply binary thresholding on the optical flow magnitude to create a binary mask."""
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #calculates magnitude of optical flow at each pixel 
    binary_mask = (mag >= threshold).astype(np.uint8) #applies the threshold to create the binary mask 
    return binary_mask #returns binary mask where pixels are 1 above threshold, and 0 elsewhere