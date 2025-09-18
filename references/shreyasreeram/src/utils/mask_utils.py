import numpy as np
import cv2

def polygons_to_mask(polygons, height=512, width=512):
    """
    Convert a list of polygon points to a binary mask
    
    Args:
        polygons: List of polygons, where each polygon is a list of [x, y] coordinates
        height: Height of the output mask
        width: Width of the output mask
        
    Returns:
        Binary mask as numpy array
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for polygon in polygons:
        if not polygon or len(polygon) < 3:
            continue
            
        # Convert to integer coordinates
        points = np.array(polygon, dtype=np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(mask, [points], 1)
    
    return mask 