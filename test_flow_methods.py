#!/usr/bin/env python3
"""Test that all optical flow methods work."""

from dotenv import load_dotenv
import os
import cv2
import numpy as np

load_dotenv('dot.env')
methods = os.getenv('FLOW_METHOD').split(',')
print('Testing flow methods:', methods)

# Create dummy images
img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
img2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

for method in methods:
    try:
        if method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif method == 'deepflow':
            deep_flow = cv2.optflow.createOptFlow_DeepFlow()
            flow = deep_flow.calc(img1, img2, None)
        elif method == 'dis':
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            flow = dis.calc(img1, img2, None)
        else:
            print(f'{method}: Unknown method')
            continue

        print(f'{method}: ✓ Works! Flow shape: {flow.shape}')
    except Exception as e:
        print(f'{method}: ✗ Error: {e}')