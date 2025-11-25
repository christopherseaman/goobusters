import cv2
import torch
import numpy as np
import os
import torchvision.models.optical_flow as optical_flow
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.transforms.functional import resize
try:
    from .performance_config import get_optimizer
except ImportError:
    try:
        from lib.performance_config import get_optimizer
    except ImportError:
        # Fallback: create a simple optimizer function
        def get_optimizer():
            return None

class OpticalFlowProcessor:
    def __init__(self, method):
        self.method = method
        self.raft_model = None

        # Use optimized device detection (GPU if available, CPU fallback)
        import torch
        optimizer = get_optimizer()
        if optimizer and hasattr(optimizer, 'device'):
            self.device = optimizer.device
        else:
            self.device = torch.device('cpu')  # Fallback to CPU

        # Load DIS preset from environment
        dis_preset_name = os.getenv('DIS_PRESET', 'fast').lower()
        self.dis_preset = {
            'ultrafast': cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            'fast': cv2.DISOPTICAL_FLOW_PRESET_FAST,
            'medium': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
        }.get(dis_preset_name, cv2.DISOPTICAL_FLOW_PRESET_FAST)

        if self.method == 'raft':
            self.load_raft_model()

    def load_raft_model(self):
        # print(f"Using device: {self.device} for RAFT")
        self.raft_model = optical_flow.raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        self.raft_model = self.raft_model.to(self.device)
        self.raft_model = self.raft_model.eval()

    def cleanup_memory(self):
        """Clean up GPU memory after processing."""
        if self.device.type in ('mps', 'cuda'):
            import gc
            gc.collect()
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def raft_optical_flow(self, image1, image2):
        # Ensure images are in the correct format (B, C, H, W)
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)
        
        # If grayscale, repeat to create 3 channels
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)
            image2 = image2.repeat(1, 3, 1, 1)
        
        # Get original dimensions
        _, _, h, w = image1.shape

        # Calculate new dimensions divisible by 8
        new_h = ((h - 1) // 8 + 1) * 8
        new_w = ((w - 1) // 8 + 1) * 8

        # Resize images
        image1 = resize(image1, [new_h, new_w])
        image2 = resize(image2, [new_h, new_w])

        # Move images to the optimized device
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        # Compute optical flow
        with torch.no_grad():
            flow = self.raft_model(image1, image2)[-1]  # Get the last prediction

        # Resize flow back to original dimensions
        flow = resize(flow, [h, w])

        # Convert flow to numpy array
        flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        return flow

    def calculate_flow(self, prev_frame, curr_frame, method=None):
        """
        Calculate optical flow between two frames with temporal smoothing.

        Args:
            prev_frame: Previous frame (grayscale)
            curr_frame: Current frame (grayscale)
            method: Optional method override

        Returns:
            Optical flow field as numpy array
        """
        # Use provided method or fall back to instance method
        flow_method = method or self.method

        # Ensure input frames are grayscale
        if len(prev_frame.shape) == 3:
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        if len(curr_frame.shape) == 3:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if flow_method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif flow_method == 'dis':
            dis = cv2.DISOpticalFlow_create(self.dis_preset)
            flow = dis.calc(prev_frame, curr_frame, None)
        elif flow_method == 'raft':
            prev_frame_tensor = torch.from_numpy(prev_frame).unsqueeze(0).unsqueeze(0).float() / 255.0
            curr_frame_tensor = torch.from_numpy(curr_frame).unsqueeze(0).unsqueeze(0).float() / 255.0
            flow = self.raft_optical_flow(prev_frame_tensor, curr_frame_tensor)
        else:
            raise ValueError(f"Unknown optical flow method: {flow_method}")

        return flow