import cv2
import torch
import numpy as np
import torchvision.models.optical_flow as optical_flow
from torchvision.transforms.functional import resize

class OpticalFlowProcessor:
    def __init__(self, method):
        self.method = method
        self.raft_model = None
        if self.method == 'raft':
            self.load_raft_model()

    def load_raft_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device} for RAFT")
        self.raft_model = optical_flow.raft_large(pretrained=True, progress=False)
        self.raft_model = self.raft_model.to(device)
        self.raft_model = self.raft_model.eval()

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

        # Move images to the same device as the model
        device = next(self.raft_model.parameters()).device
        image1 = image1.to(device)
        image2 = image2.to(device)

        # Compute optical flow
        with torch.no_grad():
            flow = self.raft_model(image1, image2)[-1]  # Get the last prediction

        # Resize flow back to original dimensions
        flow = resize(flow, [h, w])

        # Convert flow to numpy array
        flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        return flow

    def apply_optical_flow(self, prev_frame, curr_frame, prev_mask):
        # Ensure input frames are grayscale
        if len(prev_frame.shape) == 3:
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        if len(curr_frame.shape) == 3:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif self.method == 'deepflow':
            deep_flow = cv2.optflow.createOptFlow_DeepFlow()
            flow = deep_flow.calc(prev_frame, curr_frame, None)
        elif self.method == 'dis':
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            flow = dis.calc(prev_frame, curr_frame, None)
        elif self.method == 'raft':
            prev_frame = torch.from_numpy(prev_frame).unsqueeze(0).unsqueeze(0).float() / 255.0
            curr_frame = torch.from_numpy(curr_frame).unsqueeze(0).unsqueeze(0).float() / 255.0
            flow = self.raft_optical_flow(prev_frame, curr_frame)
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")
        
        # Use the flow to warp the previous mask
        h, w = prev_frame.shape[:2]
        flow_x, flow_y = flow[..., 0], flow[..., 1]
        
        # Create meshgrid
        y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(int)
        
        # Apply the flow to the coordinates
        coords = np.vstack([x + flow_x.flatten(), y + flow_y.flatten()]).round().astype(int)
        
        # Clip the coordinates to stay within the image
        coords[0] = np.clip(coords[0], 0, w - 1)
        coords[1] = np.clip(coords[1], 0, h - 1)
        
        # Create the new mask
        new_mask = np.zeros_like(prev_mask, dtype=float)
        new_mask[coords[1], coords[0]] = prev_mask[y, x]
        
        # Apply some morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)
        
        return new_mask