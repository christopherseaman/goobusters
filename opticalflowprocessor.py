import cv2
import torch
import numpy as np
import torchvision.models.optical_flow as optical_flow
from torchvision.transforms.functional import resize

#Main output: Optical flow computed b/w 2 consecutive frames: 1)flow field (displacement vectors for each pixel)
#2) updated mask - leverages flow to track a specific region/object in consecutive frames. 


# Importing the PWC-Net model: 
from models.PWCNet import PWCDCNet  

class OpticalFlowProcessor:
    def __init__(self, method):
        self.method = method  # Stores the selected optical flow method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.raft_model = None  # Only loaded if RAFT is chosen
        self.pwc_net_model = None  # Only loaded if PWC-Net is chosen

        # Initialize the appropriate model based on the method
        if self.method == 'raft':
            self.load_raft_model()
        elif self.method == 'pwc-net':
            self.load_pwc_net_model()

    def load_raft_model(self):
        print(f"Using device: {self.device} for RAFT")
        self.raft_model = optical_flow.raft_large(pretrained=True, progress=False)
        self.raft_model = self.raft_model.to(self.device)
        self.raft_model.eval()

    def load_pwc_net_model(self):
        # Initialize PWC-Net and load pre-trained weights
        print(f"Using device: {self.device} for PWC-Net")
        self.pwc_net_model = PWCDCNet().to(self.device)
        pwc_weights_path = 'pwc_net_chairs.pth.tar'  

        try:
            checkpoint = torch.load(pwc_weights_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.pwc_net_model.load_state_dict(checkpoint['state_dict'])
            else:
                self.pwc_net_model.load_state_dict(checkpoint)
            self.pwc_net_model.eval()
            print("PWC-Net weights loaded successfully.")
        except FileNotFoundError:
            print(f"Error: PWC-Net weights not found at {pwc_weights_path}. Please provide the correct path.")
        except Exception as e:
            print(f"Error loading PWC-Net weights: {e}")

    # Compute optical flow using the RAFT model
    def raft_optical_flow(self, image1, image2):
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)

        if image1.shape[1] == 1:  # If grayscale, repeat to create 3 channels
            image1 = image1.repeat(1, 3, 1, 1)
            image2 = image2.repeat(1, 3, 1, 1)

        _, _, h, w = image1.shape
        new_h = ((h - 1) // 8 + 1) * 8
        new_w = ((w - 1) // 8 + 1) * 8
        image1 = resize(image1, [new_h, new_w])
        image2 = resize(image2, [new_h, new_w])

        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        with torch.no_grad():
            flow = self.raft_model(image1, image2)[-1]

        flow = resize(flow, [h, w])
        flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        return flow
    
    def pwc_net_optical_flow(self, image1, image2):
  
    # Ensure images have 3 channels (convert grayscale to RGB if needed)
       if len(image1.shape) == 2:  # If grayscale
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
       if len(image2.shape) == 2:  # If grayscale
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    # Convert images to torch tensors, normalize, and move to the device
        image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        image2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

    # Concatenate images along the channel dimension
        input_tensor = torch.cat((image1, image2), dim=1)  # Now input_tensor has 6 channels (3 for each image)

    # Calculate flow using the model's forward method
        with torch.no_grad():
         flow = self.pwc_net_model(input_tensor)  # Forward pass through PWC-Net

    # Convert flow to numpy for further processing
        flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        return flow
    
    # Applies the chosen optical flow method to calculate the flow between two frames
    def apply_optical_flow(self, prev_frame, curr_frame, prev_mask):
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
            prev_frame_tensor = torch.from_numpy(prev_frame).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
            curr_frame_tensor = torch.from_numpy(curr_frame).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
            flow = self.raft_optical_flow(prev_frame_tensor, curr_frame_tensor)
        elif self.method == 'pwc-net':
            flow = self.pwc_net_optical_flow(prev_frame, curr_frame)
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")

        h, w = prev_frame.shape[:2]
        flow_x, flow_y = flow[..., 0], flow[..., 1]

        y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(int)
        coords = np.vstack([x + flow_x.flatten(), y + flow_y.flatten()]).round().astype(int)
        coords[0] = np.clip(coords[0], 0, w - 1)
        coords[1] = np.clip(coords[1], 0, h - 1)

        new_mask = np.zeros_like(prev_mask, dtype=float)
        new_mask[coords[1], coords[0]] = prev_mask[y, x]

        kernel = np.ones((5, 5), np.uint8)
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)

        return new_mask