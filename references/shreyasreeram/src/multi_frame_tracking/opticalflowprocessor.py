import cv2
import torch
import numpy as np
import torchvision.models.optical_flow as optical_flow
from torchvision.transforms.functional import resize
from types import SimpleNamespace
import traceback
from skimage.metrics import structural_similarity
import os
import sys
print("Attempting to import RAFT...")
#from core.raft import RAFT
#print("RAFT class location:", RAFT.__module__)
#print("RAFT:", RAFT)

# Main output: Optical flow computed b/w 2 consecutive frames: 1)flow field (displacement vectors for each pixel)
# 2) updated mask - leverages flow to track a specific region/object in consecutive frames. 

# Importing the PWC-Net model: 
#from models.PWCNet import PWCDCNet  

class OpticalFlowProcessor:
    def __init__(self, method):
        self.method = method  # Stores the selected optical flow method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.raft_model = None  # Only loaded if RAFT is chosen
        self.pwc_net_model = None  # Only loaded if PWC-Net is chosen
        self.sea_raft_model = None

        # Initialize the appropriate model based on the method
        if self.method == 'raft':
            self.load_raft_model()
        elif self.method == 'pwc-net':
            self.load_pwc_net_model()
        elif self.method == 'sea-raft':  # Add SEA-RAFT initialization
            self.load_sea_raft_model()

    def verify_sea_raft_implementation(self):
        import inspect
        
        print("\n=== Verifying SEA-RAFT Implementation ===")
        
        # Print the actual RAFT class source location
        raft_file = inspect.getfile(RAFT)
        print(f"RAFT class file location: {raft_file}")
        
        # Compare with expected path
        expected_path = "/Users/Shreya1/tools/SEA-RAFT/core/raft.py"
        if os.path.normpath(raft_file) != os.path.normpath(expected_path):
            print(f"Warning: RAFT implementation might not be from SEA-RAFT repository")
            print(f"Expected: {expected_path}")
            print(f"Found: {raft_file}")

    def inspect_weights_file(self, weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            print("\n=== Checkpoint Contents ===")
            if isinstance(checkpoint, dict):
                print("Keys in checkpoint:", checkpoint.keys())
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                    
                print("\nModel state dict keys:")
                for key in state_dict.keys():
                    print(f"Layer: {key}, Shape: {state_dict[key].shape}")
                    
        except Exception as e:
            print(f"Error loading weights file: {e}")
            traceback.print_exc()

    def visualize_flow_vectors(self, flow):
        """Visualize actual flow vectors to show movement"""
        h, w = flow.shape[:2]
        step = 16  # Show vector every 16 pixels
        
        # Create visualization
        vis = np.zeros((h, w, 3), np.uint8)
        
        # Draw flow vectors
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                cv2.line(vis, (x, y), (int(x+fx), int(y+fy)), (0,255,0), 1)
                cv2.circle(vis, (x,y), 1, (0,255,0), -1)
                
        return vis
    
    def verify_flow_tracking(self, prev_frame, curr_frame, flow_mask, initial_mask):
        """Validate that flow tracking is meaningful and not just copying"""
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(curr_frame, prev_frame)
        frame_diff_mean = np.mean(frame_diff)
        
        # Calculate flow mask movement
        flow_displacement = np.mean(np.abs(flow_mask - initial_mask))
        
        # Calculate structural similarity between frames
        ssim_score = structural_similarity(prev_frame, curr_frame, multichannel=True)
        
        print(f"[DEBUG] Frame difference: {frame_diff_mean:.2f}")
        print(f"[DEBUG] Flow displacement: {flow_displacement:.2f}")
        print(f"[DEBUG] Structural similarity: {ssim_score:.2f}")
        
        # Add validation checks
        if flow_displacement < 0.01 and frame_diff_mean > 5.0:
            print("[WARNING] Flow mask shows little movement despite significant frame changes")
            return False
            
        return True
    
    def warp_mask(self, mask, flow):
        h, w = mask.shape
    
    # Create coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    # Extract flow components - make sure they maintain 2D shape
        flow_x = flow[:, :, 0]  # Use explicit indexing to preserve 2D shape
        flow_y = flow[:, :, 1]
    
    # Verify shapes match
        if flow_x.shape != (h, w) or flow_y.shape != (h, w):
            print(f"Shape mismatch! mask: {mask.shape}, flow_x: {flow_x.shape}, flow_y: {flow_y.shape}")
        # Try reshaping if needed
            flow_x = flow_x.reshape(h, w)
            flow_y = flow_y.reshape(h, w)
    
    # Move grid points according to flow
        x_remap = x_grid.astype(np.float32) + flow_x
        y_remap = y_grid.astype(np.float32) + flow_y
    
    # Ensure points stay within image bounds
        x_remap = np.clip(x_remap, 0, w-1)
        y_remap = np.clip(y_remap, 0, h-1)
    
    # Warp the mask using the flow field
        warped_mask = cv2.remap(mask.astype(np.float32), 
                       x_remap, 
                       y_remap,
                       interpolation=cv2.INTER_LINEAR)
    
        return warped_mask
    def load_sea_raft_model(self):
        print(f"Using device: {self.device} for SEA-RAFT")

        self.verify_sea_raft_implementation()

        weights_path = "/Users/Shreya1/tools/sea_raft/models/Tartan-C-T-TSKH-spring540x960-M.pth"

        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"Weights file not found at {weights_path}")

        # Inspect weights before loading
        print("\nInspecting weights file structure:")
        self.inspect_weights_file(weights_path)

    # Initialize args with default values
        args = SimpleNamespace()
        args.corr_levels = 4
        args.radius = 4
        args.corr_radius = args.radius
        #args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        args.dim = 128  # Ensure this matches SEA-RAFT requirements
        args.iters = 12  # Adjust as needed
        args.num_blocks = 2
        args.corr_channel = 256
        args.dropout = 0
        args.mixed_precision = False
        args.refine_hidden_dims = 384  # Added to match pretrained dimensions
        args.refine_output_dim = 128

        block_dims = [64, 128, 256]
        output_dim = 256

        print(f"[DEBUG] Initializing RAFT with block_dims: {block_dims}")
        print(f"[DEBUG] args: {vars(args)}")  # Print args for debugging

    # Initialize SEA-RAFT model with args
        
        try:
           self.sea_raft_model = RAFT(
                block_dims=block_dims,
                output_dim=output_dim,
                args=args,
                refine_hidden_dims=384,  # Added explicit parameter
                refine_output_dim=128
           ).to(self.device)

           checkpoint = torch.load(weights_path, map_location=self.device)

           print("[DEBUG] Model Architecture:")
           for name, param in self.sea_raft_model.named_parameters():
              print(f"{name}: {param.shape}")
        
            # Load state dict with strict=False to ignore missing keys
           self.sea_raft_model.load_state_dict(checkpoint, strict=False)
           self.sea_raft_model.eval()
           print("[INFO] SEA-RAFT model successfully loaded!")
        
        except Exception as e:
           print(f"[ERROR] Failed to initialize SEA-RAFT: {e}")
           traceback.print_exc()
           raise
        
    def sea_raft_optical_flow(self, image1, image2):
        if self.sea_raft_model is None:
           raise RuntimeError("[ERROR] SEA-RAFT model is not initialized!")
        try:
           print(f"[DEBUG] Original shapes - image1: {image1.shape}, image2: {image2.shape}")
        
        # Convert images to RGB if they are grayscale
           if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
           if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

        # Ensure input dimensions are multiples of 8
           h, w = image1.shape[:2]
           pad_h = ((h // 8 + 1) * 8 - h) % 8
           pad_w = ((w // 8 + 1) * 8 - w) % 8

        # Apply padding to ensure dimensions are multiples of 8
           image1 = np.pad(image1, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
           image2 = np.pad(image2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        # Print shapes for debugging
           print(f"[DEBUG] Padded shapes - image1: {image1.shape}, image2: {image2.shape}")

        # Convert to torch tensors and normalize
           image1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
           image2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        # Run SEA-RAFT inference
           with torch.no_grad():
              output = self.sea_raft_model(image1_tensor, image2_tensor)

           if output is None:
              raise RuntimeError("[ERROR] Model returned None for flow predictions")
            
        # Get the final flow prediction from the output dictionary
           flow = output['final']  # Changed this line to access the dictionary
        
        # Add debug print
           print(f"[DEBUG] Output keys: {output.keys()}")
           print(f"[DEBUG] Flow tensor shape: {flow.shape}")
        
        # Remove padding if necessary
           if pad_h > 0 or pad_w > 0:
              flow = flow[:, :, :h, :w]
        
           flow_np = flow.squeeze().permute(1, 2, 0).cpu().numpy()
           print(f"[DEBUG] Final flow shape: {flow_np.shape}")
        
           return flow_np

        except Exception as e:
            print(f"[ERROR] SEA-RAFT optical flow computation failed: {str(e)}")
            traceback.print_exc()
            raise      

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
    
    def apply_optical_flow(self, prev_frame, curr_frame, prev_mask):
        # Convert frames to grayscale if they are in color
        if len(prev_frame.shape) == 3:
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        if len(curr_frame.shape) == 3:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        print(f"[DEBUG] Optical Flow Method: {self.method}")
        print(f"[DEBUG] Prev Frame Shape: {prev_frame.shape}, Curr Frame Shape: {curr_frame.shape}")

        # Choose the optical flow method
        if self.method == 'sea-raft':
           try:
              flow = self.sea_raft_optical_flow(prev_frame, curr_frame)
           except Exception as e:
            print(f"[ERROR] SEA-RAFT optical flow failed: {e}")
            return None  # Ensure failure doesn't propagate

           if flow is None:
              print("[ERROR] SEA-RAFT returned None for flow")
              return None
        elif self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif self.method == 'deepflow':
            deep_flow = cv2.optflow.createOptFlow_DeepFlow()
            flow = deep_flow.calc(prev_frame, curr_frame, None)
        elif self.method == 'dis':
        # Fix DIS method
            flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            flow_result = flow.calc(prev_frame, curr_frame, None)
            return flow_result.reshape(prev_frame.shape[0], prev_frame.shape[1], 2)
        elif self.method == 'raft':
            prev_frame_tensor = torch.from_numpy(prev_frame).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
            curr_frame_tensor = torch.from_numpy(curr_frame).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
            flow = self.raft_optical_flow(prev_frame_tensor, curr_frame_tensor)
        elif self.method == 'pwc-net':
            flow = self.pwc_net_optical_flow(prev_frame, curr_frame)
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")
        
        if flow is None:
             print("[ERROR] Optical flow calculation returned None")
             return None

        print(f"[DEBUG] Flow Shape: {flow.shape}")


        # Get the height and width of the frame
        h, w = prev_frame.shape[:2]
        flow_x, flow_y = flow[..., 0], flow[..., 1]

        # Generate mesh grid for pixel coordinates
        y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(int)

        # Calculate new coordinates based on the flow
        new_x = (x + flow_x.flatten()).round().astype(int)
        new_y = (y + flow_y.flatten()).round().astype(int)

        # Clip coordinates to ensure they stay within image bounds
        new_x = np.clip(new_x, 0, w - 1)
        new_y = np.clip(new_y, 0, h - 1)

        # Initialize new mask
        new_mask = np.zeros_like(prev_mask, dtype=float)

        # Safely update the new mask using the clipped coordinates
        try:
            new_mask[new_y, new_x] = prev_mask[y, x]
        except IndexError as e:
            print(f"IndexError during mask update: {e}. Coordinates might be out of bounds.")

        # Apply morphological operations to refine the mask
        kernel = np.ones((5, 5), np.uint8)
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)

        return new_mask
    
    