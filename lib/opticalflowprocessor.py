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
        def get_optimizer():
            return None

class OpticalFlowProcessor:
    def __init__(self, method, preset=None):
        self.method = method
        self.raft_model = None
        self._vision_objects = None

        # Use optimized device detection (GPU if available, CPU fallback)
        import torch
        optimizer = get_optimizer()
        if optimizer and hasattr(optimizer, 'device'):
            self.device = optimizer.device
        else:
            self.device = torch.device('cpu')

        # Preset: used by DIS (ultrafast/fast/medium) and Vision (low/medium/high)
        self.preset = (preset or os.getenv('DIS_PRESET', 'medium')).lower()

        self.dis_preset = {
            'ultrafast': cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            'fast': cv2.DISOPTICAL_FLOW_PRESET_FAST,
            'medium': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
        }.get(self.preset, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

        # DIS resolution scale: compute flow at reduced resolution, upscale result.
        self.dis_scale = int(os.getenv('DIS_SCALE', '1'))

        if self.method == 'raft':
            self.load_raft_model()
        elif self.method == 'vision':
            self._init_vision()

    def _init_vision(self):
        """Initialize Apple Vision framework optical flow objects."""
        import Vision
        import Quartz

        accuracy_map = {
            'low': Vision.VNGenerateOpticalFlowRequestComputationAccuracyLow,
            'medium': Vision.VNGenerateOpticalFlowRequestComputationAccuracyMedium,
            'high': Vision.VNGenerateOpticalFlowRequestComputationAccuracyHigh,
        }
        accuracy = accuracy_map.get(self.preset)
        if accuracy is None:
            raise ValueError(
                f"Unknown Vision preset: {self.preset}. Use low, medium, or high."
            )
        self._vision_objects = {
            'accuracy': accuracy,
            'colorspace': Quartz.CGColorSpaceCreateDeviceRGB(),
            'Vision': Vision,
            'Quartz': Quartz,
        }

    def vision_optical_flow(self, prev_bgr, curr_bgr):
        """Compute optical flow using Apple Vision framework (GPU/ANE)."""
        vo = self._vision_objects
        Vision = vo['Vision']
        Quartz = vo['Quartz']

        h, w = prev_bgr.shape[:2]
        bgra_p = np.ascontiguousarray(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2BGRA))
        bgra_c = np.ascontiguousarray(cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2BGRA))

        ci_p = Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
            Quartz.NSData.dataWithBytes_length_(bgra_p.tobytes(), bgra_p.nbytes),
            w * 4, Quartz.CGSizeMake(w, h), Quartz.kCIFormatBGRA8, vo['colorspace'])
        ci_c = Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
            Quartz.NSData.dataWithBytes_length_(bgra_c.tobytes(), bgra_c.nbytes),
            w * 4, Quartz.CGSizeMake(w, h), Quartz.kCIFormatBGRA8, vo['colorspace'])

        req = Vision.VNGenerateOpticalFlowRequest.alloc().initWithTargetedCIImage_options_(ci_c, {})
        req.setComputationAccuracy_(vo['accuracy'])
        hdl = Vision.VNSequenceRequestHandler.alloc().init()
        ok, err = hdl.performRequests_onCIImage_error_([req], ci_p, None)
        if not ok:
            raise RuntimeError(f"Vision optical flow failed: {err}")

        obs = req.results()[0]
        buf = obs.pixelBuffer()
        Quartz.CVPixelBufferLockBaseAddress(buf, 0)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(buf)
        ph = Quartz.CVPixelBufferGetHeight(buf)
        pw = Quartz.CVPixelBufferGetWidth(buf)
        base = Quartz.CVPixelBufferGetBaseAddress(buf)
        raw = bytes(base.as_buffer(bpr * ph))
        flow = np.frombuffer(raw, dtype=np.float32).reshape(ph, -1)[:, :pw * 2].reshape(ph, pw, 2).copy()
        Quartz.CVPixelBufferUnlockBaseAddress(buf, 0)
        return flow

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
        Calculate optical flow between two frames.

        Args:
            prev_frame: Previous frame (BGR or grayscale)
            curr_frame: Current frame (BGR or grayscale)
            method: Optional method override

        Returns:
            Optical flow field as numpy array (H, W, 2)
        """
        flow_method = method or self.method

        # Vision needs BGR color input; all others need grayscale
        if flow_method == 'vision':
            if len(prev_frame.shape) == 2:
                prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2BGR)
                curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2BGR)
            return self.vision_optical_flow(prev_frame, curr_frame)

        # Convert to grayscale for CPU-based methods
        if len(prev_frame.shape) == 3:
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        if len(curr_frame.shape) == 3:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if flow_method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif flow_method == 'dis':
            dis = cv2.DISOpticalFlow_create(self.dis_preset)
            if self.dis_scale > 1:
                s = self.dis_scale
                h, w = prev_frame.shape[:2]
                prev_scaled = cv2.resize(prev_frame, (w // s, h // s))
                curr_scaled = cv2.resize(curr_frame, (w // s, h // s))
                flow = cv2.resize(dis.calc(prev_scaled, curr_scaled, None), (w, h)) * float(s)
            else:
                flow = dis.calc(prev_frame, curr_frame, None)
        elif flow_method == 'raft':
            prev_frame_tensor = torch.from_numpy(prev_frame).unsqueeze(0).unsqueeze(0).float() / 255.0
            curr_frame_tensor = torch.from_numpy(curr_frame).unsqueeze(0).unsqueeze(0).float() / 255.0
            flow = self.raft_optical_flow(prev_frame_tensor, curr_frame_tensor)
        else:
            raise ValueError(f"Unknown optical flow method: {flow_method}")

        return flow