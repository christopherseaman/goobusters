Because we're tracking (busting) free fluid (goo) and I grew up on 80's movies.

# TODO
- [ ] Direct Pixel Mask Tracking:
 > Instead of reconstructing the mask from tracked points using polygons, track the movement of the entire pixel mask directly. Optical flow can be applied to the entire mask, propagating pixel-level changes across frames.
- [ ] Refine the Mask Using Morphology:
 > After the optical flow step, apply morphological operations to clean up the mask, ensuring it accurately reflects the fluid regions.
- [ ] Adaptive Thresholding:
 > Implement adaptive thresholding to refine the mask after optical flow, ensuring that any noise or small artifacts are removed.


# Summary
- **Quick Implementation**: Optical flow provides a fast and straightforward method for propagating annotations.
- **Suitable for Proof of Concept**: Ideal for validating the feasibility of your approach before investing more resources.
- **Next Steps**: If successful, consider more advanced methods like keypoint tracking or deep learning for improved accuracy.
# Project Review and Problem Statement
## Premise:
- **Input**: 3-second ultrasound videos from various views, each with free fluid annotated in at least one frame.
- **Goal**: Propagate the annotations of free fluid from the single annotated frame to all other frames in the video to create fully annotated videos.
- **Resources**: Access to a reasonably powerful GPU and strong Python skills.

## Objective
Annotate free fluid in all frames of each video, leveraging existing annotations as a starting point.

# Options

1. **Optical Flow**:
   - **Description**: Use optical flow to estimate motion between frames and propagate the annotated regions.
   - **Tools**: OpenCV’s Farneback or Lucas-Kanade optical flow implementations.
   - **Advantages**: Quick to set up, leverages frame-to-frame continuity, suitable for tracking amorphous regions.
   - **Steps**:
     1. **Load Video**: Read the video and the annotated frame.
     2. **Compute Optical Flow**: Estimate motion vectors between consecutive frames.
     3. **Propagate Annotation**: Use motion vectors to transform the annotated region across frames.

2. **Keypoint Tracking**:
   - **Description**: Detect keypoints within the annotated region and track these points across frames.
   - **Tools**: OpenCV’s KLT Tracker.
   - **Advantages**: Tracks specific points within the region, can handle non-rigid motion.
   - **Steps**:
     1. **Initialize Keypoints**: Detect keypoints in the annotated frame.
     2. **Track Keypoints**: Track these keypoints across frames.
     3. **Transform Annotation**: Update the region based on tracked keypoints.

3. **Deep Learning with Unsupervised Learning**:
   - **Description**: Use a pre-trained deep learning model fine-tuned on the specific task of tracking and segmentation.
   - **Tools**: TensorFlow, PyTorch, models like U-Net for segmentation.
   - **Advantages**: High accuracy, adaptable to complex motion and appearance changes.
   - **Steps**:
     1. **Fine-Tune Model**: Use the annotated frames to fine-tune a segmentation model.
     2. **Predict Annotations**: Use the model to predict annotations for the remaining frames.

# Detailed Steps for Optical Flow Approach

1. **Install Dependencies**:
   - Install OpenCV if not already installed:
     ```bash
     pip install opencv-python
     ```

2. **Load Video and Initial Frame**:
   ```python
   import cv2
   import numpy as np

   cap = cv2.VideoCapture('path_to_your_video.mp4')
   ret, frame1 = cap.read()
   prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

   # Load the initial annotation (replace with actual loading code)
   initial_annotation = np.array([[x1, y1], [x2, y2]])  # Example format
   ```

3. **Compute Optical Flow**:
   ```python
   hsv = np.zeros_like(frame1)
   hsv[..., 1] = 255

   while cap.isOpened():
       ret, frame2 = cap.read()
       if not ret:
           break
       next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
       flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
       prvs = next

       # Convert flow to HSV for visualization (optional)
       mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
       hsv[..., 0] = ang * 180 / np.pi / 2
       hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
       rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
       cv2.imshow('Optical Flow', rgb)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()
   ```

4. **Propagate Annotation**:
   ```python
   def propagate_annotation(annotation, flow):
       new_annotation = []
       for point in annotation:
           x, y = point
           dx, dy = flow[int(y), int(x)]
           new_annotation.append([x + dx, y + dy])
       return np.array(new_annotation)

   cap = cv2.VideoCapture('path_to_your_video.mp4')
   ret, frame1 = cap.read()
   prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
   current_annotation = initial_annotation

   while cap.isOpened():
       ret, frame2 = cap.read()
       if not ret:
           break
       next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
       flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
       current_annotation = propagate_annotation(current_annotation, flow)
       prvs = next

       # Optional: visualize or save the propagated annotation
       for point in current_annotation:
           cv2.circle(frame2, tuple(point), 5, (0, 255, 0), -1)
       cv2.imshow('Annotated Frame', frame2)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()
   ```
