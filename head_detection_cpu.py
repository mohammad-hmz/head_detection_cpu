import cv2
import time
from ultralytics import YOLO
import numpy as np
from collections import deque

# Load model with maximum optimizations
model = YOLO("nano.pt")
model.fuse()  # Fuse layers

# Set model to evaluation mode and optimize
model.model.eval()

cap = cv2.VideoCapture("input.mp4")
fps_in = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

frame_count = 0
total_fps = 0

# Advanced optimization parameters
SKIP_FRAMES = 3  # Process every 4th frame
RESIZE_FACTOR = 0.4  # More aggressive resize
TARGET_SIZE = (int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR))
CONFIDENCE_THRESHOLD = 0.4  # Higher confidence for fewer false positives

# For intelligent frame skipping based on motion
last_frame_gray = None
MOTION_THRESHOLD = 1000  # Adjust based on your video content

# Buffer for smoothing results
result_buffer = deque(maxlen=5)
last_results = []

def detect_motion(current_frame, last_frame):
    """Simple motion detection"""
    if last_frame is None:
        return True
    
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(current_gray, last_frame)
    motion_score = np.sum(diff)
    
    return motion_score > MOTION_THRESHOLD

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    
    # Convert to grayscale for motion detection
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Intelligent frame skipping based on motion and frame count
    should_process = (frame_count % (SKIP_FRAMES + 1) == 0) or \
                    detect_motion(frame, last_frame_gray)
    
    if should_process:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, TARGET_SIZE)
        
        # Run detection with optimized parameters
        results = model.predict(
            small_frame, 
            conf=CONFIDENCE_THRESHOLD, 
            verbose=False,
            imgsz=min(TARGET_SIZE),  # Use smaller dimension
            half=False,  # Use FP16 if supported
            augment=False,  # Disable augmentation for speed
            agnostic_nms=True  # Faster NMS
        )
        
        # Scale results back to original size
        current_results = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # Scale coordinates back to original size
            x1 = int(x1 / RESIZE_FACTOR)
            y1 = int(y1 / RESIZE_FACTOR)
            x2 = int(x2 / RESIZE_FACTOR)
            y2 = int(y2 / RESIZE_FACTOR)
            current_results.append((x1, y1, x2, y2))
        
        # Add to buffer for smoothing
        result_buffer.append(current_results)
        
        # Use average of recent results for stability
        if result_buffer:
            last_results = result_buffer[-1]  # Use most recent for now
    
    # Draw results
    for box_coords in last_results:
        x1, y1, x2, y2 = box_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    total_fps += fps
    frame_count += 1

    # More efficient text rendering
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Head Detection", frame)
    out.write(frame)
    
    # Store current frame for next motion detection
    last_frame_gray = current_gray

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

if frame_count > 0:
    print(f"✅ Average FPS: {total_fps / frame_count:.2f}")
    print(f"✅ Processed {frame_count} frames")
    print(f"✅ Motion-based optimization enabled")