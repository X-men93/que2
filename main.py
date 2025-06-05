import cv2
import torch
from ultralytics import YOLO
import time

# ========================
# Configuration Settings
# ========================
MODEL_NAME = 'yolov8n.pt'       # Fastest model for real-time
INPUT_SIZE = 320                # Lower resolution for speed (320, 416, 640)
CONF_THRESH = 0.5               # Confidence threshold
MAX_DETECTIONS = 15             # Limit number of detections per frame
TARGET_FPS = 30                 # Target frame rate for display
USE_GPU = True                  # Set to False if no GPU available
SHOW_FPS = True                 # Display FPS counter
RECORD_VIDEO = False            # Set to True to save output

# ========================
# Initialize System
# ========================
# Check for GPU availability
device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device.upper()}")

# Load model with optimizations
model = YOLO(MODEL_NAME).to(device)
model.fuse()  # Fuse layers for faster inference

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

# Set camera resolution (lower for better FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Video writer setup
if RECORD_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('detection_output.mp4', fourcc, 20.0, 
                         (int(cap.get(3)), int(cap.get(4))))

# ========================
# Performance Monitoring
# ========================
frame_count = 0
start_time = time.time()
fps_history = []
inference_times = []

# Warmup the model
print("🔥 Warming up model...")
warmup_frame = cv2.imread('path/to/test.jpg') if device == 'cuda' else None
if warmup_frame is not None:
    _ = model(warmup_frame, imgsz=INPUT_SIZE, verbose=False)
print("✅ Ready for real-time detection")

# ========================
# Main Processing Loop
# ========================
print("\nPress 'Q' to quit")
prev_frame_time = time.time()

while cap.isOpened():
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture error")
        break
    
    # Skip processing on some frames if needed to maintain FPS
    # if frame_count % 2 == 0:  # Process every other frame
    #     continue
    
    # Run inference with optimizations
    inference_start = time.time()
    results = model(
        frame,
        imgsz=INPUT_SIZE,
        conf=CONF_THRESH,
        device=device,
        half=True if 'cuda' in device else False,  # FP16 on GPU
        augment=False,          # Disable augmentation for speed
        max_det=MAX_DETECTIONS,
        verbose=False           # Disable console output
    )
    inference_end = time.time()
    
    # Render results
    annotated_frame = results[0].plot()
    
    # Calculate performance metrics
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time
    
    # Track performance
    frame_count += 1
    fps_history.append(fps)
    inference_times.append(inference_end - inference_start)
    
    # Display FPS counter
    if SHOW_FPS:
        avg_fps = sum(fps_history[-10:]) / min(10, len(fps_history))
        cv2.putText(annotated_frame, f"FPS: {int(avg_fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Inference: {inference_times[-1]*1000:.1f}ms", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Save video if enabled
    if RECORD_VIDEO:
        out.write(annotated_frame)
    
    # Display output
    cv2.imshow('Real-Time Object Detection', annotated_frame)
    
    # Exit on 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========================
# Cleanup & Performance Report
# ========================
cap.release()
if RECORD_VIDEO:
    out.release()
cv2.destroyAllWindows()

# Calculate final performance
total_time = time.time() - start_time
avg_fps = frame_count / total_time
avg_inference = sum(inference_times) / len(inference_times) * 1000

print("\n" + "="*50)
print(f"📊 Performance Report")
print("="*50)
print(f"Total frames processed: {frame_count}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Average inference time: {avg_inference:.1f}ms")
print(f"Peak FPS: {max(fps_history):.1f}")
print(f"Model: {MODEL_NAME} | Input size: {INPUT_SIZE}")
print(f"Hardware: {device.upper()}")
print("="*50)