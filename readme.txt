YOLOv8 Real-Time Object Detection Script - Summary

1. Overview:
- Uses Ultralytics YOLOv8 model (yolov8n.pt) for real-time object detection via webcam.
- Supports GPU acceleration, FPS display.

2. Approach:
- Loads YOLOv8 model with optimizations (half precision on GPU).
- Captures video frames using OpenCV at 640x480 resolution.
- Runs model inference on each frame with confidence threshold and max detections.
- Annotates frames with detected objects.
- Displays FPS and inference time on video.

3. Usage:
- Requires Python 3.8+.
- Install dependencies from requirements.txt.
- Download 'yolov8n.pt' weights.
- Run script and press 'Q' to quit.

4. Configuration Parameters:
- MODEL_NAME: YOLO weights filename.
- INPUT_SIZE: Model input resolution (320, 416, 640).
- CONF_THRESH: Confidence threshold for detections.
- MAX_DETECTIONS: Max objects detected per frame.
- USE_GPU: Enable GPU if available.
- SHOW_FPS: Display FPS counter.

5. Performance:
- Tracks and displays real-time FPS and inference latency.

6. requirements.txt:
torch>=1.13.0
opencv-python>=4.5.5
ultralytics>=8.0.0
numpy>=1.21.0

7. Notes:
- Adjust INPUT_SIZE for speed vs accuracy tradeoff.
- Warmup model with a test image for better initial performance.
- FP16 half precision used only on GPU.
- Ensure webcam is accessible.

---

End of summary.
