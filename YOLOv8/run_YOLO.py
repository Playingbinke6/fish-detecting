from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO(r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\yolov8n.pt')

# Define the video for testing
source = r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\dataset\data_raw\Test\Test_ROV_video_h264_decim.mp4'

# Run inference
results = model(source, save=True, show=True)

# Optional: Loop through results and print detections
for result in results:
    print(result.boxes)  # Bounding boxes, confidence, class IDs
