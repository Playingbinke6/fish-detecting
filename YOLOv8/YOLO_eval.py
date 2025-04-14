from ultralytics import YOLO
"""
never could get this to work properly
"""

# Load trained model
model = YOLO(r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\yolov8n.pt')

# Path to the test dataset
test_data = r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\MYdataset.yaml'

# Evaluate the model on the test data
results = model.val(data=test_data)

# Print results
print("Results:", results)

# Print more detailed metrics (if available)
print(f"Precision: {results.pmap}")
print(f"Recall: {results.rmap}")
print(f"Mean Average Precision (mAP): {results.map}")
