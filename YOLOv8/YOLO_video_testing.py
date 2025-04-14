from ultralytics import YOLO
# Saving the video negatively affects performance of both models, so a separate recording process was used

# Load the YOLOv8 model
model = YOLO(r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\yolov8n.pt')

# Path to the input video for testing
input_video_path = r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\dataset\data_raw\Test\Test_ROV_video_h264_decim.mp4'
output_video_path = r'C:\Users\playi\PycharmProjects\fish-detecting\output'

# Perform prediction on the video and save the output
results = model.predict(source=input_video_path, save=True, save_txt=True, project=output_video_path, name='test_video')

# The resulting video with bounding boxes will be saved in the 'runs/inference/test_video' folder
