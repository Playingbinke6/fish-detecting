import cv2
import os

"""
Scan every nth frame of a video.
Used this for collecting more seabed images for the model
"""
video_path = r'/HAAR-model/data\raw\Labeled_Fishes_In_The_Wild\_LABELED-FISHES-IN-THE-WILD\Test\Test_ROV_video_h264_decim.mp4'
output_folder = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\non_fish_from_frame_script'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
nth = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % nth == 0:  # look at every nth frame
        filename = os.path.join(output_folder, f"neg_frame_{frame_count}.jpg")
        cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()
