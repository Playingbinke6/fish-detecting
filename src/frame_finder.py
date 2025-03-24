import cv2
import os

video_path = r'C:\Users\playi\PycharmProjects\fish-detecting\data\raw\Labeled_Fishes_In_The_Wild\_LABELED-FISHES-IN-THE-WILD\Test\Test_ROV_video_h264_decim.mp4'
output_folder = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\non_fish_from_frame_script'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Manually inspect frames or use intervals
    if frame_count % 1 == 0:  # every 50th frame
        filename = os.path.join(output_folder, f"neg_frame_{frame_count}.jpg")
        cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()
