import cv2
import os

# Load the HAAR Cascade model
fish_cascade = cv2.CascadeClassifier(r'C:\Users\playi\PycharmProjects\fish-detecting\HAAR-model\models\cascade.xml')

# Path to the input video for testing
input_video_path = r'C:\Users\playi\PycharmProjects\fish-detecting\HAAR-model\videos\Test_ROV_video_h264_decim.mp4'

# Ensure the output directory exists
output_directory = r'C:\Users\playi\PycharmProjects\fish-detecting\output'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Set the path where the video will be saved
output_video_path = os.path.join(output_directory, "output_video_haar.mp4")  # Specify the output path

# Open the input video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties for saving output video
# saving the video negatively affects performance of both models, so a separate recording process was used
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the processed video as .mp4
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for HAAR Cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect fish (adjust the scaleFactor, minNeighbors, and minSize parameters as needed)
    fish = fish_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected fish
    for (x, y, w, h) in fish:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the frame with bounding boxes to the output video file
    out.write(frame)

# Release the video objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")
