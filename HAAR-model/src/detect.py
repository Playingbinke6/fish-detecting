import cv2
import os

def detect_fish(video_path: str, cascade_path: str, output_path: str = None):
    """
    Runs fish detection on an ROV video using a trained Haar/LBP cascade classifier.

    :param video_path: Path to the input video file.
    :param cascade_path: Path to the trained cascade XML file.
    :param output_path: Optional path to save the output video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    # Load cascade
    fish_cascade = cv2.CascadeClassifier(cascade_path)

    # Open video capture
    capture = cv2.VideoCapture(video_path)

    # Video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fish_boxes = fish_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in fish_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Fish Detection', frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    VIDEO_PATH = r'C:\Users\playi\PycharmProjects\fish-detecting\HAAR-model\videos\Test_ROV_video_h264_decim.mp4'
    CASCADE_PATH = r'C:\Users\playi\PycharmProjects\fish-detecting\HAAR-model\models\old trained models 40\cascade.xml'
    OUTPUT_PATH = r'C:\Users\playi\PycharmProjects\fish-detecting\HAAR-model\output'

    detect_fish(VIDEO_PATH, CASCADE_PATH, OUTPUT_PATH)
