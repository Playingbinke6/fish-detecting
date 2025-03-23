import cv2
import os

def detect_fish(video_path: str, cascade_path: str, output_path: str = None):
    """
    Runs fish detection on an ROV video using a trained Haar cascade classifier.

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
    cap = cv2.VideoCapture(video_path)

    # Video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
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

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    VIDEO_PATH = os.path.join('videos', 'Test_ROV_video.mp4')
    CASCADE_PATH = os.path.join('models', 'cascade.xml')
    OUTPUT_PATH = os.path.join('videos', 'output_detected.mp4')

    detect_fish(VIDEO_PATH, CASCADE_PATH, OUTPUT_PATH)
