from ultralytics import YOLO
import torch

def train_yolo_model():
    # Check if CUDA (GPU) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define the path to your dataset and configuration file
    dataset_yaml = r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\MYdataset.yaml'

    # Define the model architecture, use the YOLOv8 nano model here (yolov8n), can use others (yolov8s, yolov8m, etc.)
    model_architecture = "yolov8s.yaml"

    # Define training parameters
    epochs = 50 # Like the stages of HAAR
    imgsz = 640  # Image size for training
    batch_size = 24  # Adjust based on system resources (tower/laptop)

    # Load the model and start training
    model = YOLO(model_architecture)  # Create a YOLO model

    # Train the model
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,  # Adjust based on GPU memory
        imgsz=imgsz,  # Adjust for better detection
        workers=4,  # Use more if CPU supports it
        device="cuda"  # Ensure training runs on GPU
    )


# Call the function to start training
if __name__ == "__main__":
    train_yolo_model()
