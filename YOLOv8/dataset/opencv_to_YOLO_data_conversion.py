import os

# Define the image dimensions
image_width = 2048
image_height = 1536

# Directory for the images
image_dir = r'/YOLOv8/dataset/data_raw/Training_and_validation/Positive_fish'

# Directory to save the label files in YOLO format
label_dir = r'/YOLOv8/dataset/labels'

# Path to the .dat file
dat_file_path = r'/YOLOv8/dataset/data_raw/Training_and_validation/Positive_fish/Positive_fish_(ALL)-MARKS_DATA.dat'


def convert_to_yolo_format(dat_file_path, image_dir, label_dir, image_width, image_height):
    # Open and read the .dat file
    with open(dat_file_path, 'r') as f:
        lines = f.readlines()

    # Create the label directory if it doesn't exist
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Process each line in the .dat file
    for line in lines:
        # Split the line into parts
        parts = line.strip().split()

        # Extract information from the .dat file
        image_filename = parts[0]
        class_id = int(parts[1]) - 1  # YOLO class IDs start at 0
        x_min = int(parts[2])
        y_min = int(parts[3])
        x_max = int(parts[4])
        y_max = int(parts[5])

        # Construct the path to the image
        image_path = os.path.join(image_dir, image_filename)

        # Calculate the normalized YOLO format values
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Create the corresponding label file
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_file_path = os.path.join(label_dir, label_filename)

        # Write the label in YOLO format
        with open(label_file_path, 'a') as label_file:
            label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print("Conversion complete.")


# Run the conversion
convert_to_yolo_format(dat_file_path, image_dir, label_dir, image_width, image_height)
