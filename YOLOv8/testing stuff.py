import os
# This became a .py script/dev file for modifying the data of the YOLO model

# Set paths to label directories
train_labels_dir = r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\dataset\labels\train'
val_labels_dir = r'C:\Users\playi\PycharmProjects\fish-detecting\YOLOv8\dataset\labels\val'

# Script to change all of the class identifiers to zero, meant to help with poor identification
def correct_non_fish_labels(labels_dir):
    print(f"\nChecking and correcting labels in: {labels_dir}")

    if not os.path.exists(labels_dir):
        print(f"Directory not found: {labels_dir}")
        return

    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_dir, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()

            modified = False
            with open(file_path, "w") as file:
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])  # First value is the class ID

                    # If class_id is not 0, change it to 0 (consider as fish)
                    if class_id != 0:
                        parts[0] = "0"  # Change the class_id to 0
                        modified = True

                    # Write the modified (or unmodified) line back to the file
                    file.write(" ".join(parts) + "\n")

            if modified:
                print(f"Updated {filename} - Changed non-zero class IDs to 0.")
            else:
                print(f"{filename} already contains only class 0.")


# Run checks for training and validation labels
correct_non_fish_labels(train_labels_dir)
correct_non_fish_labels(val_labels_dir)
