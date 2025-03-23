import os


def convert_opencv_to_yolo(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, "r") as f:
        lines = f.readlines()

    total_lines = len(lines)
    processed_lines = 0

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 3:
            print(f"❌ Skipping malformed line: {line.strip()}")
            continue

        try:
            video_filename = parts[0]  # Extract the filename
            frame_number = video_filename.split("(fr_")[1].split(")")[0]  # Extract frame number
            label_filename = os.path.join(output_folder, f"{frame_number}.txt")  # YOLO annotation file

            num_objects = int(parts[1])
            bboxes = parts[2:]

            if len(bboxes) != num_objects * 4:
                print(f"⚠️ Warning: Expected {num_objects * 4} values but got {len(bboxes)} in: {line.strip()}")
                continue

            with open(label_filename, "w") as out_file:
                for i in range(num_objects):
                    x, y, w, h = map(int, bboxes[i * 4:(i + 1) * 4])

                    # Convert to YOLO format (normalize values)
                    x_center = x / 1080  # Assuming 1080p resolution
                    y_center = y / 720  # Assuming 720p resolution
                    width = w / 1080
                    height = h / 720

                    out_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            processed_lines += 1
            print(f"✅ Processed: {video_filename} → {label_filename} ({num_objects} objects)")

        except ValueError as e:
            print(f"❌ Error parsing data on line: {line.strip()} - {e}")
            continue

    print(f"✅ Successfully processed {processed_lines}/{total_lines} frames.")

# Test labels
convert_opencv_to_yolo("C:\\Users\\playi\\PycharmProjects\\fish-detecting\\data\\raw\\Labeled_Fishes_In_The_Wild\\_LABELED-FISHES-IN-THE-WILD\\Test\\Test_ROV_video_h264_decim_marks.dat", "C:\\Users\\playi\\PycharmProjects\\fish-detecting\\data\\labels\\test")
# Val labels
convert_opencv_to_yolo("C:\\Users\\playi\\PycharmProjects\\fish-detecting\\data\\raw\\Labeled_Fishes_In_The_Wild\\_LABELED-FISHES-IN-THE-WILD\\Test\\Test_ROV_video_h264_decim_marks.dat", "C:\\Users\\playi\\PycharmProjects\\fish-detecting\\data\\labels\\test")


