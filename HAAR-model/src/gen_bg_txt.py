import os

def generate_bg_file(negatives_dir: str, output_file: str):
    """
    Generates a bg.txt file listing all negative images.

    :param negatives_dir: Directory containing negative images.
    :param output_file: Path to save the bg.txt file.
    """
    with open(output_file, 'w') as bg_file:
        for filename in os.listdir(negatives_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(negatives_dir, filename)
                bg_file.write(f"{file_path}\n")
    print(f"Background file saved to {output_file}")


if __name__ == '__main__':
    NEGATIVES_DIR = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\non_fish_images'
    OUTPUT_FILE = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\HAAR-model-data-files\bg.txt'

    generate_bg_file(NEGATIVES_DIR, OUTPUT_FILE)
