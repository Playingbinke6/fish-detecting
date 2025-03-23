import os
import subprocess

def train_haar_cascade(
    output_dir: str,
    vec_file: str,
    bg_file: str,
    num_pos: int,
    num_neg: int,
    num_stages: int = 10,
    width: int = 50,
    height: int = 50,
    feature_type: str = 'LBP'
):
    """
    Trains a Haar Cascade classifier using OpenCV's traincascade tool.

    :param output_dir: Directory to save trained cascade files.
    :param vec_file: Path to the .vec file with positive samples.
    :param bg_file: Path to the bg.txt file listing negative samples.
    :param num_pos: Number of positive samples.
    :param num_neg: Number of negative samples.
    :param num_stages: Number of stages for training.
    :param width: Width of detection window.
    :param height: Height of detection window.
    :param feature_type: Feature type: 'HAAR' or 'LBP'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the command
    command = [
        'opencv_traincascade',
        f'-data', output_dir,
        f'-vec', vec_file,
        f'-bg', bg_file,
        f'-numPos', str(num_pos),
        f'-numNeg', str(num_neg),
        f'-numStages', str(num_stages),
        f'-w', str(width),
        f'-h', str(height),
        f'-featureType', feature_type
    ]

    print(f"Running command: {' '.join(command)}")

    # Execute the training process
    subprocess.run(command)


if __name__ == '__main__':
    OUTPUT_DIR = os.path.join('data', 'cascade')
    VEC_FILE = os.path.join('datasets', 'positives.vec')
    BG_FILE = os.path.join('datasets', 'bg.txt')

    NUM_POS = 600
    NUM_NEG = 300
    NUM_STAGES = 12
    WIDTH = 50
    HEIGHT = 50
    FEATURE_TYPE = 'LBP'

    train_haar_cascade(
        output_dir=OUTPUT_DIR,
        vec_file=VEC_FILE,
        bg_file=BG_FILE,
        num_pos=NUM_POS,
        num_neg=NUM_NEG,
        num_stages=NUM_STAGES,
        width=WIDTH,
        height=HEIGHT,
        feature_type=FEATURE_TYPE
    )
