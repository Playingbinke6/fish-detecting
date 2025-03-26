import subprocess
import os

# Paths
opencv_path = r'C:\Users\playi\opencv\opencv 3.4.16\build\x64\vc15\bin'
traincascade_exe = os.path.join(opencv_path, 'opencv_traincascade.exe')

output_dir = r'C:\Users\playi\PycharmProjects\fish-detecting\HAAR-model\models'  # This is where the XML files will go
vec_file = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\HAAR-model-data-files\fish_all_40_30.vec'
bg_file = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\HAAR-model-data-files\bg.txt'

# Params
num_pos = 400
num_neg = 200
num_stages = 80
width = 40
height = 30
feature_type = 'HAAR'  # or 'LBP'

# Adjustable performance parameters
precalc_val_buf_size = 4096  # Memory buffer for precomputed values (MB) (12GB total at 6144MB each)
precalc_idx_buf_size = 4096  # Memory buffer for indices (MB)
num_threads = 8  # Number of CPU cores to use (adjust based on system)

# Build the command
command = [
    traincascade_exe,
    '-data', output_dir,
    '-vec', vec_file,
    '-bg', bg_file,
    '-numPos', str(num_pos),
    '-numNeg', str(num_neg),
    '-numStages', str(num_stages),
    '-w', str(width),
    '-h', str(height),
    '-featureType', feature_type
]

# Run the command
print("Starting training...")
subprocess.run(command)
print("Training complete!")
