import subprocess
import os

# Paths (edit these based on your folder setup)
opencv_path = r'C:\Users\playi\opencv\opencv 3.4.16\build\x64\vc15\bin'
traincascade_exe = os.path.join(opencv_path, 'opencv_traincascade.exe')

output_dir = r'C:\Users\playi\PycharmProjects\fish-detecting\models'  # This is where the XML files will go
vec_file = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\fish_all_40_30.vec'
bg_file = r'C:\Users\playi\PycharmProjects\fish-detecting\input_data\bg.txt'

# Params
num_pos = 400
num_neg = 200
num_stages = 40
width = 40
height = 30
feature_type = 'HAAR'  # or 'LBP'

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
