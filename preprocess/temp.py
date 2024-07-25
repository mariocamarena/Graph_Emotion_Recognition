'''
import os

# Function to replace 'avi_files2' with 'avi_files3' in a text file and save it to a new location
def replace_and_copy(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Replace 'avi_files2' with 'avi_files3'
    modified_content = content.replace('avi_files2', 'avi_files3')
    
    # Write the modified content to the new file
    with open(output_file, 'w') as file:
        file.write(modified_content)
    
    print(f"File copied and modified successfully: {output_file}")

# Example usage
input_file = 'C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/preprocess/RML4.txt'
output_file = 'C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/preprocess/RML5.txt'

replace_and_copy(input_file, output_file)

'''

import cv2
import os

# Function to create a video with repeated frames
def create_repeated_video(input_file, output_file, num_frames):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from video {input_file}.")
        return
    
    frame_height, frame_width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    for _ in range(num_frames):
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Video created successfully: {output_file}")

# Function to process all .avi files in a directory and its subfolders
def process_directory(input_dir, output_dir, num_frames):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.avi'):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.join(output_subdir, file)
                create_repeated_video(input_file, output_file, num_frames)

# Example usage
input_dir = 'C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/test/avi_files2'
output_dir = 'C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/test/avi_files3'
num_frames = 30

process_directory(input_dir, output_dir, num_frames)