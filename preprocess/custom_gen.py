# This code load RML database

# ==================================
#              Imports
# ==================================
import os
import numpy as np
import pandas as pd
import cv2
#import face_alignment
import shutil


# ==================================
#           Functions
# ==================================
def get_datalist(data_path, path):
    with open(path) as f:
        strings = f.readlines()
        a = strings[0].split()
        imagelist = np.array([os.path.join(data_path, string.split()[0]) for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return imagelist, labellist


# ==================================
#       Get Train/Val.
# ==================================
print('Loading data...')

# get data list
trnlist, trnlb = get_datalist(data_path='C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/test/avi_files', path='C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/test/customIDSvR3.txt')

partition = {'train': trnlist.flatten()}
labels = {'train': trnlb.flatten()}


# ==================================
#       Load Landmarks from CSV
# ==================================

landmarks_df = pd.read_csv("C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/dataset/Mine_Graph_Custom/Mine_Graph_Custom.txt")  # Replace with the filename of your CSV file

Graph_train_data = np.zeros(shape=(1, 90, 68, 2))
train_label = np.zeros(shape=(1, 1))
fix_length = 30 * 3

for c, Name in enumerate(partition['train']):

    # Find the corresponding row in the CSV file
    row = landmarks_df.loc[landmarks_df['filename'] == Name]

    if row.empty:
        # Handle the case where the CSV file does not contain landmarks for this image
        continue

    landmarks = row.values[:, 1:].astype(np.float32)
    landmarks = np.reshape(landmarks, (-1, 68, 2))

    # Build nodes' attributes
    if (landmarks.shape[0] > 1):
        Res = landmarks.shape[0] % fix_length
        for i in range(np.int(landmarks.shape[0] / fix_length)):
            void = np.reshape(landmarks[i * fix_length:(i + 1) * fix_length], newshape=(1, fix_length, 68, 2))
            Graph_train_data = np.concatenate((Graph_train_data, void), axis=0)
            train_label = np.concatenate((train_label, np.reshape(labels['train'][c], newshape=(1, 1))), axis=0)

    print('Number of files:'+str(c))

# Save Graph data
np.save('train_graph_data_RML.npy', Graph_train_data[1:])
np.save('train_graph_label_RML.npy', train_label[1:])