# This code load RML database

# ==================================
#              Imports
# ==================================
import os
import numpy as np
import cv2
import face_alignment
import shutil


# ==================================
#           Functions
# ==================================
def get_datalist(data_path, path):
    with open(path) as f:
        strings = f.readlines()
        a = strings[0].split()
        videolist = np.array([os.path.join(data_path, string.split()[0]) for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return videolist, labellist


# ==================================
#       Get Train/Val.
# ==================================
print('Loading data...')

#### get data list
#trnlist, trnlb = get_datalist(data_path='F:/RAVDESS', path='F:/Spring2023Research/RAVDESS_ids1.txt')

#new data list
#trnlist, trnlb = get_datalist(data_path='D:/VideoCollection', path='D:/VideoCollection/data/customIDSvR.txt')
trnlist, trnlb = get_datalist(data_path='C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/CARLA_Data/subject_screenshots/', path='C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/CARLA_Data/keys/customIDSvR2.txt')


partition = {'train': trnlist.flatten()}
labels = {'train': trnlb.flatten()}


# ==================================
#       Extracting Landmarks
# ==================================

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

num_class = 8
fix_length = 30 * 3 

# Landmark file names
Graph_train_data = np.zeros(shape=(1, fix_length, 68, 2))
train_label = np.zeros(shape=(1, 1))

q = 0
for c, Name in enumerate(partition['train']):

    cap = cv2.VideoCapture(str(Name))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    train_data = np.zeros(shape=(1, 68, 2))
    
    # used to count the number of frames
    cou = 0

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # make sure the files has more than 40 frames and start processing afterwards to avoid glitch issues in some videos.

            # Print the current frame count
            #print("mad3 it! (ret == True)")
            
            if(cou > 1):
                # Print the current frame count
                #print("mad3 it! (cou > 40)")

                # Print the current frame count
                #print("Processing frame ", cou)

                preds = fa.get_landmarks(frame)
                if (preds != None):

                    if (preds.__len__() > 1):
                        train_data = np.concatenate((train_data, np.reshape(preds[0], newshape=(1, 68, 2))), axis=0)
                    else:
                        train_data = np.concatenate((train_data, preds), axis=0)

        # Break the loop
        else:
            # Print the current frame count
            #print("did not make it :( ")
            # Print the current frame count
            #print("Processing frame ", cou)

            break
        cou += 1

        # Print the current frame count
        #print("Processing frame ", cou)
    
    # Print the current frame count
    #print("Processing frame ", cou)
    # When everything done, release the video capture object
    cap.release()
    print('Number of files:'+str(c))
    
    train_data = train_data[1:]


    # Build nodes' atributes
    if (train_data.shape[0] > 1):
        print("looping")
        Res = train_data.shape[0] % fix_length
        for i in range(int(train_data.shape[0] / fix_length)):
            print("in loop")
            void = np.reshape(train_data[i * fix_length:(i + 1) * fix_length], newshape=(1, fix_length, 68, 2))
            Graph_train_data = np.concatenate((Graph_train_data, void), axis=0)
            train_label = np.concatenate((train_label, np.reshape(labels['train'][c], newshape=(1, 1))), axis=0)



# Save Graph data
#np.save('train_graph_data_RAVDESS4.npy', Graph_train_data[1:])
#np.save('train_graph_label_RAVDESS4.npy', train_label[1:])

# save new data
np.save('11train_graph_data_Custom.npy', Graph_train_data[1:])
np.save('11train_graph_label_Custom.npy', train_label[1:])
