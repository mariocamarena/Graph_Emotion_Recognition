
import numpy as np

directory = 'C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/preprocess/'


tr_feat = np.load('C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/preprocess/train_graph_data_RML.npy')
tr_label = np.load('C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/preprocess/train_graph_data_RML.npy')
#tr_feat = np.load(directory + 'train_graph_data_RAVDESS4.npy')
#tr_label = np.load(directory + 'train_graph_label_RAVDESS4.npy')

tr_feat = np.reshape(tr_feat, newshape=(tr_feat.shape[0],
                                        tr_feat.shape[1], 68*2))


txt = []

txt.append(tr_feat.shape[0])
for i in range(tr_feat.shape[0]):
    for j in range(tr_feat.shape[1]):
        if(j==0):
            txt.append('%s %s'% (tr_feat.shape[1], int(tr_label[i,0])))
        if(j==tr_feat.shape[1]-1    ):
            a = [int(e) for e in tr_feat[i, j]]
            txt.append('%s 1 %s ' % (j, j -1) + ' '.join(str(e) + ' ' for e in a))
        else:
            a = [int(e) for e in tr_feat[i,j]]
            txt.append('%s 1 %s '% (j, j+1)+ ' '.join(str(e)+' ' for e in a))

np.savetxt('0000Mine_Graph_Custom.txt', txt, fmt='%s')

#np.savetxt('Mine_Graph_RAVDESS4.txt', txt, fmt='%s')


'''
import numpy as np

tr_feat = np.load('C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/preprocess/1train_graph_data_RML.npy')
tr_label = np.load('C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/preprocess/1train_graph_data_RML.npy')

tr_feat = np.reshape(tr_feat, newshape=(tr_feat.shape[0],
                                        tr_feat.shape[1], 68*2))


txt = []

txt.append(tr_feat.shape[0])
for i in range(tr_feat.shape[0]):
    for j in range(tr_feat.shape[1]):
        if(j==0):
            txt.append('%s %s'% (tr_feat.shape[1], int(tr_label[i,0])))
        if(j==tr_feat.shape[1]-1    ):
            a = [int(e) for e in tr_feat[i, j]]
            txt.append('%s 1 %s ' % (j, j -1) + ' '.join(str(e) + ' ' for e in a))
        else:
            a = [int(e) for e in tr_feat[i,j]]
            txt.append('%s 1 %s '% (j, j+1)+ ' '.join(str(e)+' ' for e in a))

np.savetxt('Mine_Graph_RML1.txt', txt, fmt='%s')
'''