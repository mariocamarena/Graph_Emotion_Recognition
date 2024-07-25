import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from collections import Counter
import matplotlib.pyplot as plt
from Utils.util import load_data
from models.graphcnn_Inception_learning_adj_pool import Graph_Inception
from Utils.pytorchtools import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

directory = "C:/Users/upf962/Downloads/CREST_MECIS_19401/Folders_from_workstation/graph_emotion_recognition-main/graph_emotion_recognition-main/Saved_models/checkpoint.pt"

criterion = nn.CrossEntropyLoss()
def Comp_loss(pred, label, pred_Adj, Adj, Adj_factor, pred_Pool, Pool, Pool_factor):

    loss = criterion(pred, label)
    m = nn.Threshold(0, 0)
    pred_Adj = m(pred_Adj)
    loss += Adj_factor * (torch.mean(torch.mul(pred_Adj, Adj)) + torch.mean((pred_Adj - torch.zeros_like(pred_Adj)) ** 2) )
    loss += Pool_factor * torch.mean((pred_Pool - torch.zeros_like(pred_Pool)) ** 2)

    return loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    ind = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx])[0].detach())
        ind.append(model([graphs[j] for j in sampled_idx])[1].detach())
    return torch.cat(output, 0), torch.cat(ind, 0)

'''
def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    # Combining both train and test indices to use all data for training
    combined_idx = np.concatenate((train_idx, test_idx))
    test_graph_list = [graph_list[i] for i in combined_idx]

    return test_graph_list
'''

'''
def create_test_graph_list(graph_list, test_size, seed):
    labels = [graph.label for graph in graph_list]
    _, test_graph_list = train_test_split(graph_list, test_size=test_size, stratify=labels, random_state=seed)
    return test_graph_list
'''

def create_test_graph_list(graph_list):
    return graph_list

test_acc =[]
test_lossv = []
def test(model, device, test_graphs, num_class, A):
    model.eval()

    # Remove the calculations related to train_graphs

    test_loss = 0  # initialize test loss variable
    test_correct = 0
    test_total = 0

    # Keep track of actual and predicted labels
    actual_labels = []
    predicted_labels = []

    for batch in test_graphs:
        output, _ = model([batch])
        labels = torch.LongTensor([batch.label]).to(device)

        # Keep track of actual and predicted labels for visualization
        actual_labels.append(batch.label)
        pred = output.max(1, keepdim=True)[1]
        predicted_labels.append(pred.item())

        # setting loss function coefficients
        Adj_factor = 0.1
        Pool_factor = 0.0001

        loss = Comp_loss(output, labels, model.Adj.to(device), A.to(device), Adj_factor,
                         model.Pool.to(device), torch.ones(size=([len(batch.g)])).to(device), Pool_factor)

        # accumulate test loss
        test_loss += loss.item()

        # accuracy calculation
        pred = output.max(1, keepdim=True)[1]
        test_correct += pred.eq(labels.view_as(pred)).sum().item()
        test_total += labels.size(0)
    
    average_test_loss = test_loss / len(test_graphs)
    print("test loss: %f" % (average_test_loss))
    print("accuracy test: %f" % (test_correct / test_total))

    test_acc.append(test_correct / test_total)
    test_lossv.append(average_test_loss)

    # Visualize Classification Results
    results = Counter(predicted_labels)
    labels = [str(label) for label in results.keys()]  # Convert numerical labels to strings if needed
    counts = list(results.values())

    plt.bar(labels, counts)
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Classification Results')
    plt.show()

    # The function returns the accuracy and loss on the test set.
    # If you need to return additional values, adjust the return statement accordingly.
    #return test_correct / test_total, average_test_loss
    return test_correct / test_total, average_test_loss, actual_labels, predicted_labels


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="Mine_Graph_Custom",
                        help='name of dataset (default: RML)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 5 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 5-fold validation. Should be less then 5.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers INCLUDING the input one (default: 2)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--Normalize', type=bool, default=True, choices=[True, False],
                        help='Normalizing data')
    parser.add_argument('--patience', type=int, default=40,
                        help='Normalizing data')
    args = parser.parse_args()
    #set up seeds and gpu device
    # set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    #load data
    #graphs, num_classes = load_data(args.dataset, args.degree_as_tag, args.Normalize)
    #print(num_classes)

    # Usage
    #seed = 42
    #test_size = 0.1  # for example, 20% of the data will be in the test set
    #test_graphs = create_test_graph_list(graphs, test_size, seed)

    # load data
    graphs, num_classes = load_data(args.dataset, args.degree_as_tag, args.Normalize)
    print(num_classes)

    # create the test graph list using the entire dataset
    test_graphs = create_test_graph_list(graphs)

    ##5-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    #test_graphs= separate_data(graphs, args.seed, args.fold_idx)

    #iniial adjacency matrix
    num_nodes = test_graphs[0].node_features.shape[0]

    A = np.zeros([num_nodes, num_nodes])
    for i in range(num_nodes):
        for j in range(num_nodes):
            A[i, j] = (i - j) ** 2
    # for i in range(num_nodes-1):
    #     A[i, i+1] = 1
    #     A[i+1, i] = 1
    A = torch.FloatTensor(A).to(device)

    # Define the model architecture
    model = Graph_Inception(args.num_layers, test_graphs[0].node_features.shape[1],
                     num_classes, args.final_dropout,
                     device, args.dataset, args.batch_size, num_nodes, A).to(device)
    
    model.load_state_dict(torch.load(directory),strict=False)
    model.eval()
    '''
    test_acc2 = []
    test_loss2 = [] # initialize empty lists

    # Call the test function without the train_graphs argument
    #test_acc, test_loss = test(model, device, test_graphs, num_classes, A)
    
    test_acc, test_loss, actual_labels, predicted_labels = test(model, device, test_graphs, num_classes, A)
    test_acc2.append(test_acc)
    test_loss2.append(test_loss)
    #print("Predicted Class Labels: ", predicted_labels)


    # Plotting
    plt.figure(figsize=(10,5))
    
    # Subplot for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(len(test_acc2)), test_acc2, 'o-', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Test Index')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Subplot for loss
    plt.subplot(1, 2, 2)
    plt.plot(range(len(test_loss2)), test_loss2, 'o-', label='Test Loss')
    plt.title('Test Loss')
    plt.xlabel('Test Index')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    '''
if __name__ == '__main__':
    main()