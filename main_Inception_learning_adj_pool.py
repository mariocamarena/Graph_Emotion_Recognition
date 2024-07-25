import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
from Utils.util import load_data, separate_data
from models.graphcnn_Inception_learning_adj_pool import Graph_Inception
from Utils.pytorchtools import EarlyStopping

directory = "C:/Users/upf962/Downloads/CREST_MECIS_19401/Folders_from_workstation/graph_emotion_recognition-main/graph_emotion_recognition-main/Saved_models/checkpoint.pt"

criterion = nn.CrossEntropyLoss()
def Comp_loss(pred, label, pred_Adj, Adj, Adj_factor, pred_Pool, Pool, Pool_factor):

    loss = criterion(pred, label)
    m = nn.Threshold(0, 0)
    pred_Adj = m(pred_Adj)
    loss += Adj_factor * (torch.mean(torch.mul(pred_Adj, Adj)) + torch.mean((pred_Adj - torch.zeros_like(pred_Adj)) ** 2) )
    loss += Pool_factor * torch.mean((pred_Pool - torch.zeros_like(pred_Pool)) ** 2)

    return loss

train_acc, train_loss = [], []  # initialize empty lists

def train(args, model, device, train_graphs, optimizer, epoch, A):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    correct = 0
    total = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output,_ = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # setting loss function coefficients
        Adj_factor = 0.1
        Pool_factor = 0.0001

        loss = Comp_loss(output, labels, model.Adj.to(device), A.to(device)
                         , Adj_factor, model.Pool.to(device), torch.ones(size=([len(batch_graph[0].g)])).to(device), Pool_factor)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # accuracy calculation
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))
    print("accuracy training: %f" % (correct / total))

    train_acc.append(correct / total)  # append accuracy to train_acc
    train_loss.append(average_loss)  # append loss to train_loss

    return average_loss


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

def test(args, model, device, train_graphs, test_graphs, num_class):
    model.eval()

    output, _ = pass_data_iteratively(model, train_graphs)
    pred_ = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred_.eq(labels.view_as(pred_)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output, ind = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))


    return acc_train, acc_test, ind, labels, pred

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="Mine_Graph_RAVDESS",
                        help='name of dataset (default: RML)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=50,
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
    graphs, num_classes = load_data(args.dataset, args.degree_as_tag, args.Normalize)
    print(num_classes)


    ##5-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    #iniial adjacency matrix
    num_nodes = train_graphs[0].node_features.shape[0]

    A = np.zeros([num_nodes, num_nodes])
    for i in range(num_nodes):
        for j in range(num_nodes):
            A[i, j] = (i - j) ** 2
    # for i in range(num_nodes-1):
    #     A[i, i+1] = 1
    #     A[i+1, i] = 1
    A = torch.FloatTensor(A).to(device)

    model = Graph_Inception(args.num_layers, train_graphs[0].node_features.shape[1],
                     num_classes, args.final_dropout,
                     device, args.dataset, args.batch_size, num_nodes, A).to(device)

    Num_Param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Trainable Parameters= %d" % (Num_Param))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.5)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)


    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch, A)

        if(epoch>500):
            #### Validation check
            with torch.no_grad():
                val_out, _ = pass_data_iteratively(model, test_graphs)
                val_labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
                val_loss = criterion(val_out, val_labels)
                val_loss = np.average(val_loss.detach().cpu().numpy())

            #### Check early stopping
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break


        if(epoch % 50 ==0):
            acc_train, acc_test, _, _, _ = test(args, model, device, train_graphs, test_graphs, num_classes)


    torch.save(model.state_dict(), directory)
    
    RML_loss = [121.69954689025879, 16.327983894348144, 4.022535667419434, 2.346832823753357, 1.6893052768707275, 1.4727895641326905, 1.286280598640442, 1.1823967242240905, 1.0581410336494446, 0.9883001792430878, 0.8964689028263092, 0.947664475440979, 0.8511214625835418, 0.9031185519695282, 0.6706109803915024, 0.6184562540054321, 0.712186130285263, 0.708260229229927, 0.6538788944482803, 0.576204035282135, 0.515378515124321, 0.521196876168251, 0.43622023731470105, 0.3985357046127319, 0.45498777598142626, 0.3892622327804565, 0.4290456846356392, 0.30226821959018707, 0.44640863507986067, 0.3823381447792053, 0.3310366740822792, 0.29141702711582185, 0.24415736824274062, 0.22674551323056222, 0.333210888504982, 0.2691887667775154, 0.27845098480582237, 0.2928032438457012, 0.3223879590630531, 0.25650096982717513, 0.3017713248729706, 0.15815697431564332, 0.13824544712901116, 0.1787675227969885, 0.1715510668233037, 0.16389299377799035, 0.188879147619009, 0.13027439527213575, 0.12579908821731806, 0.11623937338590622]
    RML_train = [0.2359375, 0.32953125, 0.4359375, 0.47734375, 0.54921875, 0.57734375, 0.6065625, 0.63984375, 0.66421875, 0.69421875, 0.7184375, 0.7128125, 0.73640625, 0.73578125, 0.79171875, 0.80203125, 0.78890625, 0.791875, 0.80953125, 0.8246875, 0.84359375, 0.8478125, 0.865625, 0.8840625, 0.8678125, 0.8865625, 0.87875, 0.90640625, 0.8784375, 0.89359375, 0.90171875, 0.91109375, 0.9325, 0.93359375, 0.91015625, 0.928125, 0.9171875, 0.924375, 0.91234375, 0.936875, 0.92296875, 0.95203125, 0.96265625, 0.9515625, 0.9521875, 0.9559375, 0.9528125, 0.96046875, 0.96671875, 0.96828125]
    
    print(train_loss)
    print("\n")
    print(train_acc)
    # plot 1
    plt.subplot(1, 2, 1)
    plt.plot(RML_loss, label = "RML")
    plt.plot(train_loss, label='RAVDESS')
    # Set limits for x-axis and y-axis
    plt.xlim(0)
    plt.ylim(0,1)
    plt.title("Dynamic GCN")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # plot 2
    plt.subplot(1, 2, 2)
    plt.plot(RML_train, label = "RML")
    plt.plot(train_acc, label='RAVDESS')
    # Set limits for x-axis and y-axis
    plt.xlim(0)
    plt.ylim(0,1)
    plt.title("Dynamic GCN")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    #model.load_state_dict(torch.load(directory + '/Saved_models/checkpoint.pt'))
    acc_train, acc_test, ind, label, pred = test(args, model, device, train_graphs, test_graphs, num_classes)
    

if __name__ == '__main__':
    main()
