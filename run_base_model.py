import copy
import os
import os.path as osp
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from WebKB import *

import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikipediaNetwork
import torch_geometric.transforms as T
import torch.optim as optim
from model import GAT, GATv2, GATv3
from FilmDataset import FilmDataset
from utils import data_split, experiment, get_embeddings, embedd_raw_data, load_new_dataset, edge_to_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'data'


def load_data(dataset_name):
    if dataset_name == 'Film':
        dataset = FilmDataset('data/Film')
        data = dataset.data
    elif dataset_name == 'Chameleon':
        pre_dataset = WikipediaNetwork(
            root=path, name=dataset_name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=path, name=dataset_name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.y = pre_dataset[0].y
    elif dataset_name == 'Squirrel':
        pre_dataset = WikipediaNetwork(root=path, name='Squirrel', geom_gcn_preprocess=True,
                                       transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(root=path, name='Squirrel', geom_gcn_preprocess=False,
                                   transform=T.NormalizeFeatures())
        data = dataset[0]
        data.y = pre_dataset[0].y
    elif dataset_name == 'Cornell':
        dataset = WebKB(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset.data
        data.edge_attr = None
    elif dataset_name == 'Texas':
        dataset = WebKB(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset.data
        data.edge_attr = None
    elif dataset_name == 'Wisconsin':
        dataset = WebKB(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset.data
        data.edge_attr = None
    elif dataset_name == 'roman_empire':
        data = load_new_dataset(dataset_name)


    self_loop_mask = data.edge_index[0] != data.edge_index[1]
    remove_self_loop = data.edge_index.t()[self_loop_mask]
    data.edge_index = remove_self_loop.t()

    data = data_split(data, args.train_per, args.val_per)

    return data


def run(arg):
    data = load_data(arg.name_data)
    adj = edge_to_adj(data).to(device)
    data.to(device)

    if args.model_name == 'GAT':
        model_type = GAT
    elif args.model_name == 'GATv2':
        model_type = GATv2
    elif args.model_name == 'GATv3':
        model_type = GATv3

    adj_list = [None]
    no_loop_mat = torch.eye(adj.shape[0]).to(device)
    for ii in range(2):
        no_loop_mat = torch.mm(adj, no_loop_mat)
        adj_list.append(no_loop_mat)

    test_acc_list = []

    best_result_path = r'Weight/{}/{}_{}.pkl'.format(arg.name_data, arg.model_name, str(arg.hidden_unit))

    for i in range(arg.num_experiment):
        input_dim = data.num_node_features
        output_dim = len(np.unique(data.y.cpu().detach().numpy()))
        model = model_type(input_dim, arg.hidden_unit, output_dim, adj_list, adj).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
        # optimizer_sett = [
        #     {'params': model.classifier.parameters(), 'weight_decay': arg.weight_decay, 'lr': arg.lr},
        #     {'params': model.fc1.parameters(), 'weight_decay': arg.weight_decay, 'lr': arg.lr},
        #     {'params': model.hop_select, 'weight_decay': arg.weight_decay, 'lr': arg.lr},
        #     {'params': model.Q.parameters(), 'weight_decay': arg.weight_decay / 4, 'lr': arg.lr},
        #     {'params': model.K.parameters(), 'weight_decay': arg.weight_decay / 4, 'lr': arg.lr},
        # ]
        # optimizer = optim.Adam(optimizer_sett)
        best_model, test_acc = experiment(i, data, model, optimizer, arg)
        test_acc_list.append(test_acc)
        del model

        if test_acc == min(test_acc_list):
            torch.save(best_model, best_result_path)

    log = 'Model_type: {}, hidden_unit:{}, is_saveModel: {}, Dateset_name: {}, Experiments: {:03d}, Acc: {:.1f}±{' \
          ':.1f}\n'
    print(log.format(model_type, arg.hidden_unit, arg.is_saveModel, arg.name_data, arg.num_experiment,
                     np.mean(test_acc_list),
                     np.std(test_acc_list)))

    with open('Log/result.txt', 'a+') as f:
        log = log.format(model_type, arg.hidden_unit, arg.is_saveModel, arg.name_data, arg.num_experiment,
                         np.mean(test_acc_list), np.std(test_acc_list))
        f.write(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_saveModel', type=bool, default=False)
    parser.add_argument('--is_attention', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_experiment', type=int, default=50)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='GGAT')
    parser.add_argument('--name_data', type=str, default='roman_empire')
    parser.add_argument('--train_per', type=float, default=0.6)
    parser.add_argument('--val_per', type=float, default=0.2)
    parser.add_argument('--hidden_unit', type=int, default=64)
    args = parser.parse_args()
    run(args)
