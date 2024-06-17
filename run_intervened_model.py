import copy
import os
import os.path as osp
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from tqdm import tqdm
from WebKB import *
import argparse

import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikipediaNetwork
import torch_geometric.transforms as T
from model import GAT, GATv2, GATv3
from FilmDataset import FilmDataset
from utils import data_split, experiment, get_embeddings, embedd_raw_data, fine_tuning, load_new_dataset, fine_tuning_GATv2

from TCE import remove_cluster, pre_clustering


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

    # precluster = pre_clustering(data)
    cluster_path = r'Log/pre_cluster_{}_seed_{}.csv'.format(arg.name_data, arg.random_state)
    precluster = np.loadtxt(cluster_path, dtype=int)

    if arg.clustering_paradigm == 'unsup':
        precluster = pre_clustering(data, arg)
    elif arg.clustering_paradigm == 'semisup':
        ## please run semi-cluster.py to obatain these files
        precluster = np.loadtxt(r'Log/MLP_{}.txt'.format(arg.data_name), dtype=int)
    elif arg.clustering_paradigm == 'sup':
        precluster = data.y
    else:
        raise Exception("The paradigm does not exist.")


    data = remove_cluster(data, precluster, arg.remove_cluster)
    data.to(device)

    test_acc_list = []

    if args.model_name == 'GAT':
        model_type = GAT
    elif args.model_name == 'GATv2':
        model_type = GATv2
    elif args.model_name == 'GATv3':
        model_type = GATv3

    if args.clustering_paradigm == 'unsup':
        best_result_path = r'Weight/{}/{}_{}_{}_{}_{}_{}.pkl'.format(arg.name_data, args.cluster_paradigm, arg.model_name, str(arg.hidden_unit),
                                                              arg.pre_trained, str(arg.remove_cluster),
                                                              str(arg.random_state))
    else:
        best_result_path = r'Weight/{}/{}_{}_{}_{}_{}.pkl'.format(arg.name_data, args.cluster_paradigm,
                                                                     arg.model_name, str(arg.hidden_unit),
                                                                     arg.pre_trained, str(arg.remove_cluster))

    for i in range(arg.num_experiment):
        input_dim = data.num_node_features
        output_dim = len(np.unique(data.y.cpu().detach().numpy()))
        model = torch.load(r'Weight/{}/{}_{}.pkl'.format(arg.name_data, arg.model_name, arg.hidden_unit)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
        best_model, test_acc = fine_tuning(i, data, model, optimizer, arg)
        test_acc_list.append(test_acc)
        del model

        if test_acc == min(test_acc_list):
            torch.save(best_model, best_result_path)

    log = 'Model_type: {}, pre_trained:{}, remove_cluster:{}, hidden_unit:{}, is_saveModel: {}, Dateset_name: {}, ' \
          'Experiments: {:03d}, Acc: {:.1f}±{:.1f}\n'
    print(log.format(model_type, arg.pre_trained, arg.remove_cluster, arg.hidden_unit, arg.is_saveModel, arg.name_data, arg.num_experiment,
                     np.mean(test_acc_list),
                     np.std(test_acc_list)))

    with open('Log/my_result.txt', 'a+') as f:
        log = log.format(model_type, arg.pre_trained, arg.remove_cluster, arg.hidden_unit, arg.is_saveModel, arg.name_data, arg.num_experiment,
                         np.mean(test_acc_list), np.std(test_acc_list))
        f.write(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_saveModel', type=bool, default=False)
    parser.add_argument('--is_attention', type=bool, default=False)
    parser.add_argument('--remove_cluster', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--pre_trained', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_experiment', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='GAT')
    parser.add_argument('--name_data', type=str, default='Chameleon')
    parser.add_argument('--train_per', type=float, default=0.6)
    parser.add_argument('--val_per', type=float, default=0.2)
    parser.add_argument('--hidden_unit', type=int, default=128)
    parser.add_argument('--clustering_paradigm', type=str, default='sup')
    args = parser.parse_args()
    run(args)




