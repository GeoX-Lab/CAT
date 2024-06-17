import numpy as np
import pandas as pd
import torch
import csv
from model import GAT, GATv2, GATv3
from utils import data_split, experiment, get_embeddings, embedd_raw_data, load_new_dataset, edge_to_adj, remove_cluster
from FilmDataset import FilmDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikipediaNetwork
from WebKB import *
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import os
import random
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'data'


def load_data(data_name):
    if data_name == 'Film':
        dataset = FilmDataset('data/Film')
        data = dataset.data
    elif data_name == 'Chameleon':
        pre_dataset = WikipediaNetwork(
            root=path, name=data_name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=path, name=data_name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.y = pre_dataset[0].y
    elif data_name == 'Squirrel':
        pre_dataset = WikipediaNetwork(root=path, name='Squirrel', geom_gcn_preprocess=True,
                                       transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(root=path, name='Squirrel', geom_gcn_preprocess=False,
                                   transform=T.NormalizeFeatures())
        data = dataset[0]
        data.y = pre_dataset[0].y
    elif data_name == 'Cornell':
        dataset = WebKB(path, data_name, transform=T.NormalizeFeatures())
        data = dataset.data
        data.edge_attr = None
    elif data_name == 'Texas':
        dataset = WebKB(path, data_name, transform=T.NormalizeFeatures())
        data = dataset.data
        data.edge_attr = None
    elif data_name == 'Wisconsin':
        dataset = WebKB(path, data_name, transform=T.NormalizeFeatures())
        data = dataset.data
        data.edge_attr = None
    elif data_name == 'roman_empire':
        data = load_new_dataset(data_name)

    self_loop_mask = data.edge_index[0] != data.edge_index[1]
    remove_self_loop = data.edge_index.t()[self_loop_mask]
    data.edge_index = remove_self_loop.t()

    return data


def pre_write_raw(data, csv_file, args):
    gat_model = torch.load(r'Weight/{}/{}_{}.pkl'.format(args.data_name, args.model_name, args.hidden_unit)).to(device)
    write_raw_attention(data, gat_model, csv_file)


def get_attention_sum(model, data):
    # attention_edge: tensor(2,+selfloop)
    # attention1&2: tensor(+selfloop,8)
    out, attention = model(data)
    attention_sum = np.zeros(data.num_nodes)
    for i in range(data.num_nodes):
        attention_node = attention[i].detach().cpu().numpy()
        attention_self = attention_node[i]
        attention_sum[i] = attention_node.sum() - attention_self
    return attention_sum


def pre_clustering(data, args):
    tSNE = manifold.TSNE(n_components=2, init='pca', random_state=args.random_state)
    X_tsne = tSNE.fit_transform(data.x)
    kmeans = KMeans(n_clusters=args.class_num, max_iter=100, init="k-means++", random_state=args.random_state).fit(
        X_tsne)
    result = kmeans.predict(X_tsne)
    return result


def write_raw_attention(data, gat_model, csv_file):
    attention_sum = get_attention_sum(gat_model, data.to(device))
    with open(csv_file, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(attention_sum)
        f.close()


def remove_and_calculate(data, precluster, remove_idx, csv_file, args):
    if args.clustering_paradigm == 'unsup':
        gat_model_tune = torch.load(
            r'Weight/{}/{}_{}_{}_{}_{}_{}.pkl'.format(args.data_name, args.cluster_paradigm, args.model_name,
                                                   args.hidden_unit, False, remove_idx, args.random_state)).to(device)
    else:
        gat_model_tune = torch.load(
            r'Weight/{}/{}_{}_{}_{}_{}.pkl'.format(args.data_name, args.cluster_paradigm, args.model_name, args.hidden_unit,
                                                   False, remove_idx)).to(device)
    calculate_attention(data, gat_model_tune, precluster, csv_file, remove_idx)


def calculate_attention(data, gat_model, precluster, csv_file, i):
    data = remove_cluster(data, precluster, i)
    attention_sum = get_attention_sum(gat_model, data.to(device))
    with open(csv_file, 'a+', newline='') as f1:
        csv_write = csv.writer(f1)
        csv_write.writerow(attention_sum)
        f1.close()


def graph_trimming(data, precluster, csv_file, graph_path):
    attention_matirx = np.loadtxt(csv_file, delimiter=',')
    TCE_matrix = np.zeros((args.class_num, data.num_nodes))
    for i in range(args.class_num):
        TCE_matrix[i] = attention_matirx[i + 1] - attention_matirx[0]
    TCE_max = np.argmax(TCE_matrix, axis=0)
    raw_edges = data.edge_index.detach().numpy()
    remain_edges_max = remove_edge(raw_edges, TCE_max, precluster)
    np.savetxt(graph_path, remain_edges_max, fmt='%d')


def remove_edge(raw_edges, TCE, precluster):
    edges = []
    for i in range(raw_edges.shape[1]):
        remain_cluster = TCE[raw_edges[0][i]]
        if precluster[raw_edges[1][i]] == remain_cluster:
            edges.append([raw_edges[0][i], raw_edges[1][i]])
    return np.array(edges)


def run(arg):
    data = load_data(arg.data_name)

    if arg.clustering_paradigm == 'unsup':
        precluster = pre_clustering(data, arg)
    elif arg.clustering_paradigm == 'semisup':
        ## please run semi-cluster.py to obatain these files
        precluster = np.loadtxt(r'Log/MLP_{}.txt'.format(arg.data_name), dtype=int)
    elif arg.clustering_paradigm == 'sup':
        precluster = data.y
    else:
        raise Exception("The paradigm does not exist.")

    csv_file = r'Attention/{}/{}_{}_remove_{}.csv'.format(arg.data_name, arg.model_name, arg.hidden_unit,
                                                          arg.cluster_paradigm)
    graph_path = r'Trimgraph/{}_max_{}.txt'.format(arg.data_name, arg.cluster_paradigm)

    pre_write_raw(data, csv_file, arg)
    for i in range(arg.class_num):
        data = load_data(arg.data_name)
        remove_idx = i
        remove_and_calculate(data, precluster, remove_idx, csv_file, arg)

    graph_trimming(data, precluster, csv_file, graph_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='Texas')
    parser.add_argument('--class_num', type=int, default=5)
    parser.add_argument('--hidden_unit', type=int, default=64)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='GAT')
    parser.add_argument('--clustering_paradigm', type=str, default='sup')
    args = parser.parse_args()
    run(args)
