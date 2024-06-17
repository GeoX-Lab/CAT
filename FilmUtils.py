from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import numpy as np
import networkx as nx
import os
import torch
import scipy.sparse as sp


def load_film_data(graph_adjacency_list_file_path, graph_node_features_and_labels_file_path):
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    # edge_row = []
    # edge_col = []
    edge_list = []

    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            feature_blank = np.zeros(932, dtype=np.uint8)
            feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
            graph_node_features_dict[int(line[0])] = feature_blank
            graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))
            edge_list.append([int(line[0]), int(line[1])])

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    edge_list = torch.tensor(edge_list)
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))

    edge_index = edge_list.t()
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    data = Data(x=features, edge_index=edge_index, y=labels)
    return data


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # r_inv = np.power(rowsum, -1).flatten()
    r_inv = 1. / rowsum
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

