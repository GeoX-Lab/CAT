import numpy as np
import pandas as pd
import torch
import csv
from model import GAT
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
from utils import data_split, experiment, get_embeddings, embedd_raw_data, load_new_dataset
import os
import random
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'data'

data_name = 'roman_empire'
hidden_layer = 128
model_path = r'Weight/{}/MLP_{}.pkl'.format(data_name, hidden_layer)
cluster_path = r'Log/MLP_{}.txt'.format(data_name)


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
    elif data_name == 'roman_empire':
        data = load_new_dataset(data_name)
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

    self_loop_mask = data.edge_index[0] != data.edge_index[1]
    remove_self_loop = data.edge_index.t()[self_loop_mask]
    data.edge_index = remove_self_loop.t()

    data = data_split(data, 0.6, 0.2)

    return data


if __name__ == '__main__':
    model = torch.load(model_path).to(device)
    data = load_data(data_name).to(device)
    model.eval()
    out = model(data)
    mask = data['test_mask']
    pred = out[mask].max(1)[1]
    labels = data.y
    labels[mask] = pred
    np.savetxt(cluster_path, labels.detach().cpu().numpy(), delimiter=',', fmt='%d')
