import copy
import os
import os.path as osp
import math
import numpy as np

import torch
from torch_geometric.datasets import Planetoid, NELL
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv
from tqdm import tqdm
from GNN_model import *


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
        # return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden)
        self.relu = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(hidden, hidden)
        # self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden, output_dim)

    def forward(self, data):
        x = data.x
        x = self.linear1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        # x = self.linear2(x)
        # x = self.relu2(x)
        x = self.linear3(x)

        return F.log_softmax(x, dim=1)
        # return x


class GAT(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden, heads=8, concat=False, add_self_loops=True)
        self.conv2 = GATConv(hidden, output_dim, heads=8, concat=False, add_self_loops=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x, attention_conv1 = self.conv1(x, edge_index)
        attention_1 = attention_conv1[1]
        attention_edge = attention_conv1[0]
        x = F.relu(x)
        x, attention_conv2 = self.conv2(x, edge_index)
        attention_2 = attention_conv2[1]

        return F.log_softmax(x, dim=1), attention_edge, attention_1, attention_2
        # return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden, normalize=True)
        self.conv2 = SAGEConv(hidden, output_dim, normalize=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
        # return x


class GATv2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden, heads=8, concat=False, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden, output_dim, heads=8, concat=False, add_self_loops=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x, attention_conv1 = self.conv1(x, edge_index)
        attention_1 = attention_conv1[1]
        attention_edge = attention_conv1[0]
        x = F.relu(x)
        x, attention_conv2 = self.conv2(x, edge_index)
        attention_2 = attention_conv2[1]

        return F.log_softmax(x, dim=1), attention_edge, attention_1, attention_2
        # return x


def build_QK(input_dim, output_dim, build_att, att_act, adj):
    if build_att == 'GCN1':
        return GCN1(input_dim, output_dim, adj, att_act)
    elif build_att == 'GCN2':
        return GCN2(input_dim, output_dim, adj, att_act)
    elif build_att == 'GAT1':
        return GAT1(input_dim, output_dim, adj, att_act)
    elif build_att == 'GAT2':
        return GAT2(input_dim, output_dim, adj, att_act)
    elif build_att == 'GBP':
        return GBP(input_dim, output_dim, adj, att_act)
    elif build_att == "SGC":
        return SGC(input_dim, output_dim, adj, att_act)
    elif build_att == 'linear':
        return linear_transformation(input_dim, output_dim, adj, att_act)
    else:
        raise


class GATv3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj_list, adj_att):
        super(GATv3, self).__init__()

        Q_method = 'GCN1'
        Q_act = 'None'
        K_method = 'GCN1'
        K_act = 'None'

        # self.nlayers = len(adj_list)
        self.nlayers = 3
        self.classifier = nn.Linear((nhid) * self.nlayers, nclass)
        self.adj_list = adj_list
        self.adj_att = adj_att
        self.dropout = 0.5
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat, nhid) for _ in range(self.nlayers)])
        self.Q = nn.ModuleList([build_QK(nhid, nhid, Q_method, Q_act, adj_att) for _ in range(self.nlayers - 1)])
        self.K = nn.ModuleList([build_QK(nhid, nhid, K_method, K_act, adj_att) for _ in range(self.nlayers - 1)])

        self.hop_select = nn.Parameter(torch.ones(self.nlayers))

    def _soft_max_att(self, adj, attention):
        attention = torch.where(adj > 0, attention, torch.ones_like(attention) * -9e15)
        return F.softmax(attention, dim=-1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        mask = F.softmax(self.hop_select, dim=-1)
        list_out = list()
        for i in range(self.nlayers):
            tmp_out = self.fc1[i](x)
            if self.adj_list[i] is not None:

                tmp_out_att = torch.mm(self.adj_att, tmp_out)
                Q = self.Q[i - 1](tmp_out_att)
                K = self.K[i - 1](tmp_out_att)
                attention = self._soft_max_att(self.adj_list[i], torch.mm(Q, K.T))
                tmp_1 = torch.where(self.adj_list[i] > 0, attention, torch.zeros_like(attention))
                tmp_out = torch.mm(attention, tmp_out)

            tmp_out = F.normalize(tmp_out, p=2, dim=1)
            tmp_out = torch.mul(mask[i], tmp_out)
            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.classifier(out)

        return F.log_softmax(out, dim=1), tmp_1
