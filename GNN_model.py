import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


class GCN1(nn.Module):
    def __init__(self, input_dim, output_dim, adj, act='None'):
        super(GCN1, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.adj = adj
        self.act = act

    def forward(self, x):
        x = self.fc(x)
        x = torch.mm(self.adj, x)
        if self.act != 'None':
            x = F.relu(x)
        return x


class GCN2(nn.Module):
    def __init__(self, input_dim, output_dim, adj, act='None'):
        super(GCN2, self).__init__()
        self.gcn1 = GCN1(input_dim, output_dim, adj, 'relu')
        self.gcn2 = GCN1(output_dim, output_dim, adj, act)

    def forward(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        return x


class GAT1(nn.Module):
    def __init__(self, input_dim, output_dim, adj, act='None') -> None:
        super(GAT1, self).__init__()
        self.fc = nn.linear(input_dim, output_dim, bias=False)
        self.a1 = nn.Linear(output_dim, 1, bias=False)
        self.a2 = nn.linear(output_dim, 1, bias=False)
        self.act = act
        self.adj = adj

    def _soft_max(self, attention):
        attention = torch.where(self.adj > 0, attention, torch.ones_like(attention) * -9e15)
        attention = torch.softmax(attention, dim=-1)
        return attention

    def forward(self, x):
        x = self.fc(x)
        a1 = self.a1(x)
        a2 = self.a2(x)
        attention = self._soft_max(a1 + a2.T)
        x = torch.mm(attention, x)
        if self.act != 'None':
            x = F.relu(x)
        return x


class GAT2(nn.Module):
    def __init__(self, input_dim, output_dim, adj, act='None') -> None:
        super(GAT2, self).__init__()
        self.gat1 = GAT1(input_dim, output_dim, adj, 'relu')
        self.gat2 = GAT1(output_dim, output_dim, adj, act)

    def forward(self, x):
        x = self.gat1(x)
        x = self.gat2(x)
        return x


class GBP(nn.Module):
    def __init__(self, input_dim, output_dim, adj, act='None') -> None:
        super(GBP).__init__()
        self.gcn1 = GCN1(input_dim, output_dim, adj, 'relu')
        self.gcn2 = GCN1(output_dim, output_dim, adj, act)

    def forward(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x) + x
        return x


class SGC(nn.Module):
    def __init__(self, input_dim, output_dim, adj, act='None') -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.adj = torch.mm(adj, adj)
        self.act = act

    def forward(self, x):
        x = self.fc(x)
        x = torch.mm(self.adj, x)
        if self.act != 'None':
            x = F.relu(x)
        return x


class linear_transformation(nn.Module):
    def __init__(self, input_dim, output_dim, adj, act='None') -> None:
        super(linear_transformation, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.act = act

    def forward(self, x):
        x = self.fc(x)
        if self.act != 'None':
            x = F.relu(x)
        return x