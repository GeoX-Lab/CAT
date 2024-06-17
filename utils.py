import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit
import numpy as np
import torch
import copy
import os
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv
import networkx as nx

from sklearn.metrics import silhouette_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.transforms import to_undirected
import scipy.sparse as sp
from FilmDataset import FilmDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikipediaNetwork
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, data, optimizer, is_attention):
    model.train()
    optimizer.zero_grad()
    out, attention = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # attention_grad(_, attention_1, attention2, data.y)
    optimizer.step()
    return loss


def retrain(model, data, optimizer, is_attention):
    model.hop_select.requires_grad = False
    model.classifier.weight.requires_grad = False
    model.classifier.bias.requires_grad = False
    model.train()
    optimizer.zero_grad()
    out, attention = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # attention_grad(_, attention_1, attention2, data.y)
    optimizer.step()
    return loss

def retrain_v3(model, data, optimizer, is_attention):
    model.fc1[0].weight.requires_grad = False
    model.fc1[0].bias.requires_grad = False
    model.fc1[1].weight.requires_grad = False
    model.fc1[1].bias.requires_grad = False
    model.fc1[2].weight.requires_grad = False
    model.fc1[2].bias.requires_grad = False
    model.Q[0].fc.weight.requires_grad = False
    model.Q[0].fc.bias.requires_grad = False
    model.K[0].fc.weight.requires_grad = False
    model.K[0].fc.bias.requires_grad = False
    model.train()
    optimizer.zero_grad()
    out, attention = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # attention_grad(_, attention_1, attention2, data.y)
    optimizer.step()
    return loss


def val(model, data, is_attention):
    model.eval()
    # out = model(data).max(dim=1)
    out, attention = model(data)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    val_mask = data.val_mask
    accs.append(F.nll_loss(out[val_mask], data.y[val_mask]).cpu().detach().numpy())
    return accs


def data_split(data, train_percentage, val_percentage):
    assert (train_percentage is not None and val_percentage is not None)
    assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

    labels = data.y

    train_and_val_index, test_index = next(
        ShuffleSplit(n_splits=5, train_size=train_percentage + val_percentage, random_state=42).split(
            np.empty_like(labels), labels))
    train_index, val_index = next(ShuffleSplit(n_splits=5, train_size=train_percentage, random_state=42).split(
        np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]

    train_mask = np.zeros_like(labels)
    train_mask[train_index] = 1
    val_mask = np.zeros_like(labels)
    val_mask[val_index] = 1
    test_mask = np.zeros_like(labels)
    test_mask[test_index] = 1

    train_mask = torch.tensor(train_mask, dtype=bool)
    val_mask = torch.tensor(val_mask, dtype=bool)
    test_mask = torch.tensor(test_mask, dtype=bool)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def experiment(i, data, model, optimizer, arg):
    best_model = None
    best_val_acc = test_acc = 0.0
    best_val_loss = np.inf
    wait_step = 0
    val_loss_list = []
    tem_test_acc_list = []

    writer_path = r'{}_{}_{}'.format(str(arg.model_name), str(arg.hidden_unit), str(i))
    writer = SummaryWriter(log_dir=osp.join('curve', arg.name_data, arg.model_name, writer_path))

    for epoch in range(arg.epochs):
        train_loss = train(model, data, optimizer, arg.is_attention)
        train_acc, val_acc, tmp_test_acc, val_loss = val(model, data, arg.is_attention)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)

        val_loss_list.append(val_loss.item())
        tem_test_acc_list.append(tmp_test_acc)

        if val_acc >= best_val_acc or val_loss <= best_val_loss:
            if val_acc >= best_val_acc:
                test_acc = tmp_test_acc
                early_val_acc = val_acc
                early_val_loss = val_loss
                if arg.is_saveModel:
                    best_model_path = r'Weight/{}/{}_{}.pkl'.format(arg.name_data, arg.model_name, str(arg.hidden_unit))
                    torch.save(model, best_model_path)
                else:
                    best_model = copy.deepcopy(model)
            best_val_acc = np.max((val_acc, best_val_acc))
            best_val_loss = np.min((val_loss, best_val_loss))
            wait_step = 0
        else:
            wait_step += 1
            if wait_step == arg.patience:
                print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                print('Early stop model validation loss: ', early_val_loss, ', accuracy: ', early_val_acc)
                break

    log = 'Model_type: {}, is_saveModel: {}, Dateset_name: {}, Experiment: {:03d}, Test: {:.6f}'
    print(log.format(arg.model_name, arg.is_saveModel, arg.name_data, i + 1, test_acc))

    ## test acc
    if arg.is_saveModel:
        best_model = torch.load(best_model_path)
        train_acc, val_acc, test_acc, val_loss = val(data, best_model, arg.is_attention, flag=True)
        print('copied model\'s test_acc: {:.6f}'.format(test_acc))

    # del best_model

    return best_model, test_acc * 100


def get_embeddings(data, data_name, model_name, hidden, is_attention):
    embedding_path = r'Weight/{}/{}_{}.pkl'.format(data_name, model_name, hidden)
    embedding_model = torch.load(embedding_path).to(device)
    if is_attention:
        embedding, _, attention_1, attention2 = embedding_model(data)
    else:
        embedding = embedding_model(data)
    label_list = np.unique(data.y.cpu().detach().numpy())
    color_list = plt.cm.viridis(np.linspace(0, 1, len(label_list)))
    labels = data.y
    colors = [color_list[y] for y in labels]

    # xs, ys = zip(TSNE().fit_transform(embedding.cpu().detach().numpy()))
    # plt.scatter(xs, ys, color=colors)
    tsne = TSNE(n_components=2, random_state=42)
    feature_2d = tsne.fit_transform(embedding.cpu().detach().numpy())

    # calculating SC
    score = silhouette_score(feature_2d, labels.cpu().detach().numpy())

    # drawing embedding
    plt.scatter(feature_2d[:, 0], feature_2d[:, 1], color=colors)
    # plt.show(block=True)
    embedding_figure_path = r'./Embedding'
    embedding_figure_name = '{}_{}_{:.4f}.png'.format(model_name, hidden, score)
    plt.savefig(osp.join(embedding_figure_path, data_name, embedding_figure_name))


def embedd_raw_data(data, data_name):
    label_list = np.unique(data.y.cpu().detach().numpy())
    color_list = plt.cm.viridis(np.linspace(0, 1, len(label_list)))
    labels = data.y
    colors = [color_list[y] for y in labels]

    raw_feature = data.x
    tsne = TSNE(n_components=2, random_state=42)
    feature_2d = tsne.fit_transform(raw_feature)
    score = silhouette_score(feature_2d, labels)
    plt.scatter(feature_2d[:, 0], feature_2d[:, 1], color=colors)

    embedding_figure_path = r'./Embedding'
    embedding_figure_name = 'raw_{:.4f}.png'.format(score)
    plt.savefig(osp.join(embedding_figure_path, data_name, embedding_figure_name))


def attention_grad(attention_edge, attention1, attention2, y):
    # remove self_loop
    mask = attention_edge[0] != attention_edge[1]
    attention_edge = attention_edge[:, mask]  # tensor (2,33269)
    attention_edge = attention_edge.t().detach().cpu().numpy()  # ndarray (33269,2)
    # mean_attention
    attention1 = torch.mean(attention1, dim=1)[mask].detach().cpu().numpy()  # ndarray (33269,)
    attention2 = torch.mean(attention2, dim=1)[mask].detach().cpu().numpy()  # ndarray (33269,)
    # edge_if_homo
    y = y.cpu().detach().numpy()
    attention_edge_attribute = []  # True:homo list(33269)
    for edge in attention_edge:
        attention_edge_attribute.append(y[edge[0]] == y[edge[1]])

    # write attention to csv file
    csv_file_1 = r'Attention/attention_grad_1.csv'
    with open(csv_file_1, 'a+', newline='\n') as f1:
        csv_write = csv.writer(f1)
        # csv_write.writerow(['edge_idx', 'edge_attention_layer1', 'edge_attention_layer2', 'is_homo'])
        csv_write.writerow(attention1.T)
        # f1.close()

    csv_file_2 = r'Attention/attention_grad_2.csv'
    with open(csv_file_2, 'a+', newline='\n') as f2:
        csv_write = csv.writer(f2)
        # csv_write.writerow(['edge_idx', 'edge_attention_layer1', 'edge_attention_layer2', 'is_homo'])
        csv_write.writerow(attention2.T)
        # f2.close()


def fine_tuning(i, data, model, optimizer, arg):
    best_model = None
    best_val_acc = test_acc = 0.0
    best_val_loss = np.inf
    wait_step = 0
    val_loss_list = []
    tem_test_acc_list = []

    writer_path = r'{}_{}_{}'.format(str(arg.model_name), str(arg.hidden_unit), str(i))
    writer = SummaryWriter(log_dir=osp.join('curve', arg.name_data, arg.model_name, writer_path))

    for epoch in range(arg.epochs):
        if arg.model_name == "GATv3":
            train_loss = retrain_v3(model, data, optimizer, True)
            train_acc, val_acc, tmp_test_acc, val_loss = val(model, data, True)

            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("val_loss", val_loss, epoch)

            val_loss_list.append(val_loss.item())
            tem_test_acc_list.append(tmp_test_acc)

            if val_acc >= best_val_acc or val_loss <= best_val_loss:
                if val_acc >= best_val_acc:
                    test_acc = tmp_test_acc
                    early_val_acc = val_acc
                    early_val_loss = val_loss
                    if arg.is_saveModel:
                        best_model_path = r'Weight/{}/{}_{}_{}.pkl'.format(arg.name_data, arg.model_name,
                                                                           str(arg.hidden_unit), arg.pre_trained)
                        torch.save(model, best_model_path)
                    else:
                        best_model = copy.deepcopy(model)
                best_val_acc = np.max((val_acc, best_val_acc))
                best_val_loss = np.min((val_loss, best_val_loss))
                wait_step = 0
            else:
                wait_step += 1
                if wait_step == arg.patience:
                    print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                    print('Early stop model validation loss: ', early_val_loss, ', accuracy: ', early_val_acc)
                    break
        else:
            train_loss = retrain(model, data, optimizer, True)
            train_acc, val_acc, tmp_test_acc, val_loss = val(model, data, True)

            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("val_loss", val_loss, epoch)

            val_loss_list.append(val_loss.item())
            tem_test_acc_list.append(tmp_test_acc)

            if val_acc >= best_val_acc or val_loss <= best_val_loss:
                if val_acc >= best_val_acc:
                    test_acc = tmp_test_acc
                    early_val_acc = val_acc
                    early_val_loss = val_loss
                    if arg.is_saveModel:
                        best_model_path = r'Weight/{}/{}_{}_{}.pkl'.format(arg.name_data, arg.model_name,
                                                                           str(arg.hidden_unit), arg.pre_trained)
                        torch.save(model, best_model_path)
                    else:
                        best_model = copy.deepcopy(model)
                best_val_acc = np.max((val_acc, best_val_acc))
                best_val_loss = np.min((val_loss, best_val_loss))
                wait_step = 0
            else:
                wait_step += 1
                if wait_step == arg.patience:
                    print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                    print('Early stop model validation loss: ', early_val_loss, ', accuracy: ', early_val_acc)
                    break

    log = 'Model_type: {}, is_saveModel: {}, Dateset_name: {}, Experiment: {:03d}, Test: {:.6f}'
    print(log.format(arg.model_name, arg.is_saveModel, arg.name_data, i + 1, test_acc))

    ## test acc
    if arg.is_saveModel:
        best_model = torch.load(best_model_path)
        train_acc, val_acc, test_acc, val_loss = val(data, best_model, True, flag=True)
        print('copied model\'s test_acc: {:.6f}'.format(test_acc))

    # del best_model

    return best_model, test_acc * 100


def load_new_dataset(data_name):
    data_path = os.path.join('data', data_name)
    raw_data = np.load(os.path.join(data_path, '{}.npz'.format(data_name)))
    x = torch.tensor(raw_data['node_features'])
    y = torch.tensor(raw_data['node_labels'])
    edge_index = torch.tensor(raw_data['edges']).transpose(0, 1)
    edge_index_copy = torch.stack([edge_index[1], edge_index[0]], 0)
    edge_index_concat = torch.cat([edge_index, edge_index_copy], -1)
    data = Data(x=x, edge_index=edge_index_concat, y=y)

    return data


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1), dtype=np.float)
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sys_normalized_adjacency_i(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1),dtype=np.float)
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -1).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).tocoo()


def edge_to_adj(data):
    edges = data.edge_index.t()
    G = nx.DiGraph()
    # for i in range(len(data.y)):
    #     G.add_node(i, label=data.y[i])
    # G.add_edges_from(edges)
    for i in range(len(data.y)):
        G.add_node(i)
    for i in edges:
        # if int(i[0]) not in G:
        #     G.add_node(int(i[0]), features=data.x[int(i[0])])
        # if int(i[0]) not in G:
        #     G.add_node(int(i[0]), features=data.x[int(i[0])])
        G.add_edge(int(i[0]), int(i[1]))
        if not G.has_edge(int(i[1]), int(i[0])):
            G.add_edge(int(i[1]), int(i[0]))
    adj = nx.to_numpy_array(G, sorted(G.nodes()))
    adj = sys_normalized_adjacency(adj)
    # adj = sys_normalized_adjacency_i(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def remove_cluster(data, cluster, cluster_idx):
    remove_mask = cluster == cluster_idx
    remove_idx = np.arange(data.num_nodes)[remove_mask]
    edge_mask = []
    edge_index = data.edge_index.detach().cpu().numpy()
    for i in range(data.num_edges):
        if edge_index[0][i] in remove_idx:
            edge_mask.append(i)
    edge_after_remove = np.delete(edge_index, edge_mask, axis=1)
    data.edge_index = torch.from_numpy(edge_after_remove)
    return data



# if __name__ == '__main__':
#     load_new_dataset('amazon_ratings')
