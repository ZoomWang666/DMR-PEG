import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import scipy.sparse as sp

def data_split(full_list, ratio1, ratio2, shuffle=True):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset1 = int(n_total * ratio1)
    offset2 = offset1 + int(n_total * ratio2)
    if n_total == 0 or offset1 < 1 or offset2 < 1:
        return [],[], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1, sublist_2, sublist_3

def data_split_without(full_list, ratio1, shuffle=True):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset1 = int(n_total * ratio1)
    if n_total == 0 or offset1 < 1 :
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:]
    return sublist_1, sublist_2

class MDADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='data',
                 id_map=None, transform=None,
                 pre_transform=None, label=None):

        super(MDADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.id_map = id_map
        self.len = len(id_map)
        self.label = label

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + 'id.txt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        pass

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        index_x = self.id_map[idx][0]
        index_y = self.id_map[idx][1]
        y = self.label[idx]
        return y, [index_x, index_y]


def convert_to_geometric_data(matrix):
    youshang = np.loadtxt('sim/drug_similarity1.txt', delimiter=',', dtype=float)
    zuoxia = np.loadtxt('sim/miRNA_similarity1.txt', delimiter=',', dtype=float)
    youshang = np.array(youshang)
    zuoxia = np.array(zuoxia)
    tt = matrix.T
    s = np.hstack((matrix,youshang))
    x = np.hstack((zuoxia,tt))
    zong = np.vstack((s,x))
    edge_sum = np.where(zong == 1)
    edge = torch.tensor([edge_sum[0].tolist(),edge_sum[1].tolist()],dtype=torch.long)
    onehot=np.eye(len(zong))
    feature = []
    for i in range(len(zong)):
        feature.append(onehot[i].tolist())
    node_feature = torch.tensor(feature,dtype = torch.float)
    return node_feature, edge

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm