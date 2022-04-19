import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from similarity import *
from utils import *
from LE import *



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index


def create_dataset(seed = 0):
    association=np.loadtxt('drug-miRNA.csv',delimiter=',',dtype=float)
    miRNA_feature1=np.loadtxt('ncrna_expression_full.txt',dtype=float)
    miRNA_feature2=np.loadtxt('ncrna_GOsimilarity_full.txt',dtype=float)
    drug_feature1 = np.loadtxt('drug_feature_matrix.txt',dtype=float)
    drug_feature2 = np.loadtxt('drug_labelencoding.txt', dtype=float)

    miRNA_feature=np.hstack((miRNA_feature1, miRNA_feature2))
    drug_feature = np.hstack((drug_feature1, drug_feature2))

    miRNA_feature = normalize_features(miRNA_feature)
    drug_feature = normalize_features(drug_feature)

    none_zero_position=np.where(association==1)
    none_zero_row_index=none_zero_position[0]
    none_zero_col_index=none_zero_position[1]

    asso_index=[]
    for i in range(0,len(none_zero_row_index)):
        asso_index.append(i)

    test_index,val_index,train_index=data_split(asso_index, 0.2, 0.05)

    test_row_index=none_zero_row_index[test_index]
    test_col_index=none_zero_col_index[test_index]
    val_row_index=none_zero_row_index[val_index]
    val_col_index=none_zero_col_index[val_index]
    train_matrix=np.copy(association)
    train_matrix[test_row_index, test_col_index]=0
    train_matrix[val_row_index, val_col_index]=0

    np.random.seed(seed)
    zero_position = np.where(association != 1)
    negative_randomlist = [i for i in range(len(zero_position[0]))]
    random.shuffle(negative_randomlist)

    train_negative_index = negative_randomlist[:10 * len(train_index)]
    val_negative_index = negative_randomlist[10 * len(train_index):(10 * len(train_index))+(100 * len(val_index))]
    test_negative_index = negative_randomlist

    id_train = []
    train_label=[]
    id_val = []
    val_label=[]
    id_test = []
    test_label=[]
    for i in train_negative_index:
        id_train.append([zero_position[0][i],zero_position[1][i]])
        train_label.append(0)
    for i in val_negative_index:
        id_val.append([zero_position[0][i],zero_position[1][i]])
        val_label.append(0)
    for i in test_negative_index:
        id_test.append([zero_position[0][i],zero_position[1][i]])
        test_label.append(0)

    for i in range(len(train_index)):
        id_train.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_label.append(1)
    for i in range(len(val_index)):
        id_val.append([val_row_index[i], val_col_index[i]])
        val_label.append(1)
    for i in range(len(test_index)):
        id_test.append([test_row_index[i],test_col_index[i]])
        test_label.append(1)

    train_dataset = MDADataset(root='data', dataset='data/' + '_train',id_map=id_train, label = train_label)
    val_dataset = MDADataset(root='data', dataset='data/' + '_val',id_map=id_val, label = val_label)
    test_dataset = MDADataset(root='data', dataset='data/' + '_test',id_map=id_test, label = test_label)
    return train_dataset, val_dataset, test_dataset, train_matrix, miRNA_feature, drug_feature
device = torch.device('cuda:0')
def create_drug_mol():
    drug_s = pd.read_excel('drug_list.xlsx', engine='openpyxl')[4].values.tolist()
    drugs = []
    drug_smiles = []

    for d in range(len(drug_s)):
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(drug_s[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(drug_s[d])

    smile_graph = {}
    compound_iso_smiles = drugs

    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    data_list_mol = []
    for i in range(len(smile_graph)):
        c_size, features, edge_index = smile_graph[drugs[i]]
        GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([1]))
        GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
        GCNData_mol = GCNData_mol.to(device)
        data_list_mol.append(GCNData_mol)

    return data_list_mol

def create_dataset_as_gong(seed = 0):
    association=np.loadtxt('drug-miRNA.csv',delimiter=',',dtype=float)
    miRNA_feature1=np.loadtxt('ncrna_expression_full.txt',dtype=float)
    miRNA_feature2=np.loadtxt('ncrna_GOsimilarity_full.txt',dtype=float)
    drug_feature1 = np.loadtxt('drug_feature_matrix.txt',dtype=float)
    drug_feature2 = np.loadtxt('drug_labelencoding.txt', dtype=float)

    miRNA_feature=np.hstack((miRNA_feature1, miRNA_feature2))
    drug_feature = np.hstack((drug_feature1, drug_feature2))


    none_zero_position=np.where(association==1)
    none_zero_row_index=none_zero_position[0]
    none_zero_col_index=none_zero_position[1]

    asso_index=[]
    for i in range(0,len(none_zero_row_index)):
        asso_index.append(i)

    test_val_index, train_index=data_split_without(asso_index, 0.25)

    test_val_row_index=none_zero_row_index[test_val_index]
    test_val_col_index=none_zero_col_index[test_val_index]
    train_matrix=np.copy(association)
    train_matrix[test_val_row_index, test_val_col_index]=0

    np.random.seed(seed)
    zero_position = np.where(association != 1)
    negative_randomlist = [i for i in range(len(zero_position[0]))]
    random.shuffle(negative_randomlist)

    train_negative_index = negative_randomlist[:(10*len(train_index))]

    #negative_randomlist = negative_randomlist[len(train_index):]
    #random.shuffle(negative_randomlist)
    negative_val = negative_randomlist[len(train_index):len(train_index) + int(100 * 0.05 * len(asso_index))]
    #negative_test = list(set(negative_randomlist).difference(set(negative_val)))
    negative_test = negative_randomlist

    positive_randomlist = [i for i in range(len(test_val_row_index))]
    random.shuffle(positive_randomlist)
    positive_val = positive_randomlist[:int(0.05 * len(asso_index))]
    positive_test = list(set(positive_randomlist).difference(set(positive_val)))

    test_row_index=none_zero_row_index[test_val_index][positive_test]
    test_col_index=none_zero_col_index[test_val_index][positive_test]
    val_row_index=none_zero_row_index[test_val_index][positive_val]
    val_col_index=none_zero_col_index[test_val_index][positive_val]


    id_train = []
    train_label=[]
    id_val = []
    val_label=[]
    id_test = []
    test_label=[]

    for i in train_negative_index:
        id_train.append([zero_position[0][i],zero_position[1][i]])
        train_label.append(0)
    for i in negative_val:
        id_val.append([zero_position[0][i],zero_position[1][i]])
        val_label.append(0)
    for i in negative_test:
        id_test.append([zero_position[0][i],zero_position[1][i]])
        test_label.append(0)

    for i in range(len(train_index)):
        id_train.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_label.append(1)
    for i in range(len(positive_val)):
        id_val.append([val_row_index[i], val_col_index[i]])
        val_label.append(1)
    for i in range(len(positive_test)):
        id_test.append([test_row_index[i],test_col_index[i]])
        test_label.append(1)


    train_dataset = MDADataset(root='data', dataset='data/' + '_train',id_map=id_train, label = train_label)
    val_dataset = MDADataset(root='data', dataset='data/' + '_val',id_map=id_val, label = val_label)
    test_dataset = MDADataset(root='data', dataset='data/' + '_test',id_map=id_test, label = test_label)
    return train_dataset, val_dataset, test_dataset, train_matrix, miRNA_feature, drug_feature


''':arg
def create_dataset_as_gong(seed = 0):
    association=np.loadtxt('drug-miRNA.csv',delimiter=',',dtype=float)
    miRNA_feature1=np.loadtxt('ncrna_expression_full.txt',dtype=float)
    miRNA_feature2=np.loadtxt('ncrna_GOsimilarity_full.txt',dtype=float)
    drug_feature1 = np.loadtxt('drug_feature_matrix.txt',dtype=float)
    drug_feature2 = np.loadtxt('drug_labelencoding.txt', dtype=float)

    miRNA_feature=np.hstack((miRNA_feature1, miRNA_feature2))
    drug_feature = np.hstack((drug_feature1, drug_feature2))


    none_zero_position=np.where(association==1)
    none_zero_row_index=none_zero_position[0]
    none_zero_col_index=none_zero_position[1]

    asso_index=[]
    for i in range(0,len(none_zero_row_index)):
        asso_index.append(i)

    test_index, train_index=data_split_without(asso_index, 0.2)

    test_row_index=none_zero_row_index[test_index]
    test_col_index=none_zero_col_index[test_index]
    train_matrix=np.copy(association)
    train_matrix[test_row_index, test_col_index]=0

    np.random.seed(seed)
    zero_position = np.where(association != 1)
    negative_randomlist = [i for i in range(len(zero_position[0]))]
    random.shuffle(negative_randomlist)
    selected_negative = []
    for i in range(len(train_index)):
        selected_negative.append(negative_randomlist[i])

    train_negative_index = selected_negative[:len(train_index)]



    #val_negative_index = selected_negative[len(train_index):len(train_index)+len(val_index)]
    #test_negative_index = selected_negative[len(train_index)+len(val_index):]
    all_negative = np.where(association == 0)
    all_negative_row = all_negative[0]
    all_negative_col = all_negative[1]

    negative_randomlist = negative_randomlist[len(train_index):]
    random.shuffle(negative_randomlist)
    negative_val = negative_randomlist[:int(0.2 * len(negative_randomlist))]
    negative_test = list(set(negative_randomlist).difference(set(negative_val)))

    positive_randomlist = [i for i in range(len(test_row_index))]
    random.shuffle(positive_randomlist)
    positive_val = positive_randomlist[:int(0.2 * len(positive_randomlist))]
    positive_test = list(set(positive_randomlist).difference(set(positive_val)))

    test_row_index=none_zero_row_index[positive_test]
    test_col_index=none_zero_col_index[positive_test]
    val_row_index=none_zero_row_index[positive_val]
    val_col_index=none_zero_col_index[positive_val]

    id_train = []
    train_label=[]
    id_val = []
    val_label=[]
    id_test = []
    test_label=[]

    for i in train_negative_index:
        id_train.append([zero_position[0][i],zero_position[1][i]])
        train_label.append(0)
    for i in negative_val:
        id_val.append([zero_position[0][i],zero_position[1][i]])
        val_label.append(0)
    for i in negative_test:
        id_test.append([zero_position[0][i],zero_position[1][i]])
        test_label.append(0)

    for i in range(len(train_index)):
        id_train.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_label.append(1)
    for i in range(len(positive_val)):
        id_val.append([val_row_index[i], val_col_index[i]])
        val_label.append(1)
    for i in range(len(positive_test)):
        id_test.append([test_row_index[i],test_col_index[i]])
        test_label.append(1)


    train_dataset = MDADataset(root='data', dataset='data/' + '_train',id_map=id_train, label = train_label)
    val_dataset = MDADataset(root='data', dataset='data/' + '_val',id_map=id_val, label = val_label)
    test_dataset = MDADataset(root='data', dataset='data/' + '_test',id_map=id_test, label = test_label)
    return train_dataset, val_dataset, test_dataset, train_matrix, miRNA_feature, drug_feature
'''