import scipy.sparse as sp
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import argparse
import numpy as np
from data_process import *
from utils import *
from train import *
from Graph_embedding import *
from GNNat import GNNat,GNNnew
from torch_geometric.utils import to_networkx
import os
import time
device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='DMR-PEG')
parser.add_argument('--lr', type=float, default=0.0001, help = 'learning rate')

args = parser.parse_args()

sum_metric = np.zeros((1, 7))
for i in [100,110,120]:
    metric = np.zeros((1, 7))
    train_dataset, val_dataset, test_dataset, train_matrix, miRNA_feature, drug_feature = create_dataset(seed=i)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)

    data_list_mol = create_drug_mol()
    batchA = Batch.from_data_list(data_list_mol)

    node_feature, edge = convert_to_geometric_data(train_matrix)
    asso_data = DATA.Data(x=node_feature, edge_index=edge)
    G = to_networkx(asso_data)
    model_emb = DeepWalk(G,walk_length=80, num_walks=10,workers=1)#init model
    model_emb.train(embed_size = 128)# train model
    emb = model_emb.get_embeddings()# get embedding vectors
    embeddings = []
    for i in range(len(emb)):
        embeddings.append(emb[i])
    embeddings = np.array(embeddings)

    miRNA_feature = sp.coo_matrix(miRNA_feature)
    miRNA_feature = torch.FloatTensor(np.array(miRNA_feature.todense()))

    drug_feature = sp.coo_matrix(drug_feature)
    drug_feature = torch.FloatTensor(np.array(drug_feature.todense()))
    positional_encoding = torch.FloatTensor(embeddings)

    miRNA_feature = miRNA_feature.cuda(device)
    drug_feature = drug_feature.cuda(device)
    asso_data = asso_data.to(device)
    positional_encoding = positional_encoding.to(device)




    model = GNNat()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    result = train_model(model, optimizer, batchA, asso_data, miRNA_feature, drug_feature, positional_encoding, train_loader, val_loader, test_loader)

    sum_metric +=result
print('attention!')
print(sum_metric/3)