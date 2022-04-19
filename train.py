import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from metrics import *

device = torch.device('cuda:0')

def test(model, loader, data_list_mol, asso_data, miRNA_feature, drug_feature, positional_encoding):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    model.eval()
    y_pred = []
    y_label = []

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            label = label.cuda(device)
            output = model(data_list_mol, asso_data, miRNA_feature, drug_feature, positional_encoding, inp)

            n = torch.squeeze(m(output))
            loss = loss_fct(n, label.float())

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
            result = model_evaluate(np.array(y_label), np.array(y_pred))

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss, result


def train_model(model, optimizer, data_list_mol, asso_data, miRNA_feature, drug_feature, positional_encoding, train_loader, val_loader, test_loader):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    loss_history = []
    max_auc = 0
    # model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()

    # Train model
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    for epoch in range(20):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (label, inp) in enumerate(train_loader):
            label = label.cuda(device)
            # print(inp[0].shape)
            model.train()
            optimizer.zero_grad()
            output = model(data_list_mol, asso_data, miRNA_feature, drug_feature, positional_encoding, inp)

            n = torch.squeeze(m(output))
            loss_train = loss_fct(n, label.float())
            loss_history.append(loss_train)
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation after each epoch
        roc_val, prc_val, f1_val, loss_val, aaa = test(model, val_loader, data_list_mol, asso_data, miRNA_feature, drug_feature, positional_encoding)
        if roc_val > max_auc:
            model_max = copy.deepcopy(model)
            max_auc = roc_val
            # torch.save(model, path)
        print('epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'auroc_train: {:.4f}'.format(roc_train),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'auroc_val: {:.4f}'.format(roc_val),
              'auprc_val: {:.4f}'.format(prc_val),
              'f1_val: {:.4f}'.format(f1_val),
              'time: {:.4f}s'.format(time.time() - t))

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    # plt.plot(loss_history)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test, result = test(model_max, test_loader, data_list_mol, asso_data, miRNA_feature, drug_feature, positional_encoding)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))

    return result

