import os
import random
import warnings

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
import torch.nn.functional as F

from PRI_STGCN_ADNI.model import m_stgcn

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###############################################固定随机数种子####################################
seed = 7
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.


###############################################定义测试函数####################################
def stest(model, datasets_test, num_win, lam, beta, alpha):
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    pro_all = []
    model.eval()  # 将模型改为预测模式
    with torch.no_grad():
        for fmri, net, hub, label in datasets_test:
            fmri, net, hub, label = fmri.to(DEVICE), net.to(DEVICE), hub.to(DEVICE), label.to(DEVICE)
            fmri = fmri.float()
            net = net.float()
            hub = hub.float()
            label = label.long()
            # print(label.shape)
            loss_evot, outs = model(fmri, net, hub, num_win, lam, beta)
            lossc = F.nll_loss(outs, label)
            losss = lossc + alpha * loss_evot
            # 记录误差
            eval_loss += float(losss)
            # 记录准确率
            gailv, pred = outs.max(1)
            # print(len(pred))
            num_correct = (pred == label).sum()
            acc = int(num_correct) / net.shape[0]
            eval_acc += acc
            pre = pred.cpu().detach().numpy()
            pre_all.extend(pre)
            label_true = label.cpu().detach().numpy()
            labels_all.extend(label_true)
            pro_all.extend(outs[:, 1].cpu().detach().numpy())

        tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        eval_acc_epoch = accuracy_score(labels_all, pre_all)
        precision = precision_score(labels_all, pre_all)
        recall = recall_score(labels_all, pre_all)
        f1 = f1_score(labels_all, pre_all)
        my_auc = roc_auc_score(labels_all, pro_all)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


wd = 0
log = open('logs_try4_NC_ILL_ADNI.txt', mode='a', encoding='utf-8')
for lam in [0,0.1,0.2, 0.3,0.4, 0.5, 0.6,0.7, 0.8,0.9]:
    for beta in [0,0.1,0.2, 0.3,0.4, 0.5, 0.6,0.7, 0.8,0.9]:
        for yu in [0.65,0.7,0.75]:
            for drop in [0.4]:
                for lr1 in [5e-3]:

                    ################################# 获取数据集    NC  vs   ill########################################

                    from torch.utils.data import Dataset, DataLoader

                    m = loadmat('../datasets/ADNI_NC_SMC_EMCI_New.mat')  # fmri
                    keysm = list(m.keys())
                    fdata = m[keysm[3]]  # 特征数据
                    fdata[np.isnan(fdata)] = -1
                    for i in range(fdata.shape[0]):
                        max_t = np.max(fdata[i])
                        min_t = np.min(fdata[i])
                        fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
                    labels = m[keysm[4]][0]
                    for i in range(labels.shape[0]):
                        if labels[i] == 2:
                            labels[i] = 1

                    # # # 对应打乱数据集
                    index = [i for i in range(fdata.shape[0])]
                    np.random.shuffle(index)
                    fdata = fdata[index]
                    labels = labels[index]


                    # # 获取动态时间片段，脑网络，节点重要性分布
                    # #######################################创建动态脑网络##########################################
                    def create_DFCN(dataset, num_window, yu):
                        fmri_all = []
                        nets_all = []
                        hubness_all = []
                        win_length = dataset.shape[2] // num_window
                        for i in range(dataset.shape[0]):
                            # print(i)
                            fmri = []
                            nets = []
                            hubness = []
                            datas = dataset[i]  # 90*240
                            for j in range(num_window):
                                window_fmri = datas[:, win_length * j:win_length * (j + 1)]
                                fmri.append(window_fmri)
                                net = np.corrcoef(window_fmri)
                                max_n = np.max(net)
                                min_n = np.min(net)
                                net = MaxMinNormalization(net, max_n, min_n)
                                for i in range(90):
                                    for j in range(90):
                                        if net[i][j] < 0:
                                            net[i][j] = -net[i][j]
                                        if net[i][j] < yu:
                                            net[i][j] = 0
                                net[np.isnan(net)] = 0
                                # print(net)
                                graph = nx.from_numpy_matrix(net)
                                graph1 = nx.DiGraph(graph)
                                graph1.remove_edges_from(nx.selfloop_edges(graph1))
                                hub = nx.pagerank(graph1, alpha=0.85)
                                hub_value = np.array(list(hub.values()))
                                nets.append(net)
                                hubness.append(hub_value)
                            fmri_all.append(fmri)
                            nets_all.append(nets)
                            hubness_all.append(hubness)

                        return fmri_all, nets_all, hubness_all  # torch.Size([306, 6, 90, 40])


                    fmri_all, nets_all, hubness_all = create_DFCN(fdata, 4, yu)
                    fmri_all = np.array(fmri_all)
                    nets_all = np.array(nets_all)
                    hubness_all = np.expand_dims(np.array(hubness_all), axis=3)


                    class ADNIs(Dataset):
                        def __init__(self):
                            super(ADNIs, self).__init__()
                            self.fmri_all = fmri_all
                            self.nets_all = nets_all
                            self.hubness_all = hubness_all
                            self.label = labels

                        def __getitem__(self, item):
                            fmri = self.fmri_all[item]
                            nets = self.nets_all[item]
                            hubn = self.hubness_all[item]
                            label = self.label[item]
                            return fmri, nets, hubn, label

                        def __len__(self):
                            return self.fmri_all.shape[0]


                    num_win = 4
                    n_class = 2
                    avg_acc = 0
                    avg_spe = 0
                    avg_recall = 0
                    avg_f1 = 0
                    avg_auc = 0
                    avg_sens = 0
                    avg_spec = 0
                    pre_ten = []
                    label_ten = []
                    pro_ten = []
                    test_acc = []
                    test_pre = []
                    test_recall = []
                    test_f1 = []
                    test_auc = []
                    test_sens = []
                    test_spec = []
                    dataset = ADNIs()
                    k = 10
                    i = 0
                    # beta = 0.5
                    KF = KFold(n_splits=k, shuffle=True, random_state=7)
                    for train_idx, test_idx in KF.split(dataset):
                        train_subsampler = SubsetRandomSampler(train_idx)
                        test_sunsampler = SubsetRandomSampler(test_idx)
                        datasets_train = DataLoader(dataset, batch_size=20, shuffle=False, sampler=train_subsampler)
                        datasets_test = DataLoader(dataset, batch_size=20, shuffle=False, sampler=test_sunsampler)
                        min_loss = 1e10
                        losses = []  # 记录训练误差，用于作图分析
                        acces = []
                        eval_losses = []
                        eval_acces = []
                        patience = 0
                        patiences = 25
                        min_acc = 0
                        pre_gd = 0
                        recall_gd = 0
                        f1_gd = 0
                        auc_gd = 0
                        sens_gd = 0
                        spec_gd = 0
                        labels_all_gd = 0
                        pro_all_gd = 0
                        model = m_stgcn(49, 40, drop)
                        model.to(DEVICE)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=1e-3)
                        for e in range(300):
                            model.train()
                            train_loss = 0
                            train_acc = 0
                            pre_all_train = []
                            labels_all_train = []
                            model.train()
                            for fmri, net, hub, label in datasets_train:
                                fmri, net, hub, label = fmri.to(DEVICE), net.to(DEVICE), hub.to(DEVICE), label.to(
                                    DEVICE)
                                fmri = fmri.float()
                                net = net.float()
                                hub = hub.float()
                                label = label.long()
                                # print(label.shape)
                                loss_evo, out = model(fmri, net, hub, num_win, lam, beta)
                                loss_c = F.nll_loss(out, label)
                                loss = loss_c + 0 * loss_evo
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                train_loss += float(loss.item())
                                _, pred = out.max(1)
                                pre = pred.cpu().detach().numpy()
                                pre_all_train.extend(pre)
                                label_true = label.cpu().detach().numpy()
                                labels_all_train.extend(label_true)
                            losses.append(train_loss / len(datasets_train))
                            acces.append(train_acc / len(datasets_train))
                            train_acc = accuracy_score(labels_all_train, pre_all_train)
                            eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all = stest(
                                model, datasets_test, num_win, lam, beta, lam)

                            if eval_acc_epoch > min_acc:
                                # torch.save(model.state_dict(), '../result2/latest' + str(i) + '.pth')
                                print("Model saved at epoch{}".format(e))
                                min_acc = eval_acc_epoch
                                pre_gd = precision
                                recall_gd = recall
                                f1_gd = f1
                                auc_gd = my_auc
                                sens_gd = sensitivity
                                spec_gd = specificity
                                labels_all_gd = labels_all
                                pro_all_gd = pro_all
                                patience = 0
                            else:
                                patience += 1
                            if patience > patiences:
                                break

                            eval_losses.append(eval_loss / len(datasets_test))
                            eval_acces.append(eval_acc / len(datasets_test))
                            print(
                                'i:{},epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
                                ':.6f},recall : {:.6f},f1 : {:.6f},my_auc : {:.6f} '
                                .format(i, e, train_loss / len(datasets_train), train_acc,
                                        eval_loss / len(datasets_test),
                                        eval_acc_epoch,
                                        precision, recall, f1, my_auc))
                        test_acc.append(min_acc)
                        test_pre.append(pre_gd)
                        test_recall.append(recall_gd)
                        test_f1.append(f1_gd)
                        test_auc.append(auc_gd)
                        test_sens.append(sens_gd)
                        test_spec.append(spec_gd)
                        label_ten.extend(labels_all_gd)
                        pro_ten.extend(pro_all_gd)

                        i = i + 1
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "test_acc",
                          test_acc, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "test_pre",
                          test_pre, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "test_recall",
                          test_recall, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "test_f1",
                          test_f1, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "test_auc",
                          test_auc, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "test_sens",
                          test_sens, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "test_spec",
                          test_spec, file=log)
                    avg_acc = sum(test_acc) / k
                    avg_pre = sum(test_pre) / k
                    avg_recall = sum(test_recall) / k
                    avg_f1 = sum(test_f1) / k
                    avg_auc = sum(test_auc) / k
                    avg_sens = sum(test_sens) / k
                    avg_spec = sum(test_spec) / k
                    print("*****************************************************", file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          'acc', avg_acc,
                          file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          'pre', avg_pre,
                          file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          'recall',
                          avg_recall, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          'f1', avg_f1,
                          file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          'auc', avg_auc,
                          file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "sensitivity",
                          avg_sens, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "specificity",
                          avg_spec, file=log)

                    acc_std = np.sqrt(np.var(test_acc))
                    pre_std = np.sqrt(np.var(test_pre))
                    recall_std = np.sqrt(np.var(test_recall))
                    f1_std = np.sqrt(np.var(test_f1))
                    auc_std = np.sqrt(np.var(test_auc))
                    sens_std = np.sqrt(np.var(test_sens))
                    spec_std = np.sqrt(np.var(test_spec))
                    print("*****************************************************", file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "acc_std",
                          acc_std, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "pre_std",
                          pre_std, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "recall_std",
                          recall_std, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "f1_std",
                          f1_std, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "auc_std",
                          auc_std, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "sens_std",
                          sens_std, file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          "spec_std",
                          spec_std, file=log)
                    print("*****************************************************", file=log)

                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          label_ten,
                          file=log)
                    print("num_win", num_win, 'lam', lam, 'beta', beta, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                          pro_ten,
                          file=log)
                    print("*****************************************************", file=log)
