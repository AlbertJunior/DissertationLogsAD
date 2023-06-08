#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pickle
from random import shuffle

import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from sklearn.metrics import auc

from logadempirical.logdeep.dataset.log import log_dataset
from logadempirical.logdeep.dataset.sample import sliding_window
from logadempirical.logdeep.models.lstm import deeplog, loganomaly, robustlog
from logadempirical.logdeep.models.cnn import TextCNN
from logadempirical.logdeep.models.autoencoder import AutoEncoder
from logadempirical.logdeep.tools.train import mean_selection
from logadempirical.logdeep.tools.utils import plot_losses
from logadempirical.neural_log.transformers import NeuralLog

global_cache = dict()


def generate(output_dir, name, anomalies_ratio, is_neural):
    global global_cache
    if name in global_cache:
        return global_cache[name]
    print("Loading", output_dir + name)
    with open(output_dir + name, 'rb') as f:
        data_iter = pickle.load(f)
    print("Length test", len(data_iter))

    no_abnormal = 0
    no_normal = 0
    for seq in data_iter:
        if not isinstance(seq['Label'], int):
            label = seq['Label'].tolist()
            if max(label) > 0:
                no_abnormal += 1
            else:
                no_normal += 1
        else:
            label = seq['Label']
            if label > 0:
                no_abnormal += 1
            else:
                no_normal += 1

    num_anomalies = anomalies_ratio * (no_abnormal + no_normal)

    normal_iter = {}
    abnormal_iter = {}
    nr = 0

    for seq in data_iter:
        if not isinstance(seq['Label'], int):
            label = max(seq['Label'].tolist())
        else:
            label = seq['Label']
        if is_neural:
            key = tuple(seq['Seq'])
        else:
            key = tuple(seq['EventId'])

        if label > 0:
            nr += 1
            if nr <= num_anomalies:
                if key not in abnormal_iter:
                    abnormal_iter[key] = 1
                else:
                    abnormal_iter[key] += 1
        else:
            if key not in normal_iter:
                normal_iter[key] = 1
            else:
                normal_iter[key] += 1

    global_cache[name] = (normal_iter, abnormal_iter)

    return global_cache[name]


# def prepare_compute_anomaly(v):
#     v_new = []
#     for line in v:
#         for window, target in line:
#             window = torch.cat((window, torch.Tensor([target])))
#             v_new.append(torch.where(window == window[-1])[0][0])
#     return torch.Tensor(v_new)

# OLD GOOD VERSION
# def prepare_compute_anomaly(v):
#     v_new = []
#     tbar = tqdm(v, desc="\r")
#     for line in tbar:
#         maxi = 0
#         for window, target in line:
#             for i in range(len(window)):
#                 if window[i] == target:
#                     break
#             if i > maxi:
#                 maxi = i
#         v_new.append(maxi)
#     return torch.Tensor(v_new)


def prepare_compute_anomaly(v, line_threshold=5):
    v_new_minus_one = []
    # v_new = []
    # v_new_plus_one = []
    tbar = tqdm(v, desc="\r")
    for line in tbar:
        maxi_minus_one = 0
        # maxi = 0
        # maxi_plus_one = 0
        for window, target in line:
            # window = torch.cat((window, torch.Tensor([target - 1, target, target + 1])))
            window = torch.cat((window, torch.Tensor([target - 1])))
            value_minus_one = torch.where(window == target - 1)[0][0]
            # value = torch.where(window == target)[0][0]
            # value_plus_one = torch.where(window == target + 1)[0][0]
            if value_minus_one > maxi_minus_one:
                maxi_minus_one = value_minus_one
            # if value > maxi:
            #     maxi = value
            # if value_plus_one > maxi_plus_one:
            #     maxi_plus_one = value_plus_one
        v_new_minus_one.append(maxi_minus_one)
        # v_new.append(maxi)
        # v_new_plus_one.append(maxi_plus_one)

    return torch.Tensor(v_new_minus_one)


# def prepare_compute_anomaly(v):
#     def get_index_of_line(line):
#         maxi = 0
#         for window, target in line:
#             window = torch.cat((window, torch.Tensor([target])))
#             value = torch.where(window == window[-1])[0][0]
#             if value > maxi:
#                 maxi = value
#         return maxi

#     with ThreadPool(2) as pool:
#         rez = list(pool.map(get_index_of_line, v))
#     return torch.Tensor(rez)

# def prepare_compute_anomaly(v):
#     def get_index_of_line(line):
#         maxi = 0
#         for window, target in line:
#             for i in range(len(window)):
#                 if window[i] == target:
#                     break
#             if i > maxi:
#                 maxi = i
#         return maxi

#     with ThreadPool(processes=4) as pool:
#         rez = list(pool.map(get_index_of_line, v))
#     return torch.Tensor(rez)


def prepare_compute_anomaly_losses(v):
    return torch.stack([torch.stack(x).mean() for x in v])


def mean_selection_IQR(losses):
    print("Mean", torch.quantile(losses, 0.5))
    print("Q3", torch.quantile(losses, 0.75))
    Q1 = torch.quantile(losses, 0.25)
    Q3 = torch.quantile(losses, 0.75)

    IQR = Q3 - Q1
    print("IQR", IQR)
    ul = Q3 + 1.5 * IQR
    ll = Q1 - 1.5 * IQR

    return ul


class Predicter():
    def __init__(self, options):
        self.data_dir = options['data_dir']
        self.output_dir = options['output_dir']
        self.model_dir = options['model_dir']
        self.run_dir = options["run_dir"]
        self.vocab_path = options["vocab_path"]
        self.model_path = options['model_path']
        self.model_name = options['model_name']
        self.anomalies_ratio = options['max_anomalies_ratio']

        self.device = options['device']
        self.window_size = options['window_size']
        self.min_len = options["min_len"]
        self.seq_len = options["seq_len"]
        self.history_size = options['history_size']

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.test_ratio = options["test_ratio"]
        self.num_candidates = options['num_candidates']

        self.input_size = options["input_size"]
        self.hidden_size = options["hidden_size"]
        self.embedding_dim = options["embedding_dim"]
        self.num_layers = options["num_layers"]
        self.num_workers = options["num_workers"]

        self.batch_size = options['batch_size']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.embeddings = options['embeddings']

        # transformers' parameters
        self.num_encoder_layers = options["num_encoder_layers"]
        self.num_decoder_layers = options["num_decoder_layers"]
        self.dim_model = options["dim_model"]
        self.num_heads = options["num_heads"]
        self.dim_feedforward = options["dim_feedforward"]
        self.transformers_dropout = options["transformers_dropout"]
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.lower_bound = 0
        self.upper_bound = 3

    def create_plot(self, title, anomalies_per_thresold, x, epoch, elbow_g):
        x_no = range(len(x))
        plt.plot(x_no, anomalies_per_thresold, 'r-')
        plt.plot([elbow_g], [0], marker='o', color="green")
        plt.xlabel("G threshold")
        plt.ylabel("Anomalies No.")
        plt.title("elbow " + str(epoch))
        plt.savefig(self.run_dir + f"/{title}_{epoch}.png")
        plt.close()

    def create_plot_3d(self, title, anomalies_per_thresold, x, y, epoch, elbow_g, elbow_loss, elbow_g_loss):
        # reversed_list = np.array(anomalies_per_thresold[::-1])
        nonreversed_list = np.array(anomalies_per_thresold)

        ax = plt.axes(projection='3d')
        surf = ax.plot_trisurf(x, y, nonreversed_list, cmap=cm.jet, linewidth=0)
        # plt.plot(x, reversed_list, 'r-', label=title)
        # ax.plot(x, y, **{'color': 'lightsteelblue', 'marker': 'o'})
        ax.set_xlabel('G')
        ax.set_ylabel('Min_Loss')
        ax.set_zlabel(title, fontsize=30, rotation=60)
        ax.azim = 60
        # ax.legend()

        plt.title("elbow " + str(epoch))
        plt.savefig(self.run_dir + f"/{title}_{epoch}.png")
        plt.close()

    def detect_logkey_anomaly(self, output, label):
        num_anomaly = []
        for i in range(len(label)):
            # print(output[i])
            # print(torch.argsort(output[i], descending=True))
            predicted = torch.argsort(output[i], descending=True)[
                        :self.num_candidates].clone().detach().cpu()
            # print(predicted, label[i], label[i] in predicted, predicted.index(label[i]))
            if label[i] not in predicted:
                num_anomaly.append(self.num_candidates + 1)
            else:
                num_anomaly.append(predicted.index(label[i]) + 1)
        return num_anomaly

    def detect_params_anomaly(self, output, label):
        num_anomaly = 0
        for i in range(len(label)):
            error = output[i].item() - label[i].item()
            if error < self.lower_bound or error > self.upper_bound:
                num_anomaly += 1
        return num_anomaly

    def compute_anomaly(self, results, losses, num, threshold=0, th_loss=0):
        indices = torch.logical_and(results >= threshold, losses > th_loss)
        return num[indices].sum().item()

    def compute_anomaly_unique(self, results, losses, num, threshold=0, th_loss=0):
        indices = torch.logical_and(results >= threshold, losses >= th_loss)
        return len(num[indices])

    def find_best_threshold_train(self, train_normal_results, train_num_normal, train_normal_results_losses,
                                  train_abnormal_results, train_num_abnormal, train_abnormal_results_losses,
                                  epoch, threshold_range,
                                  elbow_g, elbow_loss, trainer):
        global global_cache
        train_abnormal_length = sum(train_num_abnormal).item()
        train_normal_length = sum(train_num_normal).item()
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        res_g = [0, 0, 0, 0, 0, 0, 0, 0]
        res_loss = [0, 0, 0, 0, 0, 0, 0, 0]

        fps, tps, tns, fns, ps, rs, f1s = [], [], [], [], [], [], []
        #         train_normal_results = prepare_compute_anomaly(train_normal_results)
        #         train_abnormal_results = prepare_compute_anomaly(train_abnormal_results)

        train_normal_results = global_cache["train_normal_results"]
        train_abnormal_results = global_cache["train_abnormal_results"]

        train_normal_results_losses = prepare_compute_anomaly_losses(train_normal_results_losses)
        train_abnormal_results_losses = prepare_compute_anomaly_losses(train_abnormal_results_losses)
        x, y = [], []
        plot_losses(train_normal_results_losses, train_abnormal_results_losses, epoch, self.run_dir + '/Train_Losses/',
                    elbow_loss)
        ok1 = ok2 = ok3 = 0
        tbar = tqdm(range(1, threshold_range), desc="\r")
        for th in tbar:
            x.append(th)
            ok = 0
            min_loss = min(torch.min(train_normal_results_losses).item(),
                           torch.min(train_abnormal_results_losses).item())
            max_loss = min(torch.max(train_normal_results_losses).item(),
                           torch.max(train_abnormal_results_losses).item())
            for th_loss in list(np.arange(min_loss, max_loss, 0.01)) + [0.0, elbow_loss]:
                FP = self.compute_anomaly(train_normal_results, train_normal_results_losses, train_num_normal, th,
                                          th_loss)
                TP = self.compute_anomaly(train_abnormal_results, train_abnormal_results_losses, train_num_abnormal, th,
                                          th_loss)

                TN = train_normal_length - FP
                FN = train_abnormal_length - TP
                if ok == 0:
                    fps.append(FP)
                    tps.append(TP)
                    tns.append(TN)
                    fns.append(FN)
                if TP == 0:
                    if th == elbow_g and th_loss < 0.001 and ok1 == 0:
                        ok1 = 1
                        trainer.log["train_metrics_g"]["epoch"].append(epoch)
                        trainer.log["train_metrics_g"]["g"].append(th)
                        trainer.log["train_metrics_g"]["th"].append(th_loss)
                        trainer.log["train_metrics_g"]["f1"].append(0)
                        trainer.log["train_metrics_g"]["p"].append(100)
                        trainer.log["train_metrics_g"]["r"].append(0)
                        trainer.log["train_metrics_g"]["tp"].append(TP)
                        trainer.log["train_metrics_g"]["fp"].append(FP)
                        trainer.log["train_metrics_g"]["fn"].append(FN)
                        print("1. For ELBOW G", th, "train metrics, TP: ", TP, "FP: ", FP, ", FN:", FN,
                              ", P:100, R:0, F1:0")
                    if abs(th_loss - elbow_loss) < 0.006 and th == 1 and ok2 == 0:
                        ok2 = 1
                        trainer.log["train_metrics_loss"]["epoch"].append(epoch)
                        trainer.log["train_metrics_loss"]["g"].append(th)
                        trainer.log["train_metrics_loss"]["th"].append(elbow_loss)
                        trainer.log["train_metrics_loss"]["f1"].append(0)
                        trainer.log["train_metrics_loss"]["p"].append(100)
                        trainer.log["train_metrics_loss"]["r"].append(0)
                        trainer.log["train_metrics_loss"]["tp"].append(TP)
                        trainer.log["train_metrics_loss"]["fp"].append(FP)
                        trainer.log["train_metrics_loss"]["fn"].append(FN)
                        print("2. For ELBOW LOSS", th_loss, "train metrics, TP: ", TP, "FP: ", FP, ", FN:", FN,
                              ", P:100, R:0, F1:0")
                    if abs(th_loss - elbow_loss) < 0.006 and th == elbow_g and ok3 == 0:
                        ok3 = 1
                        trainer.log["train_metrics_both"]["epoch"].append(epoch)
                        trainer.log["train_metrics_both"]["g"].append(th)
                        trainer.log["train_metrics_both"]["th"].append(elbow_loss)
                        trainer.log["train_metrics_both"]["f1"].append(0)
                        trainer.log["train_metrics_both"]["p"].append(100)
                        trainer.log["train_metrics_both"]["r"].append(0)
                        trainer.log["train_metrics_both"]["tp"].append(TP)
                        trainer.log["train_metrics_both"]["fp"].append(FP)
                        trainer.log["train_metrics_both"]["fn"].append(FN)
                        print("3. For ELBOW G", th, "& LOSS", th_loss, " train metrics, TP: ", TP, "FP: ", FP, ", FN:",
                              FN, ", P:100, R:0, F1:0")
                    if ok == 0:
                        ps.append(100)
                        rs.append(0)
                        f1s.append(0)
                        ok = 1
                    continue

                # Compute precision, recall and F1-measure
                P = 100 * TP / (TP + FP)
                R = 100 * TP / (TP + FN)
                F1 = 2 * P * R / (P + R)
                # print(th + 1, FP, FN, P, R)
                if ok == 0:
                    ps.append(P)
                    rs.append(R)
                    f1s.append(F1)
                ok = 1
                if th == elbow_g and th_loss < 0.001 and ok1 == 0:
                    ok1 = 1
                    trainer.log["train_metrics_g"]["epoch"].append(epoch)
                    trainer.log["train_metrics_g"]["g"].append(th)
                    trainer.log["train_metrics_g"]["th"].append(th_loss)
                    trainer.log["train_metrics_g"]["f1"].append(F1)
                    trainer.log["train_metrics_g"]["p"].append(P)
                    trainer.log["train_metrics_g"]["r"].append(R)
                    trainer.log["train_metrics_g"]["tp"].append(TP)
                    trainer.log["train_metrics_g"]["fp"].append(FP)
                    trainer.log["train_metrics_g"]["fn"].append(FN)
                    print("1. For ELBOW G", th, "train metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P, ", R:",
                          R, ", F1:", F1)
                if abs(th_loss - elbow_loss) < 0.006 and th == 1 and ok2 == 0:
                    ok2 = 1
                    trainer.log["train_metrics_loss"]["epoch"].append(epoch)
                    trainer.log["train_metrics_loss"]["g"].append(th)
                    trainer.log["train_metrics_loss"]["th"].append(elbow_loss)
                    trainer.log["train_metrics_loss"]["f1"].append(F1)
                    trainer.log["train_metrics_loss"]["p"].append(P)
                    trainer.log["train_metrics_loss"]["r"].append(R)
                    trainer.log["train_metrics_loss"]["tp"].append(TP)
                    trainer.log["train_metrics_loss"]["fp"].append(FP)
                    trainer.log["train_metrics_loss"]["fn"].append(FN)
                    print("2. For ELBOW LOSS", th_loss, "train metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P,
                          ", R:", R, ", F1:", F1)
                if abs(th_loss - elbow_loss) < 0.006 and th == elbow_g and ok3 == 0:
                    ok3 = 1
                    trainer.log["train_metrics_both"]["epoch"].append(epoch)
                    trainer.log["train_metrics_both"]["g"].append(th)
                    trainer.log["train_metrics_both"]["th"].append(elbow_loss)
                    trainer.log["train_metrics_both"]["f1"].append(F1)
                    trainer.log["train_metrics_both"]["p"].append(P)
                    trainer.log["train_metrics_both"]["r"].append(R)
                    trainer.log["train_metrics_both"]["tp"].append(TP)
                    trainer.log["train_metrics_both"]["fp"].append(FP)
                    trainer.log["train_metrics_both"]["fn"].append(FN)
                    print("3. For ELBOW G", th, "& LOSS", th_loss, "train metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN,
                          ", P:", P, ", R:", R, ", F1:", F1)
                if F1 > res[-1]:
                    res = [th, th_loss, TP, TN, FP, FN, P, R, F1]
                if th_loss < 0.001 and F1 > res_g[-1]:
                    res_g = [th, th_loss, TP, TN, FP, FN, P, R, F1]
                if th == 1 and F1 > res_loss[-1]:
                    res_loss = [th, th_loss, TP, TN, FP, FN, P, R, F1]

        trainer.log["train_metrics_both_best"]["epoch"].append(epoch)
        trainer.log["train_metrics_both_best"]["g"].append(res[0])
        trainer.log["train_metrics_both_best"]["th"].append(res[1])
        trainer.log["train_metrics_both_best"]["f1"].append(res[8])
        trainer.log["train_metrics_both_best"]["p"].append(res[6])
        trainer.log["train_metrics_both_best"]["r"].append(res[7])
        trainer.log["train_metrics_both_best"]["tp"].append(res[2])
        trainer.log["train_metrics_both_best"]["fp"].append(res[4])
        trainer.log["train_metrics_both_best"]["fn"].append(res[5])

        trainer.log["train_metrics_g_best"]["epoch"].append(epoch)
        trainer.log["train_metrics_g_best"]["g"].append(res_g[0])
        trainer.log["train_metrics_g_best"]["th"].append(res_g[1])
        trainer.log["train_metrics_g_best"]["f1"].append(res_g[8])
        trainer.log["train_metrics_g_best"]["p"].append(res_g[6])
        trainer.log["train_metrics_g_best"]["r"].append(res_g[7])
        trainer.log["train_metrics_g_best"]["tp"].append(res_g[2])
        trainer.log["train_metrics_g_best"]["fp"].append(res_g[4])
        trainer.log["train_metrics_g_best"]["fn"].append(res_g[5])

        trainer.log["train_metrics_loss_best"]["epoch"].append(epoch)
        trainer.log["train_metrics_loss_best"]["g"].append(res_loss[0])
        trainer.log["train_metrics_loss_best"]["th"].append(res_loss[1])
        trainer.log["train_metrics_loss_best"]["f1"].append(res_loss[8])
        trainer.log["train_metrics_loss_best"]["p"].append(res_loss[6])
        trainer.log["train_metrics_loss_best"]["r"].append(res_loss[7])
        trainer.log["train_metrics_loss_best"]["tp"].append(res_loss[2])
        trainer.log["train_metrics_loss_best"]["fp"].append(res_loss[4])
        trainer.log["train_metrics_loss_best"]["fn"].append(res_loss[5])

        if trainer.verbose:
            self.create_plot("Metrics/Train_Normal/Recall", rs, x, epoch, elbow_g)
            self.create_plot("Metrics/Train_Normal/Precision", ps, x, epoch, elbow_g)
            self.create_plot("Metrics/Train_Normal/F1Score", f1s, x, epoch, elbow_g)
            self.create_plot("Metrics/Train_Normal/TP", tps, x, epoch, elbow_g)
            self.create_plot("Metrics/Train_Normal/TN", tns, x, epoch, elbow_g)
            self.create_plot("Metrics/Train_Normal/FP", fps, x, epoch, elbow_g)
            self.create_plot("Metrics/Train_Normal/FN", fns, x, epoch, elbow_g)
        return res

    def find_best_threshold(self, test_normal_results, num_normal_session_logs, test_normal_results_losses,
                            test_abnormal_results, num_abnormal_session_logs, test_abnormal_results_losses,
                            epoch, threshold_range,
                            elbow_g, elbow_loss, trainer):
        global global_cache
        test_abnormal_length = sum(num_abnormal_session_logs).item()
        test_normal_length = sum(num_normal_session_logs).item()
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        res_g = [0, 0, 0, 0, 0, 0, 0, 0]
        res_loss = [0, 0, 0, 0, 0, 0, 0, 0]
        # print(threshold_range)
        fps, tps, tns, fns, ps, rs, f1s = [], [], [], [], [], [], []
        test_normal_results = prepare_compute_anomaly(test_normal_results)
        test_abnormal_results = prepare_compute_anomaly(test_abnormal_results)
        global_cache["test_normal_results"] = test_normal_results
        global_cache["test_abnormal_results"] = test_abnormal_results

        test_normal_results_losses = prepare_compute_anomaly_losses(test_normal_results_losses)
        test_abnormal_results_losses = prepare_compute_anomaly_losses(test_abnormal_results_losses)

        plot_losses(test_normal_results_losses, test_abnormal_results_losses, epoch, self.run_dir + '/Test_Losses/',
                    elbow_loss)

        x, y = [], []
        ok1 = ok2 = ok3 = 0
        tbar = tqdm(range(1, threshold_range), desc="\r")
        for th in tbar:
            x.append(th)
            ok = 0
            min_loss = min(torch.min(test_normal_results_losses).item(), torch.min(test_abnormal_results_losses).item())
            max_loss = min(torch.max(test_normal_results_losses).item(), torch.max(test_abnormal_results_losses).item())
            for th_loss in list(np.arange(min_loss, max_loss, 0.01)) + [0.0, elbow_loss]:
                FP = self.compute_anomaly(test_normal_results, test_normal_results_losses, num_normal_session_logs, th,
                                          th_loss)
                TP = self.compute_anomaly(test_abnormal_results, test_abnormal_results_losses,
                                          num_abnormal_session_logs, th, th_loss)
                TN = test_normal_length - FP
                FN = test_abnormal_length - TP
                if ok == 0:
                    fps.append(FP)
                    tps.append(TP)
                    tns.append(TN)
                    fns.append(FN)
                if TP == 0:
                    if th == elbow_g and th_loss < 0.001 and ok1 == 0:
                        ok1 = 1
                        trainer.log["test_normal_metrics_g"]["epoch"].append(epoch)
                        trainer.log["test_normal_metrics_g"]["g"].append(th)
                        trainer.log["test_normal_metrics_g"]["th"].append(th_loss)
                        trainer.log["test_normal_metrics_g"]["f1"].append(0)
                        trainer.log["test_normal_metrics_g"]["p"].append(100)
                        trainer.log["test_normal_metrics_g"]["r"].append(0)
                        trainer.log["test_normal_metrics_g"]["tp"].append(TP)
                        trainer.log["test_normal_metrics_g"]["fp"].append(FP)
                        trainer.log["test_normal_metrics_g"]["fn"].append(FN)
                        print("1. For ELBOW G", th, "test metrics, TP: ", TP, "FP: ", FP, ", FN:", FN,
                              ", P:100, R:0, F1:0")
                    if abs(th_loss - elbow_loss) < 0.006 and th == 1 and ok2 == 0:
                        ok2 = 1
                        trainer.log["test_normal_metrics_loss"]["epoch"].append(epoch)
                        trainer.log["test_normal_metrics_loss"]["g"].append(th)
                        trainer.log["test_normal_metrics_loss"]["th"].append(th_loss)
                        trainer.log["test_normal_metrics_loss"]["f1"].append(0)
                        trainer.log["test_normal_metrics_loss"]["p"].append(100)
                        trainer.log["test_normal_metrics_loss"]["r"].append(0)
                        trainer.log["test_normal_metrics_loss"]["tp"].append(TP)
                        trainer.log["test_normal_metrics_loss"]["fp"].append(FP)
                        trainer.log["test_normal_metrics_loss"]["fn"].append(FN)
                        print("2. For ELBOW LOSS", th_loss, "test metrics, TP: ", TP, "FP: ", FP, ", FN:", FN,
                              ", P:100, R:0, F1:0")
                    if abs(th_loss - elbow_loss) < 0.006 and th == elbow_g and ok3 == 0:
                        ok3 = 1
                        trainer.log["test_normal_metrics_both"]["epoch"].append(epoch)
                        trainer.log["test_normal_metrics_both"]["g"].append(th)
                        trainer.log["test_normal_metrics_both"]["th"].append(th_loss)
                        trainer.log["test_normal_metrics_both"]["f1"].append(0)
                        trainer.log["test_normal_metrics_both"]["p"].append(100)
                        trainer.log["test_normal_metrics_both"]["r"].append(0)
                        trainer.log["test_normal_metrics_both"]["tp"].append(TP)
                        trainer.log["test_normal_metrics_both"]["fp"].append(FP)
                        trainer.log["test_normal_metrics_both"]["fn"].append(FN)
                        print("3. For ELBOW G", th, "& LOSS", th_loss, " test metrics, TP: ", TP, "FP: ", FP, ", FN:",
                              FN, ", P:100, R:0, F1:0")
                    if ok == 0:
                        ps.append(100)
                        rs.append(0)
                        f1s.append(0)
                        ok = 1
                    continue

                # Compute precision, recall and F1-measure
                P = 100 * TP / (TP + FP)
                R = 100 * TP / (TP + FN)
                F1 = 2 * P * R / (P + R)
                # print(th + 1, FP, FN, P, R)
                if ok == 0:
                    ps.append(P)
                    rs.append(R)
                    f1s.append(F1)
                ok = 1
                if th == elbow_g and th_loss < 0.001 and ok1 == 0:
                    ok1 = 1
                    trainer.log["test_normal_metrics_g"]["epoch"].append(epoch)
                    trainer.log["test_normal_metrics_g"]["g"].append(th)
                    trainer.log["test_normal_metrics_g"]["th"].append(th_loss)
                    trainer.log["test_normal_metrics_g"]["f1"].append(F1)
                    trainer.log["test_normal_metrics_g"]["p"].append(P)
                    trainer.log["test_normal_metrics_g"]["r"].append(R)
                    trainer.log["test_normal_metrics_g"]["tp"].append(TP)
                    trainer.log["test_normal_metrics_g"]["fp"].append(FP)
                    trainer.log["test_normal_metrics_g"]["fn"].append(FN)
                    print("1. For ELBOW G", th, "test metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P, ", R:",
                          R, ", F1:", F1)
                if abs(th_loss - elbow_loss) < 0.006 and th == 1 and ok2 == 0:
                    ok2 = 1
                    trainer.log["test_normal_metrics_loss"]["epoch"].append(epoch)
                    trainer.log["test_normal_metrics_loss"]["g"].append(th)
                    trainer.log["test_normal_metrics_loss"]["th"].append(th_loss)
                    trainer.log["test_normal_metrics_loss"]["f1"].append(F1)
                    trainer.log["test_normal_metrics_loss"]["p"].append(P)
                    trainer.log["test_normal_metrics_loss"]["r"].append(R)
                    trainer.log["test_normal_metrics_loss"]["tp"].append(TP)
                    trainer.log["test_normal_metrics_loss"]["fp"].append(FP)
                    trainer.log["test_normal_metrics_loss"]["fn"].append(FN)
                    print("2. For ELBOW LOSS", th_loss, "test metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P,
                          ", R:", R, ", F1:", F1)
                if abs(th_loss - elbow_loss) < 0.006 and th == elbow_g and ok3 == 0:
                    ok3 = 1
                    trainer.log["test_normal_metrics_both"]["epoch"].append(epoch)
                    trainer.log["test_normal_metrics_both"]["g"].append(th)
                    trainer.log["test_normal_metrics_both"]["th"].append(th_loss)
                    trainer.log["test_normal_metrics_both"]["f1"].append(F1)
                    trainer.log["test_normal_metrics_both"]["p"].append(P)
                    trainer.log["test_normal_metrics_both"]["r"].append(R)
                    trainer.log["test_normal_metrics_both"]["tp"].append(TP)
                    trainer.log["test_normal_metrics_both"]["fp"].append(FP)
                    trainer.log["test_normal_metrics_both"]["fn"].append(FN)
                    print("3. For ELBOW G", th, "& LOSS", th_loss, "test metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN,
                          ", P:", P, ", R:", R, ", F1:", F1)

                if F1 > res[-1]:
                    res = [th, th_loss, TP, TN, FP, FN, P, R, F1]
                if th_loss < 0.001 and F1 > res_g[-1]:
                    res_g = [th, th_loss, TP, TN, FP, FN, P, R, F1]
                if th == 1 and F1 > res_loss[-1]:
                    res_loss = [th, th_loss, TP, TN, FP, FN, P, R, F1]

        trainer.log["test_normal_metrics_both_best"]["epoch"].append(epoch)
        trainer.log["test_normal_metrics_both_best"]["g"].append(res[0])
        trainer.log["test_normal_metrics_both_best"]["th"].append(res[1])
        trainer.log["test_normal_metrics_both_best"]["f1"].append(res[8])
        trainer.log["test_normal_metrics_both_best"]["p"].append(res[6])
        trainer.log["test_normal_metrics_both_best"]["r"].append(res[7])
        trainer.log["test_normal_metrics_both_best"]["tp"].append(res[2])
        trainer.log["test_normal_metrics_both_best"]["fp"].append(res[4])
        trainer.log["test_normal_metrics_both_best"]["fn"].append(res[5])

        trainer.log["test_normal_metrics_g_best"]["epoch"].append(epoch)
        trainer.log["test_normal_metrics_g_best"]["g"].append(res_g[0])
        trainer.log["test_normal_metrics_g_best"]["th"].append(res_g[1])
        trainer.log["test_normal_metrics_g_best"]["f1"].append(res_g[8])
        trainer.log["test_normal_metrics_g_best"]["p"].append(res_g[6])
        trainer.log["test_normal_metrics_g_best"]["r"].append(res_g[7])
        trainer.log["test_normal_metrics_g_best"]["tp"].append(res_g[2])
        trainer.log["test_normal_metrics_g_best"]["fp"].append(res_g[4])
        trainer.log["test_normal_metrics_g_best"]["fn"].append(res_g[5])

        trainer.log["test_normal_metrics_loss_best"]["epoch"].append(epoch)
        trainer.log["test_normal_metrics_loss_best"]["g"].append(res_loss[0])
        trainer.log["test_normal_metrics_loss_best"]["th"].append(res_loss[1])
        trainer.log["test_normal_metrics_loss_best"]["f1"].append(res_loss[8])
        trainer.log["test_normal_metrics_loss_best"]["p"].append(res_loss[6])
        trainer.log["test_normal_metrics_loss_best"]["r"].append(res_loss[7])
        trainer.log["test_normal_metrics_loss_best"]["tp"].append(res_loss[2])
        trainer.log["test_normal_metrics_loss_best"]["fp"].append(res_loss[4])
        trainer.log["test_normal_metrics_loss_best"]["fn"].append(res_loss[5])

        if trainer.verbose:
            self.create_plot("Metrics/Normal/Recall", rs, x, epoch, elbow_g)
            self.create_plot("Metrics/Normal/Precision", ps, x, epoch, elbow_g)
            self.create_plot("Metrics/Normal/F1Score", f1s, x, epoch, elbow_g)
            self.create_plot("Metrics/Normal/TP", tps, x, epoch, elbow_g)
            self.create_plot("Metrics/Normal/TN", tns, x, epoch, elbow_g)
            self.create_plot("Metrics/Normal/FP", fps, x, epoch, elbow_g)
            self.create_plot("Metrics/Normal/FN", fns, x, epoch, elbow_g)
        return res

    def find_best_threshold_unique(self, test_normal_results, num_normal_session_logs, test_normal_results_losses,
                                   test_abnormal_results, num_abnormal_session_logs, test_abnormal_results_losses,
                                   epoch, threshold_range,
                                   elbow_g, elbow_loss, trainer):
        global global_cache
        test_abnormal_length = len(num_abnormal_session_logs)
        test_normal_length = len(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        res_g = [0, 0, 0, 0, 0, 0, 0, 0]
        res_loss = [0, 0, 0, 0, 0, 0, 0, 0]
        # print(threshold_range)
        fps, tps, tns, fns, ps, rs, f1s = [], [], [], [], [], [], []
        x, y = [], []

        #         test_normal_results = prepare_compute_anomaly(test_normal_results)
        #         test_abnormal_results = prepare_compute_anomaly(test_abnormal_results)

        test_normal_results = global_cache["test_normal_results"]
        test_abnormal_results = global_cache["test_abnormal_results"]

        test_normal_results_losses = prepare_compute_anomaly_losses(test_normal_results_losses)
        test_abnormal_results_losses = prepare_compute_anomaly_losses(test_abnormal_results_losses)
        ok1 = ok2 = ok3 = 0
        tbar = tqdm(range(1, threshold_range), desc="\r")
        for th in tbar:
            x.append(th)
            ok = 0
            min_loss = min(torch.min(test_normal_results_losses).item(), torch.min(test_abnormal_results_losses).item())
            max_loss = min(torch.max(test_normal_results_losses).item(), torch.max(test_abnormal_results_losses).item())
            for th_loss in list(np.arange(min_loss, max_loss, 0.01)) + [0.0, elbow_loss]:
                FP = self.compute_anomaly_unique(test_normal_results, test_normal_results_losses,
                                                 num_normal_session_logs, th, th_loss)
                TP = self.compute_anomaly_unique(test_abnormal_results, test_abnormal_results_losses,
                                                 num_abnormal_session_logs, th, th_loss)
                TN = test_normal_length - FP
                FN = test_abnormal_length - TP
                if ok == 0:
                    fps.append(FP)
                    tps.append(TP)
                    tns.append(TN)
                    fns.append(FN)
                if TP == 0:
                    if th == elbow_g and th_loss < 0.001 and ok1 == 0:
                        ok1 = 1
                        trainer.log["test_unique_metrics_g"]["epoch"].append(epoch)
                        trainer.log["test_unique_metrics_g"]["g"].append(th)
                        trainer.log["test_unique_metrics_g"]["th"].append(th_loss)
                        trainer.log["test_unique_metrics_g"]["f1"].append(0)
                        trainer.log["test_unique_metrics_g"]["p"].append(100)
                        trainer.log["test_unique_metrics_g"]["r"].append(0)
                        trainer.log["test_unique_metrics_g"]["tp"].append(TP)
                        trainer.log["test_unique_metrics_g"]["fp"].append(FP)
                        trainer.log["test_unique_metrics_g"]["fn"].append(FN)
                        print("1. For ELBOW G", th, "test metrics, TP: ", TP, "FP: ", FP, ", FN:", FN,
                              ", P:100, R:0, F1:0")
                    if abs(th_loss - elbow_loss) < 0.006 and th == 1 and ok2 == 0:
                        ok2 = 1
                        trainer.log["test_unique_metrics_loss"]["epoch"].append(epoch)
                        trainer.log["test_unique_metrics_loss"]["g"].append(th)
                        trainer.log["test_unique_metrics_loss"]["th"].append(th_loss)
                        trainer.log["test_unique_metrics_loss"]["f1"].append(0)
                        trainer.log["test_unique_metrics_loss"]["p"].append(100)
                        trainer.log["test_unique_metrics_loss"]["r"].append(0)
                        trainer.log["test_unique_metrics_loss"]["tp"].append(TP)
                        trainer.log["test_unique_metrics_loss"]["fp"].append(FP)
                        trainer.log["test_unique_metrics_loss"]["fn"].append(FN)
                        print("2. For ELBOW LOSS", th_loss, "test metrics, TP: ", TP, "FP: ", FP, ", FN:", FN,
                              ", P:100, R:0, F1:0")
                    if abs(th_loss - elbow_loss) < 0.006 and th == elbow_g and ok3 == 0:
                        ok3 = 1
                        trainer.log["test_unique_metrics_both"]["epoch"].append(epoch)
                        trainer.log["test_unique_metrics_both"]["g"].append(th)
                        trainer.log["test_unique_metrics_both"]["th"].append(th_loss)
                        trainer.log["test_unique_metrics_both"]["f1"].append(0)
                        trainer.log["test_unique_metrics_both"]["p"].append(100)
                        trainer.log["test_unique_metrics_both"]["r"].append(0)
                        trainer.log["test_unique_metrics_both"]["tp"].append(TP)
                        trainer.log["test_unique_metrics_both"]["fp"].append(FP)
                        trainer.log["test_unique_metrics_both"]["fn"].append(FN)
                        print("3. For ELBOW G", th, "& LOSS", th_loss, " test metrics, TP: ", TP, "FP: ", FP, ", FN:",
                              FN, ", P:100, R:0, F1:0")
                    if ok == 0:
                        ps.append(100)
                        rs.append(0)
                        f1s.append(0)
                        ok = 1
                    continue
                # Compute precision, recall and F1-measure
                P = 100 * TP / (TP + FP)
                R = 100 * TP / (TP + FN)
                F1 = 2 * P * R / (P + R)
                # print(th + 1, FP, FN, P, R)
                if ok == 0:
                    ps.append(P)
                    rs.append(R)
                    f1s.append(F1)
                ok = 1
                if th == elbow_g and th_loss < 0.001 and ok1 == 0:
                    ok1 = 1
                    trainer.log["test_unique_metrics_g"]["epoch"].append(epoch)
                    trainer.log["test_unique_metrics_g"]["g"].append(th)
                    trainer.log["test_unique_metrics_g"]["th"].append(th_loss)
                    trainer.log["test_unique_metrics_g"]["f1"].append(F1)
                    trainer.log["test_unique_metrics_g"]["p"].append(P)
                    trainer.log["test_unique_metrics_g"]["r"].append(R)
                    trainer.log["test_unique_metrics_g"]["tp"].append(TP)
                    trainer.log["test_unique_metrics_g"]["fp"].append(FP)
                    trainer.log["test_unique_metrics_g"]["fn"].append(FN)
                    print("1. For ELBOW G", th, "test metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P, ", R:",
                          R, ", F1:", F1)
                if abs(th_loss - elbow_loss) < 0.006 and th == 1 and ok2 == 0:
                    ok2 = 1
                    trainer.log["test_unique_metrics_loss"]["epoch"].append(epoch)
                    trainer.log["test_unique_metrics_loss"]["g"].append(th)
                    trainer.log["test_unique_metrics_loss"]["th"].append(th_loss)
                    trainer.log["test_unique_metrics_loss"]["f1"].append(F1)
                    trainer.log["test_unique_metrics_loss"]["p"].append(P)
                    trainer.log["test_unique_metrics_loss"]["r"].append(R)
                    trainer.log["test_unique_metrics_loss"]["tp"].append(TP)
                    trainer.log["test_unique_metrics_loss"]["fp"].append(FP)
                    trainer.log["test_unique_metrics_loss"]["fn"].append(FN)
                    print("2. For ELBOW LOSS", th_loss, "test metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P,
                          ", R:", R, ", F1:", F1)
                if abs(th_loss - elbow_loss) < 0.006 and th == elbow_g and ok3 == 0:
                    ok3 = 1
                    trainer.log["test_unique_metrics_both"]["epoch"].append(epoch)
                    trainer.log["test_unique_metrics_both"]["g"].append(th)
                    trainer.log["test_unique_metrics_both"]["th"].append(th_loss)
                    trainer.log["test_unique_metrics_both"]["f1"].append(F1)
                    trainer.log["test_unique_metrics_both"]["p"].append(P)
                    trainer.log["test_unique_metrics_both"]["r"].append(R)
                    trainer.log["test_unique_metrics_both"]["tp"].append(TP)
                    trainer.log["test_unique_metrics_both"]["fp"].append(FP)
                    trainer.log["test_unique_metrics_both"]["fn"].append(FN)
                    print("3. For ELBOW G", th, "& LOSS", th_loss, "test metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN,
                          ", P:", P, ", R:", R, ", F1:", F1)
                if F1 > res[-1]:
                    res = [th, th_loss, TP, TN, FP, FN, P, R, F1]
                if th_loss < 0.001 and F1 > res_g[-1]:
                    res_g = [th, th_loss, TP, TN, FP, FN, P, R, F1]
                if th == 1 and F1 > res_loss[-1]:
                    res_loss = [th, th_loss, TP, TN, FP, FN, P, R, F1]
        trainer.log["test_unique_metrics_both_best"]["epoch"].append(epoch)
        trainer.log["test_unique_metrics_both_best"]["g"].append(res[0])
        trainer.log["test_unique_metrics_both_best"]["th"].append(res[1])
        trainer.log["test_unique_metrics_both_best"]["f1"].append(res[8])
        trainer.log["test_unique_metrics_both_best"]["p"].append(res[6])
        trainer.log["test_unique_metrics_both_best"]["r"].append(res[7])
        trainer.log["test_unique_metrics_both_best"]["tp"].append(res[2])
        trainer.log["test_unique_metrics_both_best"]["fp"].append(res[4])
        trainer.log["test_unique_metrics_both_best"]["fn"].append(res[5])

        trainer.log["test_unique_metrics_g_best"]["epoch"].append(epoch)
        trainer.log["test_unique_metrics_g_best"]["g"].append(res_g[0])
        trainer.log["test_unique_metrics_g_best"]["th"].append(res_g[1])
        trainer.log["test_unique_metrics_g_best"]["f1"].append(res_g[8])
        trainer.log["test_unique_metrics_g_best"]["p"].append(res_g[6])
        trainer.log["test_unique_metrics_g_best"]["r"].append(res_g[7])
        trainer.log["test_unique_metrics_g_best"]["tp"].append(res_g[2])
        trainer.log["test_unique_metrics_g_best"]["fp"].append(res_g[4])
        trainer.log["test_unique_metrics_g_best"]["fn"].append(res_g[5])

        trainer.log["test_unique_metrics_loss_best"]["epoch"].append(epoch)
        trainer.log["test_unique_metrics_loss_best"]["g"].append(res_loss[0])
        trainer.log["test_unique_metrics_loss_best"]["th"].append(res_loss[1])
        trainer.log["test_unique_metrics_loss_best"]["f1"].append(res_loss[8])
        trainer.log["test_unique_metrics_loss_best"]["p"].append(res_loss[6])
        trainer.log["test_unique_metrics_loss_best"]["r"].append(res_loss[7])
        trainer.log["test_unique_metrics_loss_best"]["tp"].append(res_loss[2])
        trainer.log["test_unique_metrics_loss_best"]["fp"].append(res_loss[4])
        trainer.log["test_unique_metrics_loss_best"]["fn"].append(res_loss[5])
        if trainer.verbose:
            self.create_plot("Metrics/Unique/Recall", rs, x, epoch, elbow_g)
            self.create_plot("Metrics/Unique/Precision", ps, x, epoch, elbow_g)
            self.create_plot("Metrics/Unique/F1Score", f1s, x, epoch, elbow_g)
            self.create_plot("Metrics/Unique/TP", tps, x, epoch, elbow_g)
            self.create_plot("Metrics/Unique/TN", tns, x, epoch, elbow_g)
            self.create_plot("Metrics/Unique/FP", fps, x, epoch, elbow_g)
            self.create_plot("Metrics/Unique/FN", fns, x, epoch, elbow_g)
        return res

    def compute_metrics_on_threshold(self, test_normal_results, num_normal_session_logs,
                                     test_abnormal_results, num_abnormal_session_logs,
                                     th):
        test_abnormal_length = sum(num_abnormal_session_logs)
        test_normal_length = sum(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        # print(threshold_range)
        test_normal_results = prepare_compute_anomaly(test_normal_results)
        test_abnormal_results = prepare_compute_anomaly(test_abnormal_results)

        FP = self.compute_anomaly(test_normal_results, num_normal_session_logs, th)
        TP = self.compute_anomaly(test_abnormal_results, num_abnormal_session_logs, th)

        # Compute precision, recall and F1-measure
        TN = test_normal_length - FP
        FN = test_abnormal_length - TP
        if TP == 0:
            print("RAMONA: Pentru th", th, "TP e 0")
            return [th, TP, TN, FP, FN, 0, 0, 0]
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print("Ce a dat pe test")
        print(th, FP, FN, P, R)
        print()
        if F1 > res[-1]:
            res = [th, TP, TN, FP, FN, P, R, F1]
        return res

    def compute_metrics_on_threshold_unique(self, test_normal_results, num_normal_session_logs, test_abnormal_results,
                                            num_abnormal_session_logs, th):
        test_abnormal_length = len(num_abnormal_session_logs)
        test_normal_length = len(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1

        FP = self.compute_anomaly_unique(test_normal_results, num_normal_session_logs, th)
        TP = self.compute_anomaly_unique(test_abnormal_results, num_abnormal_session_logs, th)

        # Compute precision, recall and F1-measure
        TN = test_normal_length - FP
        FN = test_abnormal_length - TP
        if TP == 0:
            print("RAMONA: Pentru th", th, "TP e 0")
            return [th, TP, TN, FP, FN, 0, 0, 0]
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print("Ce a dat pe test")
        print(th, FP, FN, P, R)
        print()
        if F1 > res[-1]:
            res = [th, TP, TN, FP, FN, P, R, F1]
        return res

    def semi_supervised_helper(self, model, logs, vocab, data_type, scale=None, min_len=0):
        test_results = [[] for _ in range(len(logs))]
        test_results_losses = [[] for _ in range(len(logs))]
        l = data_type == "test_abnormal"
        sess_events = [(k, l, []) for (k, v) in logs.items()]
        num_sess = [logs[x] for (x, l, _) in sess_events]
        seqs, labels, anomaly = sliding_window(sess_events, vocab,
                                               history_size=self.history_size,
                                               is_train=False,
                                               data_dir=self.data_dir, semantics=self.semantics)

        dataset = log_dataset(logs=seqs,
                              labels=labels,
                              labels_anomaly=anomaly)
        data_loader = DataLoader(dataset,
                                 batch_size=min(len(dataset), 4096),
                                 shuffle=False,
                                 pin_memory=True)
        tbar = tqdm(data_loader, desc="\r")
        with torch.no_grad():
            for _, (log, label, anomaly) in enumerate(tbar):
                seq_idx = log['idx'].cpu().numpy()
                del log['idx']
                features = [x.to(self.device) for x in log['features']]
                output, _ = model(features=features)
                # output = output.softmax(dim=-1)

                label_for_loss = torch.tensor(label).view(-1).to(self.device)
                label_for_loss = label_for_loss - 1
                loss = self.criterion(output, label_for_loss)

                if self.is_logkey:
                    for i in range(len(seq_idx)):
                        test_results[seq_idx[i]].append(
                            (torch.argsort(output[i], descending=True)[:self.num_candidates].cpu(),
                             label[i]))
                        test_results_losses[seq_idx[i]].append(
                            loss[i].cpu())

        torch.cuda.empty_cache()
        return test_results, test_results_losses, torch.Tensor(num_sess), len(labels)

    def find_elbow(self, test_normal_results, num_normal_session_logs, train_normal_results_losses,
                   test_abnormal_results, num_abnormal_session_logs, train_abnormal_results_losses,
                   epoch, x_values, y_values, threshold_range):
        global global_cache
        test_abnormal_length = sum(num_abnormal_session_logs)
        test_normal_length = sum(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        no_anomalies_predicted = []
        fps, tps, tns, fns, ps, rs, f1s = [], [], [], [], [], [], []
        test_normal_results = prepare_compute_anomaly(test_normal_results)
        test_abnormal_results = prepare_compute_anomaly(test_abnormal_results)
        global_cache["train_normal_results"] = test_normal_results
        global_cache["train_abnormal_results"] = test_abnormal_results
        x, y = [], []
        train_normal_results_losses = prepare_compute_anomaly_losses(train_normal_results_losses)
        train_abnormal_results_losses = prepare_compute_anomaly_losses(train_abnormal_results_losses)
        tbar = tqdm(x_values, desc="\r")
        for th in tbar:
            x.append(th)
            ok = 0
            min_loss = min(torch.min(train_normal_results_losses).item(),
                           torch.min(train_abnormal_results_losses).item())
            max_loss = min(torch.max(train_normal_results_losses).item(),
                           torch.max(train_abnormal_results_losses).item())
            for th_loss in list(np.arange(min_loss, max_loss, 0.01)) + [0.0]:
                # y.append(th_loss)
                FP = self.compute_anomaly(test_normal_results, train_normal_results_losses, num_normal_session_logs, th,
                                          th_loss)
                TP = self.compute_anomaly(test_abnormal_results, train_abnormal_results_losses,
                                          num_abnormal_session_logs, th, th_loss)
                TN = test_normal_length - FP
                FN = test_abnormal_length - TP
                if ok == 0:
                    fps.append(FP)
                    tps.append(TP)
                    tns.append(TN)
                    fns.append(FN)
                    anomalies = FP + TP
                    no_anomalies_predicted.append(anomalies)
                if TP == 0:
                    # print("For g ", th, " Train metrics, TP: ", TP, "FP: ", FP, ", FN:", FN, ", P:100, R:0, F1:0")
                    if ok == 0:
                        ps.append(100)
                        rs.append(0)
                        f1s.append(0)
                        ok = 1
                    continue

                # Compute precision, recall and F1-measure
                P = 100 * TP / (TP + FP)
                R = 100 * TP / (TP + FN)
                F1 = 2 * P * R / (P + R)
                # print(th + 1, FP, FN, P, R)
                if ok == 0:
                    ps.append(P)
                    rs.append(R)
                    f1s.append(F1)
                    ok = 1
                # print("For g ", th, " train metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P, ", R:", R, ", F1:",F1)
                if F1 > res[-1]:
                    res = [th, TP, TN, FP, FN, P, R, F1]

        # self.create_plot("Metrics/Train_Elbow/Recall", rs, x, y, epoch, 0)
        # self.create_plot("Metrics/Train_Elbow/Precision", ps, x, y, epoch, 0)
        # self.create_plot("Metrics/Train_Elbow/F1Score", f1s, x, y, epoch, 0)
        # self.create_plot("Metrics/Train_Elbow/TP", tps, x, y, epoch, 0)
        # self.create_plot("Metrics/Train_Elbow/TN", tns, x, y, epoch, 0)
        # self.create_plot("Metrics/Train_Elbow/FP", fps, x, y, epoch, 0)
        # self.create_plot("Metrics/Train_Elbow/FN", fns, x, y, epoch, 0)
        return no_anomalies_predicted, x

    def compute_elbow(self, epoch, info_dir, anomalies_ratio):
        global global_cache
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        if self.model_name == "deeplog":
            lstm_model = deeplog
        else:
            lstm_model = loganomaly

        model_init = lstm_model(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                vocab_size=len(vocab),
                                embedding_dim=self.embedding_dim)
        model = model_init.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        train_normal, train_abnormal = generate(self.output_dir, 'train.pkl', 1, self.embeddings == 'neural')
        print("Nr secvente unice train: normale vs anormale")
        print(len(train_normal), len(train_abnormal))

        start_time = time.time()
        train_normal_results, train_normal_results_losses, num_normal, train_normal_no = \
            self.semi_supervised_helper(model, train_normal, vocab, 'test_normal')
        train_abnormal_results, train_abnormal_results_losses, num_abnormal, train_abnormal_no = \
            self.semi_supervised_helper(model, train_abnormal, vocab, 'test_abnormal')

        global_cache[
            "semi_supervised_helper_train_normal"] = train_normal_results, train_normal_results_losses, num_normal, train_normal_no
        global_cache[
            "semi_supervised_helper_train_abnormal"] = train_abnormal_results, train_abnormal_results_losses, num_abnormal, train_abnormal_no

        print("------------------------JUST NORMAL FOR ELBOW TRAIN SEQUENCES----------------------------", flush=True)
        x_values = range(self.num_candidates)
        y_values = np.arange(0.0, 5.0, 0.01)
        anomalies_per_thresold, x = self.find_elbow(train_normal_results, num_normal, train_normal_results_losses,
                                                    train_abnormal_results, num_abnormal, train_abnormal_results_losses,
                                                    epoch, x_values, y_values,
                                                    threshold_range=self.num_candidates)
        # print(anomalies_per_thresold)
        mx_g_loss = -1
        mx_g = -1
        mx_loss = -1
        elbow_g = None
        elbow_loss = None
        train_normal_results_losses = prepare_compute_anomaly_losses(train_normal_results_losses)
        train_abnormal_results_losses = prepare_compute_anomaly_losses(train_abnormal_results_losses)
        for x_i in range(1, len(x_values)):
            # for y_i in range(1, len(y_values)):
            diff_g = - anomalies_per_thresold[x_i] + anomalies_per_thresold[x_i - 1]
            if diff_g > mx_g:
                mx_g = diff_g
                elbow_g = x_values[x_i]

        if anomalies_ratio < 1:
            elbow_loss = mean_selection_IQR(train_normal_results_losses)
        else:
            elbow_loss = mean_selection_IQR(torch.cat((train_normal_results_losses, train_abnormal_results_losses)))

        print("1. G: diff ", mx_g, " at g ", elbow_g)
        print("2. Loss: diff ", mx_loss, " at loss ", elbow_loss)
        self.create_plot("Number_anomalies/Epoch", anomalies_per_thresold, x, epoch, elbow_g)
        # plot_losses(train_normal_results_losses, train_abnormal_results_losses, epoch, self.run_dir + '/Metrics/Elbow/', elbow_loss)

        print("Changes in number of anomalies")

        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        return elbow_g, elbow_loss

    def predict_semi_supervised(self, epoch, elbow_g, elbow_loss, info_dir, anomalies_ratio, trainer):
        global global_cache
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        if self.model_name == "deeplog":
            lstm_model = deeplog
        else:
            lstm_model = loganomaly

        model_init = lstm_model(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                vocab_size=len(vocab),
                                embedding_dim=self.embedding_dim)
        model = model_init.to(self.device)

        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        test_normal, test_abnormal = generate(self.output_dir, 'test.pkl', 1, self.embeddings == 'neural')
        train_normal, train_abnormal = generate(self.output_dir, 'train.pkl', 1, self.embeddings == 'neural')
        print("Nr secvente unice test: normale vs anormale")
        print(len(test_normal), len(test_abnormal))

        # Test the model
        start_time = time.time()
        test_normal_results, test_normal_results_losses, num_normal, test_normal_no = \
            self.semi_supervised_helper(model, test_normal, vocab, 'test_normal')
        test_abnormal_results, test_abnormal_results_losses, num_abnormal, test_abnormal_no = \
            self.semi_supervised_helper(model, test_abnormal, vocab, 'test_abnormal')

        train_normal_results, train_normal_results_losses, train_num_normal, train_normal_no = global_cache[
            "semi_supervised_helper_train_normal"]
        train_abnormal_results, train_abnormal_results_losses, train_num_abnormal, train_abnormal_no = global_cache[
            "semi_supervised_helper_train_abnormal"]

        #         train_normal_results, train_normal_results_losses, train_num_normal, train_normal_no = \
        #             self.semi_supervised_helper(model, train_normal, vocab, 'test_normal')
        #         train_abnormal_results, train_abnormal_results_losses, train_num_abnormal, train_abnormal_no = \
        #             self.semi_supervised_helper(model, train_abnormal, vocab, 'test_abnormal')

        trainer.log["train_statistics"]["total_sessions_no"].append(
            sum(train_normal.values()) + sum(train_abnormal.values()))
        trainer.log["train_statistics"]["normal_sessions_no"].append(sum(train_normal.values()))
        trainer.log["train_statistics"]["abnormal_sessions_no"].append(sum(train_abnormal.values()))
        trainer.log["train_statistics"]["abnormal_sessions_per"].append(
            sum(train_abnormal.values()) / (sum(train_normal.values()) + sum(train_abnormal.values())))

        trainer.log["train_statistics"]["total_unique_sessions_no"].append(len(train_normal) + len(train_abnormal))
        trainer.log["train_statistics"]["unique_normal_sessions_no"].append(len(train_normal))
        trainer.log["train_statistics"]["unique_abnormal_sessions_no"].append(len(train_abnormal))
        trainer.log["train_statistics"]["unique_abnormal_sessions_per"].append(
            len(train_abnormal) / (len(train_normal) + len(train_abnormal)))

        trainer.log["train_statistics"]["total_sequences_no"].append(train_normal_no + train_abnormal_no)
        trainer.log["train_statistics"]["normal_sequences_no"].append(train_normal_no)
        trainer.log["train_statistics"]["abnormal_sequences_no"].append(train_abnormal_no)
        trainer.log["train_statistics"]["abnormal_sequences_per"].append(
            train_abnormal_no / (train_normal_no + train_abnormal_no))

        trainer.log["test_statistics"]["total_sessions_no"].append(
            sum(test_normal.values()) + sum(test_abnormal.values()))
        trainer.log["test_statistics"]["normal_sessions_no"].append(sum(test_normal.values()))
        trainer.log["test_statistics"]["abnormal_sessions_no"].append(sum(test_abnormal.values()))
        trainer.log["test_statistics"]["abnormal_sessions_per"].append(
            sum(test_abnormal.values()) / (sum(test_normal.values()) + sum(test_abnormal.values())))

        trainer.log["test_statistics"]["total_unique_sessions_no"].append(len(test_normal) + len(test_abnormal))
        trainer.log["test_statistics"]["unique_normal_sessions_no"].append(len(test_normal))
        trainer.log["test_statistics"]["unique_abnormal_sessions_no"].append(len(test_abnormal))
        trainer.log["test_statistics"]["unique_abnormal_sessions_per"].append(
            len(test_abnormal) / (len(test_normal) + len(test_abnormal)))

        trainer.log["test_statistics"]["total_sequences_no"].append(test_normal_no + test_abnormal_no)
        trainer.log["test_statistics"]["normal_sequences_no"].append(test_normal_no)
        trainer.log["test_statistics"]["abnormal_sequences_no"].append(test_abnormal_no)
        trainer.log["test_statistics"]["abnormal_sequences_per"].append(
            test_abnormal_no / (test_normal_no + test_abnormal_no))

        print("------------------------NORMAL TRAIN SEQUENCES----------------------------")
        TH, th_loss, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold_train(train_normal_results, train_num_normal,
                                                                               train_normal_results_losses,
                                                                               train_abnormal_results,
                                                                               train_num_abnormal,
                                                                               train_abnormal_results_losses,
                                                                               epoch, self.num_candidates,
                                                                               elbow_g, elbow_loss, trainer)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)

        print('Best threshold', TH)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

        print("------------------------NORMAL TEST SEQUENCES----------------------------")
        TH, th_loss, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(test_normal_results, num_normal,
                                                                         test_normal_results_losses,
                                                                         test_abnormal_results, num_abnormal,
                                                                         test_abnormal_results_losses,
                                                                         epoch, self.num_candidates,
                                                                         elbow_g, elbow_loss, trainer)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)

        print('Best threshold', TH)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

        print("--------------------------UNIQUE TEST SEQUENCES------------------------------------")
        TH, th_loss, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold_unique(test_normal_results, num_normal,
                                                                                test_normal_results_losses,
                                                                                test_abnormal_results, num_abnormal,
                                                                                test_abnormal_results_losses,
                                                                                epoch, self.num_candidates,
                                                                                elbow_g, elbow_loss, trainer)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)

        print('Best threshold', TH)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def predict_semi_supervised_ramona(self):

        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(len(vocab))
        if self.model_name == "deeplog":
            lstm_model = deeplog
        else:
            lstm_model = loganomaly

        model_init = lstm_model(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                vocab_size=len(vocab),
                                embedding_dim=self.embedding_dim)
        model = model_init.to(self.device)

        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        test_normal, test_abnormal = generate(self.output_dir, 'test.pkl', 1, self.embeddings == 'neural')

        # Test the model

        normal_keys = list(test_normal.keys())
        shuffle(normal_keys)

        abnormal_keys = list(test_abnormal.keys())
        shuffle(abnormal_keys)

        for percentage in [0.001, 0.005, 0.01, 0.1, 0.25, 0.5]:
            print("*** Percentage ***", percentage)
            valid_normal_keys = normal_keys[:int(percentage * len(normal_keys))]
            test_normal_keys = normal_keys[int(percentage * len(normal_keys)):]

            valid_abnormal_keys = abnormal_keys[:int(percentage * len(abnormal_keys))]
            test_abnormal_keys = abnormal_keys[int(percentage * len(abnormal_keys)):]

            valid_normal_dict = dict()
            test_normal_dict = dict()
            for key in valid_normal_keys:
                valid_normal_dict[key] = test_normal[key]
            for key in test_normal_keys:
                test_normal_dict[key] = test_normal[key]

            valid_abnormal_dict = dict()
            test_abnormal_dict = dict()
            for key in valid_abnormal_keys:
                valid_abnormal_dict[key] = test_abnormal[key]
            for key in test_abnormal_keys:
                test_abnormal_dict[key] = test_abnormal[key]

            print("Cate secvente de validare unice: normale vs anormale")
            print(len(valid_normal_dict), len(valid_abnormal_dict))
            print("Cate secvente de test unice: normale vs anormale")
            print(len(test_normal_dict), len(test_abnormal_dict))

            start_time = time.time()
            valid_normal_results, valid_normal_results_losses, num_valid_normal, valid_normal_no = \
                self.semi_supervised_helper(model, valid_normal_dict, vocab, 'test_normal')
            valid_abnormal_results, valid_abnormal_results_losses, num_valid_abnormal, valid_abnormal_no = \
                self.semi_supervised_helper(model, valid_abnormal_dict, vocab, 'test_abnormal')

            test_normal_results, test_normal_results_losses, num_test_normal, test_normal_no = \
                self.semi_supervised_helper(model, test_normal_dict, vocab, 'test_normal')
            test_abnormal_results, test_abnormal_results_losses, num_test_abnormal, test_abnormal_no = \
                self.semi_supervised_helper(model, test_abnormal_dict, vocab, 'test_abnormal')

            print("--------------------------NORMAL TEST SEQUENCES------------------------------------")
            TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(valid_normal_results, num_valid_normal,
                                                                    valid_normal_results_losses,
                                                                    valid_abnormal_results, num_valid_abnormal,
                                                                    valid_abnormal_results_losses,
                                                                    threshold_range=self.num_candidates)

            TH, TP, TN, FP, FN, P, R, F1 = self.compute_metrics_on_threshold(test_normal_results, num_test_normal,
                                                                             test_normal_results_losses,
                                                                             test_abnormal_results, num_test_abnormal,
                                                                             test_abnormal_results_losses,
                                                                             TH)
            FPR = FP / (FP + TN)
            FNR = FN / (TP + FN)
            SP = TN / (TN + FP)

            print('Best threshold', TH)
            print("Confusion matrix")
            print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
            print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

            print("--------------------------UNIQUE TEST SEQUENCES------------------------------------")

            TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold_unique(valid_normal_results, num_valid_normal,
                                                                           valid_abnormal_results, num_valid_abnormal,
                                                                           threshold_range=self.num_candidates)

            TH, TP, TN, FP, FN, P, R, F1 = self.compute_metrics_on_threshold_unique(test_normal_results,
                                                                                    num_test_normal,
                                                                                    test_abnormal_results,
                                                                                    num_test_abnormal,
                                                                                    TH)
            FPR = FP / (FP + TN)
            FNR = FN / (TP + FN)
            SP = TN / (TN + FP)

            print('Best threshold', TH)
            print("Confusion matrix")
            print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
            print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

            elapsed_time = time.time() - start_time
            print('elapsed_time: {}'.format(elapsed_time))

    def predict_supervised(self):
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        if self.model_name == "cnn":
            model = TextCNN(self.dim_model, self.seq_len, 128).to(self.device)
        elif self.model_name == "neurallog":
            model = NeuralLog(num_encoder_layers=2, num_heads=12, dim_model=768, dim_feedforward=2048,
                              droput=0.1).to(self.device)
        else:
            lstm_model = robustlog

            model_init = lstm_model(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    vocab_size=len(vocab),
                                    embedding_dim=self.embedding_dim)
            model = model_init.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        test_normal_loader, test_abnormal_loader = generate(self.output_dir, 'test.pkl', 1,
                                                            is_neural=self.embeddings == 'neural')
        print(len(test_normal_loader), len(test_abnormal_loader))
        start_time = time.time()
        total_normal, total_abnormal = 0, 0
        FP = 0
        TP = 0
        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                logs, labels = sliding_window([(line, 0, list(line))], vocab, history_size=self.history_size,
                                              is_train=False,
                                              data_dir=self.data_dir, semantics=self.semantics, is_predict_logkey=False,
                                              e_name=self.embeddings)
                dataset = log_dataset(logs=logs, labels=labels)
                data_loader = DataLoader(dataset, batch_size=4096, shuffle=True, pin_memory=True)
                for _, (log, label) in enumerate(data_loader):
                    del log['idx']
                    features = [x.to(self.device) for x in log['features']]
                    output, _ = model(features, self.device)
                    # print(output)
                    output = output.softmax(dim=1)
                    pred = torch.argsort(output, 1, descending=True)
                    pred = pred[:, 0]
                    # print(pred)
                    if 1 in pred:
                        FP += test_normal_loader[line]
                        break
                total_normal += test_normal_loader[line]
        TN = total_normal - FP
        lead_time = []
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                logs, labels = sliding_window([(line, 1, list(line))], vocab, history_size=self.history_size,
                                              is_train=False,
                                              data_dir=self.data_dir, semantics=self.semantics, is_predict_logkey=False,
                                              e_name=self.embeddings)
                n_log = len(logs)
                dataset = log_dataset(logs=logs, labels=labels)
                data_loader = DataLoader(dataset, batch_size=4096, shuffle=False, pin_memory=True)
                for i, (log, label) in enumerate(data_loader):
                    del log['idx']
                    features = [x.to(self.device) for x in log['features']]
                    output, _ = model(features, self.device)
                    output = output.softmax(dim=1)
                    pred = torch.argsort(output, 1, descending=True)
                    pred = pred[:, 0]
                    # print(pred)
                    if 1 in pred:
                        TP += test_abnormal_loader[line]
                        lead_time.append(i)
                        break
                total_abnormal += test_abnormal_loader[line]
        FN = total_abnormal - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def predict_supervised2(self):
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        if self.model_name == "cnn":
            model = TextCNN(self.dim_model, self.seq_len, 128).to(self.device)
        elif self.model_name == "neurallog":
            model = NeuralLog(num_encoder_layers=1, num_heads=12, dim_model=768, dim_feedforward=2048,
                              droput=0.2).to(self.device)
        else:
            lstm_model = robustlog

            model_init = lstm_model(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    vocab_size=len(vocab),
                                    embedding_dim=self.embedding_dim)
            model = model_init.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        test_normal_loader, test_abnormal_loader = generate(self.output_dir, 'test.pkl', 1,
                                                            is_neural=self.embeddings == 'neural')
        start_time = time.time()
        data = [(k, v, list(k)) for k, v in test_normal_loader.items()]
        logs, labels = sliding_window(data, vocab, history_size=self.history_size, is_train=False,
                                      data_dir=self.data_dir, semantics=self.semantics, is_predict_logkey=False,
                                      e_name=self.embeddings, in_size=self.input_size)
        dataset = log_dataset(logs=logs, labels=labels)
        data_loader = DataLoader(dataset, batch_size=4096, shuffle=False, pin_memory=True)
        normal_results = [0] * len(data)
        for _, (log, label) in enumerate(tqdm(data_loader)):
            seq_idx = log['idx'].tolist()
            # print(seq_idx)
            features = [x.to(self.device) for x in log['features']]
            output, _ = model(features, self.device)
            output = output.softmax(dim=1)
            pred = torch.argsort(output, 1, descending=True)
            pred = pred[:, 0]
            pred = pred.cpu().numpy().tolist()
            for i in range(len(pred)):
                normal_results[seq_idx[i]] = max(normal_results[seq_idx[i]], int(pred[i]))

        total_normal, FP = 0, 0
        for i in range(len(normal_results)):
            if normal_results[i] == 1:
                FP += data[i][1]
            total_normal += data[i][1]
        data = [(k, v, list(k)) for k, v in test_abnormal_loader.items()]
        logs, labels = sliding_window(data, vocab, history_size=self.history_size, is_train=False,
                                      data_dir=self.data_dir, semantics=self.semantics, is_predict_logkey=False,
                                      e_name=self.embeddings, in_size=self.input_size)
        dataset = log_dataset(logs=logs, labels=labels)
        data_loader = DataLoader(dataset, batch_size=4096, shuffle=False, pin_memory=True)
        abnormal_results = [[]] * len(data)
        for _, (log, label) in enumerate(tqdm(data_loader)):
            seq_idx = log['idx'].tolist()
            # print(seq_idx)
            features = [x.to(self.device) for x in log['features']]
            output, _ = model(features, self.device)
            output = output.softmax(dim=1)
            pred = torch.argsort(output, 1, descending=True)
            pred = pred[:, 0]
            pred = pred.cpu().numpy().tolist()
            for i in range(len(pred)):
                # print(len(seq_idx))
                # print(pred[i])
                abnormal_results[seq_idx[i]] = abnormal_results[seq_idx[i]] + [int(pred[i])]
        lead_time = []
        total_abnormal, TP = 0, 0
        for i in range(len(abnormal_results)):
            # print(len(abnormal_results[i]))
            if max(abnormal_results[i]) == 1:
                TP += data[i][1]
                lead_time.append(abnormal_results[i].index(1) + self.history_size + 1)
            total_abnormal += data[i][1]
        TN = total_normal - FP
        FN = total_abnormal - TP

        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def predict_unsupervised(self, model, th=3e-8):
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        # model = AutoEncoder(self.hidden_size, self.num_layers, embedding_dim=self.embedding_dim).to(self.device)
        # model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        # print('model_path: {}'.format(self.model_path))

        test_normal_loader, test_abnormal_loader = generate(self.output_dir, 'test.pkl', 1)
        print(len(test_normal_loader), len(test_abnormal_loader))
        start_time = time.time()
        test_normal_results = [[] for _ in range(len(test_normal_loader))]
        sess_normal_events = [(k, 0) for (k, v) in test_normal_loader.items()]
        num_normal_sess = [test_normal_loader[x] for (x, l) in sess_normal_events]
        seqs, labels = sliding_window(sess_normal_events, vocab, history_size=self.history_size, is_train=False,
                                      is_predict_logkey=False,
                                      data_dir=self.data_dir, semantics=self.semantics)

        dataset = log_dataset(logs=seqs,
                              labels=labels)
        data_loader = DataLoader(dataset,
                                 batch_size=min(len(dataset), 4096),
                                 shuffle=True,
                                 pin_memory=True)
        tbar = tqdm(data_loader, desc="\r")
        with torch.no_grad():
            for _, (log, label) in enumerate(tbar):
                seq_idx = log['idx'].clone().detach().cpu().numpy()
                del log['idx']
                features = [x.to(self.device) for x in log['features']]
                output, _ = model(features=features, device=self.device)
                pred = output['y_pred']
                # print(pred.shape)
                # label = torch.tensor(label).view(-1).to(self.device)
                for i in range(len(seq_idx)):
                    test_normal_results[seq_idx[i]].append(pred[i])

        test_abnormal_results = [[] for _ in range(len(test_abnormal_loader))]
        sess_abnormal_events = [(k, 1) for (k, v) in test_abnormal_loader.items()]
        num_abnormal_sess = [test_abnormal_loader[x] for (x, l) in sess_abnormal_events]
        seqs, labels = sliding_window(sess_abnormal_events, vocab, history_size=self.history_size, is_train=False,
                                      data_dir=self.data_dir, semantics=self.semantics)

        dataset = log_dataset(logs=seqs,
                              labels=labels)
        data_loader = DataLoader(dataset,
                                 batch_size=min(len(dataset), 4096),
                                 shuffle=True,
                                 pin_memory=True)
        tbar = tqdm(data_loader, desc="\r")
        with torch.no_grad():
            for _, (log, label) in enumerate(tbar):
                seq_idx = log['idx'].clone().detach().cpu().numpy()
                del log['idx']
                features = [x.to(self.device) for x in log['features']]
                output, _ = model(features=features, device=self.device)
                pred = output['y_pred']
                # print(pred.shape)
                # label = torch.tensor(label).view(-1).to(self.device)
                for i in range(len(seq_idx)):
                    test_abnormal_results[seq_idx[i]].append(pred[i])

        for i in range(1, 50):
            threshold = th * (i + 1)
            print("Threshold:", threshold)
            FP = 0
            TP = 0
            for j, pred in enumerate(test_abnormal_results):
                # print(pred)
                if max(pred) > threshold:
                    TP += num_abnormal_sess[j]
            FN = sum(num_abnormal_sess) - TP
            for j, pred in enumerate(test_normal_results):
                if max(pred) > threshold:
                    FP += num_normal_sess[j]
            TN = sum(num_normal_sess) - FP
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            FPR = FP / (FP + TN)
            FNR = FN / (TP + FN)
            SP = TN / (TN + FP)
            print("Confusion matrix for threshold", threshold)
            print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
            print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))
            # elapsed_time = time.time() - start_time
            # print('elapsed_time: {}'.format(elapsed_time))
        # for i in range(10):
        #     threshold = th * (i + 1)
        #     print("Threshold:", threshold)
        #     total_normal, total_abnormal = 0, 0
        #     FP = 0
        #     TP = 0
        #     with torch.no_grad():
        #         for line in tqdm(test_normal_loader.keys()):
        #             logs, labels = sliding_window([(line, 0)], vocab, window_size=self.history_size, is_train=False,
        #                                           data_dir=self.data_dir, semantics=self.semantics, is_predict_logkey=False)
        #             dataset = log_dataset(logs=logs, labels=labels)
        #             data_loader = DataLoader(dataset, batch_size=4096, shuffle=True, pin_memory=True)
        #             for _, (log, label) in enumerate(data_loader):
        #                 del log['idx']
        #                 features = [x.to(self.device) for x in log['features']]
        #                 output, _ = model(features, self.device)
        #
        #                 if max(output['y_pred'].clone().detach().cpu().numpy().tolist()) > threshold:
        #                     FP += test_normal_loader[line]
        #                     break
        #             total_normal += test_normal_loader[line]
        #     TN = total_normal - FP
        #     with torch.no_grad():
        #         for line in tqdm(test_abnormal_loader.keys()):
        #             logs, labels = sliding_window([(line, 1)], vocab, window_size=self.history_size, is_train=False,
        #                                           data_dir=self.data_dir, semantics=self.semantics, is_predict_logkey=False)
        #             dataset = log_dataset(logs=logs, labels=labels)
        #             data_loader = DataLoader(dataset, batch_size=4096, shuffle=True, pin_memory=True)
        #             for _, (log, label) in enumerate(data_loader):
        #                 del log['idx']
        #                 features = [x.to(self.device) for x in log['features']]
        #                 output, _ = model(features, self.device)
        #
        #                 if max(output['y_pred'].clone().detach().cpu().numpy().tolist()) > threshold:
        #                     TP += test_abnormal_loader[line]
        #                     break
        #             total_abnormal += test_abnormal_loader[line]
        #     FN = total_abnormal - TP
        #     P = 100 * TP / (TP + FP)
        #     R = 100 * TP / (TP + FN)
        #     F1 = 2 * P * R / (P + R)
        #     FPR = FP / (FP + TN)
        #     FNR = FN / (TP + FN)
        #     SP = TN / (TN + FP)
        #     print("Confusion matrix")
        #     print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        #     print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))
        #
        #     elapsed_time = time.time() - start_time
        #     print('elapsed_time: {}'.format(elapsed_time))
