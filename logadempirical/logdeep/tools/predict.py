#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pickle
from random import shuffle

import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import auc

from logadempirical.logdeep.dataset.log import log_dataset
from logadempirical.logdeep.dataset.sample import sliding_window
from logadempirical.logdeep.models.lstm import deeplog, loganomaly, robustlog
from logadempirical.logdeep.models.cnn import TextCNN
from logadempirical.logdeep.models.autoencoder import AutoEncoder
from logadempirical.neural_log.transformers import NeuralLog


def generate(output_dir, name, is_neural):
    print("Loading", output_dir + name)
    with open(output_dir + name, 'rb') as f:
        data_iter = pickle.load(f)
    print("Length test", len(data_iter))
    normal_iter = {}
    abnormal_iter = {}
    for seq in data_iter:
        if not isinstance(seq['Label'], int):
            label = max(seq['Label'].tolist())
        else:
            label = seq['Label']
        if is_neural:
            key = tuple(seq['Seq'])
        else:
            key = tuple(seq['EventId'])
        if label == 0:
            if key not in normal_iter:
                normal_iter[key] = 1
            else:
                normal_iter[key] += 1
        else:
            if key not in abnormal_iter:
                abnormal_iter[key] = 1
            else:
                abnormal_iter[key] += 1

    return normal_iter, abnormal_iter





class Predicter():
    def __init__(self, options):
        self.data_dir = options['data_dir']
        self.output_dir = options['output_dir']
        self.model_dir = options['model_dir']
        self.run_dir = options["run_dir"]
        self.vocab_path = options["vocab_path"]
        self.model_path = options['model_path']
        self.model_name = options['model_name']

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

        self.lower_bound = 0
        self.upper_bound = 3

    def create_plot(self, title, anomalies_per_thresold, epoch):
        x = range(self.num_candidates)
        reversed_list = anomalies_per_thresold[::-1]

        plt.plot(x, reversed_list, 'r-', label=title)
        plt.legend()
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

    def compute_anomaly(self, results, num, threshold=0):
        # print(num)
        total_errors = 0
        for i, line in enumerate(results):
            for seq in line:
                # print("George: Ramona found this bug. It was seq[1] before..")
                if (seq[1] - 1) not in seq[0][:threshold]:
                    total_errors += num[i]
                    break
        return total_errors

    def compute_anomaly_unique(self, results, num, threshold=0):
        # print(num)
        total_errors = 0
        for i, line in enumerate(results):
            for seq in line:
                # print("George: Ramona found this bug. It was seq[1] before..")
                if (seq[1] - 1) not in seq[0][:threshold]:
                    total_errors += 1
                    break
        return total_errors

    def find_best_threshold(self, test_normal_results, num_normal_session_logs, test_abnormal_results,
                            num_abnormal_session_logs, epoch, threshold_range):
        test_abnormal_length = sum(num_abnormal_session_logs)
        test_normal_length = sum(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        # print(threshold_range)
        fps, tps, tns, fns, ps, rs, f1s = [], [], [], [], [], [], []
        for th in range(threshold_range, 0, -1):
            FP = self.compute_anomaly(test_normal_results, num_normal_session_logs, th + 1)
            TP = self.compute_anomaly(test_abnormal_results, num_abnormal_session_logs, th + 1)
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            fps.append(FP)
            tps.append(TP)
            tns.append(TN)
            fns.append(FN)
            if TP == 0:
                print("For g ", th, " test metrics, TP: ", TP, "FP: ", FP, ", FN:", FN, ", P:100, R:0, F1:0")
                ps.append(100)
                rs.append(0)
                f1s.append(0)
                continue

            # Compute precision, recall and F1-measure
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            # print(th + 1, FP, FN, P, R)
            ps.append(P)
            rs.append(R)
            f1s.append(F1)
            print("For g ", th, " test metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P, ", R:", R, ", F1:", F1)
            if F1 > res[-1]:
                res = [th, TP, TN, FP, FN, P, R, F1]
        print("Ce a dat pe validare")
        print(res[0] + 1, res[3], res[4], res[5], res[6])
        print()
        self.create_plot("Metrics/Normal/Recall", rs, epoch)
        self.create_plot("Metrics/Normal/Precision", ps, epoch)
        self.create_plot("Metrics/Normal/F1Score", f1s, epoch)
        self.create_plot("Metrics/Normal/TP", tps, epoch)
        self.create_plot("Metrics/Normal/TN", tns, epoch)
        self.create_plot("Metrics/Normal/FP", fps, epoch)
        self.create_plot("Metrics/Normal/FN", fns, epoch)
        return res

    def find_best_threshold_unique(self, test_normal_results, num_normal_session_logs, test_abnormal_results,
                            num_abnormal_session_logs, epoch, threshold_range):
        test_abnormal_length = len(num_abnormal_session_logs)
        test_normal_length = len(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        # print(threshold_range)
        fps, tps, tns, fns, ps, rs, f1s = [], [], [], [], [], [], []
        for th in range(threshold_range, 0, -1):
            FP = self.compute_anomaly_unique(test_normal_results, num_normal_session_logs, th + 1)
            TP = self.compute_anomaly_unique(test_abnormal_results, num_abnormal_session_logs, th + 1)
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            fps.append(FP)
            tps.append(TP)
            tns.append(TN)
            fns.append(FN)
            if TP == 0:
                print("For g ", th, " test metrics, TP: ", TP, "FP: ", FP, ", FN:", FN, ", P:100, R:0, F1:0")
                ps.append(100)
                rs.append(0)
                f1s.append(0)
                continue

            # Compute precision, recall and F1-measure
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            # print(th + 1, FP, FN, P, R)
            ps.append(P)
            rs.append(R)
            f1s.append(F1)
            print("For g ", th, " test metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P, ", R:", R, ", F1:", F1)
            if F1 > res[-1]:
                res = [th, TP, TN, FP, FN, P, R, F1]
        print("Ce a dat pe validare")
        print(res[0] + 1, res[3], res[4], res[5], res[6])
        print()
        self.create_plot("Metrics/Unique/Recall", rs, epoch)
        self.create_plot("Metrics/Unique/Precisiton", ps, epoch)
        self.create_plot("Metrics/Unique/F1Score", f1s, epoch)
        self.create_plot("Metrics/Unique/TP", tps, epoch)
        self.create_plot("Metrics/Unique/TN", tns, epoch)
        self.create_plot("Metrics/Unique/FP", fps, epoch)
        self.create_plot("Metrics/Unique/FN", fns, epoch)
        return res

    def compute_metrics_on_threshold(self, test_normal_results, num_normal_session_logs, test_abnormal_results, num_abnormal_session_logs, th):
        test_abnormal_length = sum(num_abnormal_session_logs)
        test_normal_length = sum(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        # print(threshold_range)
        FP = self.compute_anomaly(test_normal_results, num_normal_session_logs, th + 1)
        TP = self.compute_anomaly(test_abnormal_results, num_abnormal_session_logs, th + 1)

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
        print(th + 1, FP, FN, P, R)
        print()
        if F1 > res[-1]:
            res = [th, TP, TN, FP, FN, P, R, F1]
        return res

    def compute_metrics_on_threshold_unique(self, test_normal_results, num_normal_session_logs, test_abnormal_results, num_abnormal_session_logs, th):
        test_abnormal_length = len(num_abnormal_session_logs)
        test_normal_length = len(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        # print(threshold_range)
        FP = self.compute_anomaly_unique(test_normal_results, num_normal_session_logs, th + 1)
        TP = self.compute_anomaly_unique(test_abnormal_results, num_abnormal_session_logs, th + 1)

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
        print(th + 1, FP, FN, P, R)
        print()
        if F1 > res[-1]:
            res = [th, TP, TN, FP, FN, P, R, F1]
        return res

    def semi_supervised_helper(self, model, logs, vocab, data_type, scale=None, min_len=0):

        test_results = [[] for _ in range(len(logs))]
        l = data_type == "test_abnormal"
        sess_events = [(k, l, []) for (k, v) in logs.items()]
        num_sess = [logs[x] for (x, l, _) in sess_events]
        seqs, labels, anomaly = sliding_window(sess_events, vocab, history_size=self.history_size, is_train=False,
                                      data_dir=self.data_dir, semantics=self.semantics)

        dataset = log_dataset(logs=seqs,
                              labels=labels,
                              labels_anomaly=anomaly)
        data_loader = DataLoader(dataset,
                                 batch_size=min(len(dataset), 512),
                                 shuffle=False,
                                 pin_memory=True)
        tbar = tqdm(data_loader, desc="\r")
        with torch.no_grad():
            for _, (log, label, anomaly) in enumerate(tbar):
                seq_idx = log['idx'].clone().detach().cpu().numpy()
                del log['idx']
                features = [x.to(self.device) for x in log['features']]
                output, _ = model(features=features, device=self.device)
                output = output.softmax(dim=-1)
                # label = torch.tensor(label).view(-1).to(self.device)
                if self.is_logkey:
                    for i in range(len(seq_idx)):
                        test_results[seq_idx[i]].append(
                            (torch.argsort(output[i], descending=True)[:self.num_candidates].clone().detach().cpu(),
                             label[i]))
        return test_results, num_sess

    def find_elbow(self, test_normal_results, num_normal_session_logs, test_abnormal_results, num_abnormal_session_logs, epoch, threshold_range):
        test_abnormal_length = sum(num_abnormal_session_logs)
        test_normal_length = sum(num_normal_session_logs)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        no_anomalies_predicted = []
        fps, tps, tns, fns, ps, rs, f1s = [], [], [], [], [], [], []

        for th in range(threshold_range, 0, -1):
            FP = self.compute_anomaly(test_normal_results, num_normal_session_logs, th + 1)
            TP = self.compute_anomaly(test_abnormal_results, num_abnormal_session_logs, th + 1)
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            fps.append(FP)
            tps.append(TP)
            tns.append(TN)
            fns.append(FN)
            anomalies = FP + TP
            no_anomalies_predicted.append(anomalies)
            if TP == 0:
                print("For g ", th, " Train metrics, TP: ", TP, "FP: ", FP, ", FN:", FN, ", P:100, R:0, F1:0")
                ps.append(100)
                rs.append(0)
                f1s.append(0)
                continue

            # Compute precision, recall and F1-measure
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            # print(th + 1, FP, FN, P, R)
            ps.append(P)
            rs.append(R)
            f1s.append(F1)
            print("For g ", th, " train metrics, TP: ", TP, ", FP: ", FP, ", FN:", FN, ", P:", P, ", R:", R, ", F1:", F1)
            if F1 > res[-1]:
                res = [th, TP, TN, FP, FN, P, R, F1]

        print("Ce a dat pe train")
        print(res[0] + 1, res[3], res[4], res[5], res[6])
        print()
        self.create_plot("Metrics/Train/Recall", rs, epoch)
        self.create_plot("Metrics/Train/Precision", ps, epoch)
        self.create_plot("Metrics/Train/F1Score", f1s, epoch)
        self.create_plot("Metrics/Train/TP", tps, epoch)
        self.create_plot("Metrics/Train/TN", tns, epoch)
        self.create_plot("Metrics/Train/FP", fps, epoch)
        self.create_plot("Metrics/Train/FN", fns, epoch)
        return no_anomalies_predicted

    def compute_elbow(self, epoch):
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

        train_normal, train_abnormal = generate(self.output_dir, 'train.pkl', self.embeddings == 'neural')
        print("Nr secvente unice train: normale vs anormale")
        print(len(train_normal), len(train_abnormal))

        start_time = time.time()
        train_normal_results, num_normal = self.semi_supervised_helper(model, train_normal, vocab, 'test_normal')
        train_abnormal_results, num_abnormal = self.semi_supervised_helper(model, train_abnormal, vocab, 'test_abnormal')

        print("------------------------JUST NORMAL FOR ELBOW TRAIN SEQUENCES----------------------------")
        anomalies_per_thresold = self.find_elbow(train_normal_results, num_normal,
                                                 train_abnormal_results, num_abnormal,
                                                 epoch,
                                                 threshold_range=self.num_candidates)
        print(anomalies_per_thresold)
        self.create_plot("Number anomalies", anomalies_per_thresold, epoch)
        print("Changes in number of anomalies")
        for idx in range(1, len(anomalies_per_thresold)):
            print("Pentru g " + str(self.num_candidates-idx) + " anomalies diff:" + str(anomalies_per_thresold[idx] - anomalies_per_thresold[idx-1]))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def predict_semi_supervised(self, epoch):

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

        test_normal, test_abnormal = generate(self.output_dir, 'test.pkl', self.embeddings == 'neural')
        print("Nr secvente unice test: normale vs anormale")
        print(len(test_normal), len(test_abnormal))

        # Test the model
        start_time = time.time()
        test_normal_results, num_normal = self.semi_supervised_helper(model, test_normal, vocab, 'test_normal')
        test_abnormal_results, num_abnormal = self.semi_supervised_helper(model, test_abnormal, vocab, 'test_abnormal')
        print("------------------------NORMAL TEST SEQUENCES----------------------------")
        TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(test_normal_results, num_normal,
                                                                test_abnormal_results, num_abnormal,
                                                                epoch,
                                                                threshold_range=self.num_candidates)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)

        print('Best threshold', TH)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

        print("--------------------------UNIQUE TEST SEQUENCES------------------------------------")
        TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold_unique(test_normal_results, num_normal,
                                                                test_abnormal_results, num_abnormal,
                                                                epoch,
                                                                threshold_range=self.num_candidates)
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

        test_normal, test_abnormal = generate(self.output_dir, 'test.pkl', self.embeddings == 'neural')

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
            valid_normal_results, num_valid_normal = self.semi_supervised_helper(model, valid_normal_dict, vocab, 'test_normal')
            valid_abnormal_results, num_valid_abnormal = self.semi_supervised_helper(model, valid_abnormal_dict, vocab, 'test_abnormal')

            test_normal_results, num_test_normal = self.semi_supervised_helper(model, test_normal_dict, vocab, 'test_normal')
            test_abnormal_results, num_test_abnormal = self.semi_supervised_helper(model, test_abnormal_dict, vocab, 'test_abnormal')

            print("--------------------------NORMAL TEST SEQUENCES------------------------------------")
            TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(valid_normal_results, num_valid_normal,
                                                                    valid_abnormal_results, num_valid_abnormal,
                                                                    threshold_range=self.num_candidates)

            TH, TP, TN, FP, FN, P, R, F1 = self.compute_metrics_on_threshold(test_normal_results, num_test_normal,
                                                                    test_abnormal_results, num_test_abnormal, TH)
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

            TH, TP, TN, FP, FN, P, R, F1 = self.compute_metrics_on_threshold_unique(test_normal_results, num_test_normal,
                                                                             test_abnormal_results, num_test_abnormal,
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

        test_normal_loader, test_abnormal_loader = generate(self.output_dir, 'test.pkl',
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

        test_normal_loader, test_abnormal_loader = generate(self.output_dir, 'test.pkl',
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

        test_normal_loader, test_abnormal_loader = generate(self.output_dir, 'test.pkl')
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
