#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import gc

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest as iForest

from logadempirical.logdeep.dataset.log import log_dataset
from logadempirical.logdeep.dataset.sample import sliding_window, load_features
from logadempirical.logdeep.tools.utils import plot_train_valid_loss, plot_next_token_histogram_of_probabilities
from logadempirical.logdeep.models.lstm import deeplog, loganomaly, robustlog
from logadempirical.logdeep.models.autoencoder import AutoEncoder
from logadempirical.logdeep.models.cnn import TextCNN
from logadempirical.neural_log.transformers import NeuralLog


def sa_value(losses, ul, T):
    return torch.exp(-torch.abs(losses - ul) / (1 + 0.05 * T))


def mean_selection(losses, T):
    Q1 = torch.quantile(losses, 0.25)
    Q2 = torch.quantile(losses, 0.5)
    Q3 = torch.quantile(losses, 0.75)
    IQR = Q3 - Q1
    ul = Q2
    ll = Q1 - 1.5 * IQR

    sa_verdict = torch.rand(losses.shape[0], device=losses.device) > sa_value(losses, ul, T)
    upper_limit = torch.logical_and(losses > ul, sa_verdict)
    return torch.where(torch.logical_not(upper_limit))[0], torch.where(upper_limit)[0]

def skewness_fn(x, device, dim=1):
    """Calculates skewness of data "x" along dimension "dim"."""
    x = torch.FloatTensor(x)
    std, mean = torch.std_mean(x, dim)
    n = torch.Tensor([x.shape[dim]]).to(device)
    eps = 1e-6  # for stability

    sample_bias_adjustment = torch.sqrt(n * (n - 1)) / (n - 2)
    skewness = sample_bias_adjustment * (
            (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(3), dim) / n)
            / std.pow(3).clamp(min=eps)
    )
    return skewness


def kurtosis_fn(x, device, dim=1):
    """Calculates kurtosis of data "x" along dimension "dim"."""
    x = torch.FloatTensor(x)
    std, mean = torch.std_mean(x, dim)
    n = torch.Tensor([x.shape[dim]]).to(device)
    eps = 1e-6  # for stability

    sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
    kurtosis = sample_bias_adjustment * (
            (n + 1)
            * (
                    (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(4), dim) / n)
                    / std.pow(4).clamp(min=eps)
            )
            - 3 * (n - 1)
    )
    return kurtosis


class Trainer():
    def __init__(self, options):
        self.model_name = options['model_name']
        self.model_dir = options['model_dir']
        self.model_path = options['model_path']
        self.data_dir = options['output_dir']
        self.run_dir = options['run_dir']
        self.vocab_path = options["vocab_path"]
        # self.scale_path = options["scale_path"]
        self.emb_dir = options['data_dir']

        self.window_size = options['window_size']
        self.min_len = options["min_len"]
        self.seq_len = options["seq_len"]
        self.history_size = options['history_size']

        self.input_size = options["input_size"]
        self.hidden_size = options["hidden_size"]
        self.embedding_dim = options["embedding_dim"]
        self.num_layers = options["num_layers"]
        self.num_workers = options["num_workers"]

        self.max_epoch = options['max_epoch']
        self.n_epochs_stop = options["n_epochs_stop"]
        self.device = options['device']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.batch_size = options['batch_size']
        self.min_loss_reduction_per_epoch = options['min_loss_reduction_per_epoch']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.embeddings = options['embeddings']

        self.sample = options['sample']
        self.train_ratio = options['train_ratio']
        self.train_size = options['train_size']
        self.valid_ratio = options['valid_ratio']

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.mean_selection_activated = options['mean_selection_activated']
        self.verbose = options['verbose']

        # transformers' parameters
        self.num_encoder_layers = options["num_encoder_layers"]
        self.num_decoder_layers = options["num_decoder_layers"]
        self.dim_model = options["dim_model"]
        self.num_heads = options["num_heads"]
        self.dim_feedforward = options["dim_feedforward"]
        self.transformers_dropout = options["transformers_dropout"]
        self.random_sample = options["random_sample"]
        self.anomalies_ratio = options["max_anomalies_ratio"]

        # detection model: predict the next log or classify normal/abnormal
        if self.model_name in ["cnn", "logrobust", "autoencoder", "neurallog"]:
            self.is_predict_logkey = False
        else:
            self.is_predict_logkey = True

        self.early_stopping = False
        self.epochs_no_improve = 0
        self.criterion = None

        print("Loading vocab")
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(len(vocab))

        if self.sample == 'sliding_window':
            print("Loading train dataset\n")
            # data = load_features(self.data_dir + "train.pkl", only_normal=self.is_predict_logkey)
            data = load_features(self.data_dir + "train.pkl", anomalies_ratio=self.anomalies_ratio)

            n_train = int(len(data))
            print("Nr secvente train from train.pkl", n_train)
            train_logs, train_labels, anomaly_labels = sliding_window(data,
                                                                      vocab=vocab,
                                                                      history_size=self.history_size,
                                                                      data_dir=self.emb_dir,
                                                                      is_predict_logkey=self.is_predict_logkey,
                                                                      semantics=self.semantics,
                                                                      sample_ratio=self.train_ratio,
                                                                      e_name=self.embeddings,
                                                                      in_size=self.input_size
                                                                      )

            train_logs, train_labels, anomaly_labels = shuffle(train_logs, train_labels, anomaly_labels)
            # train_logs = train_logs[:200000]
            # train_labels = train_labels[:200000]
            n_val = int(len(train_logs) * self.valid_ratio)
            val_logs, val_labels, val_anomaly = train_logs[-n_val:], train_labels[-n_val:], anomaly_labels[-n_val:]
            train_logs, train_labels, anomaly_labels = train_logs[:-n_val], train_labels[:-n_val], anomaly_labels[:-n_val]
            del data
            gc.collect()
        else:
            raise NotImplementedError

        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    labels_anomaly=anomaly_labels)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    labels_anomaly=val_anomaly)

        del train_logs
        del val_logs
        gc.collect()

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=self.num_workers)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=self.num_workers)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d secvente de train, %d secvente de validation' %
              (self.num_train_log, self.num_valid_log))

        self.threshold_rate = self.num_train_log // self.num_valid_log

        if self.model_name == "cnn":
            print(self.dim_model, self.seq_len)
            self.model = TextCNN(self.dim_model, self.seq_len, 128).to(self.device)
        elif self.model_name == "autoencoder":
            self.model = AutoEncoder(self.hidden_size, self.num_layers, embedding_dim=self.embedding_dim).to(
                self.device)
        elif self.model_name == "neurallog":
            self.model = NeuralLog(num_encoder_layers=1, num_heads=12, dim_model=768, dim_feedforward=2048,
                                   droput=0.2).to(self.device)
        else:
            if self.model_name == "deeplog":
                lstm_model = deeplog
            elif self.model_name == "loganomaly":
                lstm_model = loganomaly
            else:
                lstm_model = robustlog

            model_init = lstm_model(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    vocab_size=len(vocab),
                                    embedding_dim=self.embedding_dim)
            self.model = model_init.to(self.device)

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999)
            )
        else:
            raise NotImplementedError

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.softmax_function = torch.nn.Softmax(dim=1)
        self.time_criterion = nn.MSELoss()

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        if self.mean_selection_activated:
            self.log = {
                "train": {key: []
                          for key in
                          ["epoch", "lr", "time", "loss", "acc", "kurtosis", "skewness",
                           "elim_no", "elim_per", "an_elim_no", "an_elim_per"]},

                "train_statistics": {key: []
                          for key in
                          ["vocab_size", "total_sessions_no", "normal_sessions_no", "abnormal_sessions_no", "abnormal_sessions_per",
                           "total_unique_sessions_no", "unique_normal_sessions_no", "unique_abnormal_sessions_no", "unique_abnormal_sessions_per",
                           "total_sequences_no", "normal_sequences_no", "abnormal_sequences_no", "abnormal_sequences_per"]},
                "test_statistics": {key: []
                          for key in
                          ["total_sessions_no", "normal_sessions_no", "abnormal_sessions_no", "abnormal_sessions_per",
                           "total_unique_sessions_no", "unique_normal_sessions_no", "unique_abnormal_sessions_no", "unique_abnormal_sessions_per",
                           "total_sequences_no", "normal_sequences_no", "abnormal_sequences_no", "abnormal_sequences_per"]},

                "valid": {key: []
                          for key in ["epoch", "lr", "time", "loss", "acc", "kurtosis", "skewness"]},
                  "train_metrics_both_best": {key: []
                         for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_g_best": {key: []
                                            for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_loss_best": {key: []
                                            for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_g": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_loss": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_both": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},

                "test_normal_metrics_both_best": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_g_best": {key: []
                                                  for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_loss_best": {key: []
                                                  for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_g": {key: []
                                 for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_loss": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_both": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},

                "test_unique_metrics_both_best": {key: []
                                     for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_g_best": {key: []
                                                  for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_loss_best": {key: []
                                                  for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_g": {key: []
                                       for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_loss": {key: []
                                          for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_both": {key: []
                                          for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]}
            }
        else:
            self.log = {
                "train": {key: []
                          for key in ["epoch", "lr", "time", "loss", "acc", "kurtosis", "skewness"]},
                "valid": {key: []
                          for key in ["epoch", "lr", "time", "loss", "acc", "kurtosis", "skewness"]},
                "train_statistics": {key: []
                          for key in
                          ["vocab_size", "total_sessions_no", "normal_sessions_no", "abnormal_sessions_no", "abnormal_sessions_per",
                           "total_unique_sessions_no", "unique_normal_sessions_no", "unique_abnormal_sessions_no", "unique_abnormal_sessions_per",
                           "total_sequences_no", "normal_sequences_no", "abnormal_sequences_no", "abnormal_sequences_per"]},
                "test_statistics": {key: []
                          for key in
                          ["total_sessions_no", "normal_sessions_no", "abnormal_sessions_no", "abnormal_sessions_per",
                           "total_unique_sessions_no", "unique_normal_sessions_no", "unique_abnormal_sessions_no", "unique_abnormal_sessions_per",
                           "total_sequences_no", "normal_sequences_no", "abnormal_sequences_no", "abnormal_sequences_per"]},


                "train_metrics_both_best": {key: []
                         for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_g_best": {key: []
                                            for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_loss_best": {key: []
                                            for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_g": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_loss": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "train_metrics_both": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},

                "test_normal_metrics_both_best": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_g_best": {key: []
                                                  for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_loss_best": {key: []
                                                  for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_g": {key: []
                                 for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_loss": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_normal_metrics_both": {key: []
                                    for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},

                "test_unique_metrics_both_best": {key: []
                                     for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_g_best": {key: []
                                                  for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_loss_best": {key: []
                                                  for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_g": {key: []
                                       for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_loss": {key: []
                                          for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]},
                "test_unique_metrics_both": {key: []
                                          for key in ["epoch", "g", "th", "f1", "p", "r", "tp", "fp", "fn"]}
            }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.model_path
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self, save_dir):
        for key, values in self.log.items():
            pd.DataFrame(values).to_csv(save_dir + "/" + key + "_log.csv", index=False)

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("\nStarting EPOCH: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()

        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        acc = 0
        probabilities_real_next_token = []
        skewness = 0
        kurtosis = 0
        total_log = 0
        no_not_selected = 0
        no_trained = 0
        no_anomalies = 0

        anomalies_not_selected = 0
        normals_not_selected = 0

        anomalies_selected = 0
        normals_selected = 0
        for i, (log, label, anomaly_label) in enumerate(tbar):
            del log['idx']
            features = [x.to(self.device) for x in log['features']]
            output, _ = self.model(features=features, device=self.device)
            probab_output = self.softmax_function(output)

            if isinstance(output, dict):
                loss = output['loss']
                total_log += len(label)

                total_losses += float(loss)
                loss /= self.accumulation_step
                loss.backward()
                if (i + 1) % self.accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                tbar.set_description(
                    "Train loss: {0:.8f}".format(total_losses / (i + 1)))
            else:
                label = label.view(-1).to(self.device)
                label = label - 1

                for idx in range(probab_output.size()[0]):
                    probabilities_real_next_token.append(probab_output[idx][label[idx].item()].item())

                skewness = skewness_fn(probabilities_real_next_token, self.device, dim=0).item()
                kurtosis = kurtosis_fn(probabilities_real_next_token, self.device, dim=0).item()

                loss = self.criterion(output, label)
                TEMP_INIT = 1
                if self.mean_selection_activated:
                    T = (self.max_epoch - 1 - epoch) / (self.max_epoch - 1) * TEMP_INIT
                    selected, not_selected = mean_selection(loss, epoch)
                    selected = selected.to(self.device)
                    not_selected = not_selected.to(self.device)
                    if epoch > 0:
                        loss = loss[selected].mean()
                    else:
                        loss = loss.mean()

                    no_not_selected += len(not_selected)
                    no_selected = len(selected)
                    no_trained += (len(not_selected) + len(selected))

                    anomaly_label = anomaly_label.to(self.device)

                    labels_for_not_selected = anomaly_label[not_selected]
                    labels_for_selected = anomaly_label[selected]

                    not_selected_anomalies_condition = (labels_for_not_selected == 1)
                    not_selected_normal_condition = (labels_for_not_selected == 0)

                    selected_anomalies_condition = (labels_for_selected == 1)
                    selected_normal_condition = (labels_for_selected == 0)

                    not_selected_anomalies = torch.where(not_selected_anomalies_condition)[0]
                    not_selected_normal = torch.where(not_selected_normal_condition)[0]

                    selected_anomalies = torch.where(selected_anomalies_condition)[0]
                    selected_normal = torch.where(selected_normal_condition)[0]

                    anomalies_not_selected += len(not_selected_anomalies)
                    normals_not_selected += len(not_selected_normal)

                    anomalies_selected += len(selected_anomalies)
                    normals_selected += len(selected_normal)

                    no_anomalies += (len(selected_anomalies) + len(not_selected_anomalies))
                else:
                    loss = loss.mean()

                predicted = output.argmax(dim=1).cpu().numpy()
                label = np.array([y.cpu() for y in label])
                acc += (predicted == label).sum()
                total_log += len(label)

                total_losses += float(loss)
                loss /= self.accumulation_step
                loss.backward()

                if (i + 1) % self.accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                tbar.set_description(
                    "Train loss: {0:.8f} - Train acc: {1:.2f} - Train kurtosis: {2:.2f} - Train skewness: {3:.2f}".format(
                        total_losses / (i + 1), acc / total_log, kurtosis, skewness))
        plot_next_token_histogram_of_probabilities("train", epoch, probabilities_real_next_token, self.run_dir)
        self.log['train']['loss'].append(total_losses / num_batch)
        self.log['train']['acc'].append(acc / total_log)
        self.log['train']['kurtosis'].append(kurtosis)
        self.log['train']['skewness'].append(skewness)

        if self.mean_selection_activated:
            print("Eliminated instances percentage: ", no_not_selected / no_trained,
                  str(no_not_selected) + "/" + str(no_trained))
            self.log['train']['elim_no'].append(no_not_selected)
            self.log['train']['elim_per'].append(no_not_selected / no_trained)

            if normals_not_selected + anomalies_not_selected == 0:
                print("Eliminated anomalies percentage: ", 0,
                      str(anomalies_not_selected) + "/" + str(no_anomalies))
                self.log['train']['an_elim_no'].append(0)
                self.log['train']['an_elim_per'].append(0)
            else:
                print("Eliminated anomalies percentage: ", anomalies_not_selected / no_anomalies,
                      str(anomalies_not_selected) + "/" + str(no_anomalies))
                self.log['train']['an_elim_no'].append(anomalies_not_selected)
                self.log['train']['an_elim_per'].append(anomalies_not_selected / no_anomalies)
        # if len(self.log['train']['loss']) > 2 and self.log['train']['loss'][-2] - self.log['train']['loss'][-1] < self.log['train']['loss'][-2] * 0.01:
        #     return True
        # return False

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("\nStarting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        acc = 0
        probabilities_real_next_token = []
        total_log = 0
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)

        for i, (log, label, anomaly) in enumerate(tbar):
            with torch.no_grad():
                del log['idx']
                features = [x.to(self.device) for x in log['features']]
                output, _ = self.model(features=features, device=self.device)
                probab_output = self.softmax_function(output)
                if isinstance(output, dict):
                    loss = output['loss']
                else:
                    label = label.view(-1).to(self.device)
                    label = label - 1
                    loss = 0 if not self.is_logkey else self.criterion(output, label).mean()

                    predicted = torch.max(output.softmax(dim=-1), 1).indices.cpu().numpy()
                    label = np.array([y.cpu() for y in label])
                    acc += (predicted == label).sum()

                    for idx in range(probab_output.size()[0]):
                        probabilities_real_next_token.append(probab_output[idx][label[idx].item()].item())

                    total_log += len(label)

                total_losses += float(loss)

        skewness = skewness_fn(probabilities_real_next_token, self.device, dim=0).item()
        kurtosis = kurtosis_fn(probabilities_real_next_token, self.device, dim=0).item()
        if total_log:
            print("\nValidation loss:", total_losses / num_batch, "Validation accuracy:", acc / total_log,
                  "Validation kurtosis:", kurtosis, "Validation skewness:", skewness)
        else:
            print("\nValidation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)
        self.log['valid']['acc'].append(acc / total_log)
        self.log['valid']['kurtosis'].append(kurtosis)
        self.log['valid']['skewness'].append(skewness)
        plot_next_token_histogram_of_probabilities("valid", epoch, probabilities_real_next_token, self.run_dir)

        if total_losses / num_batch < self.min_loss_reduction_per_epoch * self.best_loss:
            self.best_loss = total_losses / num_batch
            self.epochs_no_improve = 0
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix=self.model_name)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve == self.n_epochs_stop:
            self.early_stopping = True
            print("Early stopping")
        return total_losses / num_batch

    def train_autoencoder2(self):
        print("Compute representation of log sequences...")
        logs = []
        tbar = tqdm(self.valid_loader, desc="\r")
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        self.model.eval()
        with torch.no_grad():
            for i, (log, label) in enumerate(tbar):
                embs = log['features'][2].numpy()
                del log['idx']
                features = [x.to(self.device) for x in log['features']]
                output, _ = self.model(features=features, device=self.device)
                repr = output['repr'].clone().detach().cpu().numpy()
                for j in range(len(repr)):
                    logs.append((embs[j], repr[j]))
        print(logs[0][0].shape)
        reprs = np.array([log[1] for i, log in enumerate(logs)])
        print("Find normal logs...")
        iforest = iForest(n_estimators=100, max_samples="auto", contamination="auto", verbose=1)
        iforest.fit(reprs)
        y_pred = iforest.predict(reprs)
        y_pred = np.where(y_pred > 0, 0, 1)
        normal_logs = [log[0] for i, log in enumerate(logs) if y_pred[i] == 0]
        print(len(normal_logs))

        class AEDataset(Dataset):
            def __init__(self, logs):
                self.logs = logs

            def __len__(self):
                return len(self.logs)

            def __getitem__(self, idx):
                return self.logs[idx]

        dataset = AEDataset(normal_logs)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            pin_memory=False)

        print("Train second autoencoder...")
        model_ae2 = AutoEncoder(self.hidden_size, self.num_layers, embedding_dim=self.embedding_dim).to(self.device)
        model_ae2.train()
        optimizer = torch.optim.Adam(
            model_ae2.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
        )

        total_losses = 0
        optimizer.zero_grad()
        for epoch in range(0, 20):
            print("Epoch {}...".format(epoch + 1))
            tbar = tqdm(loader, desc="\r")
            for i, log in enumerate(tbar):
                features = log.to(self.device)
                output, _ = model_ae2(features=[0, 0, features], device=self.device)
                loss = output['loss']
                total_losses += float(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tbar.set_description(
                    "Train loss: {0:.5f}".format(total_losses / (i + 1)))
        recst_value = []
        # model_ae2 = best_model
        model_ae2.eval()
        print("Compute threshold...")
        for i, log in enumerate(tbar):
            features = log.to(self.device)
            output, _ = model_ae2(features=[0, 0, features], device=self.device)
            y_pred = output['y_pred']
            recst_value.extend(y_pred.clone().detach().cpu().numpy().tolist())
        from statistics import stdev
        return model_ae2, stdev(recst_value)  # * self.threshold_rate

    def start_train(self, predicter, vocab_size):
        val_loss = 0
        n_epoch = 0
        n_val_epoch = 0
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.early_stopping:
                break
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in (50, 75):
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.train(epoch)

            n_epoch += 1
            if epoch > 0:
                self.log["train_statistics"]["vocab_size"].append(vocab_size)
                val_loss += self.valid(epoch)
                self.save_checkpoint(epoch,
                                     save_optimizer=False,
                                     suffix=self.model_name)
                n_val_epoch += 1
                print("======== My contribution ===========")
                elbow_g, elbow_loss = predicter.compute_elbow(epoch, self.run_dir + "/Csvs")
                print("======== Their approach ===========")
                predicter.predict_semi_supervised(epoch, elbow_g, elbow_loss, self.run_dir + "/Csvs", self)
                # print("======== My contribution ===========")
                # predicter.predict_semi_supervised_ramona()
            self.save_log(self.run_dir + "/Csvs")

        plot_train_valid_loss(self.run_dir + "/Csvs", self.run_dir + "/Pngs", self.mean_selection_activated)
        if self.model_name == "autoencoder":
            return self.train_autoencoder2()  # self.model, val_loss / n_val_epoch
