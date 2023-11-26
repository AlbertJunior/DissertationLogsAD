import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from torch import nn
import csv
from kneed import KneeLocator


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))


# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


# https://blog.csdn.net/folk_/article/details/80208557
def train_val_split(logs_meta, labels, val_ratio=0.1):
    total_num = len(labels)
    train_index = list(range(total_num))
    train_logs = {}
    val_logs = {}
    for key in logs_meta.keys():
        train_logs[key] = []
        val_logs[key] = []
    train_labels = []
    val_labels = []
    val_num = int(total_num * val_ratio)

    for i in range(val_num):
        random_index = int(np.random.uniform(0, len(train_index)))
        for key in logs_meta.keys():
            val_logs[key].append(logs_meta[key][random_index])
        val_labels.append(labels[random_index])
        del train_index[random_index]

    for i in range(total_num - val_num):
        for key in logs_meta.keys():
            train_logs[key].append(logs_meta[key][train_index[i]])
        train_labels.append(labels[train_index[i]])

    return train_logs, train_labels, val_logs, val_labels


def plot_next_token_histogram_of_probabilities(phase, epoch, probabilities, save_dir):
    plt.hist(probabilities)

    with open(save_dir + "/Histograms/" + str(epoch) + "_Histogram_next_token_probabilities_" + phase + ".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(probabilities.tolist())

    plt.title(str(epoch) + " Histogram of next token probailities " + phase)
    plt.xlabel("Next token probability")
    plt.ylabel("Train sequences no.")
    plt.savefig(save_dir + "/Histograms/" + str(epoch) + "_Histogram_next_token_probabilities_" + phase +".png")
    plt.close()


def plot_losses(losses_normal, losses_anomalies, epoch, save_dir, elbow_loss):

    plt.hist(losses_normal.tolist(), stacked=True, density = True, color='green')
    plt.hist(losses_anomalies.tolist(), stacked=True, density = True, color='red')
    with open(save_dir +f"Losses_{epoch}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(losses_normal.tolist(), ["Normal"] * len(losses_normal.tolist())))
        writer.writerows(zip(losses_anomalies.tolist(), ["Abnormal"] * len(losses_anomalies.tolist())))
    plt.axvline(elbow_loss, linestyle="dashed", color="blue")
    plt.xlabel("Loss value")
    plt.ylabel("Sequence no.")
    plt.title("Losses hist")
    plt.savefig(save_dir + f"Losses_{epoch}.png")
    plt.close()

def plot_train_valid_loss(save_dir, save_dir_photos, root_save_dir, mean_selection_activated):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(save_dir_photos):
        os.mkdir(save_dir_photos)

    train_loss = pd.read_csv(save_dir + "/train_log.csv")
    valid_loss = pd.read_csv(save_dir + "/valid_log.csv")

    train_metrics_both_best = pd.read_csv(save_dir + "/train_metrics_both_best_log.csv")
    train_metrics_g_best = pd.read_csv(save_dir + "/train_metrics_g_best_log.csv")
    train_metrics_loss_best = pd.read_csv(save_dir + "/train_metrics_loss_best_log.csv")
    train_metrics_g = pd.read_csv(save_dir + "/train_metrics_g_log.csv")
    train_metrics_loss = pd.read_csv(save_dir + "/train_metrics_loss_log.csv")
    train_metrics_both = pd.read_csv(save_dir + "/train_metrics_both_log.csv")

    test_normal_metrics_both_best = pd.read_csv(save_dir + "/test_normal_metrics_both_best_log.csv")
    test_normal_metrics_g_best = pd.read_csv(save_dir + "/test_normal_metrics_g_best_log.csv")
    test_normal_metrics_loss_best = pd.read_csv(save_dir + "/test_normal_metrics_loss_best_log.csv")
    test_normal_metrics_g = pd.read_csv(save_dir + "/test_normal_metrics_g_log.csv")
    test_normal_metrics_loss = pd.read_csv(save_dir + "/test_normal_metrics_loss_log.csv")
    test_normal_metrics_both = pd.read_csv(save_dir + "/test_normal_metrics_both_log.csv")

    test_unique_metrics_both_best = pd.read_csv(save_dir + "/test_unique_metrics_both_best_log.csv")
    test_unique_metrics_g_best = pd.read_csv(save_dir + "/test_unique_metrics_g_best_log.csv")
    test_unique_metrics_loss_best = pd.read_csv(save_dir + "/test_unique_metrics_loss_best_log.csv")
    test_unique_metrics_g = pd.read_csv(save_dir + "/test_unique_metrics_g_log.csv")
    test_unique_metrics_loss = pd.read_csv(save_dir + "/test_unique_metrics_loss_log.csv")
    test_unique_metrics_both = pd.read_csv(save_dir + "/test_unique_metrics_both_log.csv")

    colors = [0, 'darkred', 'darkred', 'darkred', 'royalblue', 'royalblue', 'royalblue', 'orange', 'orange', 'orange']
    sns.set_context("notebook", font_scale=1.1, rc={"lines.linewidth": 2.5})

    # G
    with sns.axes_style("whitegrid"):
        sns.lineplot(x="epoch", y="g", data=train_metrics_both_best, label="Train Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
        sns.lineplot(x="epoch", y="g", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
        sns.lineplot(x="epoch", y="g", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
        sns.lineplot(x="epoch", y="g", data=train_metrics_both, label="Train Both", linewidth='3', color=colors[2], alpha=0.7)
        sns.lineplot(x="epoch", y="g", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
        sns.lineplot(x="epoch", y="g", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
        plt.title("Train G")
        plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_G.png")
        plt.close()

    with sns.axes_style("whitegrid"):
        ax = plt.figure().gca()
        ax.yaxis.get_major_locator().set_params(integer=True)
        sns.lineplot(x="epoch", y="g", data=test_normal_metrics_both_best, label="Best Mixed", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
        sns.lineplot(x="epoch", y="g", data=test_normal_metrics_g_best, label="Best Extreme", linewidth='2', color=colors[4], linestyle='--')
        sns.lineplot(x="epoch", y="g", data=test_normal_metrics_loss_best, label="Best Global", linewidth='2', color=colors[7], linestyle='--')
        sns.lineplot(x="epoch", y="g", data=test_normal_metrics_both, label="Mixed", linewidth='3', color=colors[2], alpha=0.7)
        sns.lineplot(x="epoch", y="g", data=test_normal_metrics_g, label="Extreme", linewidth='2', color=colors[5])
        sns.lineplot(x="epoch", y="g", data=test_normal_metrics_loss, label="Global", linewidth='2', color=colors[8])
        plt.xlabel("Epoch")
        plt.ylabel("G threshold")

        # plt.title("Only normal, non-unique; G threshold")
        plt.title("Unlabeled w/o Mean selection, non-unique; G threshold")
        # plt.title("Unlabeled w/ Mean selection, non-unique; G threshold")
        plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_G.png")
        plt.close()

    with sns.axes_style("whitegrid"):
        ax = plt.figure().gca()
        ax.yaxis.get_major_locator().set_params(integer=True)
        sns.lineplot(x="epoch", y="g", data=test_unique_metrics_both_best, label="Best Mixed", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
        sns.lineplot(x="epoch", y="g", data=test_unique_metrics_g_best, label="Best Extreme", linewidth='2', color=colors[4], linestyle='--')
        sns.lineplot(x="epoch", y="g", data=test_unique_metrics_loss_best, label="Best Global", linewidth='2', color=colors[7], linestyle='--')
        sns.lineplot(x="epoch", y="g", data=test_unique_metrics_both, label="Mixed", linewidth='3', color=colors[2], alpha=0.7)
        sns.lineplot(x="epoch", y="g", data=test_unique_metrics_g, label="Extreme", linewidth='2', color=colors[5])
        sns.lineplot(x="epoch", y="g", data=test_unique_metrics_loss, label="Global", linewidth='2', color=colors[8])
        plt.xlabel("Epoch")
        plt.ylabel("G threshold")

        # plt.title("Only normal, unique; G threshold")
        plt.title("Unlabeled w/o Mean selection, unique; G threshold")
        # plt.title("Unlabeled w/ Mean selection, unique; G threshold")
        plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_G.png")
        plt.close()

    # Loss
    sns.lineplot(x="epoch", y="th", data=train_metrics_both_best, label="Train Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="th", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=train_metrics_both, label="Train Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="th", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="th", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train Loss")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_Loss.png")
    plt.close()

    with sns.axes_style("whitegrid"):
        sns.lineplot(x="epoch", y="th", data=test_normal_metrics_both_best, label="Best Mixed", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
        sns.lineplot(x="epoch", y="th", data=test_normal_metrics_g_best, label="Best Extreme", linewidth='2', color=colors[4], linestyle='--')
        sns.lineplot(x="epoch", y="th", data=test_normal_metrics_loss_best, label="Best Global", linewidth='2', color=colors[7], linestyle='--')
        sns.lineplot(x="epoch", y="th", data=test_normal_metrics_both, label="Mixed", linewidth='3', color=colors[2], alpha=0.7)
        sns.lineplot(x="epoch", y="th", data=test_normal_metrics_g, label="Extreme", linewidth='2', color=colors[5])
        sns.lineplot(x="epoch", y="th", data=test_normal_metrics_loss, label="Global", linewidth='2', color=colors[8])
        plt.xlabel("Epoch")
        plt.ylabel("Loss threshold")
        # plt.title("Only normal, non-unique; Loss threshold")
        # plt.title("Unlabeled w/o Mean selection, unique; Loss threshold")
        plt.title("Unlabeled w/ Mean selection, unique; Loss threshold")

        plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_Loss.png")
        plt.close()

    with sns.axes_style("whitegrid"):
        sns.lineplot(x="epoch", y="th", data=test_unique_metrics_both_best, label="Best Mixed", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
        sns.lineplot(x="epoch", y="th", data=test_unique_metrics_g_best, label="Best Extreme", linewidth='2', color=colors[4], linestyle='--')
        sns.lineplot(x="epoch", y="th", data=test_unique_metrics_loss_best, label="Best Global", linewidth='2', color=colors[7], linestyle='--')
        sns.lineplot(x="epoch", y="th", data=test_unique_metrics_both, label="Mixed", linewidth='3', color=colors[2], alpha=0.7)
        sns.lineplot(x="epoch", y="th", data=test_unique_metrics_g, label="Extreme", linewidth='2', color=colors[5])
        sns.lineplot(x="epoch", y="th", data=test_unique_metrics_loss, label="Global", linewidth='2', color=colors[8])
        plt.xlabel("Epoch")
        plt.ylabel("Loss threshold")
        # plt.title("Only normal, unique; Loss threshold")
        # plt.title("Unlabeled w/o Mean selection, non-unique; Loss threshold")
        plt.title("Unlabeled w/ Mean selection, non-unique; Loss threshold")
        plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_Loss.png")
        plt.close()

    # F1
    with sns.axes_style("whitegrid"):
        sns.lineplot(x="epoch", y="f1", data=train_metrics_both_best, label="Train Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
        sns.lineplot(x="epoch", y="f1", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
        sns.lineplot(x="epoch", y="f1", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
        sns.lineplot(x="epoch", y="f1", data=train_metrics_both, label="Train Both", linewidth='3', color=colors[2], alpha=0.7)
        sns.lineplot(x="epoch", y="f1", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
        sns.lineplot(x="epoch", y="f1", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
        plt.ylim(0, 100)
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Train F1")
        plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_F1.png")
        plt.close()

    with sns.axes_style("whitegrid"):
        train_metrics_both_best['f1'] = train_metrics_both_best['f1'] / 100
        test_normal_metrics_g_best['f1'] = test_normal_metrics_g_best['f1'] / 100
        test_normal_metrics_loss_best['f1'] = test_normal_metrics_loss_best['f1'] / 100
        test_normal_metrics_both['f1'] = test_normal_metrics_both['f1'] / 100
        test_normal_metrics_g['f1'] = test_normal_metrics_g['f1'] / 100
        test_normal_metrics_loss['f1'] = test_normal_metrics_loss['f1'] / 100

        sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_both_best, label="Best Mixed", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
        sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_g_best, label="Best Extreme", linewidth='2', color=colors[4], linestyle='--')
        sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_loss_best, label="Best Global", linewidth='2', color=colors[7], linestyle='--')
        sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_both, label="Mixed", linewidth='3', color=colors[2], alpha=0.7)
        sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_g, label="Extreme", linewidth='2', color=colors[5])
        sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_loss, label="Global", linewidth='2', color=colors[8])
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        # plt.title("Only normal, non-unique; F1 score")
        # plt.title("Unlabeled w/o Mean selection, non-unique; F1 score")
        plt.title("Unlabeled w/ Mean selection, non-unique; F1 score")
        plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_F1.png")
        plt.close()

    with sns.axes_style("whitegrid"):
        test_unique_metrics_both_best['f1'] = test_unique_metrics_both_best['f1'] / 100
        test_unique_metrics_g_best['f1'] = test_unique_metrics_g_best['f1'] / 100
        test_unique_metrics_loss_best['f1'] = test_unique_metrics_loss_best['f1'] / 100
        test_unique_metrics_both['f1'] = test_unique_metrics_both['f1'] / 100
        test_unique_metrics_g['f1'] = test_unique_metrics_g['f1'] / 100
        test_unique_metrics_loss['f1'] = test_unique_metrics_loss['f1'] / 100

        sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_both_best, label="Best Mixed", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
        sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_g_best, label="Best Extreme", linewidth='2', color=colors[4], linestyle='--')
        sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_loss_best, label="Best Global", linewidth='2', color=colors[7], linestyle='--')
        sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_both, label="Mixed", linewidth='3', color=colors[2], alpha=0.7)
        sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_g, label="Extreme", linewidth='2', color=colors[5])
        sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_loss, label="Global", linewidth='2', color=colors[8])
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        # plt.title("Only normal, unique; F1 score")
        # plt.title("Unlabeled w/o Mean selection, unique; F1 score")
        plt.title("Unlabeled w/ Mean selection, unique; F1 score")
        plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_F1.png")
        plt.close()

    # P
    sns.lineplot(x="epoch", y="p", data=train_metrics_both_best, label="Train Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="p", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=train_metrics_both, label="Train Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="p", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="p", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Train P")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_P.png")
    plt.close()

    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_both, label="Test Normal Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Test Normal P")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_P.png")
    plt.close()

    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2',color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Test Unique P")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_P.png")
    plt.close()

    # R
    sns.lineplot(x="epoch", y="r", data=train_metrics_both_best, label="Train Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="r", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=train_metrics_both, label="Train Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="r", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="r", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Train R")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_R.png")
    plt.close()

    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_both, label="Test Normal Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Test Normal R")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_R.png")
    plt.close()

    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_both, label="Test Unique Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Test Unique R")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_R.png")
    plt.close()

    # TP
    sns.lineplot(x="epoch", y="tp", data=train_metrics_both_best, label="Train Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="tp", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=train_metrics_both, label="Train Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="tp", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="tp", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train TP")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_TP.png")
    plt.close()

    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_both, label="Test Normal Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal TP")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_TP.png")
    plt.close()

    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_both, label="Test Unique Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique TP")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_TP.png")
    plt.close()

    # FP
    sns.lineplot(x="epoch", y="fp", data=train_metrics_both_best, label="Train Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="fp", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=train_metrics_both, label="Train Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="fp", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fp", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train FP")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_FP.png")
    plt.close()

    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_both, label="Test Normal Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal FP")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_FP.png")
    plt.close()

    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_both, label="Test Unique Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique FP")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_FP.png")
    plt.close()

    # FN
    sns.lineplot(x="epoch", y="fn", data=train_metrics_both_best, label="Train Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="fn", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=train_metrics_both, label="Train Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="fn", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fn", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train FN")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_FN.png")
    plt.close()

    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_both, label="Test Normal Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal FN")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_FN.png")
    plt.close()

    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='3', color=colors[1], linestyle='--', alpha=0.7)
    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_both, label="Test Unique Both", linewidth='3', color=colors[2], alpha=0.7)
    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique FN")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_FN.png")
    plt.close()

    # -----------------------------------------
    sns.lineplot(x="epoch",y="loss" , data = train_loss, label="train loss")
    sns.lineplot(x="epoch",y="loss" , data = valid_loss, label="valid loss")
    plt.title("epoch vs train loss vs valid loss")
    plt.savefig(save_dir_photos+"/train_valid_loss.png")
    plt.close()

    sns.lineplot(x="epoch", y="acc", data=train_loss, label="train acc")
    sns.lineplot(x="epoch", y="acc", data=valid_loss, label="valid acc")
    plt.title("epoch vs train acc vs valid acc")
    plt.savefig(save_dir_photos + "/train_valid_acc.png")
    plt.close()

    sns.lineplot(x="epoch", y="skewness", data=train_loss, label="train skewness")
    sns.lineplot(x="epoch", y="skewness", data=valid_loss, label="valid skewness")
    plt.title("epoch vs train skewness vs valid skewness")
    plt.savefig(save_dir_photos + "/train_valid_skewness.png")
    plt.close()

    sns.lineplot(x="epoch", y="kurtosis", data=train_loss, label="train kurtosis")
    sns.lineplot(x="epoch", y="kurtosis", data=valid_loss, label="valid kurtosis")
    plt.title("epoch vs train kurtosis vs valid kurtosis")
    plt.savefig(save_dir_photos + "/train_valid_kurtosis.png")
    plt.close()

    sns.lineplot(x="epoch", y="kurtosis", data=train_loss, label="train kurtosis")
    sns.lineplot(x="epoch", y="kurtosis", data=valid_loss, label="valid kurtosis")
    plt.title("epoch vs train kurtosis vs valid kurtosis")
    plt.savefig(save_dir_photos + "/train_valid_kurtosis.png")
    plt.close()

    if mean_selection_activated:
        sns.lineplot(x="epoch", y="elim_no", data=train_loss, label="train eliminated number")
        sns.lineplot(x="epoch", y="an_elim_no", data=train_loss, label="train anomalies eliminated number")
        plt.title("epoch vs eliminated number vs anomalies eliminated number")
        plt.savefig(save_dir_photos + "/train_valid_eliminated_no.png")
        plt.close()

        sns.lineplot(x="epoch", y="elim_per", data=train_loss, label="train eliminated percentage")
        sns.lineplot(x="epoch", y="an_elim_per", data=train_loss, label="train anomalies eliminated percentage")
        plt.title("epoch vs eliminated percentage vs anomalies eliminated percentage")
        plt.savefig(save_dir_photos + "/train_valid_eliminated_percentage.png")
        print("plot done")
        plt.close()



def plot_sequence_len(save_dir):
    normal_seq_len = []
    with open(save_dir+"train", "r") as f:
        for line in f.readlines():
            line = line.split()
            normal_seq_len.append(len(line))
    with open(save_dir+"test_normal", 'r') as f:
        for line in f.readlines():
            normal_seq_len.append(len(line.split()))
    abnormal_seq_line = []
    with open(save_dir+"test_abnormal", "r") as f:
        for line in f.readlines():
            abnormal_seq_line.append(len(line.split()))
    sns.distplot(normal_seq_len, label="normal")
    sns.distplot(abnormal_seq_line, label = "abnormal")
    plt.title("session length distribution")
    plt.xlabel("num of log keys in a session")
    plt.legend()
    plt.show()
    plt.close()


def plot_train_percentage(csv_path, file_name):
    train_loss = pd.read_csv(csv_path + "/" + file_name)
    colors = [0, 'darkred', 'darkred', 'darkred', 'royalblue', 'royalblue', 'royalblue', 'orange', 'orange', 'orange']
    sns.set_context("notebook", font_scale=1.1, rc={"lines.linewidth": 2.5})

    with sns.axes_style("whitegrid"):
        sns.lineplot(x="epoch", y="elim_no", data=train_loss, label="train eliminated number", linewidth='2', color=colors[4])
        sns.lineplot(x="epoch", y="an_elim_no", data=train_loss, label="train anomalies eliminated number", linewidth='2', color=colors[1])
        plt.xlabel("Epoch")
        plt.ylabel("Eliminated")

        plt.title("epoch vs eliminated number vs anomalies eliminated number")
        plt.savefig(csv_path + "/train_valid_eliminated_no.png")
        plt.close()

    with sns.axes_style("whitegrid"):
        sns.lineplot(x="epoch", y="elim_per", data=train_loss, label="Total eliminated %", linewidth='2', color=colors[4])
        sns.lineplot(x="epoch", y="an_elim_per", data=train_loss, label="Anomalies eliminated %", linewidth='2', color=colors[7])
        plt.xlabel("Epoch")
        plt.ylabel("Eliminated %")
        plt.ylim(0, 1)
        plt.title("Total eliminated % vs. anomalies eliminated %")
        plt.savefig(csv_path + "/train_valid_eliminated_percentage.png")

        print("plot done")
        plt.close()


def plot_anomalies(csv_path, file_name, epoch):
    dataset = pd.read_csv(csv_path + "/" + file_name + ".csv")
    dataset.columns = ['x', 'y']

    colors = [0, 'darkred', 'darkred', 'darkred', 'royalblue', 'royalblue', 'royalblue', 'orange', 'orange', 'orange']
    sns.set_context("notebook", font_scale=1.1, rc={"lines.linewidth": 1.5})

    with sns.axes_style("whitegrid"):
        kn = KneeLocator(dataset['x'], dataset['y'], curve='convex', direction='decreasing', online=True)
        kn.plot_knee()
        elbow = kn.knee

        fig, ax = plt.subplots(figsize=(6.2, 5))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K'))
        sns.lineplot(x="x", y="y", data=dataset, linewidth='2', color=colors[4])
        # plt.hist(losses_normal.tolist(), stacked=True, density=True, color='green')
        # plt.hist(losses_anomalies.tolist(), stacked=True, density=True, color='red')
        plt.axvline(elbow, linestyle="dashed", color=colors[1], linewidth='1.1')
        plt.text(elbow + 0.6, 15000, 'G threshold', rotation=90)


        plt.xlabel("G candidates")
        plt.ylabel("Candidate anomalies")
        # plt.ylim(0, 35000)
        plt.title("G chosen by Extreme heuristic. Epoch " + str(epoch))
        plt.savefig(csv_path + "/" + file_name + ".png")
        plt.close()

def plot_losses(csv_path, file_name, elbow_loss, epoch=None):
    dataset = pd.read_csv(csv_path + "/" + file_name + ".csv")
    dataset.columns = ['Loss', 'Sequence type']

    colors = [0, 'darkred', 'darkred', 'darkred', 'royalblue', 'royalblue', 'royalblue', 'orange', 'orange', 'orange']
    sns.set_context("notebook", font_scale=1.1, rc={"lines.linewidth": 1.5})

    with sns.axes_style("whitegrid"):
        sns.histplot(data=dataset, x="Loss", hue="Sequence type", palette=[colors[4], colors[7]])
        plt.axvline(elbow_loss, linestyle="dashed", color=colors[1], label="Loss threshold")
        plt.text(elbow_loss + 0.1, 250, 'Loss threshold', rotation=90)

        plt.xlabel("Loss value")
        plt.ylabel("Sequence no.")
        plt.xlim(0, 11)
        if epoch is not None:
            plt.title("Loss values histogram. Epoch " + str(epoch))
        else:
            plt.title("Loss values histogram")
        plt.savefig(csv_path + "/" + file_name + ".png")
        plt.close()


if __name__ == "__main__":
    # plot_losses("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Losses", 'Losses_60', 1.6)
    # plot_train_percentage("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Train_percentage", "TRAIN_1.CSV")
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number", "Epoch_16", 9)
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number", "BGL_Epoch_35", 9)
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number/Progress",
    #                "Epoch_1", 1)
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number/Progress",
    #                "Epoch_2", 2)
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number/Progress",
    #                "Epoch_3", 3)
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number/Progress",
    #                "Epoch_78",78)
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number/Progress",
    #                "Epoch_66", 66)
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number/Progress",
    #                "Epoch_67", 67)
    # plot_anomalies("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Anomalies_number/Progress",
    #                "Epoch_68", 68)

    # plot_losses("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Losses/Progress", 'Losses_2', 3.78, 2)
    plot_losses("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Losses/Progress", 'Losses_17', 4.05, 17)
    plot_losses("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Losses/Progress", 'Losses_18', 3.9, 18)
    plot_losses("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Losses/Progress", 'Losses_19', 3.4, 19)

    plot_losses("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Losses/Progress", 'Losses_40', 1.85, 40)
    plot_losses("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Losses/Progress", 'Losses_41', 1.80, 41)
    plot_losses("/Volumes/workplace/disertatie/LogADEmpirical/dataset/create_images/Losses/Progress", 'Losses_42', 1.85, 42)


    # plot_train_valid_loss("./../../../dataset/bgl/parser_type=drain/window_type=session/train_size=0.8/deeplog/runs/history_size=60/max_anomalies_ratio=0.0/max_epoch=50/n_epochs_stop=3/min_loss_reduction_per_epoch=0.95/lr=0.0001/batch_size=4096/mean_selection_activated=False/Csvs", \
    #                       "./../../../dataset/bgl/parser_type=drain/window_type=session/train_size=0.8/deeplog/runs/history_size=60/max_anomalies_ratio=0.0/max_epoch=50/n_epochs_stop=3/min_loss_reduction_per_epoch=0.95/lr=0.0001/batch_size=4096/mean_selection_activated=False/Pngs",\
    #                       "./../../../dataset/bgl/parser_type=drain/window_type=session/train_size=0.8/deeplog/runs/history_size=60/max_anomalies_ratio=0.0/max_epoch=50/n_epochs_stop=3/min_loss_reduction_per_epoch=0.95/lr=0.0001/batch_size=4096/mean_selection_activated=False/",\
    #                       False)
    # plot_train_valid_loss("./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=True/Csvs", \
    #                   "./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=True/PngsToday",\
    #                   "./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=True/",\
    #                   True)
    # plot_train_valid_loss(
    #     "./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=False/Csvs", \
    #     "./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=False/PngsToday", \
    #     "./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=False/", \
    #     False)

    # plot_train_valid_loss(
    #     "./../../../dataset/bgl/deeplog/runs/history_size=60/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.01/batch_size=4096/mean_selection_activated=True/Csvs", \
    #     "./../../../dataset/bgl/deeplog/runs/history_size=60/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.01/batch_size=4096/mean_selection_activated=True/PngsToday", \
    #     "./../../../dataset/bgl/deeplog/runs/history_size=60/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.01/batch_size=4096/mean_selection_activated=True/", \
    #     True)

    # plot_train_valid_loss(
    #     "./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=False/Csvs", \
    #     "./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=False/PngsToday", \
    #     "./../../../dataset/hdfs/parser_type=drain/window_type=session/train_size=0.1/deeplog/runs/history_size=10/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.001/batch_size=4096/mean_selection_activated=False/", \
    #     True)

    # plot_train_valid_loss(
    #     "./../../../dataset/bgl/deeplog/runs/history_size=60/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.01/batch_size=4096/mean_selection_activated=True/Csvs", \
    #     "./../../../dataset/bgl/deeplog/runs/history_size=60/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.01/batch_size=4096/mean_selection_activated=True/PngsToday", \
    #     "./../../../dataset/bgl/deeplog/runs/history_size=60/max_anomalies_ratio=1.0/max_epoch=100/n_epochs_stop=10/min_loss_reduction_per_epoch=0.99/lr=0.01/batch_size=4096/mean_selection_activated=True/", \
    #     True)