import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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
    plt.hist(probabilities, bins=100)
    plt.title(str(epoch) + " Histogram of next token probailities " + phase)
    plt.savefig(save_dir + "/Histograms/" + str(epoch) + "_Histogram_next_token_probabilities_" + phase +".png")
    plt.close()


def plot_losses(losses_normal, losses_anomalies, epoch, save_dir, elbow_loss):

    plt.hist(losses_normal.tolist(), bins=100, color='red')
    plt.hist(losses_anomalies.tolist(), bins=100, color='green')
    plt.vlines(elbow_loss, 0, 100, linestyles="dotted", colors="blue")
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

    colors = [0, '#b5179e', '#b5179e', '#b5179e', '#560bad', '#560bad', '#560bad', '#4361ee', '#4361ee', '#4361ee']

    # G
    sns.lineplot(x="epoch", y="g", data=train_metrics_both_best, label="Train Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=train_metrics_both, label="Train Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="g", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="g", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train G")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_G.png")
    plt.close()

    sns.lineplot(x="epoch", y="g", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=test_normal_metrics_both, label="Test Normal Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="g", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="g", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal G")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_G.png")
    plt.close()

    sns.lineplot(x="epoch", y="g", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="g", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="g", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="g", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique G")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_G.png")
    plt.close()

    # Loss
    sns.lineplot(x="epoch", y="th", data=train_metrics_both_best, label="Train Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=train_metrics_both, label="Train Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="th", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="th", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train Loss")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_Loss.png")
    plt.close()

    sns.lineplot(x="epoch", y="th", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=test_normal_metrics_both, label="Test Normal Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="th", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="th", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal Loss")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_Loss.png")
    plt.close()

    sns.lineplot(x="epoch", y="th", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="th", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="th", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="th", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique Loss")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_Loss.png")
    plt.close()

    # F1
    sns.lineplot(x="epoch", y="f1", data=train_metrics_both_best, label="Train Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=train_metrics_both, label="Train Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="f1", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="f1", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train F1")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_F1.png")
    plt.close()

    sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_both, label="Test Normal Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="f1", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal F1")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_F1.png")
    plt.close()

    sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="f1", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique F1")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_F1.png")
    plt.close()

    # P
    sns.lineplot(x="epoch", y="p", data=train_metrics_both_best, label="Train Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=train_metrics_both, label="Train Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="p", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="p", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train P")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_P.png")
    plt.close()

    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_both, label="Test Normal Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="p", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal P")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_P.png")
    plt.close()

    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2',color=colors[2])
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="p", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique P")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_P.png")
    plt.close()

    # R
    sns.lineplot(x="epoch", y="r", data=train_metrics_both_best, label="Train Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=train_metrics_both, label="Train Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="r", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="r", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train R")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_R.png")
    plt.close()

    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_both, label="Test Normal Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="r", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal R")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_R.png")
    plt.close()

    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="r", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique R")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_R.png")
    plt.close()

    # TP
    sns.lineplot(x="epoch", y="tp", data=train_metrics_both_best, label="Train Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=train_metrics_both, label="Train Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="tp", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="tp", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train TP")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_TP.png")
    plt.close()

    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_both, label="Test Normal Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="tp", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal TP")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_TP.png")
    plt.close()

    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="tp", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique TP")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_TP.png")
    plt.close()

    # FP
    sns.lineplot(x="epoch", y="fp", data=train_metrics_both_best, label="Train Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=train_metrics_both, label="Train Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="fp", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fp", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train FP")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_FP.png")
    plt.close()

    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_both, label="Test Normal Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fp", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal FP")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_FP.png")
    plt.close()

    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_g, label="Test Unique G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fp", data=test_unique_metrics_loss, label="Test Unique Loss", linewidth='2', color=colors[8])
    plt.title("Test Unique FP")
    plt.savefig(root_save_dir + "/Metrics/Unique/Test_Unique_FP.png")
    plt.close()

    # FN
    sns.lineplot(x="epoch", y="fn", data=train_metrics_both_best, label="Train Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=train_metrics_g_best, label="Train G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=train_metrics_loss_best, label="Train Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=train_metrics_both, label="Train Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="fn", data=train_metrics_g, label="Train G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fn", data=train_metrics_loss, label="Train Loss", linewidth='2', color=colors[8])
    plt.title("Train FN")
    plt.savefig(root_save_dir + "/Metrics/Train_Normal/Train_FN.png")
    plt.close()

    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_both_best, label="Test Normal Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_g_best, label="Test Normal G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_loss_best, label="Test Normal Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_both, label="Test Normal Both", linewidth='2', color=colors[2])
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_g, label="Test Normal G", linewidth='2', color=colors[5])
    sns.lineplot(x="epoch", y="fn", data=test_normal_metrics_loss, label="Test Normal Loss", linewidth='2', color=colors[8])
    plt.title("Test Normal FN")
    plt.savefig(root_save_dir + "/Metrics/Normal/Test_Normal_FN.png")
    plt.close()

    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_both_best, label="Test Unique Both Best", linewidth='2', color=colors[1], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_g_best, label="Test Unique G Best", linewidth='2', color=colors[4], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_loss_best, label="Test Unique Loss Best", linewidth='2', color=colors[7], linestyle='--')
    sns.lineplot(x="epoch", y="fn", data=test_unique_metrics_both, label="Test Unique Both", linewidth='2', color=colors[2])
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

if __name__ == "__main__":
    plot_train_valid_loss("./../../../dataset/bgl/parser_type=drain/window_type=session/train_size=0.8/deeplog/runs/history_size=60/max_anomalies_ratio=0.0/max_epoch=50/n_epochs_stop=3/min_loss_reduction_per_epoch=0.95/lr=0.0001/batch_size=4096/mean_selection_activated=False/Csvs", \
                          "./../../../dataset/bgl/parser_type=drain/window_type=session/train_size=0.8/deeplog/runs/history_size=60/max_anomalies_ratio=0.0/max_epoch=50/n_epochs_stop=3/min_loss_reduction_per_epoch=0.95/lr=0.0001/batch_size=4096/mean_selection_activated=False/Pngs",\
                          "./../../../dataset/bgl/parser_type=drain/window_type=session/train_size=0.8/deeplog/runs/history_size=60/max_anomalies_ratio=0.0/max_epoch=50/n_epochs_stop=3/min_loss_reduction_per_epoch=0.95/lr=0.0001/batch_size=4096/mean_selection_activated=False/",\
                          False)