#Standard modules
import os
import time
import random
import argparse

import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

#Custom modules
import helpers as hlp
import models as mdl
from run import run, benchmark


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="all", help="Specific task (or all)")
parser.add_argument('--device', type=str, default="cuda", help="Device to use")
args = parser.parse_args()

def generate_plots(n_clients_list, ph_kd_list, ph_fl_list, ph_fd_list, ph_il_list, export_dir, r=-1):

    # Data organization
    for evaluation in ["Train", "Validation", "Train (global)", "Validation (global)"]:
        for metric in ["loss", "accuracy"]:
            data_kd_mean = []
            data_kd_std = []
            data_fl_mean = []
            data_fl_std = []
            data_fd_mean = []
            data_fd_std = []
            data_il_mean = []
            data_il_std = []

            for i, (ph_kd, ph_fl, ph_fd, ph_il) in enumerate(zip(ph_kd_list, ph_fl_list, ph_fd_list, ph_il_list)):
                data_kd = np.array([ph[evaluation][metric][r] for ph in ph_kd])
                data_kd_mean.append(data_kd.mean())
                data_kd_std.append(data_kd.std())

                data_fl = np.array([ph[evaluation][metric][r] for ph in ph_fl])
                data_fl_mean.append(data_fl.mean())
                data_fl_std.append(data_fl.std())

                data_fd = np.array([ph[evaluation][metric][r] for ph in ph_fd])
                data_fd_mean.append(data_fd.mean())
                data_fd_std.append(data_fd.std())

                data_il = np.array([ph[evaluation][metric][r] for ph in ph_il])
                data_il_mean.append(data_il.mean())
                data_il_std.append(data_il.std())

            fig, ax = plt.subplots(1,1, figsize=(4,4))

            ax.errorbar(n_clients_list, data_kd_mean, yerr=data_kd_std, marker="o", label="CFKD")
            ax.errorbar(n_clients_list, data_fl_mean, yerr=data_fl_std, marker="o", label="FL")
            ax.errorbar(n_clients_list, data_fd_mean, yerr=data_fd_std, marker="o", label="FD")
            ax.errorbar(n_clients_list, data_il_mean, yerr=data_il_std, marker="o", label="IL")
            ax.set_xlabel("Number of clients")
            ax.set_ylabel(metric)
            ax.legend()
            
            savepath = export_dir + "/figures/round{}".format(r)
            os.makedirs(savepath, exist_ok=True)
            fig.savefig(savepath + "/{}_{}.png".format(evaluation, metric), bbox_inches='tight')
            plt.close("all")
            

# Global parameter
reduced = 0.1
rounds = 20
n_avg=16
track_history = 10
n_clients_list = range(1,11)
device=args.device
r_list = [1, 2]

##################################################################
if args.dataset in ["MNIST", "all"]:
    # Figure MNIST
    dataset = "MNIST"
    model = "LeNet5"
    export_dir = "./saves/MNIST_n_clients"


    # Experiment
    ph_kd_list = []
    ph_fl_list = []
    ph_fd_list = [] 
    ph_il_list = []
    for n_clients in n_clients_list:
        pt_kd, pt_fl, pt_fd, pt_il = benchmark(n_clients=n_clients, dataset=dataset, model=model, reduced=reduced, 
                                               n_avg=n_avg, rounds=rounds, track_history=track_history, 
                                               export_dir=None, device=device)
        ph_kd_list.append([pt.perf_histories for pt in pt_kd])
        ph_fl_list.append([pt.perf_histories for pt in pt_fl])
        ph_fd_list.append([pt.perf_histories for pt in pt_fd])
        ph_il_list.append([pt.perf_histories for pt in pt_il])
        plt.close("all")

    for r in r_list:
        generate_plots(n_clients_list, ph_kd_list, ph_fl_list, ph_fd_list, ph_il_list, export_dir, r)

##################################################################
# Figure FMNIST
if args.dataset in ["FMNIST", "all"]:
    dataset = "FMNIST"
    model = "ResNet9"
    export_dir = "./saves/FMNIST_n_clients"


    # Experiment
    ph_kd_list = []
    ph_fl_list = []
    ph_fd_list = [] 
    ph_il_list = []
    for n_clients in n_clients_list:
        pt_kd, pt_fl, pt_fd, pt_il = benchmark(n_clients=n_clients, dataset=dataset, model=model, reduced=reduced, 
                                               n_avg=n_avg, rounds=rounds, track_history=track_history, 
                                               export_dir=None, device=device)
        ph_kd_list.append([pt.perf_histories for pt in pt_kd])
        ph_fl_list.append([pt.perf_histories for pt in pt_fl])
        ph_fd_list.append([pt.perf_histories for pt in pt_fd])
        ph_il_list.append([pt.perf_histories for pt in pt_il])
        plt.close("all")

    for r in r_list:
        generate_plots(n_clients_list, ph_kd_list, ph_fl_list, ph_fd_list, ph_il_list, export_dir, r)

##################################################################
# Figure CIFAR10
if args.dataset in ["CIFAR10", "all"]:
    dataset = "CIFAR10"
    model = "ResNet18"
    export_dir = "./saves/CIFAR10_n_clients"


    # Experiment
    ph_kd_list = []
    ph_fl_list = []
    ph_fd_list = [] 
    ph_il_list = []
    for n_clients in n_clients_list:
        pt_kd, pt_fl, pt_fd, pt_il = benchmark(n_clients=n_clients, dataset=dataset, model=model, reduced=reduced, 
                                               n_avg=n_avg, rounds=rounds, track_history=track_history, 
                                               export_dir=None, device=device)
        ph_kd_list.append([pt.perf_histories for pt in pt_kd])
        ph_fl_list.append([pt.perf_histories for pt in pt_fl])
        ph_fd_list.append([pt.perf_histories for pt in pt_fd])
        ph_il_list.append([pt.perf_histories for pt in pt_il])
        plt.close("all")

    for r in r_list:
        generate_plots(n_clients_list, ph_kd_list, ph_fl_list, ph_fd_list, ph_il_list, export_dir, r)

