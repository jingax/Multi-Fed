# -----------------------------------------------------------------------------
# This file contains the helper code that was developped
# during my master thesis at MIT.
#
# 2022 Frédéric Berdoz, Boston, USA
# -----------------------------------------------------------------------------

# Miscellaneous
import os
from datetime import datetime
import time
import copy
import random

# Data processing
import pandas as pd
import numpy as np
import scipy
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

# AI
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import datasets

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns



def load_data(dataset="MNIST", data_dir="./data", reduced=False
              , normalize="image-wise", flatten=False, device="cpu"):
    """Load the specified dataset.
    
    Arguments:
        - dataset: Name of the dataset to load.
        - data_dir: Directory where to store (or load if already stored) the dataset.
        - reduced: Boolean/'small'/'tiny'/float between 0 and 1. Reduce the dataset size.
        - normalize: Whether to normalize the data ('image-wise', 'channel-wise' or 'sample-wise').
        - flatten: Whether to flatten the data (i.g. for FC nets).
        - device: Device where to ship the data.
        
    Returns:
        - train_input: A tensor with the train inputs.
        - train_target: A tensor with the train targets.
        - test_input: A tensor with the test inputs.
        - test_target: A tensor with the test targets.
        - meta: A dictionry with useful metadata on the dataset.
    """
    
    # Initialize meta data:
    meta = {"n_class": None,
            "in_dimension": None}
    
    if dataset == "CIFAR10":
        # Load
        print("** Using CIFAR **")
        print("Load train data...")
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train=True,download=True)
        print("Load validation data...")
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train=False,download=True)

        # Process train data
        train_input = torch.from_numpy(cifar_train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        train_target = torch.tensor(cifar_train_set.targets, dtype=torch.int64)
        
        # Process validation data
        test_input = torch.from_numpy(cifar_test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        test_target = torch.tensor(cifar_test_set.targets, dtype=torch.int64)
        
        # Update metadata
        meta["n_class"] = 10
    
    elif dataset == "CIFAR100":
        # Load
        print("** Using CIFAR **")
        print("Load train data...")
        cifar_train_set = datasets.CIFAR100(data_dir + '/cifar100/', train=True,download=True)
        print("Load validation data...")
        cifar_test_set = datasets.CIFAR100(data_dir + '/cifar100/', train=False,download=True)

        # Process train data
        train_input = torch.from_numpy(cifar_train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        train_target = torch.tensor(cifar_train_set.targets, dtype=torch.int64)
        
        # Process validation data
        test_input = torch.from_numpy(cifar_test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        test_target = torch.tensor(cifar_test_set.targets, dtype=torch.int64)
        
        # Update metadata
        meta["n_class"] = 100
        
    elif dataset == "MNIST":
        print("** Using MNIST **")
        print("Load train data...")
        mnist_train_set = datasets.MNIST(data_dir, train=True, download=True)
        print("Load validation data...")
        mnist_test_set = datasets.MNIST(data_dir, train=False, download=True)

        # Process train data
        train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.targets
        
        # Process validation data
        test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.targets
        
        # Update metadata
        meta["n_class"] = 10
    
    elif dataset == "FMNIST":
        print("** Using FMNIST **")
        print("Load train data...")
        mnist_train_set = datasets.FashionMNIST(data_dir, train=True, download=True)
        print("Load validation data...")
        mnist_test_set = datasets.FashionMNIST(data_dir, train=False, download=True)

        # Process train data
        train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.targets
        
        # Process validation data
        test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.targets
        
        # Update metadata
        meta["n_class"] = 10
    else:
        raise ValueError("Unknown dataset.")
    
    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if reduced == "small" or reduced is True:
        train_input = train_input.narrow(0, 0, 2000)
        train_target = train_target.narrow(0, 0, 2000)
        test_input = test_input.narrow(0, 0, 500)
        test_target = test_target.narrow(0, 0, 500)
    
    elif reduced == "tiny":
        train_input = train_input.narrow(0, 0, 400)
        train_target = train_target.narrow(0, 0, 400)
        test_input = test_input.narrow(0, 0, 100)
        test_target = test_target.narrow(0, 0, 100)
    
    elif isinstance(reduced, float) and reduced > 0 and reduced < 1.0:
        n_tr = int(reduced * train_input.shape[0])
        n_te = int(reduced * test_input.shape[0])
        
        train_input = train_input.narrow(0, 0, n_tr)
        train_target = train_target.narrow(0, 0, n_tr)
        test_input = test_input.narrow(0, 0, n_te)
        test_target = test_target.narrow(0, 0, n_te)
        
    print("Dataset sizes:\n\t- Train: {}\n\t- Validation {}".format(tuple(train_input.shape), tuple(test_input.shape)))

    if normalize == "channel-wise":
        dims = [i for i in range(test.dim()) if i != 1]
        mu = train_input.mean(dim=dims, keepdim=True)
        sig = train_input.std(dim=dims, keepdim=True)
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)
        
    elif normalize == "image-wise":
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    elif normalize == "sample-wise":
        dims = [i for i in range(test.dim()) if i != 0]
        mu = train_input.mean(dim=dims, keepdim=True)
        sig = train_input.std(dim=dims, keepdim=True)
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)
        
    # Update metadata
    meta["in_dimension"] = train_input.shape[1:]
    
    return train_input.to(device), train_target.to(device), test_input.to(device), test_target.to(device), meta

class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset wrapper."""
    def __init__(self, inputs, targets):
        """Constructor.
        
        Arguments:
            - inputs: A tensor contaiing the inputs aligned in the 0th dimension.
            - targets: A tensor contaiing the targets aligned in the 0th dimension.
        """
        super(CustomDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.len = inputs.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    
def split_dataset_randomly(dataset, sizes):
    """Split a dataset into subsets of a given relative size.
    
    Arguments:
        - dataset: The dataset to split.
        - sizes: The relative sizes of the subsets.
    Return:
        - dataset_list: The list of the subsets.
    """
    
    if sum(sizes) > 1.0:
        raise ValueError("The sum of the fractions should add up to at most 1.")
    
    sizes_abs = [int(sz * dataset.len) for sz in sizes]
    sizes_abs.append(dataset.len - sum(sizes_abs))
    dataset_list = torch.utils.data.random_split(dataset, sizes_abs)

    return dataset_list[:-1]

def split_dataset(n_clients, train_ds, val_ds, alpha, sizes=None):
    """Create a matrix whose columns sum to (at most) the class counts and
    whose rows sum to exactly to the specified sizes. The rows are distributed following 
    (approximatively) a Dirichlet distribution of concentration alpha (times the prior obtained using
    the class counts.
    
    Arguments:
        - n_clients: The number of client (# of rows).
        - train_ds: Train dataset that must be splited.
        - val_ds: Validation dataset that must be splitted.
        - alpha: The concentration parameter of the Dirichlet distriutions (-> infinity for uniform)
        - sizes: Fraction of the dataset that must be given to each user (None for uniform sizes)
        
    Return:
        - train_ds_list: The list of the train datasets.
        - val_ds_list: The list of the validation datasets.
    """
    
    # Argument processing and checking
    if isinstance(sizes, list):
        if n_clients != len(sizes):
            raise ValueError("'n_client' must be the same as the length of 'sizes' when is is given.")
        if sum(sizes) > 1.0 or min(sizes) < 0.0 or max(sizes) > 1.0:
            raise ValueError("'sizes' must be an array of non-neative numbers summing to at most 1.")
    elif sizes is None:
        sizes = [1.0/n_clients for _ in range(n_clients)]
    else:
        raise TypeError("'sizes' must be an array or None.")
    
    if isinstance(alpha, str):
        if alpha == "uniform":
            alpha = 1e15
        elif alpha == "disjoint":
            alpha = 1e-15
        else:
            raise ValueError("'alpha' must be either 'uniform', 'disjoint' or a positive number.")
    elif isinstance(alpha, float) or isinstance(alpha, int):
        if alpha <= 0:
            raise ValueError("'alpha' must be strictly positive.")
    else:
        raise TypeError("'alpha' must be either 'uniform', 'disjoint' or a positive number.")
    
    # Extracting the count of each class
    counts_tr = train_ds.targets.unique(return_counts=True)[1].cpu().numpy()
    counts_val = val_ds.targets.unique(return_counts=True)[1].cpu().numpy()
    counts = counts_tr + counts_val
    n_class = len(counts)
    
    # Creation of the distribution matrices
    train_dist = np.zeros((n_clients, n_class), dtype=np.int32)
    val_dist = np.zeros((n_clients, n_class), dtype=np.int32)
    
    for client_id in range(n_clients):
        # Compute the sizes of the train and validation dataset for client client_id
        n_tr = int(sizes[client_id] * train_ds.len)
        n_val = int(sizes[client_id] * val_ds.len)
        
        # Sampling the row distribution following a dirichlet pdf.
        dist = scipy.stats.dirichlet.rvs(alpha * (counts + 1e-15) / counts.sum(), size=1)
        dist_tr = dist * n_tr
        dist_val = dist * n_val
        
        # Rounding so that the sum of the rows stays the same
        for col in range(n_class-1):
            #Train
            res_tr = dist_tr[:,col] - np.round(dist_tr[:,col])
            dist_tr[:,col] = np.round(dist_tr[:,col])
            dist_tr[:,col + 1] = dist_tr[:,col + 1] + res_tr
            
            # Validation
            res_val = dist_val[:,col] - np.round(dist_val[:,col])
            dist_val[:,col] = np.round(dist_val[:,col])
            dist_val[:,col + 1] = dist_val[:,col + 1] + res_val
        
        # Accounting for classes for which there are not enough samples.
        dist_tr = np.squeeze(np.minimum(dist_tr, counts_tr))
        dist_val = np.squeeze(np.minimum(dist_val, counts_val))
        
        # Casting into integers (adding 0.5 dure to rounding inaccuracies (i.e. sometimes int(10.) = 9)
        dist_tr = (dist_tr + 0.5).astype(int)
        dist_val = (dist_val + 0.5).astype(int)
        
        counts_tr = counts_tr - dist_tr
        counts_val = counts_val - dist_val
        
        # Completing the distribution for the train dataset
        res_tr = n_tr - dist_tr.sum()
        
        if res_tr > 0:
            # Forward pass in array
            for c in range(n_class):
                fill = min(res_tr, counts_tr[c])
                counts_tr[c] -= fill
                dist_tr[c] += fill
                res_tr -= fill
            
            # Backward pass to ensure success
            for c in reversed(range(n_class)):
                fill = min(res_tr, counts_tr[c])
                counts_tr[c] -= fill
                dist_tr[c] += fill
                res_tr -= fill
        
        # Completing the distribution for the validation dataset
        res_val = n_val - dist_val.sum()
        if res_val > 0:
            # Forward pass in array
            for c in range(n_class):
                fill = min(res_val, counts_val[c])
                counts_val[c] -= fill
                dist_val[c] += fill
                res_val -= fill
            
            # Backward pass to ensure success
            for c in reversed(range(n_class)):
                fill = min(res_val, counts_val[c])
                counts_val[c] -= fill
                dist_val[c] += fill
                res_val -= fill
        
        train_dist[client_id, :] = np.array(dist_tr)
        val_dist[client_id, :] = np.array(dist_val)
        
    # Initialize datasets
    train_x_list = [[] for i in range(n_clients)]
    train_y_list = [[] for i in range(n_clients)]
    val_x_list = [[] for i in range(n_clients)]
    val_y_list = [[] for i in range(n_clients)]

    train_ds_list = []
    val_ds_list = []

    #Spliting the datasets
    for c in range(n_class):
        index_class_tr = np.random.permutation(np.squeeze(np.where(train_ds.targets.cpu()==c)))
        index_class_val = np.random.permutation(np.squeeze(np.where(val_ds.targets.cpu()==c)))

        for client_id in range(n_clients):
            if train_dist[client_id, c] > 0:
                train_x_list[client_id].append(train_ds.inputs[index_class_tr[:train_dist[client_id, c]]])
                train_y_list[client_id].append(train_ds.targets[index_class_tr[:train_dist[client_id, c]]])
                index_class_tr = index_class_tr[train_dist[client_id, c]:]
            if val_dist[client_id, c] > 0:
                val_x_list[client_id].append(val_ds.inputs[index_class_val[:val_dist[client_id, c]]])
                val_y_list[client_id].append(val_ds.targets[index_class_val[:val_dist[client_id, c]]])
                index_class_val = index_class_val[val_dist[client_id, c]:]

    for client_id in range(n_clients):
        train_ds_list.append(CustomDataset(torch.cat(train_x_list[client_id], dim=0),
                                              torch.cat(train_y_list[client_id], dim=0)))
        val_ds_list.append(CustomDataset(torch.cat(val_x_list[client_id], dim=0),
                                            torch.cat(val_y_list[client_id], dim=0)))

    return train_ds_list, val_ds_list

def visualize_class_dist(ds_list, n_class, title=None, savepath=None):
    """Plot the class distribution across the clients.
    
    Arguments:
        - ds_list: List of datasets.
        - n_class: Number of class.
    """
    
    # Build label distribution table
    n_clients = len(ds_list)
    class_dist = np.zeros((n_clients, n_class), dtype=np.int32)
    for client_id, client_ds in enumerate(ds_list):
        values, counts = client_ds.targets.cpu().unique(return_counts=True)
        class_dist[client_id, values] = counts
    
    class_cum = np.cumsum(class_dist, axis=1)
    
    #Create name and colors
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, n_class))
    labels = ["Client {}".format(i) for i in range(n_clients)]
    
    # Plot horizontal barplot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, class_cum[:, -1].max())
    ax.set_ylim(n_clients-0.5, -0.5)
    starts = np.zeros(n_clients)
    for class_id in range(n_class):
        ax.barh(labels, class_dist[:, class_id], left=starts, 
                color=category_colors[class_id], label=class_id)
        starts = class_cum[:, class_id]
    
    if n_class < 17:
        ax.legend(range(n_class), bbox_to_anchor=(1.02, 1), loc='upper left')
    
    if title is not None:
        ax.set_title(title)
    # Save the figure at given location
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
        
def ds_to_dl(datasets, batch_size=None, shuffle=True):
    """Create a (list of) torch dataloader(s) given a (list of) dataset(s).
    
    Arguments:
        - datasets: A torch dataset (or a list of torch datasets).
        - batch_size: The batch size to use in the dataloader.
        - shuffle: Wheter to shuffle the dataset after each epoch.
    Return:
        - dl: A torch dataloader (or a list of torch dataloaders).
    
    """

    if isinstance(datasets, list):
        if batch_size is None:
            dl = [torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=shuffle) for ds in datasets]
        else:
            dl = [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle) for ds in datasets]
    else:
        if batch_size is None:
            dl = torch.utils.data.DataLoader(datasets, batch_size=len(datasets), shuffle=shuffle)
        else:
            dl = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)
    return dl

def evaluate_model(model, data_loader, criterion, n_class):
    """
    Compute loss and different performance metric of a single model using a data_loader.
    Returns a dictionary.
    
    Arguments:
        - model: Torch model.
        - data_loader: Torch data loader.
        - criterion: Loss function (not evaluated if None).
        - n_class: Number of class for classification. Treated as regression if None.
    Return:
        - perf: A dictionary containing the different performance metrics.
    """
    # Initialize performance dictionary
    perf = {}
    
    # Inference
    pred, targets = infer(model, data_loader, form="torch") 
    
    # Compute and store different performance metrics
    perf["loss"] = criterion(pred, targets).numpy()
    
    if n_class is not None and n_class > 0:
        cm = confusion_matrix(targets, pred.argmax(dim=1), labels=range(n_class))
        perf["confusion matrix"] = cm
        perf["accuracy"] = np.trace(cm, axis1=0, axis2=1) / np.sum(cm, axis=(0,1))
        
    elif n_class is None:
        raise NotImplementedError
    else:
        raise ValueError("Number of class must be None (regression) or a positive integer.")
    return perf
    
    
class PerfTracker():
    """Track train and validation performace of a given model during training."""
    def __init__(self, model, dl_dict, criterion, n_class, export_dir=None, ID="N/A"):
        """Constructor.
        
        Arguments:
            - model: Torch model.
            - dl_tr: Torch dataloader (train data).
            - dl_val: Torch dalatoader (validation data).
            - criterion: Loss function.
            - n_class: Number of class for classification. Treated as regression if None.
            - export_dir: Export directory.
            - ID: ID of the performance tracker (useful when the are plotted against eachother).
        """
        #  Assigning the attributes
        self.model = model
        self.dl_dict = dl_dict
        self.criterion = criterion
        self.n_class = n_class
        self.directory = export_dir
        self.index = [0]
        self.ID = ID
        
        self.perf_histories = {}
        for key, dl in self.dl_dict.items():
            perf = evaluate_model(self.model, dl, self.criterion, self.n_class)
            self.perf_histories[key] = {metric : np.expand_dims(value, 0) for metric, value in perf.items()}
        
        # Creating the directory for exports
        if self.directory is not None:
            os.makedirs(self.directory, exist_ok=True)


    def new_eval(self, index=None):
        """Function to call in each epoch.
        
        Arguments:
            - index: Index of the new entry in the training history.
        Return:
            - current_perf: Various performance of the model at thise epoch.
        """
    
        # Compute performance and add it to the performance histories
        current_perf = {}
        for key, dl in self.dl_dict.items():
            perf = evaluate_model(self.model, dl, self.criterion, self.n_class)
            current_perf[key] = perf
            for metric, value in perf.items():
                self.perf_histories[key][metric] = np.concatenate((self.perf_histories[key][metric], np.expand_dims(value, 0)), axis=0)
        
        # Update index
        if index is None:
            index = self.index[-1] + 1
        self.index.append(index)
            
        return current_perf

    def plot_training_history(self, metric="loss", logscale=False, savepath=None):
        """Plot the training history.
        
        Arguments:
            - metrics: metrics to plot.
            - logscale: Bollean, wheather to set y-axis to log scale.
            - savepath: filepath (and filename) where the plot must be stored. Not stored if None is given.
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_title("Training History")
        
        if metric == "confusion matrix":
            raise NotImplementedError
        
        for key, perf_dict in self.perf_histories.items():
            ax.plot(self.index, perf_dict[metric], label=key)
            ax.legend(loc="best")
            ax.grid(True, which="both")
            ax.set_ylabel(metric)
        
        # Set y-axis to logscale
        if logscale:
            ax.set_yscale('log')
            
        # Save the figure at given location
        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
            
    def plot_confusion_matrix(self, index=-1, savepath=None):
        """Plot a heatmap of the confusion matrices (on the train and validation data).
        
        Arguments:
            - index: Index at which to cumpute the correlation matrix (usually epoch/round).
            - savepath: filepath (and filename) where the plot must be stored. Not stored if None is given.
        """
        # Figure creation
        n_datasets = len(self.perf_histories)
        fig, axs = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))
        fig.suptitle("Confusion Matrix ({})".format(self.ID))
        
        # Plot the heatmap 
        for i, (key, perf_dict) in enumerate(self.perf_histories.items()):
            #axs[0].imshow(self.perf_history_tr["confusion matrix"][index], cmap="Blues")
            #axs[1].imshow(self.perf_history_val["confusion matrix"][index])
            sns.heatmap(perf_dict["confusion matrix"][index], cmap="Blues", 
                        annot=True, ax=axs[i], cbar=False, fmt='d', annot_kws={"fontsize":"small"})

            # Annotation
            axs[i].set_title("{} Dataset".format(key))
            axs[i].set_ylabel("True label")
            axs[i].set_xlabel("Predicted label")

            # Show ax frame for clarity
            [spine.set_visible(True) for spine in axs[i].spines.values()]
        
        # Save the figure at given location
        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
            
def infer(model, data_loader, form="numpy", normalize=False, classify=False):
    """Function to streamline the inference of a data sample through the given model.
    
    Arguments:
        - model: Torch model
        - data_loader: Torch dataloader
        - form: Type of output. Either 'torch' or 'numpy'.
    Return:
        - List of predictions and targets.
    """ 
    # Inference
    model.eval()
    with torch.no_grad():
        predictions = model(data_loader.dataset.inputs).cpu()
        targets = data_loader.dataset.targets.squeeze().cpu()
    model.train()
    if normalize:
        predictions = F.softmax(predictions, dim=1)
    
    if classify:
        predictions = predictions.argmax(dim=1)
    
    if form == "numpy":
        return predictions.numpy(), targets.numpy()
    elif form == "torch":
        return predictions, targets
    
    
def plot_global_training_history(perf_trackers, metric, title=None, logscale=False, savepath=None):
    """Plot the training history of multiple performance trackers.
    
    Arguments:
        - perf_trackers: A list of PerfTracker objects.
        - metric: The metric to plot.
        - logscale: Bollean, wheather to set y-axis to log scale.
        - savepath: The savepath (with filename) where to store the figure. Not stored if None is given.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title("Training History")
    
    for pt in perf_trackers:
        if metric == "confusion matrix":
            raise NotImplementedError
    
        for key, perf_dict in pt.perf_histories.items():
            ax.plot(pt.index, perf_dict[metric], label="{} ({})".format(pt.ID, key))
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True, which="both")
    ax.set_ylabel(metric)
    
    # Set y-axis to logscale
    if logscale:
        ax.set_yscale('log')
    
    if title is not None:
        ax.set_title(title)

    # Save the figure at given location
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')


def distance_correlation(X, Y):
    """Compute the distance correlation between the data X and Y.
    
    Arguments:
        - X: 1st data (samples along dim 0 and features along dim 1).
        - Y: 2nd data (samples along dim 0 and features along dim 1).
    
    Return:
        - dCorr: The distance correlation between X and Y.
    """
    n = X.shape[0]
    A = euclidean_distances(X, X)
    B = euclidean_distances(Y, Y)
    
    A = A - A.mean(axis=0, keepdims=True) - A.mean(axis=1, keepdims=True) + A.mean()
    B = B - B.mean(axis=0, keepdims=True) - B.mean(axis=1, keepdims=True) + B.mean()
    
    dVarX = np.multiply(A, A).sum() / (n**2)
    dVarY = np.multiply(B, B).sum() / (n**2)
    dCov = np.multiply(A, B).sum() / (n**2)
    
    dCorr = dCov / np.sqrt(dVarX * dVarY)

    return dCorr
 

    
def initialize_centroids(feat_dim, n_class):
    """Initialize the featrue centroids for each class."""
    return torch.rand((n_class, feat_dim)).mul_(2).sub_(1).float()
    

class DiscLoss(nn.Module):
    """Discriminator contrastive loss."""
    def __init__(self):
        super(DiscLoss, self).__init__()
        self.criterion = nn.BCELoss()
    
    def forward(self, prob, prob_global, labels, labels_global):
        scores = prob.unsqueeze(1) * prob_global.unsqueeze(0)
        scores = scores.sum(dim=2).view(prob.shape[0] * prob_global.shape[0]).float()
        targets = (labels.unsqueeze(1) == (labels_global.unsqueeze(0))).view(labels.shape[0] * labels_global.shape[0]).float()
        return self.criterion(scores, targets)

class FeatureTracker():
    """Track the feature of each clients for learning/analysis/visualization."""
    def __init__(self, client_models, train_ds_list, meta):
        # Models and datasets
        self.client_models = client_models
        self.train_ds_list = train_ds_list
        self.n_clients = len(client_models)
        self.meta = meta
        
        # Buffers to store features at each round
        self.buffers = [[] for _ in range(self.n_clients)]
        self.average_features = [[] for _ in range(self.n_clients)]
        self.global_features = []
        
        # Class counts
        self.class_counts = torch.zeros(self.n_clients, self.meta["n_class"]).to(int)
        for client_id, class_count in enumerate(self.class_counts):
            val, counts = self.train_ds_list[client_id].targets.unique(return_counts=True)
            self.class_counts[client_id, val.cpu()] = counts.cpu()

        # Initialization
        with torch.no_grad():
            for client_id, (model, tr_ds) in enumerate(zip(self.client_models, self.train_ds_list)):
                model.eval()
                features = model.features(tr_ds.inputs).cpu()
                average_features = torch.zeros(self.meta["n_class"], self.meta["feat_dim"])
                for c in range(self.meta["n_class"]):
                    average_features[c] = features[tr_ds.targets.cpu() == c].mean(dim=0)
                self.buffers[client_id].append(features)
                self.average_features[client_id].append(average_features)
            
            # Compute global averaged
            global_features = torch.stack([self.average_features[i][0] * self.class_counts[i].unsqueeze(1) 
                                           for i in range(self.n_clients)], dim=0).sum(dim=0).div(self.class_counts.sum(dim=0).unsqueeze(1))
            self.global_features.append(global_features)
        
    def new_round(self):
        """Compute the features for each data sample and aggregates the results for the next rounds."""
        # Compute features and averegage
        with torch.no_grad():
            for client_id, (model, tr_ds) in enumerate(zip(self.client_models, self.train_ds_list)):
                model.eval()
                features = model.features(tr_ds.inputs).cpu()
                average_features = torch.zeros(self.meta["n_class"], self.meta["feat_dim"])
                for c in range(self.meta["n_class"]):
                    average_features[c] = features[tr_ds.targets.cpu() == c].mean(dim=0)
                self.buffers[client_id].append(features)
                self.average_features[client_id].append(average_features)
            
            # Compute global averaged
            global_features = torch.stack([self.average_features[i][0] * self.class_counts[i].unsqueeze(1) 
                                           for i in range(self.n_clients)], dim=0).sum(dim=0).div(self.class_counts.sum(dim=0).unsqueeze(1))
            self.global_features.append(global_features)
    
    def get_global_features(self, r=-1):
        """Return the global aggregated feature at the given round."""
        return self.global_features[r]

    
    def plot_class_distance(self, class1, class2):
        """Plot the evolution of the distances between classes"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        distances = [[torch.linalg.norm(feat[class1] - feat[class2]) for feat in self.average_features[i]] for i in range(self.n_clients)]
        
        print(len(distances))
        for i, dist in enumerate(distances):
            ax.plot(dist, label="Client {}".format(i))
        
        ax.legend()
        

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
