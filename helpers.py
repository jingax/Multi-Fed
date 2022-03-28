# --------------------------------------------------------------
# This file contains the code used to run AI models on several
# database.
#
# 2022 Frédéric Berdoz, Zurich, Switzerland
# --------------------------------------------------------------

# Miscellaneous
from datetime import datetime
import copy
import os

# Data processing
import pandas as pd
import numpy as np
import scipy

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
from sklearn.metrics import confusion_matrix


def load_data(dataset="MNIST", data_dir="./data", reduced=False, one_hot_labels=False, normalize=True, flatten=False, device="cpu"):
    """Load the specified dataset."""
    
    # Initialize meta data:
    meta = {"n_class": None,
            "in_dimension": None}
    
    if dataset == "CIFAR10":
        # Load
        print("** Using CIFAR **")
        print("Load train data...")
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train = True, download = True)
        print("Load validation data...")
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train = False, download = True)

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
    else:
        raise ValueError("Unknown dataset.")
    
    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if reduced:
        train_input = train_input.narrow(0, 0, 500)
        train_target = train_target.narrow(0, 0, 500)
        test_input = test_input.narrow(0, 0, 100)
        test_target = test_target.narrow(0, 0, 100)

    print("Dataset sizes:\n\t- Train: {}\n\t- Valitation {}".format(tuple(train_input.shape), 
                                                                    tuple(test_input.shape)))

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    # Update metadata
    meta["in_dimension"] = train_input.shape[1:]
    
    return train_input.to(device), train_target.to(device), test_input.to(device), test_target.to(device), meta

class ImageDataset(torch.utils.data.Dataset):
    """Custom dataset wrapper."""
    def __init__(self, features, targets):
        super(ImageDataset, self).__init__()
        self.features = features
        self.targets = targets
        self.len = features.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.features[index], self.targets[index]
    
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
        index_class_tr = np.random.permutation(np.squeeze(np.where(train_ds.targets==c)))
        index_class_val = np.random.permutation(np.squeeze(np.where(val_ds.targets==c)))

        for client_id in range(n_clients):
            if train_dist[client_id, c] > 0:
                train_x_list[client_id].append(train_ds.features[index_class_tr[:train_dist[client_id, c]]])
                train_y_list[client_id].append(train_ds.targets[index_class_tr[:train_dist[client_id, c]]])
                index_class_tr = index_class_tr[train_dist[client_id, c]:]
            if val_dist[client_id, c] > 0:
                val_x_list[client_id].append(val_ds.features[index_class_val[:val_dist[client_id, c]]])
                val_y_list[client_id].append(val_ds.targets[index_class_val[:val_dist[client_id, c]]])
                index_class_val = index_class_val[val_dist[client_id, c]:]

    for client_id in range(n_clients):
        train_ds_list.append(ImageDataset(torch.cat(train_x_list[client_id], dim=0),
                                              torch.cat(train_y_list[client_id], dim=0)))
        val_ds_list.append(ImageDataset(torch.cat(val_x_list[client_id], dim=0),
                                            torch.cat(val_y_list[client_id], dim=0)))

    return train_ds_list, val_ds_list

def visualize_class_dist(ds_list, n_class, title=None):
    """Plot the class distribution across the clients.
    
    Arguments:
        - ds_list: List of datasets.
        - n_class: Number of class.
    """
    
    # Build label distribution table
    n_clients = len(ds_list)
    class_dist = np.zeros((n_clients, n_class), dtype=np.int32)
    for client_id, client_ds in enumerate(ds_list):
        values, counts = client_ds.targets.unique(return_counts=True)
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
    ax.legend(range(n_class), bbox_to_anchor=(1.02, 1), loc='upper left')
    
    if title is not None:
        ax.set_title(title)
        
def ds_to_dl(datasets, batch_size=None, shuffle=True):
    """Create a (list of) torch dataloader(s) given a (list of) dataset(s)"""

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

def evaluate_model(model, data_loader, criterion, n_class=None):
    """
    Compute loss and different performance metric of a single model using a data_loader.
    Returns a dictionary.
    
    Arguments:
        - model: Torch model.
        - data_loader: Torch data loader.
        - criterion: Loss function (not evaluated if None).
        - n_class: Number of class for classification. Treated as regression if None.
    Return:
        - loss: The loss of the predictions compared to the targets.
        - cm: the confusion matrix
    """
    # Initialize performance dictionary
    perf = {}
    
    # Inference
    pred, targets = infer(model, data_loader, form="torch") 
    
    # Compute and store different performance metrics
    perf["loss"] = criterion(pred, targets).numpy()
    
    if n_class is not None and n_class > 0:
        perf["confusion matrix"] = confusion_matrix(pred.argmax(dim=1), 
                                                    targets, labels=range(n_class))
    elif n_class is None:
        raise NotImplementedError
    else:
        raise ValueError("Number of class must be None (regression) or a positive integer.")
    return perf
    
    
class PerfTracker():
    """Track train and validation performace of a given model during training."""
    def __init__(self, model, dl_tr, dl_val, criterion, n_class, export_dir):
        """Constructor.
        
        Arguments:
            - model: Torch model.
            - dl_tr: Torch dataloader (train data).
            - dl_val: Torch dalatoader (validation data).
            - criterion: Loss function.
            - n_class: Number of class for classification. Treated as regression if None.
            - export_dir: Directory to save model and plots.
        """
        #  Assigning the attributes
        self.model = model
        self.best_model = copy.deepcopy(model)
        self.dl_tr = dl_tr
        self.dl_val = dl_val
        self.criterion = criterion
        self.n_class = n_class
        self.directory = export_dir
        self.index = [0]
        self.perf_history_tr = {metric : np.expand_dims(value, 0) 
                                for metric, value in evaluate_model(model, dl_tr, criterion, n_class).items()}
        self.perf_history_val = {metric : np.expand_dims(value, 0) 
                                 for metric, value in evaluate_model(model, dl_val, criterion, n_class).items()}
        
        # Creating the directory for exports
        os.makedirs(self.directory, exist_ok=True)
        
        # Initializing the minimum loss to store the best model
        self.loss_min = float("inf")
        
        
    def new_eval(self, index=None, checkpoints=True):
        """Function to call in each epoch.
        
        Arguments:
            - index: Index of the new entry in the training history.
        Return:
            - perf_tr, perf_val: performance of the model at this epoch.
        """
        
        # Compute performance
        perf_tr = evaluate_model(self.model, self.dl_tr, self.criterion, self.n_class)
        perf_val = evaluate_model(self.model, self.dl_val, self.criterion, self.n_class)
        
        # Save model if validation performance is the best so far
        if perf_val["loss"] < self.loss_min and checkpoints:
            torch.save(self.model.state_dict(), self.directory + "/model.pt")
            self.loss_min = perf_val["loss"]
            self.best_model = copy.deepcopy(self.model)
        
        # Append performance metric
        for metric, value in perf_tr.items():
            self.perf_history_tr[metric] = np.concatenate((self.perf_history_tr[metric], 
                                                           np.expand_dims(value, 0)), axis=0)
        
        for metric, value in perf_val.items():
            self.perf_history_val[metric] = np.concatenate((self.perf_history_val[metric], 
                                                            np.expand_dims(value, 0)), axis=0)
        
        if index is None:
            index = self.index[-1] + 1
        self.index.append(index)
            
        return perf_tr, perf_val

    def plot_training_history(self, metric="loss", save=False):
        """PLot the training history.
        
        Arguments:
            - metrics: metrics to plot.
            - save: Boolean. Whether to save the plot.
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_title("Training History")
        
        if metric == "loss":
            metric_tr = self.perf_history_tr["loss"]
            metric_val = self.perf_history_val["loss"]
        elif metric == "accuracy":
            cm_tr = self.perf_history_tr["confusion matrix"]
            cm_val = self.perf_history_val["confusion matrix"]
            metric_tr = np.trace(cm_tr, axis1=1, axis2=2) / np.sum(cm_tr, axis=(1,2))
            metric_val = np.trace(cm_val, axis1=1, axis2=2) / np.sum(cm_val, axis=(1,2))
        
        ax.plot(self.index, metric_tr, label="Train")
        ax.plot(self.index, metric_val, label="Validation")
        ax.legend(loc="lower right")
        ax.grid()
        ax.set_ylabel(metric)

        # Save the figure at given location
        if save:
            fig.savefig(self.directory + "/train_history.png", bbox_inches='tight')


def infer(model, data_loader, form="numpy"):
    """Function to streamline the inference of a data sample through the given model.
    
    Arguments:
        - model: Torch model
        - data_loader: Torch dataloader
        - form: Type of output. Either 'torch' or 'numpy'.
    Return:
        - List of predictions and targets.
    """
    
    # Initiallize output lists
    predictions_list = []
    targets_list = []
    
    # Inference
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            # Infer
            output = model(data)
            
            # Store batch results
            targets_list.append(target)
            predictions_list.append(output)    
    
    # Processing outputs
    predictions = torch.cat(predictions_list)
    targets = torch.cat(targets_list)
    model.train()
    
    if form == "numpy":
        return predictions.squeeze().cpu().numpy(), targets.squeeze().cpu().numpy()
    elif form == "torch":
        return predictions.squeeze().cpu(), targets.squeeze().cpu()
    
    
def foo():
    print("ok")