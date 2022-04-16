# -----------------------------------------------------------------------------
# This file contains the archived code that was developped
# during my master thesis at MIT.
#
# 2022 Frédéric Berdoz, Boston, USA
# -----------------------------------------------------------------------------

# Miscellaneous
import os
from datetime import datetime
import time
import copy

# Data processing
import pandas as pd
import numpy as np
import scipy
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


class PerfTracker():
    """Track train and validation performace of a given model during training."""
    def __init__(self, model, dl_tr, dl_val, criterion, n_class, export_dir, ID="N/A"):
        """Constructor.
        
        Arguments:
            - model: Torch model.
            - dl_tr: Torch dataloader (train data).
            - dl_val: Torch dalatoader (validation data).
            - criterion: Loss function.
            - n_class: Number of class for classification. Treated as regression if None.
            - export_dir: Directory to save model and plots.
            - ID: ID of the performance tracker (useful when the are plotted against eachother).
        """
        #  Assigning the attributes
        self.model = model
        self.dl_tr = dl_tr
        self.dl_val = dl_val
        self.criterion = criterion
        self.n_class = n_class
        self.directory = export_dir
        self.index = [0]
        self.ID = ID
        
        perf_tr = evaluate_model(model, dl_tr, criterion, n_class)
        perf_val = evaluate_model(model, dl_val, criterion, n_class)
        
        self.perf_history_tr = {metric : np.expand_dims(value, 0) for metric, value in perf_tr.items()}
        self.perf_history_val = {metric : np.expand_dims(value, 0) for metric, value in perf_val.items()}
        
        # Creating the directory for exports
        if self.directory is not None:
            os.makedirs(self.directory, exist_ok=True)
        
        # Initializing the minimum loss to store the best model
        self.loss_min = float("inf")
        
        # Initialize checkpoint
        self.best_model = copy.deepcopy(model)

    def new_eval(self, index=None):
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
        if perf_val["loss"] < self.loss_min and self.directory is not None:
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

    def plot_training_history(self, metric="loss", savepath=None):
        """Plot the training history.
        
        Arguments:
            - metrics: metrics to plot.
            - savepath: filepath (and filename) where the plot must be stored. Not stored if None is given.
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_title("Training History")
        
        if metric == "confusion matrix":
            raise NotImplementedError
        else:
            metric_tr = self.perf_history_tr[metric]
            metric_val = self.perf_history_val[metric]
            
        ax.plot(self.index, metric_tr, label="Train")
        ax.plot(self.index, metric_val, label="Validation")
        ax.legend(loc="lower right")
        ax.grid()
        ax.set_ylabel(metric)

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
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 5))

        # Plot the heatmap 
        #axs[0].imshow(self.perf_history_tr["confusion matrix"][index], cmap="Blues")
        #axs[1].imshow(self.perf_history_val["confusion matrix"][index])
        sns.heatmap(self.perf_history_tr["confusion matrix"][index], cmap="Blues", 
                    annot=True, ax=axs[0], cbar=False, fmt='d', annot_kws={"fontsize":"small"})
        sns.heatmap(self.perf_history_val["confusion matrix"][index], cmap="Blues", 
                    annot=True, ax=axs[1], cbar=False, fmt='d', annot_kws={"fontsize":"small"})
        
        # Annotation
        fig.suptitle("Confusion Matrix ({})".format(self.ID))
        axs[0].set_title("Train Dataset")
        axs[1].set_title("Validation Dataset")
        axs[0].set_ylabel("True label")
        axs[1].set_ylabel("True label")
        axs[0].set_xlabel("Predicted label")
        axs[1].set_xlabel("Predicted label")
        
        # Show ax frame for clarity
        [spine.set_visible(True) for spine in axs[0].spines.values()]
        [spine.set_visible(True) for spine in axs[1].spines.values()]
        
        # Save the figure at given location
        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
            
def plot_global_training_history(perf_trackers, metric, savepath=None):
    """Plot the training history of multiple performance trackers.
    
    Arguments:
        - perf_trackers: A list of PerfTracker objects.
        - metric: The metric to plot.
        - savepath: The savepath (with filename) where to store the figure. Not stored if None is given.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_title("Training History")
    
    for pt in perf_trackers:
        if metric == "confusion matrix":
            raise NotImplementedError
        else:
            metric_tr = pt.perf_history_tr[metric]
            metric_val = pt.perf_history_val[metric]

        ax.plot(pt.index, metric_tr, label="{} (Train)".format(pt.ID))
        ax.plot(pt.index, metric_val, label="{} (Validation)".format(pt.ID))
    
    ax.legend(loc="best")
    ax.grid()
    ax.set_ylabel(metric)

    # Save the figure at given location
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')