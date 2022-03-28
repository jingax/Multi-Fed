# --------------------------------------------------------------
# This file contains the code used to run AI models on several
# database.
#
# 2021 Frédéric Berdoz, Zurich, Switzerland
# --------------------------------------------------------------

# Miscellaneous
from datetime import datetime
import copy
import os

# Data processing
import pandas as pd
import numpy as np

# AI
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class TabularDataset(torch.utils.data.Dataset):
    """Create a torch dataset using a dataframe as an input."""
    def __init__(self, df, categorical_columns=[], continuous_columns="all", 
                 output_column=None):
        """Constructor.
        
        Arguments:
            - df: Dataframe containing the raw database.
            - categorical_columns: Columns in the dataframe that are categorical.
            - continuous_cpolumns: Columns in the dataframe that are continuousl.
            - output column: Column in the dataframe that is considered as the label.
        """
        super().__init__()
        
        # Dataset length
        self.len = df.shape[0]
        
        # Store categorical and continuous data separately
        if isinstance(categorical_columns, list):
            self.categorical_columns = categorical_columns
        else:
            raise TypeError("Wrong argument type for 'categorical_columns'. Must be a list of string.")
        
        if continuous_columns == "all":
            self.continous_columns = [col for col in df.columns 
                                      if col not in (self.categorical_columns + [output_column])]
        elif isinstance(continuous_columns, list):
            self.continous_columns = continuous_columns
        else:
            raise TypeError("Wrong argument type for 'categorical_columns'. Must be 'all' or list of strings.")
        
        # Continuous columns
        if self.continous_columns:
            self.cont_X = df[self.continous_columns].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.len, 1)) #for compatibility, will not be used
        
        # Categorical columns
        if self.categorical_columns:
            self.cat_X = df[self.categorical_columns].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.len, 1)) #for compatibility, will not be used
        
        # Output columns
        if output_column is not None:
            self.has_label = True
            self.label = df[output_column].astype(np.float32).values.reshape(-1, 1)
        else:
            self.has_label = False

    def __len__(self):
        """Return dataset length"""
        return self.len

    def __getitem__(self, index):
        """Function to iterate through dataset.
        
        Arguments:
            - index: Index of the sample to return.
        Return:
            - label and feature (ior just feature if no label)."""
        if self.has_label:
              return self.label[index], (self.cont_X[index], self.cat_X[index])
        else:
              return (self.cont_X[index], self.cat_X[index])
        
class FC_Net(nn.Module):
    """ FC-type net, strongly inspired by 
    https://www.kaggle.com/chriszou/titanic-with-pytorch-nn-solution
    """
    def __init__(self, n_cont, lin_layer_sizes, 
                 output_size, emb_dims=[], dropout_rate=0.2, 
                 hl_activation="relu", ll_activation=True):
        """Constructor.
        
        Arguments:
            - n_cont: Number of continuous features.
            - lin_layer_sizes: List of sizes for the hidden linear layers.
            - output_size: Size of the output layer.
            - emb_dims: List of tuple containing the embedding dimentions (x: nb of categories, y: dimension of embedding).
            - dropout_rate: Dropout for all hidden layers.
            - hl_activation: Hidden layer activation (relu, sigmoid, tanh).
            - ll_activation: Boolean. True if an activation function must be used on the last layer.
        """
        
        super().__init__()
        # Create one embedding "space" for each categroical feature. 
        # (x: nb of categories, y: dimension of embedding)
        self.ll_activation = ll_activation
        self.hl_activation = hl_activation
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims]) if emb_dims else None
        
        # Total dimensions of embeddings, continuous data and output
        self.n_embs = sum([y for x, y in emb_dims])
        self.n_cont = n_cont
        self.output_size = output_size

        # Linear Layers
        first_lin_layer = nn.Linear(self.n_embs + self.n_cont, lin_layer_sizes[0])
        self.lin_layers = nn.ModuleList([first_lin_layer] + 
                                        [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) 
                                         for i in range(len(lin_layer_sizes) - 1)])
        # Linear layers initialization
        for lin_layer in self.lin_layers:
             nn.init.kaiming_normal_(lin_layer.weight.data)

        # Dropout "layers"
        self.dropouts = [nn.Dropout(dropout_rate) for _ in lin_layer_sizes]
        
        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)
    
    def forward(self, data):
        """Forward pass function.
        
        Argument:
            - data: Batch tha must be forwarded through the net.
        Return:
            - x: The output of the model.
        """
        
        # Unpack continuous and categorical data
        cont_data, cat_data = data
        
        # Embedding categorical data and concatenating with continuous data
        if self.n_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)

            if self.n_cont != 0:
                x = torch.cat([x, cont_data], 1) 
        else:
            x = cont_data
        
        # Hidden layers + dropout
        for lin_layer, dropout in zip(self.lin_layers, self.dropouts):
            if self.hl_activation == "relu":
                x = torch.relu(lin_layer(x))
            elif self.hl_activation == "sigmoid":
                x = torch.sigmoid(lin_layer(x))
            elif self.hl_activation == "tanh":
                x = torch.tanh(lin_layer(x))
            x = dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Activation function
        if self.ll_activation:
            if self.output_size == 1:
                x = torch.sigmoid(x)
            else:
                x = F.log_softmax(x, dim=1)

        return x

def evaluate_model(model, data_loader, criterion, mode, threshold=0.5):
    """
    Compute loss and different performance metric of a single model using a data_loader.
    Returns a dictionary.
    
    Arguments:
        - model: Torch model.
        - data_loader: Torch data loader.
        - criterion: Loss function (not evaluated if None).
        - mode: Mode (binary, multiclass, regression)
        - threshold: Threshold in [0, 1] for binary classification. Ignore if mode is not 'binary'.
    Return:
        - perf: A dictionary of the performance metrics.
    """
    # Evaluation mode
    model.eval()
    
    # Initialization of metrics
    n = len(data_loader.dataset)
    loss = 0
    
    if mode == "binary":     
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        pos = 0
        pred_pos = 0
    if mode == "multiclass":
        raise NotImplementedError

    
    # Evalutation
    with torch.no_grad():
        for target, data in data_loader:
            
            # Inference
            output = model(data)
            
            # Loss
            loss += criterion(output, target).item()  # sum up batch loss
           
            # Autodetection of output type
            if mode == "binary":
                target = torch.where(target > threshold, torch.tensor([1]), torch.tensor([0]))
                pred = torch.where(output > threshold, torch.tensor([1]), torch.tensor([0]))
                pos += target.sum().item()
                pred_pos += pred.sum().item()
                correct += pred.eq(target).sum().item()
                TP += pred[pred==1].eq(target[pred==1]).sum().item()
                TN += pred[pred==0].eq(target[pred==0]).sum().item()
                FP += pred[pred==1].ne(target[pred==1]).sum().item()
                FN += pred[pred==0].ne(target[pred==0]).sum().item()
            if mode == "multiclass":
                raise NotImplementedError

    
    # Loss normalitzation
    loss_norm = loss / n
    
    # Initialize performance dictionary
    perf = {}
    perf['loss_norm'] =  loss_norm
    
    if mode == "binary":
        # Accuracy
        acc = correct / n

        # Additional values
        neg = n - pos
        pred_neg = n - pred_pos

        # precision
        precision = TP/pred_pos if pred_pos != 0 else np.nan

        # recall or true positive rate (TPR)
        recall = TP/pos if pos != 0 else np.nan

        # f1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else np.nan

        # fall-out or false positive rate (FPR)
        FPR = FP / neg if neg != 0 else np.nan
        
        # Update performance dictionary
        perf['accuracy'] = acc
        perf['precision'] = precision
        perf['recall'] = recall
        perf['f1'] = f1
        perf['FPR'] = FPR
        
    if mode == "multiclass":
        raise NotImplementedError
    
    model.train()
    return perf

def infer(model, data_loader, mode, threshold=None):
    """Function to streamline the inference of a data sample through the given model.
    
    Arguments:
        - model: Torch model
        - data_loader: Torch dataloader
        - mode: Mode (binary, multiclass, regression)
        - threshold: Threshold in [0, 1] for binary classification. Ignore if mode is not 'binary'.
    Return:
        - List of predictions and targets.
    """
    
    # Initiallize output lists
    predictions_list = []
    targets_list = []
    
    # Inference
    model.eval()
    with torch.no_grad():
        for target, data in data_loader:
            # Infer
            output = model(data)
            
            # Store batch results
            targets_list.append(target)
            if mode == "regression":
                predictions_list.append(output)
            elif mode == "binary": 
                if threshold is not None:
                    predictions_list.append(torch.where(output > threshold, 
                                                    torch.ones_like(output), 
                                                    torch.zeros_like(output)))
                else:
                    predictions_list.append(output)
                    
            elif mode == "multiclass":
                predictions_list.append(torch.argmax(output, 1))        
    
    # Processing outputs
    predictions = torch.cat(predictions_list)
    targets = torch.cat(targets_list)
    model.train()
    return predictions.detach().squeeze().numpy(), targets.detach().squeeze().numpy()
    
class PerfTracker():
    """Track train and validation performace of a given model during training."""
    def __init__(self, model, dl_tr, dl_val, criterion, mode, export_dir, threshold=0.5):
        """Constructor.
        
        Arguments:
            - model: Torch model.
            - dl_tr: Torch dataloader (train data).
            - dl_val: Torch dalatoader (validation data).
            - criterion: Loss function.
            - mode: Inference mode (binary, multiclass, regression)
            - export_dir: Directory to save model and plots.
            - threshold: Threshold in [0, 1] for binary classification. Ignore if mode is not 'binary'.
        """
        #  Assigning the attributes
        self.model = model
        self.best_model = copy.deepcopy(model)
        self.dl_tr = dl_tr
        self.dl_val = dl_val
        self.criterion = criterion
        self.mode = mode
        self.directory = export_dir + "/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.threshold = threshold
        self.perf_history_tr = pd.DataFrame(columns=[m for m in evaluate_model(model, dl_tr, criterion,
                                                                            mode, threshold)])
        self.perf_history_val = pd.DataFrame(columns=[m for m in evaluate_model(model, dl_val, criterion,
                                                                            mode, threshold)])
        
        # Creating the directory for exports
        os.makedirs(self.directory, exist_ok=True)
        
        # Initializing the minimum loss to store the best model
        self.loss_min = float("inf")
        
        
    def new_epoch(self, epoch=None):
        """Function to call in each epoch.
        
        Arguments:
            - epoch: Index of the new entry in the training history.
        Return:
            - perf_tr, perf_val: performance of the model at this epoch.
        """
        
        # Compute performance
        perf_tr = evaluate_model(self.model, self.dl_tr, self.criterion, self.mode, self.threshold)
        perf_val = evaluate_model(self.model, self.dl_val, self.criterion, self.mode, self.threshold)
        
        # Save model if validation performance is the best so far
        if perf_val["loss_norm"] < self.loss_min:
            torch.save(self.model.state_dict(), self.directory + "/model.pt")
            self.loss_min = perf_val["loss_norm"]
            self.best_model = copy.deepcopy(self.model)
        
        # Create temporary df
        new_line_tr = pd.DataFrame(perf_tr, index=[epoch])
        new_line_val = pd.DataFrame(perf_val, index=[epoch])
        
        # Append temporary df
        self.perf_history_tr = self.perf_history_tr.append(new_line_tr, 
                                                           ignore_index=True if epoch is None else False)
        self.perf_history_val = self.perf_history_val.append(new_line_val, 
                                                             ignore_index=True if epoch is None else False)
        
        return perf_tr, perf_val

    def plot_training_history(self, metrics=None, save=False):
        """PLot the training history.
        
        Arguments:
            - metrics: metrics to plot (all if None is given).
            - save: Boolean. Whether to save the plot.
        """
        
        if metrics is None:
            metrics = self.perf_history_tr.columns
        
        nrows = len(metrics)
        fig, axs = plt.subplots(nrows, 1, figsize=(8, 4 * nrows), squeeze=False)
        axs[0,0].set_title("Training History")
        
        for idx, perf in enumerate(metrics):
            ax = axs[idx, 0]
            ax.plot(self.perf_history_tr.index, self.perf_history_tr[perf], label="Train")
            ax.plot(self.perf_history_val.index, self.perf_history_val[perf], label="Validation")
            ax.legend(loc="upper right")
            ax.grid()
            ax.set_ylabel(perf)
        
        # Save the figure at given location
        if save:
            fig.savefig(self.directory + "/train_history.png", bbox_inches='tight')

    
    def ROC(self, npoints=101, annotate=[5, 10, 20, 30, 50, 70, 90], best=True, save=False):
        """Create a ROC plot for a binary classification model.
        
        Arguements:
            - npoints: Number of threshold to evaluate.
            - annotate: Whcich threshol to annotate on the plot.
            - best: Boolean. Whether to use the best or the last model.
            - save: Boolean. WHether to save the plot.
        """
        # List of thresholds
        thresholds = np.linspace(0, 1, npoints)
        
        # Initialization
        TPR = np.empty_like(thresholds)
        FPR = np.empty_like(thresholds)
        
        # Iter through thresholds
        for i, t in enumerate(thresholds):
            print('\rBuilding the ROC plot ({}/{} evaluations)'.format(i+1, npoints), end='  ')
            if best:
                perf = evaluate_model(self.best_model, self.dl_val, self.criterion, self.mode, t)
            else:
                perf = evaluate_model(self.model, self.dl_val, self.criterion, self.mode, t)
            TPR[i] = perf['recall']
            FPR[i] = perf['FPR']

        fig, ax = plt.subplots(1 ,1, figsize=(5, 5))
        fig.suptitle('ROC Plot')
        ax.grid()
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.plot(thresholds, thresholds, 'r--')
        if annotate is not None:
            ax.plot(FPR[annotate], TPR[annotate], 'k.')
            for i in annotate:
                if i < 11:
                    ax.annotate('{:.2f}'.format(thresholds[i]), (FPR[i], TPR[i]), xytext=(5,-10), textcoords='offset points')
                else:
                    ax.annotate('{:.1f}'.format(thresholds[i]), (FPR[i], TPR[i]), xytext=(5,-10), textcoords='offset points')

        ax.plot(FPR, TPR)
        
        # Save the figure at given location
        if save:
            fig.savefig(self.directory + "/ROC.png", bbox_inches='tight')
    
    def visualize_predictions(self, threshold=None, best=True, save=False):
        """Visualize the predictions.
        
        Arguements:
            - threshold: Threshold in [0, 1] for binary classification.
            - best: Boolean. Whether to use the best or the last model.
            - save: Boolean. WHether to save the plot.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
        fig.subplots_adjust(wspace=0.1)
        
        # Prediction plots
        for i, dl in enumerate((self.dl_tr, self.dl_val)):
            ax = axs[i, 0]
            if best:
                preds, targets = infer(self.best_model, dl, mode="binary")
            else:
                preds, targets = infer(self.model, dl, mode="binary")
            heights = [p - t for (p,t) in zip(preds, targets)]
            colors = ["green"  if h > 0 else "red" for h in heights]
            ax.bar(np.arange(len(preds)), bottom=targets, height=heights, color=colors, alpha=0.7)
            ax.scatter(np.arange(len(preds)), preds, marker="o", 
                       color = ["red" if t == 1 else "green" for t in targets])
            ax.grid(True)
            ax.set_ylim(0,1)
            ax.set_title(["Train", "Validation"][i] + " Predictions")
            
            if threshold is not None:
                ax.axhline(threshold, color="black")
        
        # Histogram plots
        for i, dl in enumerate((self.dl_tr, self.dl_val)):
            ax = axs[i, 1]
            ax.set_ylim(0,1)
            ax.grid(True)
            ax.set_title("Histogram")
            if best:
                preds, targets = infer(self.best_model, dl, mode="binary")
            else:
                preds, targets = infer(self.model, dl, mode="binary")
            sns.distplot(preds[targets==0], vertical=True, kde=False, 
                         bins=20, ax=ax, color="green", 
                         hist_kws={"height": 0.05, "align": "mid", "range":(0,1)})
            sns.distplot(preds[targets==1], vertical=True, kde=False, 
                         bins=20, ax=ax, color="red",
                         hist_kws={"height": 0.05, "align": "mid", "range":(0,1)})
            
        # Save the figure at given location
        if save:
            fig.savefig(self.directory + "/predictions.png" , bbox_inches='tight')

class ECDDs(Dataset):
    """A torch dataset based on a processed ECD database."""
    def __init__(self, data_dict, fft_data_dict, features, labels):
        """Constructor.
        
        Arguments:
            - data_dict: Raw data dictionary from a ECDDatabase.
            - fft_data_dict: Fft data dictionary from a ECDDatabase.
            - features: List of feature to use.
            - labels: List of test to consider as label (taling the maximum value).
        """
        self.data_dict = data_dict
        self.fft_data_dict = fft_data_dict
        self.features = features
        self.labels = labels
        self.FM_list = list(data_dict.keys())
        
        # Transform a the dataset into a numpy array
        data_list = [np.stack([self.data_dict[FM][feature] for feature in self.features] +
                              [np.real(self.fft_data_dict[FM][feature]) for feature in self.features] +
                              [np.imag(self.fft_data_dict[FM][feature]) for feature in self.features]) for FM in self.FM_list]
        self.data = np.stack(data_list).astype(np.float32)
        targets_list = [np.where(np.max(np.concatenate([self.data_dict[FM][label] for label in self.labels], axis=0)).reshape(1) >= 1.24, 1, 0) for FM in self.FM_list]
        self.targets = np.stack(targets_list).astype(np.float32)
        
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        """Function to iterate through the dataset based on sample index.
        
        Argument:
            - idx: Index to return.
        Return:
            - The target and feature of the given idx."""
        return self.targets[idx], self.data[idx]
    
    def get_FM(self, FM):
        """Function to iterate through the dataset based on sample serial number.
        
        Argument:
            - FM: Serial number to return.
        Return:
            - The target and feature of the given FM."""
        return self.targets[self.FM_list.index(FM)], self.data[self.FM_list.index(FM)]

class Simple1DCNN(torch.nn.Module):
    """A neural network based on 1D convolutions and a final linear layer."""
    def __init__(self, in_channels, output_dim):
        """Constructor.
        
        Arguments:
            - in_channels: Number of input channels.
            - output_dim: Size of output.
        """
        super(Simple1DCNN, self).__init__()
        self.dropout = nn.Dropout(0.2)
        
        self.layer1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=32, stride=1, dilation=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxpool1 = nn.MaxPool1d(6)
        self.act1 = torch.nn.ReLU()
        
        self.layer2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16, stride=1, dilation=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.maxpool2 = nn.MaxPool1d(6)
        self.act2 = torch.nn.ReLU()
        
        self.layer3 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16, stride=1, dilation=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.maxpool3 = nn.MaxPool1d(4)
        self.act3 = torch.nn.ReLU()
        
        self.layer4 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=8, stride=1, dilation=1)
        self.bn4 = nn.BatchNorm1d(16)
        self.maxpool4 = nn.MaxPool1d(4)
        self.act4 = torch.nn.ReLU()
        
        self.layer5 = torch.nn.Conv1d(in_channels=16, out_channels=8, kernel_size=4, stride=1, dilation=1)
        self.bn5 = nn.BatchNorm1d(8)
        self.maxpool5 = nn.MaxPool1d(4)
        self.act5 = torch.nn.ReLU()
        
        self.fc = nn.Linear(88, 1)
        self.last_act = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass.
        
        Arguments:
            - x: Data to forward.
        Return:
            - y: output of the model."""
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.maxpool1(x)
        x = self.act1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.act2(x)
        
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.act3(x)
        
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.maxpool4(x)
        x = self.act4(x)
        
        x = self.layer5(x)
        x = self.bn5(x)
        x = self.maxpool5(x)
        x = self.act5(x)

        x = self.fc(x.view(x.size(0), -1))
        y = self.last_act(x)

        return y
