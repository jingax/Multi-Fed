# Miscellaneous
import os
import random

# Data processing
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# AI
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import datasets

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#custom
import data_maker as DM

def set_seed(seed=0):
    """Set the same seed for all the RNG.
    
    Argument:
        - seed: the specified seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_data(dataset="MNIST", data_dir="./data", reduced=False, normalize="image-wise", flatten=False, device="cpu"):
    """Load the specified dataset.
    
    Arguments:
        - dataset: Name of the dataset to load.
        - data_dir: Directory where to store (or load if already stored) the dataset.
        - reduced: Boolean/'small'/'tiny'/float between 0 and 1. Reduce the dataset size.
        - normalize: Whether to normalize the data ('image-wise' or 'channel-wise').
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
        train_set = datasets.CIFAR10(data_dir + '/cifar10/', train=True,download=True)
        print("Load validation data...")
        test_set = datasets.CIFAR10(data_dir + '/cifar10/', train=False,download=True)

        # Process train data
        train_input = torch.from_numpy(train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float() / 255
        train_target = torch.tensor(train_set.targets, dtype=torch.int64)
        
        # Process validation data
        test_input = torch.from_numpy(test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float() / 255
        test_target = torch.tensor(test_set.targets, dtype=torch.int64)
        
        # Update metadata
        meta["n_class"] = 10
        meta["class_names"] = ["airplane", "automobile", "bird", "cat", 
                               "deer", "dog", "frog", "horse", "ship", "truck"]
    

    elif dataset == "COVID":
        print("** Using COVID **")
        print("Load train data...")
        train_input, train_target = DM.get_covid()
        train_input = train_input.float()
        print("Load validation data...")
        test_input, test_target = DM.get_covid()
        test_input = test_input.float()
        # Process train data
        # train_input = train_set.data.view(-1, 1, 28, 28).float()
        # train_target = train_set.targets
        
        # # Process validation data
        # test_input = test_set.data.view(-1, 1, 28, 28).float()
        # test_target = test_set.targets
        
        # Update metadata
        meta["n_class"] = 4
        meta["class_names"] = ["0", "1", "2", "3"]
    

    elif dataset == "CIFAR100":
        # Load
        print("** Using CIFAR **")
        print("Load train data...")
        train_set = datasets.CIFAR100(data_dir + '/cifar100/', train=True,download=True)
        print("Load validation data...")
        test_set = datasets.CIFAR100(data_dir + '/cifar100/', train=False,download=True)

        # Process train data
        train_input = torch.from_numpy(train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float() / 255
        train_target = torch.tensor(train_set.targets, dtype=torch.int64)
        
        # Process validation data
        test_input = torch.from_numpy(test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float() / 255
        test_target = torch.tensor(test_set.targets, dtype=torch.int64)
        
        # Update metadata
        meta["n_class"] = 100
        meta["class_names"] = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
                               "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly",
                               "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee",
                               "clock", "cloud", "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur",
                               "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
                               "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
                               "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
                               "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck",
                               "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon",
                               "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
                               "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
                               "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
                               "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
                               "woman", "worm"]
        
    elif dataset == "MNIST":
        print("** Using MNIST **")
        print("Load train data...")
        train_set = datasets.MNIST(data_dir, train=True, download=True)
        print("Load validation data...")
        test_set = datasets.MNIST(data_dir, train=False, download=True)

        # Process train data
        train_input = train_set.data.view(-1, 1, 28, 28).float()
        train_target = train_set.targets
        
        # Process validation data
        test_input = test_set.data.view(-1, 1, 28, 28).float()
        test_target = test_set.targets
        
        # Update metadata
        meta["n_class"] = 10
        meta["class_names"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
    elif dataset == "FMNIST":
        print("** Using FMNIST **")
        print("Load train data...")
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True)
        print("Load validation data...")
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True)

        # Process train data
        train_input = train_set.data.view(-1, 1, 28, 28).float()
        train_target = train_set.targets
        
        # Process validation data
        test_input = test_set.data.view(-1, 1, 28, 28).float()
        test_target = test_set.targets
        
        # Update metadata
        meta["n_class"] = 10
        meta["class_names"] = ["T-shirt/top", "Trouser", "Pullover", "Dress", 
                               "Coat", "Sandal", "Shirt", "Sneaker", "Bag",  "Ankle boot"]
    elif dataset == "EMNIST":
        print("** Using EMNIST **")
        print("Load train data...")
        train_set = datasets.EMNIST(data_dir, split="balanced", train=True, download=True)
        print("Load validation data...")
        test_set = datasets.EMNIST(data_dir, split="balanced", train=False, download=True)

        # Process train data
        train_input = train_set.data.view(-1, 1, 28, 28).permute(0, 1, 3, 2).float()
        train_target = train_set.targets
        
        # Process validation data
        test_input = test_set.data.view(-1, 1, 28, 28).permute(0, 1, 3, 2).float()
        test_target = test_set.targets
        
        # Update metadata
        meta["n_class"] = 1
        meta["class_names"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", 
                               "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", 
                               "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "d", 
                               "e", "f", "g", "h", "n", "q", "r", "t"]
    else:
        raise ValueError("Unknown dataset.")
    
    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if reduced == "small" or reduced is True:
        train_input = train_input.narrow(0, 0, 2000)
        train_target = train_target.narrow(0, 0, 2000)
        test_input = test_input.narrow(0, 0, 1000)
        test_target = test_target.narrow(0, 0, 1000)
    
    elif reduced == "tiny":
        train_input = train_input.narrow(0, 0, 400)
        train_target = train_target.narrow(0, 0, 400)
        test_input = test_input.narrow(0, 0, 200)
        test_target = test_target.narrow(0, 0, 200)
    
    elif isinstance(reduced, float) and reduced > 0 and reduced < 1.0:
        n_tr = int(reduced * train_input.shape[0])
        train_input = train_input.narrow(0, 0, n_tr)
        train_target = train_target.narrow(0, 0, n_tr)
        
        #n_te = int(reduced * test_input.shape[0])
        #test_input = test_input.narrow(0, 0, n_te)
        #test_target = test_target.narrow(0, 0, n_te)
    
    # Print dataset information
    memory_train = (train_input.element_size() * train_input.nelement() + train_target.element_size() * train_target.nelement())/ 1e6
    memory_val = (test_input.element_size() * test_input.nelement() + test_target.element_size() * test_target.nelement())/ 1e6
    print("Dataset sizes:\n\t- Train: {} ({} MB)\n\t- Validation {} ({} MB)".format(tuple(train_input.shape), memory_train, tuple(test_input.shape), memory_val))
    
    # Normalization
    if normalize == "channel-wise":
        # Normalize each channels independently
        dims = [i for i in range(train_input.dim()) if i != 1]
        mu = train_input.mean(dim=dims, keepdim=True)
        sig = train_input.std(dim=dims, keepdim=True)   
    
    elif normalize == "image-wise":
        # Normalize all channels
        mu = train_input.mean()
        sig = train_input.std()

    else:
        mu = 0
        sig = 1
    
    # Subtract mean and divide by std
    train_input.sub_(mu).div_(sig)
    test_input.sub_(mu).div_(sig)
    meta["mu"] = mu
    meta["sig"] = sig
    
    # Update metadata
    meta["in_dimension"] = train_input.shape[1:]
    
    return train_input.to(device), train_target.to(device), test_input.to(device), test_target.to(device), meta

def visualize_data(data, meta, index=[0], targets=None):
    """Visualize images in a row.
    
    Arguments:
        - data: input data (N, C, H, W).
        - meta: meta data.
        - index: List of images to plot (along first dimension in data.
        - targets: List of targets (to properly annotate images (optional)
    """
    fig, axs  = plt.subplots(1, len(index), figsize=(len(index) * 2, 2), squeeze=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, idx in enumerate(index):
        axs[0,i].imshow(data[idx].mul(meta["sig"]).add(meta["mu"]).permute(1, 2, 0))
        axs[0,i].axis("off")
        if targets is not None:
            y = targets[idx]
            axs[0,i].set_title(meta["class_names"][y])

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
        - title: Plot title.
        - savepath: Path to save the figure (not saved if None).
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
        idx = 0
        predictions = []
        targets = []
        for x, y in data_loader:
            predictions.append(model(x).cpu())
            targets.append(y.cpu())
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
    model.train()
    
    if normalize:
        predictions = F.softmax(predictions, dim=1)
    
    if classify:
        predictions = predictions.argmax(dim=1)
    
    if form == "numpy":
        return predictions.numpy(), targets.numpy()
    elif form == "torch":
        return predictions, targets
    
def evaluate_model(model, data_loader, n_class, criterion=None):
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
    if criterion is not None:
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
            perf = evaluate_model(self.model, dl, self.n_class, self.criterion)
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
            perf = evaluate_model(self.model, dl, self.n_class, self.criterion)
            current_perf[key] = perf
            for metric, value in perf.items():
                self.perf_histories[key][metric] = np.concatenate((self.perf_histories[key][metric], np.expand_dims(value, 0)), axis=0)
        
        # Update index
        if index is None:
            index = self.index[-1] + 1
        self.index.append(index)
            
        return current_perf

    def plot_training_history(self, metric="loss", logscale=False, title="Training History", savepath=None):
        """Plot the training history.
        
        Arguments:
            - metrics: metrics to plot.
            - logscale: Bollean, wheather to set y-axis to log scale.
            - title: Title to display.
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
        
        # Parameters
        if logscale:
            ax.set_yscale('log')
        
        if title is not None:
            ax.set_title(title)
        
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        
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
        annot = True if self.n_class <= 10 else False
        fig, axs = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))
        fig.suptitle("Confusion Matrix ({})".format(self.ID))
        
        # Plot the heatmap 
        for i, (key, perf_dict) in enumerate(self.perf_histories.items()):
            #axs[0].imshow(self.perf_history_tr["confusion matrix"][index], cmap="Blues")
            #axs[1].imshow(self.perf_history_val["confusion matrix"][index])
            sns.heatmap(perf_dict["confusion matrix"][index], cmap="Blues", 
                        annot=annot, ax=axs[i], cbar=False, fmt='d', annot_kws={"fontsize":"small"})

            # Annotation
            axs[i].set_title("{} Dataset".format(key))
            axs[i].set_ylabel("True label")
            axs[i].set_xlabel("Predicted label")

            # Show ax frame for clarity
            [spine.set_visible(True) for spine in axs[i].spines.values()]
        
        # Save the figure at given location
        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
    
    
def plot_global_training_history(perf_trackers, metric, which=None, shaded=True, title=None, logscale=False, savepath=None):
    """Plot the training history of multiple performance trackers.
    
    Arguments:
        - perf_trackers: A list of PerfTracker objects.
        - metric: The metric to plot.
        - which: Data to evaluate (list or all if None)
        - logscale: Bollean, wheather to set y-axis to log scale.
        - savepath: The savepath (with filename) where to store the figure. Not stored if None is given.
    """
    # Argument processing
    if which is None:
        which = list(perf_trackers[0].perf_histories.keys())
    elif isinstance(which, str):
        which = [which]
    
    # Figure creation
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))    
    
    if shaded:
        x = perf_trackers[0].index
        for ds in which:
            data = np.empty((len(perf_trackers), len(perf_trackers[0].index)))
            for i, pt in enumerate(perf_trackers):
                data[i] = pt.perf_histories[ds][metric]
            mu = data.mean(0)
            sigma = data.std(0)
            ax.plot(x, mu, label="{}".format(ds))
            ax.fill_between(x, mu-sigma, mu+sigma, alpha=0.5)
        
        if metric == "accuracy":
            pos = "lower right"
        elif metric == "loss":
            pos = "upper right"
        ax.legend(loc=pos)
    else:
        for pt in perf_trackers:
            for ds in which:
                perf_dict = pt.perf_histories[ds]
                ax.plot(pt.index, perf_dict[metric], label="{} ({})".format(pt.ID, ds))
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    # Ax properties
    ax.grid(True, which="both")
    ax.set_ylabel(metric)
    ax.set_xlabel("round")
    # Parameters
    if logscale:
        ax.set_yscale('log')
    
    if metric == "accuracy":
        ax.set_ylim(0.2, 1)
    
    if title is not None:
        ax.set_title(title)

    # Save the figure at given location
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')

    
class OutputTracker():
    """Track the output of each clients for learning/analysis/visualization."""
    def __init__(self, client_models, dl_list, dim, meta):
        """Constructor.
        
        Arguments:
            - client_models: List of client models.
            - dl_list: List of dataloader.
            - dim: Dimension of the the data to track.
            - meta: Metadata about the dataset.
        """
        # Models and datasets
        self.client_models = client_models
        self.dl_list = dl_list
        self.n_clients = len(client_models)
        self.dim = dim
        self.meta = meta
        
        # Create indices for each client
        self.sizes = [len(dl.dataset) for dl in dl_list]
        self.idx = [sum(self.sizes[:i]) + torch.arange(self.sizes[i]) for i in range(self.n_clients)]
        
        # Buffers to store outputs at each round
        self.buffers_outputs = [] #buffer[client][round]
        self.buffers_targets = [] #buffer[client][round]
        
        # Class counts
        self.class_counts = torch.zeros(self.n_clients, self.meta["n_class"]).long()
        for client_id, class_count in enumerate(self.class_counts):
            val, counts = self.dl_list[client_id].dataset.targets.unique(return_counts=True)
            self.class_counts[client_id, val.cpu()] = counts.cpu()
        
        # Initialization
        self.new_round()
        
    def new_round(self):
        """Compute the outputs for each data sample."""
        # Compute average
        buffer_outputs = torch.empty(sum(self.sizes), self.dim)
        buffer_targets = torch.empty(sum(self.sizes)).long()
        for client_id, (model, dl) in enumerate(zip(self.client_models, self.dl_list)):
            outputs, targets = infer(model, dl, form="torch")
            buffer_outputs[self.idx[client_id]] = outputs
            buffer_targets[self.idx[client_id]] = targets
            
        # Buffering
        self.buffers_outputs.append(buffer_outputs)
        self.buffers_targets.append(buffer_targets)

    
    def get_global_outputs(self, r=-1, n_avg=None, client_id=None):
        """Return the global aggregated output at the given round.
        
        Arguments:
            - r: Round.
            - n_avg: Number of samples to consider for the average (all if 0 or None).
            - client_id: Consider only the data of client_id if given.
        Return:
            - Global averaged outputs.
        """
        # Argument processing
        if n_avg is not None and n_avg < 0:
            raise ValueError("'n_avg' must be a positive integer.")
        
        # Extract data
        global_outputs = torch.empty(self.meta["n_class"], self.dim)
        
        if client_id is None:
            outputs = self.buffers_outputs[r]
            targets = self.buffers_targets[r]
        else:
            if client_id == "random":
                client_id = random.randint(0, self.n_clients-1)
            outputs = self.buffers_outputs[r][self.idx[client_id]]
            targets = self.buffers_targets[r][self.idx[client_id]]
        
        for c in range(self.meta["n_class"]):
            if torch.any(targets == c).item():
                data = outputs[targets == c]
                # Radom subsampling for the averaging
                if n_avg:
                    n = min(n_avg, int(data.shape[0]))
                    idx = torch.randperm(data.shape[0])[:n]
                    data = data[idx]
                global_outputs[c] = data.mean(dim=0)
        
        return global_outputs


    def plot_tSNE(self, r_list=[-1], p=30, single_client=None, savepath=None, title=None, fig_axs=None):
        """Plot the t-SNE dimension reduction of the averaged output.
        
        Arguments:
            -r_list: List of rounds where to plot the tSNE.
            -p: Complexitiy of the tSNE.
            -single_client: Client for which to plot the tSNE (all clients if None).
            -savepath: Path where to save the figure (not saved if None).
            -title: Title of the plot.
            -fig_axs: (Figure, axs) tuple where to plot the tSNE. Create new tuple if None given.
        """
        if fig_axs is None:
            fig, axs = plt.subplots(1, len(r_list), figsize=(3*len(r_list), 3))
            plt.subplots_adjust(wspace=0)
        else:
            fig, axs = fig_axs
        cmap_list = ["Oranges", "Blues"]
        norm = matplotlib.colors.NoNorm()
        for i, r in enumerate(r_list):
            embedder = TSNE(n_components=2, init='random', perplexity=p)
            axs[i].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
            if fig_axs is None:
                axs[i].set_title("Round {}".format(r))
            if single_client is None:
                feat_emb = embedder.fit_transform(self.buffers_outputs[r])
                for client_id in range(self.n_clients):
                    colors = 0.2 + 0.8 * self.buffers_targets[r][self.idx[client_id]] / (self.meta["n_class"]-1)
                    axs[i].scatter(feat_emb[self.idx[client_id],0], feat_emb[self.idx[client_id],1], 
                                   c=colors, norm=norm, s=1, marker="o", lw=1, cmap=cmap_list[client_id])
        
            else:
                feat_emb = embedder.fit_transform(self.buffers_outputs[r][self.idx[single_client]])
                colors = 0.2 + 0.8 * self.buffers_targets[r][self.idx[single_client]] / (self.meta["n_class"]-1)
                axs[i].scatter(feat_emb[:,0], feat_emb[:,1], 
                                   c=colors, norm=norm, s=1, marker="o", lw=1, cmap=cmap_list[single_client])
        if title is not None:
            axs[0].set_ylabel(title)
        # Save the figure at given location
        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
            
def model_size(model):
    """Compute the memory size (MB) of the given model (parameters + buffers).
    
    Arguments:
        - model: Model to analyse.
    """
    n_params = sum([param.nelement() for param in model.parameters()])
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    print("Number of parameter: {}".format(n_params))
    print("Model size: {} MB".format((mem_params + mem_bufs)/1e6))
    

def compare(pt, pt_baseline, metric="accuracy", r=-1, which=None):
    """Plot the performance improvement between baseline and new method.
    
    Arguments:
        - pt: Performance trackers for the new method.
        - pt_baseline: Performance trackers for the baseline method.
        - metric: Metric to analyse.
        - r: Round to analyse.
        - which: Dataset to evaluate.
    """
    # Arguement processing
    n_clients = len(pt)
    
    if which is None:
        which = list(pt[0].perf_histories.keys())
    elif isinstance(which, str):
        which = [which]
    
    
    print("Average {} improvement:".format(metric))
    diff_dict = {}
    for ds in which:
        diff = np.array([pt[i].perf_histories[ds][metric][r] - pt_baseline[i].perf_histories[ds][metric][r] for i in range(n_clients)])
        diff_dict[ds] = diff
        print("\t{}: {:.3f} (+- {:.3f})".format(ds, diff.mean(), diff.std()))
    
    return diff_dict
    

    
    
    
    
    
    
    
    
    
    
    
