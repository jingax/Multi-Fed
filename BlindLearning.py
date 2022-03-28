#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Standard modules
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
import matplotlib.pyplot as plt
from datetime import datetime

#Custom modules
import helpers as hlp
import models as mdl

# Check avaibale device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Used device: {}.".format(DEVICE))


# In[3]:


SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)


# In[4]:


#Dataset
DATASET = "MNIST" 
MODEL = "fc" # 'fc', 'resnet18'
REDUCED = True
FLATTEN = True if MODEL == "fc" else False

#Collaborative learning
N_CLIENTS = 4
SIZES = None # None for uniform sizes or array of length N_CLIENTS
ALPHA = "uniform" #'uniform', 'disjoint' or postive.
TOPOLOGY = [[0, 1, 0, 0], #Tij = 1 means i uses Xj and Yj for the KD
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]]

#Learning
BATCH_SIZE = 16
BATCH_SIZE_KD = 16
ROUNDS = 5
EPOCHS_PER_ROUND = 1
EPOCHS_PER_ROUND_KD = 1
RANDOM_SAMPLES = 1000
CRITERION = nn.CrossEntropyLoss()
CRITERION_KD = nn.MSELoss() #loss for knowledge diffusion

#Directories
DATE = datetime.now()
EXPORT_DIR = "./saves/" + DATE.strftime("%d-%m-%Y/%H-%M-%S")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Store parameters
with open(EXPORT_DIR + "/metadata.txt", 'w') as f:
    f.write("Parameter of the experiment conducted the {} at {}.\n\n".format(DATE.strftime("%d/%m/%Y"),
                                                                         DATE.strftime("%H:%m:%Y")))
    f.write("Model architecture:       {}\n".format(MODEL))
    f.write("Dataset:                  {}\n".format(DATASET))
    f.write("Number of clients:        {}\n".format(N_CLIENTS))
    f.write("Dataset sizes:            {}\n".format("uniform" if SIZES is None else SIZES))
    f.write("Concentration (alpha):    {}\n".format(ALPHA))
    f.write("Topology:                 {}\n".format(TOPOLOGY))
    f.write("Batch size:               {}\n".format(BATCH_SIZE))
    f.write("Batch size (KD):          {}\n".format(BATCH_SIZE_KD))
    f.write("Rounds:                   {}\n".format(ROUNDS))
    f.write("Epoch per round:          {}\n".format(EPOCHS_PER_ROUND))
    f.write("Epoch per round (KD):     {}\n".format(EPOCHS_PER_ROUND_KD))
    f.write("Number of random samples: {}\n".format(RANDOM_SAMPLES))


# In[5]:


# Load dataset
train_input, train_target, val_input, val_target, meta = hlp.load_data(dataset=DATASET,
                                                                       reduced=REDUCED, 
                                                                       flatten=FLATTEN,
                                                                       device=DEVICE)

#Create custom torch datasets
train_ds = hlp.ImageDataset(train_input, train_target)
val_ds = hlp.ImageDataset(val_input, val_target)

#Split dataset
#train_ds_list = hlp.split_dataset_randomly(train_ds, SIZES)
#val_ds_list = hlp.split_dataset_randomly(val_ds, SIZES)
train_ds_list, val_ds_list = hlp.split_dataset(N_CLIENTS, train_ds, val_ds, ALPHA, SIZES)

#Create dataloader
train_dl_list = hlp.ds_to_dl(train_ds_list, batch_size=BATCH_SIZE)
val_dl_list = hlp.ds_to_dl(val_ds_list)


#Visualize partition
hlp.visualize_class_dist(train_ds_list, meta["n_class"], title="Class distribution")
#hlp.visualize_class_dist(val_ds_list, N_CLASS, title="Validation datasets")


# In[6]:


# Model initialization
if MODEL == "fc":
    client_models = [mdl.FC_Net(meta["in_dimension"][0], meta["n_class"]).to(DEVICE) for _ in range(N_CLIENTS)]
    client_models_kd = [mdl.FC_Net(meta["in_dimension"][0], meta["n_class"]).to(DEVICE) for _ in range(N_CLIENTS)]

elif MODEL == "resnet18":
    client_models = [mdl.ResNet18(meta["in_dimension"][0], meta["n_class"]).to(DEVICE) for _ in range(N_CLIENTS)]
    client_models_kd = [mdl.ResNet18(meta["in_dimension"][0], meta["n_class"]).to(DEVICE) for _ in range(N_CLIENTS)]

# Performance tracker
perf_trackers = [hlp.PerfTracker(client_models[i], train_dl_list[i], val_dl_list[i], 
                                 CRITERION, meta["n_class"], 
                                 EXPORT_DIR + "/client_{}".format(i)) for i in range(N_CLIENTS)]
perf_trackers_kd = [hlp.PerfTracker(client_models_kd[i], train_dl_list[i], val_dl_list[i], 
                                    CRITERION, meta["n_class"], 
                                    EXPORT_DIR + "/client_{}_KD".format(i)) for i in range(N_CLIENTS)]


# In[7]:


print("Data on cuda:", train_ds.targets.is_cuda)
print("Model on cuda:", next(client_models[0].parameters()).is_cuda)


# In[8]:


#Each client updates its model locally on its own dataset (Standard)
for client_id in range(N_CLIENTS):
    #Setting up the local training
    model = client_models[client_id]
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    #Local update
    for e in range(ROUNDS*EPOCHS_PER_ROUND):
        for features, target in train_dl_list[client_id]:
            optimizer.zero_grad()
            output = model(features)
            loss = CRITERION(output, target)
            loss.backward()
            optimizer.step()
        
        #Tracking performance
        perf_trackers[client_id].new_eval()
        print("\rClient {}: epoch {}/{} done.".format(client_id, e+1, ROUNDS*EPOCHS_PER_ROUND), end="  ")
    print(" ")
    
# Visualization of training history
user = 0
perf_trackers[user].plot_training_history(metric="accuracy")
perf_trackers[user].plot_training_history(metric="loss")


# In[9]:


#Training phase
for r in range(ROUNDS):

    #Each client updates its model locally on its own dataset
    for client_id in range(N_CLIENTS):
        
        #Setting up the local training
        model = client_models_kd[client_id]
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        
        #Local update
        for e in range(EPOCHS_PER_ROUND):
            for features, target in train_dl_list[client_id]:
                optimizer.zero_grad()
                output = model(features)
                loss = CRITERION(output, target)
                loss.backward()
                optimizer.step()
    
    # Blind learning (creation of input/output pairs)
    features_rand = torch.normal(0, 1, size=(N_CLIENTS, RANDOM_SAMPLES, *meta["in_dimension"])).to(DEVICE)
    output_rand = torch.empty(N_CLIENTS, RANDOM_SAMPLES, meta["n_class"]).to(DEVICE)
    
    for client_id in range(N_CLIENTS):
        model = client_models_kd[client_id]
        model.eval()
        with torch.no_grad():
            output_rand[client_id] = model(features_rand[client_id])
    
    ds_kd_list = [hlp.ImageDataset(features_rand[i], output_rand[i]) for i in range(N_CLIENTS)]
    dl_kd_list = [hlp.ds_to_dl(ds, batch_size=BATCH_SIZE_KD) for ds in ds_kd_list]
    
    # Blind learning (knowledge diffusion)
    for client_id in range(N_CLIENTS):
        model = client_models_kd[client_id]
        optimizer = torch.optim.Adam(model.parameters())
        #model.train() #Should this be? maybe the dropout hinders the effect of KD
        
        for peer_id, isConsidered in enumerate(TOPOLOGY[client_id]):
            if isConsidered:
                for e in range(EPOCHS_PER_ROUND_KD):
                    for x_kd, y_kd in dl_kd_list[peer_id]:
                        optimizer.zero_grad()
                        y_kd_pers = model(x_kd)
                        loss = CRITERION_KD(y_kd_pers, y_kd)
                        loss.backward()
                        optimizer.step()
        #Tracking performance
        perf_trackers_kd[client_id].new_eval()

    print("\rRound {}/{} done.".format(r+1, ROUNDS), end="  ")
# Visualization of training history
user = 0
perf_trackers_kd[user].plot_training_history(metric="accuracy")
perf_trackers_kd[user].plot_training_history(metric="loss")


# In[10]:


for i in range(N_CLIENTS):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    pt = perf_trackers[i]
    pt_kd = perf_trackers_kd[i]
    axs[0].plot(pt.index, pt.perf_history_tr["loss"], label="Train")
    axs[0].plot(pt.index, pt.perf_history_val["loss"], label="Validation")
    axs[0].plot(pt_kd.index, pt_kd.perf_history_tr["loss"], label="Train (KD)")
    axs[0].plot(pt_kd.index, pt_kd.perf_history_val["loss"],label="Validation (KD)")
    axs[0].set_title("Client {} (loss)".format(i))
    axs[0].legend()
    
    
    axs[1].plot(pt.index, np.trace(pt.perf_history_tr["confusion matrix"], axis1=1, axis2=2) / np.sum(pt.perf_history_tr["confusion matrix"], axis=(1,2)), label="Train")
    axs[1].plot(pt.index, np.trace(pt.perf_history_val["confusion matrix"], axis1=1, axis2=2) / np.sum(pt.perf_history_val["confusion matrix"], axis=(1,2)), label="Validation")
    axs[1].plot(pt_kd.index, np.trace(pt_kd.perf_history_tr["confusion matrix"], axis1=1, axis2=2) / np.sum(pt_kd.perf_history_tr["confusion matrix"], axis=(1,2)), label="Train (KD)")
    axs[1].plot(pt_kd.index, np.trace(pt_kd.perf_history_val["confusion matrix"], axis1=1, axis2=2) / np.sum(pt_kd.perf_history_val["confusion matrix"], axis=(1,2)), label="Validation (KD)")
    axs[1].set_title("Client {} (accuracy)".format(i))
    axs[1].legend()
    
    fig.savefig(EXPORT_DIR + "/train_history.png", bbox_inches='tight')


# In[ ]:




