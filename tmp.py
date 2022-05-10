# -----------------------------------------------------------------------------
# This file contains the experiments code that was developped
# during my master thesis at MIT.
#
# 2022 Frédéric Berdoz, Boston, USA
# -----------------------------------------------------------------------------


import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import helpers as hlp
import models as mdl
import json


def get_args():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clients', type=int, default=2, help="")
    parser.add_argument('--task', type=str, default='MNIST', help="")
    parser.add_argument('--model', type=str, default='LeNet5', help="")
    parser.add_argument('--alpha', type=float, default=1e15, help="")
    parser.add_argument('--rounds', type=int, default=100,  help="")
    parser.add_argument('--batch_size', type=int, default=32, help="")
    parser.add_argument('--epoch_per_round', type=int, default=1, help="")
    parser.add_argument('--lr', type=float, default=1e-3,  help="")
    parser.add_argument('--optimizer', type=str, default="adam",  help="")
    parser.add_argument('--feature_dim', type=int, default=100, help="")
    parser.add_argument('--n_avg', type=int, default=None, help="")
    parser.add_argument('--lambda_kd', type=float,default=1.0, help="")
    parser.add_argument('--lambda_disc', type=float,default=1.0, help="")    
    parser.add_argument('--sizes',type=float, nargs='*', default=None, help="")
    parser.add_argument('--reduced', type=float, default=1.0, help="")
    parser.add_argument('--track_history', type=int, default=1, help="")
    parser.add_argument('--fed_avg', type=str, default=False,  help="")
    parser.add_argument('--export_dir', type=str, default="./saves", help="")
    parser.add_argument('--data_dir', type=str, default="./data", help="")
    parser.add_argument('--config_file', type=str, default=None, help="")
    parser.add_argument('--device', type=str, default=None, help="")
    parser.add_argument('--seed', type=int, default=0, help="")
    args = parser.parse_args()
    return args


def run(n_clients, dataset, model, alpha="uniform", rounds=100, 
        batch_size=32, epoch_per_round=1, lr=1e-3, optimizer="adam", feature_dim=100,
        n_avg=None, lambda_kd=1.0, lambda_disc=1.0, sizes=None, reduced=False, 
        track_history=1, fed_avg=False, export_dir=None, data_dir="./data",
        device=None, seed=0):
    
    """Run an experiment of PrivateKD.
    
    Arguments:
    
    Return:
        - perf_trackers: A list of all the performance trackers
        - feature_trackers: The feature trackers.
    """
    
    # Store parameters for easy reproductability
    if export_dir is not None:
        # Store local parameters
        config = locals()
        del config["export_dir"]
        del config["device"]
        
        # Create subdirectories and throw error if already existing (security)
        time_string = time.strftime("%d%h%y_%H:%M:%S")
        directory = os.path.join(export_dir, time_string)
        fig_directory = os.path.join(directory, "figures/")
        os.makedirs(directory, exist_ok=False)
        os.makedirs(fig_directory, exist_ok=False)
        
        # Create config file
        config = json.dumps(config, indent=4)
        with open(os.path.join(directory, 'config.json'), 'w') as outfile:
            outfile.write(config)
            
    # Chose device automatically (if not specified)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print("Device: {}".format(device))
    torch.cuda.empty_cache()
    
    # Reproductilility
    hlp.set_seed(seed)
    
    # Load data
    train_input, train_target, val_input, val_target, meta = hlp.load_data(dataset=dataset, reduced=reduced, device=device)
    
    #Create custom torch datasets
    train_ds = hlp.CustomDataset(train_input, train_target)
    val_ds = hlp.CustomDataset(val_input, val_target)
    
    #Split dataset
    train_ds_list, val_ds_list = hlp.split_dataset(n_clients, train_ds, val_ds, alpha, sizes)
    
    #Create dataloader
    train_dl_list = hlp.ds_to_dl(train_ds_list, batch_size=batch_size)
    val_dl_list = hlp.ds_to_dl(val_ds_list, batch_size=10*batch_size)
    global_val_dl = hlp.ds_to_dl(val_ds, batch_size=10*batch_size)
    global_train_dl = hlp.ds_to_dl(train_ds, batch_size=10*batch_size)
    
    #Visualize partition
    hlp.visualize_class_dist(train_ds_list, meta["n_class"], 
                             title="Class distribution (alpha = {})".format(alpha),
                             savepath=os.path.join(fig_directory, "class_dist.png") if fig_directory is not None else None)
    
    # Store dataset sizes
    meta["n_total"] = train_input.shape[0] 
    meta["n_local"] = [ds.inputs.shape[0] for ds in train_ds_list]
    
    # Define criterions
    criterion = nn.CrossEntropyLoss()
    criterion_kd = nn.MSELoss()
    criterion_disc = nn.BCELoss()
    
    # Model initialization
    if fed_avg:
        global_model = mdl.get_model(model, feature_dim, meta).to(device)
    client_models = [mdl.get_model(model, feature_dim, meta).to(device) for _ in range(n_clients)]
    
    # Performance tracker
    if fed_avg == "model":
        perf_trackers = [hlp.PerfTracker(global_model, 
                                         {"Train": train_dl_list[i], "Validation": val_dl_list[i]}, 
                                         criterion, meta["n_class"], ID="Client {}".format(i)) for i in range(n_clients)]
        perf_tracker_global = hlp.PerfTracker(global_model, 
                                         {"Train (global)": global_train_dl, "Validation (global)": global_val_dl}, 
                                         criterion, meta["n_class"], ID="Global")
        
        # Boradcast for same model initialization
        global_parameters = global_model.state_dict()
        for model in client_models:
                model.load_state_dict(global_parameters)
    else:
        perf_trackers = [hlp.PerfTracker(client_models[i], 
                                         {"Train": train_dl_list[i], "Validation": val_dl_list[i], "Train (global)": global_train_dl, "Validation (global)": global_val_dl}, 
                                         criterion, meta["n_class"], ID="Client {}".format(i)) for i in range(n_clients)]
        
        if fed_avg == "classifier":
            # Boradcast for same classifier initialization
            global_parameters = global_model.classifier.state_dict()
            for model in client_models:
                    model.classifier.load_state_dict(global_parameters)
            
    
    # Optimizers
    if optimizer in ["adam", "Adam", "ADAM"]:
        optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in client_models]
    elif optimizer in ["sgd", "Sgd", "SGD"]:
        optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in client_models] 
    
    # Feature tracker and discriminator
    feat_tracker = hlp.FeatureTracker(client_models, train_dl_list, feature_dim, meta)
    discriminators = [mdl.Discriminator("prob_product", client_models[i].classifier) for i in range(n_clients)]
    
    #Each client updates its model locally on its own dataset (Standard)
    for r in range(rounds):
        t0 = time.time()
        
        for client_id in range(n_clients):
            #Setting up the local training
            disc = discriminators[client_id]
            model = client_models[client_id]
            model.train()
            opt = optimizers[client_id]
            
            #Local update
            for e in range(epoch_per_round):
                for inputs, targets in train_dl_list[client_id]:
                    # Reset gradient
                    opt.zero_grad()
                    
                    # Local representation
                    features = model.features(inputs)
                    logits = model.classifier(features)
                    
                    if lambda_kd > 0 or lambda_disc > 0:
                        # Compute estimated probabilities
                        features_global = feat_tracker.get_global_features(n_avg=n_avg).to(device)
                        targets_global = torch.arange(meta["n_class"]).to(device)
                    
                    # Optimization step
                    loss = criterion(logits, targets)
                    if lambda_kd > 0:
                        loss += lambda_kd * criterion_kd(features, features_global[targets])
                    if lambda_disc > 0:
                        scores, disc_targets = disc(features, features_global, targets, targets_global)
                        loss += lambda_disc * criterion_disc(scores, disc_targets)
                    
                    # Optimization step
                    loss.backward()
                    opt.step()
            
        # Use FedAvg on the classifer
        if fed_avg == "classifier":
            # Aggregation (weighted average)
            global_parameters = global_model.classifier.state_dict()
            for k in global_parameters.keys():
                global_parameters[k] = torch.stack([(meta["n_local"][i] / meta["n_total"]) * client_models[i].classifier.state_dict()[k] 
                                                    for i in range(n_clients)], 0).sum(0)

            # Broadcast classifiers
            for model in client_models:
                model.classifier.load_state_dict(global_parameters)


        # Use FedAvg on the model (standard FL)
        elif fed_avg == "model":
            # Aggregation (weighted average)
            global_parameters = global_model.state_dict()
            for k in global_parameters.keys():
                global_parameters[k] = torch.stack([(meta["n_local"][i] / meta["n_total"]) * client_models[i].state_dict()[k] 
                                                    for i in range(n_clients)], 0).sum(0)

            # Broadcast classifiers
            global_model.load_state_dict(global_parameters)
            for model in client_models:
                model.load_state_dict(global_parameters)

        #Tracking performance
        if (track_history and (r+1) % track_history == 0) or (r+1) == rounds:
            for client_id in range(n_clients):
                perf_trackers[client_id].new_eval(index=r+1)
            if fed_avg == "model":
                perf_tracker_global.new_eval(index=r+1)  
                
        # Compute representations
        feat_tracker.new_round()
        
        t1 = time.time()    
        print("\rRound {} done. ({:.1f}s)".format(r+1, t1-t0), end=6*" ")  
    
    # Plot training history and return
    if fed_avg == "model":
        hlp.plot_global_training_history(perf_trackers, metric="accuracy", title="Training history: Accuracy",
                                         savepath=os.path.join(fig_directory, "accuracy_history.png") if fig_directory is not None else None)
        hlp.plot_global_training_history(perf_trackers, metric="loss", title="Training history: Loss",
                                         savepath=os.path.join(fig_directory, "loss_history.png") if fig_directory is not None else None)
        
        perf_tracker_global.plot_training_history(metric="loss", title="Training history: Loss",
                                                  savepath=os.path.join(fig_directory, "loss_history_global.png") if fig_directory is not None else None)
        perf_tracker_global.plot_training_history(metric="accuracy", title="Training history: Accuracy",
                                                  savepath=os.path.join(fig_directory, "accuracy_history_global.png") if fig_directory is not None else None)
        return perf_trackers, perf_tracker_global, feat_tracker
    else:
        hlp.plot_global_training_history(perf_trackers, metric="accuracy", title="Training history: Accuracy",
                                     savepath=os.path.join(fig_directory, "accuracy_history.png") if fig_directory is not None else None)
        hlp.plot_global_training_history(perf_trackers, metric="loss", title="Training history: Loss",
                                     savepath=os.path.join(fig_directory, "loss_history.png") if fig_directory is not None else None)
        return perf_trackers, feat_tracker


def run_fedavg(n_clients, task, alpha="uniform", rounds=100, batch_size=32,
               epoch_per_round=1, sizes=None, reduced=False, track_history=False, seed=0):
    
    """Run a federated averaging experiment (FedAvg).
    
    Arguments:
    
    Return:
    """
    
    # Check avaibale device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    # Measure time
    t0 = time.time()
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)  
    random.seed(seed)
    
    # Load data, split and create dataloaders
    x_train, y_train, x_val, y_val, meta = hlp.load_data(dataset=task, data_dir="./data",
                                                         reduced=reduced, normalize="image-wise",
                                                         flatten=False, device=device)
    train_ds = hlp.CustomDataset(x_train, y_train)
    val_ds = hlp.CustomDataset(x_val, y_val)
    train_ds_list, val_ds_list = hlp.split_dataset(n_clients, train_ds, val_ds, alpha, sizes)    
    train_dl_list = hlp.ds_to_dl(train_ds_list, batch_size)
    val_dl_list = hlp.ds_to_dl(val_ds_list)
    val_dl_global = hlp.ds_to_dl(val_ds)
    
    # Compute dataset sizes
    n_local = [ds.len for ds in train_ds_list]
    n_tot = sum(n_local)
    
    # Create models
    global_model = mdl.get_model(task).to(device)
    client_models = [mdl.get_model(task).to(device) for i in range(n_clients)]
    
    # Create loss functions
    criterion = nn.CrossEntropyLoss() 
    
    # Initialize performance trackers
    pt_list = [hlp.PerfTracker(global_model, {"Train": train_dl_list[i], "Validation": val_dl_list[i], "Global": val_dl_global}, 
                               criterion, meta["n_class"], export_dir=None, ID="Client {}".format(i)) for i in range(n_clients)]
    
    # Create optimizers
    optimizers = [torch.optim.Adam(model.parameters()) for model in client_models]
    
    # Global training
    for r in range(rounds):
        
        # Local training
        global_parameters = global_model.state_dict()
        for i, model in enumerate(client_models): 
            
            # Model broadcasting
            model.load_state_dict(global_parameters)
            
            # Initialization
            optimizer =  optimizers[i]
            model.train()

            #Teacher learning
            for e in range(epoch_per_round):

                # Training
                for features, target in train_dl_list[i]:
                    optimizer.zero_grad()
                    output = model(features)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
        
        # Aggregation (weighted average
        for k in global_parameters.keys():
            global_parameters[k] = torch.stack([(n_local[i] / n_tot) * client_models[i].state_dict()[k] for i in range(n_clients)], 0).sum(0)
        global_model.load_state_dict(global_parameters)
        
        #Tracking performance
        if track_history or r == rounds-1:
            [pt_list[i].new_eval(index=r+1) for i in range(n_clients)]
        print("\rRound {}/{} done.".format(r+1, rounds), end="   ")
            
    # Display time
    tf = time.time()
    print("\nTotal time: {:.1f}s".format(tf-t0))
    
    return pt_list

    
def run_federated_distillation(n_clients, task, alpha="uniform", rounds=100, batch_size=32, topology="fc", 
              epoch_per_round=1, sizes=None, reduced=False, track_history=False, seed=0):
    """Run fully decentralized federated learning (no central server).
    
    Arguments:
    
    Return:
    """
    raise NotImplementedError

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    
    if args.congig_file is not None:
        f = open(config_file)
        config = json.load(f)
        f.close()
    
    # Run experiments
    pt_list, ft = run_privateKD()

