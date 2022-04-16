# -----------------------------------------------------------------------------
# This file contains the experiments code that was developped
# during my master thesis at MIT.
#
# 2022 Frédéric Berdoz, Boston, USA
# -----------------------------------------------------------------------------


import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import helpers as hlp
import models as mdl


def get_args():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clients', type=int, default=2, help="")
    parser.add_argument('--task', type=str, default='MNIST', help="")
    parser.add_argument('--reg_coeff', type=float,default=1.0, help="")
    parser.add_argument('--generator', type=str, default='normal', help="")
    parser.add_argument('--n_rand', type=int, default=100, help="")
    parser.add_argument('--reg_loss', type=str, default=0.01, help="")
    parser.add_argument('--alpha', type=float, default=1e15, help="")
    parser.add_argument('--rounds', type=int, default=100,  help="")
    parser.add_argument('--batch_size', type=int, default=32,help="")
    parser.add_argument('--topology', type=str, default="fc", help="")
    parser.add_argument('--epoch_per_round', type=int, default=1, help="")
    parser.add_argument('--sizes',type=float, nargs='*', default=None, help="")
    parser.add_argument('--reduced', action="store_true")
    parser.add_argument('--track_history', type=bool, default=False, help="")
    parser.add_argument('--seed', type=int, default=0, help="")
    args = parser.parse_args()
    return args

    
def run_shadow_learning(n_clients, task, reg_coeff=1, generator="normal", n_rand=100, reg_loss="MSE", 
                        alpha="uniform", rounds=100, batch_size=32, topology="fc", epoch_per_round=1, 
                        sizes=None, reduced=False, track_history=False, seed=0):
    
    """Run a shadown learning experiment.
    
    Arguments:
    
    Return:
    """
    
    # Check avaibale device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Measure time
    t0 = time.time()
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Argument processing
    if topology == "fc":
        topology = np.ones(n_clients) - np.eye(n_clients)
    
    else:
        raise NotImplementedError
    
    
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

    # Create models
    client_models = [mdl.get_model(task).to(device) for i in range(n_clients)]
    
    # Create loss functions
    criterion_vanilla = nn.CrossEntropyLoss() 
    criterion = hlp.KDLoss(main_loss="cross-entropy", reg_loss=reg_loss, coeff=reg_coeff)
    
    # Initialize performance trackers
    pt_list = [hlp.PerfTracker(client_models[i], {"Train": train_dl_list[i], "Validation": val_dl_list[i], "Global": val_dl_global}, 
                               criterion_vanilla, meta["n_class"], export_dir=None, ID="Client {}".format(i)) for i in range(n_clients)]
    
    # Create optimizers
    optimizers = [torch.optim.Adam(model.parameters()) for model in client_models]
    
    # Training
    for r in range(rounds):
        
        # Optimizing when no KD
        if reg_coeff != 0 and n_rand > 0:
            # Reset KD data list
            KD_inputs = [None for _ in range(n_clients)]
            KD_outputs = [None for _ in range(n_clients)]

            for i, model in enumerate(client_models):
                # Create KD data
                X = hlp.generate_data((n_rand, *meta["in_dimension"]), generator).to(device)
                KD_inputs[i] = X
                with torch.no_grad():
                    KD_outputs[i] = model(X)
        
        for i, model in enumerate(client_models):            
            
            # Initialization
            optimizer =  optimizers[i]
            model.train()
            
            # Optimizing when no KD
            if reg_coeff > 0 and n_rand > 0:
                # Build others predictions list
                incident_edges = topology[i]
                predictions_others = [KD_outputs[i] for i, key in enumerate(incident_edges) if key == 1]
                input_others = [KD_inputs[i] for i, key in enumerate(incident_edges) if key == 1]

            #Teacher learning
            for e in range(epoch_per_round):

                # Training
                for features, target in train_dl_list[i]:
                    optimizer.zero_grad()
                    output = model(features)
                    
                    # Optimizing when no KD
                    if reg_coeff != 0 and n_rand > 0:
                        predictions_local = [model(X) for X in input_others]
                        loss = criterion(output, target, predictions_local, predictions_others)
                    else:
                        loss = criterion_vanilla(output, target)
                    loss.backward()
                    optimizer.step()

            #Tracking performance
            if track_history or r == rounds-1:
                perf = pt_list[i].new_eval(index=r+1)
            print("\rRound {}/{} done.".format(r+1, rounds), end=10*" ")

    # Display time
    tf = time.time()
    print("\nTotal time: {:.1f}s".format(tf-t0))
    
    return pt_list

def run_fedavg(n_clients, task, alpha="uniform", rounds=100, batch_size=32,
               epoch_per_round=1, sizes=None, reduced=False, track_history=False, seed=0):
    """Run a federated averaging experiment (FedAvg).
    
    Arguments:
    
    Return:
    """
    
    # Check avaibale device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    
def run_fedprox(n_clients, task, alpha="uniform", rounds=100, batch_size=32, 
               epoch_per_round=1, sizes=None, reduced=False, track_history=False, seed=0):
    """Run a fede experiment.
    
    Arguments:
    
    Return:
    """
    
def run_scaffold(n_clients, task, alpha="uniform", rounds=100, batch_size=32, 
               epoch_per_round=1, sizes=None, reduced=False, track_history=False, seed=0):
    """Run a fede experiment (FedAvg).
    
    Arguments:
    
    Return:
    """
    
def run_fednova(n_clients, task, alpha="uniform", rounds=100, batch_size=32, 
               epoch_per_round=1, sizes=None, reduced=False, track_history=False, seed=0):
    """Run a fede experiment (FedAvg).
    
    Arguments:
    
    Return:
    """
    
def run_decentralized_fl(n_clients, task, alpha="uniform", rounds=100, batch_size=32, topology="fc", 
              epoch_per_round=1, sizes=None, reduced=False, track_history=False, seed=0):
    """Run fully decentralized federated learning (no central server).
    
    Arguments:
    
    Return:
    """
    raise NotImplementedError
    
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
    
    # Run experiments
    pt_list = run_shadow_learning(n_clients=args.n_clients, 
                                  task=args.task, 
                                  reg_coeff=args.reg_coeff, 
                                  generator=args.generator, 
                                  n_rand=args.n_rand, 
                                  reg_loss=args.reg_loss, 
                                  alpha=args.alpha, 
                                  rounds=args.rounds, 
                                  batch_size=args.batch_size, 
                                  topology=args.topology, 
                                  epoch_per_round=args.epoch_per_round, 
                                  sizes=args.sizes, 
                                  reduced=args.reduced, 
                                  track_history=args.track_history, 
                                  seed=args.seed)

