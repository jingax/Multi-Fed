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
import matplotlib.pyplot as plt
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
    parser.add_argument('--kd_type', type=str, default="feature",  help="")
    parser.add_argument('--sizes',type=float, nargs='*', default=None, help="")
    parser.add_argument('--reduced', type=float, default=1.0, help="")
    parser.add_argument('--track_history', type=int, default=1, help="")
    parser.add_argument('--fed_avg', type=str, default=False,  help="")
    parser.add_argument('--export_dir', type=str, default="./saves", help="")
    parser.add_argument('--data_dir', type=str, default="./data", help="")
    parser.add_argument('--config_file', type=str, default=None, help="")
    parser.add_argument('--device', type=str, default=None, help="")
    parser.add_argument('--preset', type=str, default=None, help="")
    parser.add_argument('--seed', type=int, default=0, help="")
    args = parser.parse_args()
    return args


def run(n_clients, dataset, model, alpha="uniform", rounds=100, 
        batch_size=32, epoch_per_round=1, lr=1e-3, optimizer="adam", feature_dim=100,
        n_avg=None, lambda_kd=1.0, lambda_disc=1.0, kd_type="feature", sizes=None, reduced=False, 
        track_history=1, fed_avg=False, export_dir=None, data_dir="./data", disc_method="classifier",
        device=None, preset=None, seed=0):
    
    """Run an experiment of PrivateKD.
    
    Arguments:
    
    Return:
        - perf_trackers: A list of all the performance trackers
        - feature_trackers: The feature trackers.
    """
    # Argument processing
    if preset is not None:
        if preset in ["fl", "FL", "fedavg", "FedAvg"]:
            print("Running FL with {} clients".format(n_clients))
            lambda_kd=0
            lambda_disc=0
            fed_avg = "model"
        elif preset in ["fd", "FD"]:
            print("Running FD with {} clients".format(n_clients))
            n_avg = None
            lambda_disc=0
            kd_type = "output"
            fed_avg = False
        elif preset in ["il", "IL"]:
            print("Running IL with {} clients".format(n_clients))
            lambda_kd=0
            lambda_disc=0
            fed_avg = False
        elif preset in ["CL", "cl"]:
            print("Running centralized learning".format(n_clients))
            n_clients = 1
            lambda_kd = 0
            lambda_disc = 0
        else:
            raise ValueError("Unknown preset {} (FL, FD, IL or CL).".format(preset))
    else:
        print("Running CFKD with {} clients".format(n_clients))
    
    if kd_type == "output" and lambda_disc != 0:
        lambda_disc = 0
        print("Warning: lambda_disc has been set to 0 since output-based kd is used (FD).")

    # Store parameters for easy reproductability
    if export_dir is not None:
        # Store local parameters
        config = locals()
        
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
                             savepath=os.path.join(fig_directory, "class_dist.png") if export_dir is not None else None)
    
    # Store dataset sizes
    meta["n_total"] = train_input.shape[0] 
    meta["n_local"] = [ds.inputs.shape[0] for ds in train_ds_list]
    
    # Define criterions
    criterion = nn.CrossEntropyLoss()
    criterion_kd = nn.MSELoss()
    criterion_disc = nn.BCELoss()
    
    # Model initialization
    client_models = [mdl.get_model(model, feature_dim, meta).to(device) for _ in range(n_clients)]
    if fed_avg in ["model", "classifier"]:
        global_model = mdl.get_model(model, feature_dim, meta).to(device)
    
    
    # Performance tracker
    if fed_avg == "model":
        perf_trackers = [hlp.PerfTracker(global_model, 
                                         {"Train": train_dl_list[i], "Validation": val_dl_list[i], "Validation (global)": global_val_dl}, 
                                         criterion, meta["n_class"], ID="Client {}".format(i)) for i in range(n_clients)]
        
    else:
        perf_trackers = [hlp.PerfTracker(client_models[i], 
                                         {"Train": train_dl_list[i], "Validation": val_dl_list[i], "Validation (global)": global_val_dl}, 
                                         criterion, meta["n_class"], ID="Client {}".format(i)) for i in range(n_clients)]
        
    # Feature tracker and discriminator
    if kd_type == "feature":
        tracker = hlp.OutputTracker([m.features for m in client_models], train_dl_list, feature_dim, meta)
    elif kd_type == "output":
        tracker = hlp.OutputTracker(client_models, train_dl_list, meta["n_class"], meta)
    
    if lambda_disc > 0:
        if disc_method == "classifier":
            discriminators = [mdl.Discriminator("prob_product", client_models[i].classifier) for i in range(n_clients)]
        elif disc_method == "seperate":
            discriminators = [mdl.Discriminator("exponential_prob", feat_dim=feature_dim, n_class=meta["n_class"]).to(device) for i in range(n_clients)]

    # Optimizers
    # Adam
    if optimizer in ["adam", "Adam", "ADAM"]:
        optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in client_models]
        if disc_method == "seperate" and lambda_disc > 0:
            optimizers_disc = [torch.optim.Adam(disc.parameters(), lr=lr) for disc in discriminators]
    
    # SGD
    elif optimizer in ["sgd", "Sgd", "SGD"]:
        optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in client_models] 
        if disc_method == "seperate" and lambda_disc > 0:
            optimizers_disc = [torch.optim.SGD(disc.parameters(), lr=lr) for disc in discriminators] 
    else:
        raise ValueError("Optimizer unknown.")
    
    #Each client updates its model locally on its own dataset
    for r in range(rounds):
        t0 = time.time()
        
        for client_id in range(n_clients):
            #Setting up the local training
            model = client_models[client_id]
            model.train()
            opt = optimizers[client_id]
            if lambda_disc > 0:
                disc = discriminators[client_id]
                if disc_method == "seperate":
                    opt_disc = optimizers_disc[client_id]
            
            #Local update
            for e in range(epoch_per_round):
                for inputs, targets in train_dl_list[client_id]:
                    # Reset gradient
                    opt.zero_grad()
                    if disc_method == "seperate" and lambda_disc > 0:
                        opt_disc.zero_grad()
                    
                    # Local forward pass
                    features = model.features(inputs)
                    logits = model.classifier(features)
                        
                    # Optimization step
                    loss = criterion(logits, targets)
                    if lambda_kd > 0:
                        # Compute estimated probabilities
                        if kd_type == "feature":
                            teacher_data = tracker.get_global_outputs(n_avg=None, client_id=None).to(device)
                            loss += lambda_kd * criterion_kd(features, teacher_data[targets])
                        elif kd_type == "output":
                            teacher_data = tracker.get_global_outputs(n_avg=None, client_id=None).to(device)
                            loss += lambda_kd * criterion_kd(logits, teacher_data[targets])
                    if lambda_disc > 0:
                        teacher_data = tracker.get_global_outputs(n_avg=n_avg, client_id="random").to(device)
                        targets_global = torch.arange(meta["n_class"]).to(device)
                        scores, disc_targets = disc(features, teacher_data, targets, targets_global)
                        loss += lambda_disc * criterion_disc(scores, disc_targets)
                    
                    # Optimization step
                    loss.backward()
                    opt.step()
                    if disc_method == "seperate" and lambda_disc > 0:
                        opt_disc.step()
        
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
                
        # Compute representations
        tracker.new_round()
        
        t1 = time.time()    
        print("\rRound {} done. ({:.1f}s)".format(r+1, t1-t0), end=6*" ")  
    
    
    # Evalutate models (averaging performance)
    tr_loss = np.array([pt.perf_histories["Train"]["loss"][-1] for pt in perf_trackers]).mean()
    val_loss = np.array([pt.perf_histories["Validation (global)"]["loss"][-1] for pt in perf_trackers]).mean()
    tr_acc = np.array([pt.perf_histories["Train"]["accuracy"][-1] for pt in perf_trackers]).mean()
    val_acc = np.array([pt.perf_histories["Validation (global)"]["accuracy"][-1] for pt in perf_trackers]).mean()
    print("\nFinal average performance:")
    print("Train loss: {:.2f} | Validation (global) loss: {:.2f}".format(tr_loss, val_loss))
    print("Train acc: {:.2f}% | Validation (global) acc: {:.2f}%".format(100*tr_acc, 100*val_acc))
    
    # Saving plots if necessary
    hlp.plot_global_training_history(perf_trackers, metric="accuracy", title="Training history: Accuracy",
                                     savepath=os.path.join(fig_directory, "accuracy_history.png") if export_dir is not None else None)
    hlp.plot_global_training_history(perf_trackers, metric="loss", title="Training history: Loss",
                                     savepath=os.path.join(fig_directory, "loss_history.png") if export_dir is not None else None)
    plt.close("all")
    return perf_trackers, tracker

def benchmark(n_clients, dataset, model, alpha="uniform", rounds=100, 
              batch_size=32, epoch_per_round=1, lr=1e-3, optimizer="adam", feature_dim=100, 
              n_avg=None, fed_avg=None, lambda_kd=1.0, lambda_disc=1.0, sizes=None, reduced=False, 
              track_history=1, export_dir=None, data_dir="./data",
              device=None, seed=0):
    """Benchmark privateKD with FedAvg, FD and independent learning."""
    
    print(40 * "*")
    print("Starting benchmark")
    print(40 * "*")
    pt_cfkd, _ = run(n_clients=n_clients, dataset=dataset, model=model, alpha=alpha, rounds=rounds, 
                  batch_size=batch_size, epoch_per_round=epoch_per_round, lr=lr, optimizer=optimizer, 
                  feature_dim=feature_dim, n_avg=n_avg, lambda_kd=lambda_kd, lambda_disc=lambda_disc, 
                  sizes=sizes, fed_avg=fed_avg, reduced=reduced, track_history=track_history, 
                  export_dir=export_dir, data_dir=data_dir, device=device, seed=seed)
    print(20 * "*")
    pt_fd, _ = run(n_clients=n_clients, dataset=dataset, model=model, alpha=alpha, rounds=rounds, 
                batch_size=batch_size, epoch_per_round=epoch_per_round, lr=lr, optimizer=optimizer, 
                feature_dim=feature_dim, lambda_kd=lambda_kd, sizes=sizes, reduced=reduced, track_history=track_history,
                export_dir=export_dir, data_dir=data_dir, device=device, preset="fd", seed=seed)
    print(20 * "*")
    pt_il, _ = run(n_clients=n_clients, dataset=dataset, model=model, alpha=alpha, rounds=rounds, 
                batch_size=batch_size, epoch_per_round=epoch_per_round, lr=lr, optimizer=optimizer, 
                feature_dim=feature_dim, sizes=sizes, reduced=reduced, track_history=track_history, 
                export_dir=export_dir, data_dir=data_dir, device=device, preset="il", seed=seed)
    print(20 * "*")
    pt_fl, _ = run(n_clients=n_clients, dataset=dataset, model=model, alpha=alpha, rounds=rounds, 
                batch_size=batch_size, epoch_per_round=epoch_per_round, lr=lr, optimizer=optimizer, 
                feature_dim=feature_dim, sizes=sizes, reduced=reduced, track_history=track_history,
                export_dir=export_dir, data_dir=data_dir, device=device, preset="fl", seed=seed)


    
    print("Benchmark done.")
    return pt_cfkd, pt_fl, pt_fd, pt_il
    
if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    
    if args.congig_file is not None:
        f = open(config_file)
        config = json.load(f)
        f.close()
    
    # Run experiments
    pt_list, ft = run_privateKD()

