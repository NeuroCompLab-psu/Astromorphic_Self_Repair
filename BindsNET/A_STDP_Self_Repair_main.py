import os
import shutil
import argparse
from time import time
import io
import uuid

import sys
from pathlib import Path

print('Python %s on %s' % (sys.version, sys.platform))

sys.path.extend([os.path.abspath('./')])
print()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torchvision import transforms
from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, FashionMNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from A_STDP_models import DiehlAndCook2015v2_A_STDP
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights
from bindsnet.analysis.plotting import plot_spikes, plot_weights
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.network import load, Network
import seaborn as sns
import PIL.Image
from torchvision.transforms import ToTensor
from typing import Union
from collections import OrderedDict

# Save a set of weight to log folder
def save_weight_theta(_network: Network, _path: str):
    weight_theta = OrderedDict()
    weight_theta['weight'] = _network.X_to_Y.w.data
    weight_theta['theta'] = _network.Y.theta.data
    torch.save(weight_theta, _path)

# Weight decay mask generator
def decay_ratio(size, t, v_mean, v_sigma, gpu):
    if gpu:
        v = torch.normal(v_mean, v_sigma, size=size, device=torch.device('cuda'))
    else:
        v = torch.normal(v_mean, v_sigma, size=size)
    
    return torch.pow(t, -v)

# Sobel filter
def sobel(x):
    '''
    Whitening and sharpening of data
    edge detection
    :param x: input tensor with 3 dimensions
    :return: sharpened grayscale tensor (single dimension)
    '''

    sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
    sobel_filter.weight.data[0, 0].copy_(
        torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    )
    sobel_filter.weight.data[1, 0].copy_(
        torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    )
    sobel_filter.bias.data.zero_()
    sobel_op = nn.Sequential(sobel_filter)
    for p in sobel_op.parameters():
        p.requires_grad = False
    g = sobel_op(x.unsqueeze_(0)).squeeze_(0)
    g = torch.sqrt(g.mul(g).sum(0))
    return g.unsqueeze_(0)

# Test the accuracy of a network
def doOneAccuracyTest(
    network: Network, 
    batches1: list,
    batches2: list,
    time: int,
    one_step: bool, 
    n_classes: int,
    gpu: bool
) -> torch.Tensor:

    # Set norm to None
    norm_ori = network.connections[('X','Y')].norm
    network.connections[('X','Y')].norm = None

    # Assignement -------------------------------------------------------

    # Label vector
    train_labels = []
    # Spike record vector (num_sample, time, num_neuron)
    train_spike_record = torch.zeros(60000, time, network.n_neurons)

    network.reset_state_variables()
    for step, batch in enumerate(batches1):
        print(f"Assignment step: {step}")
        train_labels.extend(batch["label"].tolist())

        this_batch_size = batch['encoded_image'].shape[1]

        inpts = {"X": batch["encoded_image"]}
        if gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}
        
        network.run(inputs=inpts, time=time, one_step=one_step)
    
        # Add to spikes recording.
        s = network.monitors['Y_spikes'].get("s").permute((1, 0, 2))
        train_spike_record[(step * this_batch_size): 
                    (step * this_batch_size) + s.size(0)
        ] = s
    
        network.reset_state_variables()

    train_label_tensor = torch.tensor(train_labels)

    train_assignments, _, _ = assign_labels(
    spikes=train_spike_record,
    labels=train_label_tensor,
    n_labels=n_classes
    )

    # Inference -------------------------------------------------------

    # Label vector
    test_labels = []
    # Spike record vector (num_sample, time, num_neuron)
    test_spike_record = torch.zeros(10000, time, network.n_neurons)

    network.reset_state_variables()
    for step, batch in enumerate(batches2):
        print(f"Inference step: {step}")
        test_labels.extend(batch["label"].tolist())

        this_batch_size = batch['encoded_image'].shape[1]

        inpts = {"X": batch["encoded_image"]}
        if gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}
        
        network.run(inputs=inpts, time=time, one_step=one_step)
    
        # Add to spikes recording.
        s = network.monitors['Y_spikes'].get("s").permute((1, 0, 2))
        test_spike_record[(step * this_batch_size): 
                    (step * this_batch_size) + s.size(0)
        ] = s
    
        network.reset_state_variables()

    test_label_tensor = torch.tensor(test_labels)
    
    test_all_activity_pred = all_activity(spikes=test_spike_record, 
                                    assignments=train_assignments, 
                                    n_labels=n_classes)
    
    test_acc = 100 * torch.mean(
        (test_label_tensor.long() == test_all_activity_pred).float()
    )

    ave_out_spk = test_spike_record.sum() / 10000

    # Set norm to original value
    network.connections[('X','Y')].norm = norm_ori

    return test_acc, train_assignments, ave_out_spk

####################################
####################################
####################################
# main function
def main(args):
    update_interval = args.update_steps * args.batch_size

    # Sets up GPU use
    torch.backends.cudnn.benchmark = False
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Determines number of workers to use
    if args.n_workers == -1:
        args.n_workers = args.gpu * 6 * torch.cuda.device_count()

    n_sqrt = int(np.ceil(np.sqrt(args.n_neurons)))

    # Argument checking
    if args.reduction == "sum":
        reduction = torch.sum
    elif args.reduction == "mean":
        reduction = torch.mean
    else:
        raise NotImplementedError
    
    if (args.dataset != "MNIST") and (args.dataset != "FMNIST"):
        print("dataset is {}".format(args.dataset))
        raise ValueError("dataset must be MNIST or FMNIST !")
    
    # Generate log folder name
    args.log_dir += "/data_{}_pf_{}_vfau_({},{})_tfau_{}_tau_{}_eqs_{}_sumlo_{}_n_{}_dt_{}_t_{}_bs_{}_inh_{}_tplus_{}_int_{}_wm_{}_nu_({},{})_th_{}_sob_{}/". \
        format(
            args.dataset,
            args.pfault,
            args.v_mean,
            args.v_sigma,
            args.t_norm,
            args.tau,
            args.eq_step,
            args.sum_lowerbound,
            args.n_neurons,
            args.dt,
            args.time,
            args.batch_size,
            args.inh,
            args.theta_plus,
            args.intensity,
            args.weight_max,
            args.nu_pre,
            args.nu_post,
            args.thresh,
            args.sobel,  
            )

    # Create log folder at specified path
    run_count = 0
    log_dir_run = args.log_dir + f'run_{run_count}'
    while os.path.isdir(log_dir_run):
        run_count += 1
        log_dir_run = args.log_dir + f'run_{run_count}/'
    args.log_dir = log_dir_run
    abs_log_dir = str(os.path.abspath(args.log_dir))
    if not os.path.exists(abs_log_dir):
        os.makedirs(abs_log_dir)
    
    # Create runID folder in log folder
    run_id = str(uuid.uuid4())
    os.makedirs(abs_log_dir + "/_" + run_id)

    print("Logs to be written in \n{}".format(os.path.abspath(args.log_dir)))

    save_path = args.log_dir + "/weights"
    save_path = str(os.path.abspath(save_path))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path += "/"
    
    # create a cvs file recording accuracy and number of spike
    recorder = open(args.log_dir + "/accrec.csv", 'w')
    recorder.write("Global step")
    recorder.write(',')
    recorder.write('Test accuracy')
    recorder.write(',')
    recorder.write('Average number of spike')
    recorder.write('\n')

    ####################################
    ####################################
    ####################################

    # Initialize network class
    network = DiehlAndCook2015v2_A_STDP(
        n_inpt=784,
        n_neurons=args.n_neurons,
        inh=args.inh,
        dt=args.dt,
        nu=(args.nu_pre, args.nu_post),
        reduction=reduction,
        wmax=args.weight_max,
        norm=None,
        theta_plus=args.theta_plus,
        tc_theta_decay=args.tc_theta_decay,
        inpt_shape=(1, 28, 28),
        pfault=args.pfault,
        tau=args.tau,
        thresh = args.thresh,
    )
    
    # Load uncorrupted weight file
    if args.network_file_load:
        print("=>Loading Pretrained Model from {}...........".format(args.network_file_load))
        trained_weight_theta = load(args.network_file_load)
        network.X_to_Y.w.data[:] = trained_weight_theta['weight']
        network.Y.theta.data[:] = trained_weight_theta['theta']
        print("=>Loaded successfully .")

    # Directs network to GPU.
    if args.gpu:
        network.to("cuda")

    # Create dataset classes
    if args.dataset == "FMNIST":
        # Load FashionMNIST data.
        train_dataset_for_train = FashionMNIST(
            PoissonEncoder(time=args.time, dt=args.dt),
            None,
            root=os.path.join(ROOT_DIR, "data", "FashionMNIST"),
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
            ),
        )

        train_dataset_for_test = FashionMNIST(
            PoissonEncoder(time=args.test_time, dt=args.dt),
            None,
            root=os.path.join(ROOT_DIR, "data", "FashionMNIST"),
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
            ),
        )

        test_dataset = FashionMNIST(
            PoissonEncoder(time=args.test_time, dt=args.dt),
            None,
            root=os.path.join(ROOT_DIR, "data", "FashionMNIST"),
            download=True,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
            ),
        )

        # Add sobel filter if sobel is True
        if args.sobel:
            train_dataset_for_train = FashionMNIST(
                PoissonEncoder(time=args.time, dt=args.dt),
                None,
                root=os.path.join(ROOT_DIR, "data", "FashionMNIST"),
                download=True,
                train=True,
                transform=transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: sobel(x)),
                    transforms.Lambda(lambda x: x * args.intensity)]
                ),
            )

            train_dataset_for_test = FashionMNIST(
                PoissonEncoder(time=args.test_time, dt=args.dt),
                None,
                root=os.path.join(ROOT_DIR, "data", "FashionMNIST"),
                download=True,
                train=True,
                transform=transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: sobel(x)),
                    transforms.Lambda(lambda x: x * args.intensity)]
                ),
            )

            test_dataset = FashionMNIST(
                PoissonEncoder(time=args.test_time, dt=args.dt),
                None,
                root=os.path.join(ROOT_DIR, "data", "FashionMNIST"),
                download=True,
                train=False,
                transform=transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: sobel(x)),
                    transforms.Lambda(lambda x: x * args.intensity)]
                ),
            )

    else:
        # Load MNIST data.
        train_dataset_for_train = MNIST(
            PoissonEncoder(time=args.time, dt=args.dt),
            None,
            root=os.path.join(ROOT_DIR, "data", "MNIST"),
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
            ),
        )

        train_dataset_for_test = MNIST(
            PoissonEncoder(time=args.test_time, dt=args.dt),
            None,
            root=os.path.join(ROOT_DIR, "data", "MNIST"),
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
            ),
        )

        test_dataset = MNIST(
            PoissonEncoder(time=args.test_time, dt=args.dt),
            None,
            root=os.path.join(ROOT_DIR, "data", "MNIST"),
            download=True,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
            ),
        )

    # Create dataloaders
    train_dataloader = DataLoader(
            dataset=train_dataset_for_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=args.gpu,
        )

    test_dataloader_1 = DataLoader(
            dataset=train_dataset_for_test,
            batch_size=args.test_batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=args.gpu,
        )

    test_dataloader_2 = DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=args.gpu,
        )

    # Extract batches from dataloader for better efficiency
    batches_1 = []
    for step, batch in enumerate(test_dataloader_1):
        print(f'Extracting test batches1: {step}')
        batches_1.append(batch)

    batches_2 = []
    for step, batch in enumerate(test_dataloader_2):
        print(f'Extracting test batches2: {step}')
        batches_2.append(batch)

    ####################################
    ####################################
    ####################################

    # Number of classes in the dataset
    n_classes = 10

    ####################################
    # ACC after fault injection

    # Take weight snapshot
    network.connections[('X','Y')].weight_snapshot()
    
    # Apply weight decay
    w = network.state_dict()["X_to_Y.w"]
    healthy_mean_sum = w.sum(0).mean(0)
    dr = decay_ratio(w.size(), args.t_norm, args.v_mean, args.v_sigma, args.gpu)
    w = w * dr
    w = w * network.connections[('X','Y')].weight_mask
    network.state_dict()["X_to_Y.w"].data[:] = w

    # Set norm to None
    norm_ori = network.connections[('X','Y')].norm
    network.connections[('X','Y')].norm = None
    
    # Disable learning.
    network.train(False)

    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=args.test_time)
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    # Test once
    test_acc, assignments, _ = doOneAccuracyTest(network, 
                                batches_1, 
                                batches_2, 
                                args.test_time,
                                args.one_step, 
                                n_classes, 
                                args.gpu)
    
    # Save parameters if specified
    if args.network_param_save:
        save_path_f = save_path + "After_fault_injection_{:.2f}.wt".format(test_acc)
        print("After fault injection ACC : {:.2f}".format(test_acc))
        save_weight_theta(network, save_path_f)

    ####################################
    # ACC after normalization

    # Normalization
    w = network.state_dict()["X_to_Y.w"]
    sum_w = w.sum(0)
    w = w / sum_w * 78.4
    network.state_dict()["X_to_Y.w"].data[:] = w

    # Test once
    test_acc, assignments, _ = doOneAccuracyTest(network, 
                                batches_1, 
                                batches_2,
                                args.test_time,
                                args.one_step, 
                                n_classes, 
                                args.gpu)
    
    # Save parameters if specified
    if args.network_param_save:
        save_path_f = save_path + "After_normalization_{:.2f}.wt".format(test_acc)
        print("After normalization ACC : {:.2f}".format(test_acc))
        save_weight_theta(network, save_path_f)

    # Recover weight just after fault injection
    network.state_dict()["X_to_Y.w"].data[:] = w / 78.4 * sum_w

    # Set norm to original value
    network.connections[('X','Y')].norm = norm_ori

    # Re-enable learning.
    network.train(True)

    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=args.time)
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    ####################################
    ####################################
    ####################################

    # start of training

    # Best accuracy recorder for entire self-repair process
    best_acc = 0

    for epoch in range(args.n_epochs):

        for step, batch in enumerate(train_dataloader):

            # Global step
            global_step = len(train_dataset_for_train) * epoch + args.batch_size * step

            # Weight equalization
            if args.eq_step > 0:
                if step % args.eq_step == 0:
                    w = network.state_dict()["X_to_Y.w"]
                    sum_w = w.sum(0)
                    meansum_w = sum_w.mean(0)
                    w = w / sum_w * meansum_w

                    # Raise weight sum to lowerbound if the weight sum is too low
                    if global_step == 0 and args.sum_lowerbound > 0:
                        sumlb = args.sum_lowerbound * healthy_mean_sum
                        if sumlb > meansum_w:
                            w = w / meansum_w * sumlb

                    network.state_dict()["X_to_Y.w"].data[:] = w

            ####################################
            # Test accuracy once when reaching update intervals
            if step % args.update_steps == 0:
                # Disable learning.
                network.train(False)

                # Set up monitors for spikes and voltages
                spikes = {}
                for layer in set(network.layers):
                    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=args.test_time)
                    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

                # Print weights sum mean
                w1 = network.connections[('X','Y')].w
                print("Mean of neuron-wide sum of weights : {:.2f}".format(w1.sum(0).mean().cpu().numpy()))

                # Test once
                test_acc, assignments, ave_spk = doOneAccuracyTest(network, 
                                            batches_1, 
                                            batches_2, 
                                            args.test_time,
                                            args.one_step, 
                                            n_classes, 
                                            args.gpu)

                
                if test_acc > best_acc:
                    if global_step != 0:
                        best_acc = test_acc
                    print("Test accuracy : {:.2f}".format(test_acc))

                    # Save parameters if specified
                    if args.network_param_save:
                        save_path_f = save_path + "global_step_{}_test_acc_{:.2f}.wt".format(global_step, test_acc)
                        save_weight_theta(network, save_path_f)

                # record accuracy and average number of spikes
                recorder.write(str(global_step))
                recorder.write(',')
                recorder.write(f'{test_acc:.2f}')
                recorder.write(',')
                recorder.write(f'{ave_spk:.2f}')
                recorder.write('\n')

                # Re-enable learning.
                network.train(True)
                # Set up monitors for spikes and voltages
                spikes = {}
                for layer in set(network.layers):
                    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=args.time)
                    network.add_monitor(spikes[layer], name="%s_spikes" % layer)
            # End test section
            ####################################

            # Print information before each self-repair batch
            print(f"{run_id}    Epoch: {epoch}    Step: {step} / {len(train_dataloader)}", end = '')

            # Prepare input batch spick train
            inpts = {"X": batch["encoded_image"]}
            if args.gpu:
                inpts = {k: v.cuda() for k, v in inpts.items()}

            # Run the network
            network.run(inputs=inpts, time=args.time, one_step=args.one_step)

            # Calculate the maximum number of spike amoung all neurons in this run
            # If max # of spikes = 0, the network does not fire/learn at all.
            s = network.monitors['Y_spikes'].get("s").permute((1, 0, 2))
            maxspikecount = s.sum(0).max()
            print(f"  max # of spikes {maxspikecount.cpu().numpy()}")

            # Reset state variables.
            network.reset_state_variables()

    ####################################
    ####################################
    ####################################

    # Save weights after the end of last epoch
    
    # Disable learning.
    network.train(False)

    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=args.test_time)
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    # Test once
    test_acc, assignments, ave_spk = doOneAccuracyTest(network, 
                                batches_1, 
                                batches_2,
                                args.test_time,
                                args.one_step, 
                                n_classes, 
                                args.gpu)
    
    # Save parameters if specified
    if args.network_param_save:
        save_path_f = save_path + "After_{:d}_epochs_of_self_repair_{:.2f}.wt".format(args.n_epochs, test_acc)
        save_weight_theta(network, save_path_f)

    # record accuracy and average number of spikes
    recorder.write(str(global_step))
    recorder.write(',')
    recorder.write(f'{test_acc:.2f}')
    recorder.write(',')
    recorder.write(f'{ave_spk:.2f}')
    recorder.write('\n')
    recorder.close()



def parse_args():
    parser = argparse.ArgumentParser()

    # STDP network parameter arguments
    parser.add_argument("--network-file-load", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n-neurons", type=int, required=True)
    parser.add_argument("--time", type=int, required=True)
    parser.add_argument("--test-time", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--test-batch-size", type=int, required=True)
    parser.add_argument("--dt", type=int, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reduction", type=str, default="sum")
    parser.add_argument("--inh", type=float, required=True)
    parser.add_argument("--theta-plus", type=float, required=True)
    parser.add_argument("--tc-theta-decay", type=float, required=True)
    parser.add_argument("--intensity", type=float, required=True)
    parser.add_argument("--weight-max", type=float, required=True)
    parser.add_argument("--nu-pre", type=float, required=True)
    parser.add_argument("--nu-post", type=float, required=True)
    parser.add_argument("--thresh", type=float, required=True)
    parser.add_argument("--sobel", action="store_true")
    parser.add_argument("--one-step", action="store_true")

    # Fault arguments
    parser.add_argument("--pfault", type=float, required=True)
    parser.add_argument("--v-mean", type=float, required=True)
    parser.add_argument("--v-sigma", type=float, required=True)
    parser.add_argument("--t-norm", type=float, required=True)
    
    # self-repair parameter arguments
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--eq-step", type=int, required=True)
    parser.add_argument("--sum-lowerbound", type=float, required=True)
    parser.add_argument("--n-epochs", type=int, required=True)
    parser.add_argument("--update-steps", type=int, default=250)
    
    # script behavior arguments
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--network-param-save", action="store_true")
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
