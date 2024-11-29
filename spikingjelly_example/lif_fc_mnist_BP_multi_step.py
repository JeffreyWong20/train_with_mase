import os
import time
import argparse
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.dataset.utils import RecordDict, Timer
from utils.dataset.vision.preprocessing import load_data
from spikingjelly.activation_based import encoding  # functional, surrogate, layer
from chop.nn.snn import functional
from chop.nn.snn import modules as snn_modules
from chop.nn.snn.modules import neuron as snn_neuron
from chop.models.vision.snn import get_snn_toy
from timm.optim import create_optimizer_v2

from utils.action.train import train_one_epoch
from utils.action.eval import evaluate
import logging


def setup_logger(output_dir):
    """
    Initialize logger
    """
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s]%(message)s", datefmt=r"%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(os.path.join(output_dir, "log.log"))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def main():
    """
    :return: None

    * :ref:`API in English <lif_fc_mnist.main-en>`

    .. _lif_fc_mnist.main-en:

    The network with FC-LIF structure for classifying MNIST.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    """
    parser = argparse.ArgumentParser(description="LIF MNIST Training")
    parser.add_argument("-T", default=100, type=int, help="simulating time-steps")
    parser.add_argument("-device", default="cuda:0", help="device")
    parser.add_argument("-b", default=64, type=int, help="batch size")
    parser.add_argument(
        "-epochs",
        default=1,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-data-dir",
        type=str,
        default="/data/datasets",
        help="root dir of MNIST dataset",
    )
    parser.add_argument(
        "-out-dir",
        type=str,
        default="./logs",
        help="root dir for saving logs and checkpoint",
    )
    parser.add_argument("-resume", type=str, help="resume from the checkpoint path")
    parser.add_argument(
        "-amp", action="store_true", help="automatic mixed precision training"
    )
    parser.add_argument(
        "-opt",
        type=str,
        choices=["sgd", "adam"],
        default="adam",
        help="use which optimizer. SGD or Adam",
    )
    parser.add_argument("-momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("-lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument(
        "-tau", default=2.0, type=float, help="parameter tau of LIF neuron"
    )

    args = parser.parse_args()
    print(args)

    # net = SNN(tau=args.tau)

    info = {
        "tau": args.tau,
        "T": args.T,
        "num_classes": 10,
        "step_mode": "m",
    }

    net = get_snn_toy(info)

    print(net)

    net.to(args.device)
    dataset_type = "MNIST"
    torch.manual_seed(1234)
    train_dataset, test_dataset, data_loader_train, data_loader_test = load_data(
        args.data_dir,
        args.b,
        args.j,
        dataset_type,
        distributed=False,
        augment=None,
        mixup=False,
        cutout=False,
        label_smoothing=0,
        T=0,
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = create_optimizer_v2(
        net,
        opt=args.opt,
        lr=args.lr,
        weight_decay=0,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        max_test_acc = checkpoint["max_test_acc"]

    out_dir = os.path.join(args.out_dir, f"T{args.T}_b{args.b}_{args.opt}_lr{args.lr}")

    if args.amp:
        out_dir += "_amp"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Mkdir {out_dir}.")

    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"), purge_step=start_epoch)

    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as args_txt:
        args_txt.write(str(args))
        args_txt.write("\n")
        args_txt.write(" ".join(sys.argv))

    encoder = encoding.PoissonEncoder()
    logger = setup_logger(out_dir)

    distributed, rank, world_size, local_rank = False, 0, 1, 0
    lr_scheduler = None
    scheduler_per_epoch = None

    criterion = F.mse_loss
    info = {
        "tau": args.tau,
        "T": args.T,
        "num_classes": 10,
        "step_mode": "s",
    }
    for epoch in range(start_epoch, args.epochs):
        with Timer(" Train", logger):
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                net,
                criterion,
                optimizer,
                data_loader_train,
                logger,
                5,
                world_size,
                None,
                scaler=scaler,
                one_hot=10,
                encoder=encoder,
            )
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc1", train_acc1, epoch)

        with Timer(" Test", logger):
            test_loss, test_acc1, test_acc5 = evaluate(
                net,
                criterion,
                data_loader_test,
                5,
                logger,
                one_hot=10,
                encoder=encoder,
            )

        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_acc1", test_acc1, epoch)

        save_max = False
        if test_acc1 > max_test_acc:
            max_test_acc = test_acc1
            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "max_test_acc": max_test_acc,
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, "checkpoint_max.pth"))

        torch.save(checkpoint, os.path.join(out_dir, "checkpoint_latest.pth"))

        print(args)
        print(out_dir)
        print(
            f"epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc1: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc1: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        # print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        # print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # Save data for plotting
    net.eval()
    # register forward hook to save v and s
    output_layer = net.layer[-1]  # output layer
    output_layer.v_seq = []
    output_layer.s_seq = []

    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)

    with torch.no_grad():
        img, label = next(iter(data_loader_test))
        img = img[0].to(args.device)
        net.step_mode = "s"
        print(img.shape)
        out_fr = 0.0
        for t in range(10):
            encoded_img = encoder(img)
            encoded_img = encoded_img.unsqueeze(0)  # add time dimension
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()

        print(out_spikes_counter_frequency.shape)
        print(f"Firing rate: {out_spikes_counter_frequency}")

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = (
            output_layer.v_seq.cpu().numpy().squeeze()
        )  # v_t_array[i][j] represents the voltage value of neuron i at time j
        print(v_t_array.shape)
        np.save("v_t_array.npy", v_t_array)
        s_t_array = (
            output_layer.s_seq.cpu().numpy().squeeze()
        )  # s_t_array[i][j] represents the spikes released by neuron i at time j, 0 or 1
        np.save("s_t_array.npy", s_t_array)
        print(s_t_array.shape)


if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
file_path = "v_t_array.npy"  # Replace with your file path
data = np.load(file_path)

# Plot the 2D array as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(data, aspect="auto", cmap="viridis")
plt.colorbar(label="Value")
plt.title("2D Data Heatmap")
plt.xlabel("Neuron Index")
plt.ylabel("Time Step")
plt.savefig("v_t_array.png")

file_path = "s_t_array.npy"  # Replace with your file path
data = np.load(file_path)
# Plot the 2D array as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(data, aspect="auto", cmap="viridis")
plt.colorbar(label="Value")
plt.title("2D Data Heatmap")
plt.xlabel("Neuron Index")
plt.ylabel("Time Step")
plt.savefig("s_t_array.png")
