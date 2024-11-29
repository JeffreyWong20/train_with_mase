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

torch.manual_seed(1234)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.dataset.utils import RecordDict
from utils.dataset.vision.preprocessing import load_data
from spikingjelly.activation_based import encoding  # functional, surrogate, layer
from chop.nn.snn import functional
from chop.nn.snn import modules as snn_modules
from chop.nn.snn.modules import neuron as snn_neuron
from chop.models.vision.snn import get_snn_toy
from timm.optim import create_optimizer_v2

# class SNN(nn.Module):
#     def __init__(self, tau):
#         super().__init__()

#         self.layer = nn.Sequential(
#             snn_modules.Flatten(),
#             snn_modules.Linear(28 * 28, 10, bias=False),
#             snn_neuron.LIFNode(tau=tau, surrogate_function=snn_modules.surrogate.ATan()),
#             )

#     def forward(self, x: torch.Tensor):
#         return self.layer(x)


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

    info = {
        "tau": args.tau,
        "num_classes": 10,
        "T": args.T,
        "step_mode": "s",
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

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as args_txt:
        args_txt.write(str(args))
        args_txt.write("\n")
        args_txt.write(" ".join(sys.argv))

    # This encoder will introduce randomness to the input image even the torch.manual_seed(1234) is set at the entry point of this script.
    # TODO: Moving encoder to MASE
    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for idx, (img, label) in enumerate(data_loader_train):
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.0
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.0
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in data_loader_test:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.0
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
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
            f"epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        print(
            f"train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s"
        )
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n'
        )

    # 保存绘图用数据
    net.eval()
    # 注册钩子
    output_layer = net.layer[-1]  # 输出层
    output_layer.v_seq = []
    output_layer.s_seq = []

    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)

    with torch.no_grad():
        img, label = next(iter(data_loader_test))
        img = img[0].to(args.device)
        img = img.to(args.device)
        out_fr = 0.0
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f"Firing rate: {out_spikes_counter_frequency}")

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = (
            output_layer.v_seq.cpu().numpy().squeeze()
        )  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy", v_t_array)
        s_t_array = (
            output_layer.s_seq.cpu().numpy().squeeze()
        )  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy", s_t_array)


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
