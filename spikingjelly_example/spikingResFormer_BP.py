# -------------------------------------------
# Code adapted from https://github.com/xyshi2000/SpikingResformer
# ------------------------------------------
import os
import sys
import time
import yaml
import random
import logging
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

import torchvision
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter

# from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.distributed

import argparse
from thop import profile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

# -------------
from chop.nn.snn import functional
from chop.nn.snn import modules as snn_modules
from chop.nn.snn.modules import neuron as snn_neuron

# -------------

from chop.nn.snn.modules import Conv1x1, Conv3x3, Linear, SpikingMatmul
from utils.counter.utils import count_convNd, count_linear, count_matmul

from utils.dataset.vision.augment import DVSAugment
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter

from utils.dataset.utils import RecordDict, GlobalTimer, Timer
from utils.dataset.utils import (
    DatasetSplitter,
    DatasetWarpper,
    CriterionWarpper,
    DVStransform,
    SOPMonitor,
)
from utils.dataset.utils import (
    is_main_process,
    save_on_master,
    tb_record,
    accuracy,
    safe_makedirs,
)

from spikingjelly.activation_based import functional, layer, base

from timm.data import FastCollateMixup, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model


def parse_args():
    config_parser = argparse.ArgumentParser(
        description="Training Config", add_help=False
    )

    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description="Training")

    # training options
    parser.add_argument("--seed", default=12450, type=int)
    parser.add_argument("--epochs", default=320, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--T", default=4, type=int, help="simulation steps")
    parser.add_argument("--model", default="spikingresformer_ti", help="model type")
    parser.add_argument("--dataset", default="ImageNet", help="dataset type")
    parser.add_argument("--augment", type=str, help="data augmentation")
    parser.add_argument("--mixup", type=bool, default=False, help="Mixup")
    parser.add_argument("--cutout", type=bool, default=False, help="Cutout")
    parser.add_argument(
        "--label-smoothing", type=float, default=0, help="Label smoothing"
    )
    parser.add_argument(
        "--workers", default=16, type=int, help="number of data loading workers"
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--weight-decay", default=0, type=float, help="weight decay")

    parser.add_argument("--data-path", default="/data/datasets")
    parser.add_argument("--output-dir", default="./logs/temp")

    parser.add_argument(
        "--print-freq",
        default=5,
        type=int,
        help="Number of times a debug message is printed in one epoch",
    )
    parser.add_argument("--resume", type=str, help="resume from checkpoint")
    parser.add_argument(
        "--transfer", type=str, help="transfer from pretrained checkpoint"
    )
    parser.add_argument("--input-size", type=int, nargs="+", default=[])
    parser.add_argument("--distributed-init-mode", type=str, default="env://")

    # argument of TET
    parser.add_argument("--TET", action="store_true", help="Use TET training")
    parser.add_argument("--TET-phi", type=float, default=1.0)
    parser.add_argument("--TET-lambda", type=float, default=0.0)

    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument("--amp", type=bool, default=True, help="Use AMP training")
    parser.add_argument("--sync-bn", action="store_true", help="Use SyncBN training")

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args


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


def init_distributed(logger: logging.Logger, distributed_init_mode):
    """
    NOTE: Update documentation for distributed training
    Initialize distributed training for PyTorch
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])  #
        world_size = int(os.environ["WORLD_SIZE"])  #
        local_rank = int(os.environ["LOCAL_RANK"])  #
    else:
        logger.info("Not using distributed mode")
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = "nccl"
    logger.info("Distributed init rank {}".format(rank))
    torch.distributed.init_process_group(
        backend=backend,
        init_method=distributed_init_mode,
        world_size=world_size,
        rank=rank,
    )
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank
