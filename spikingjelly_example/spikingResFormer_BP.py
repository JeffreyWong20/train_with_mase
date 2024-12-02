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

# ------------------------------------------
from utils.action.train import train_one_epoch
from utils.action.eval import evaluate
from utils.dataset.vision.preprocessing import load_data
from utils.loss_fn.utils import CriterionWarpper
from utils.counter.utils import count_convNd, count_linear, count_matmul
from utils.dataset.vision.augment import DVSAugment
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter
from utils.dataset.utils import RecordDict, GlobalTimer, Timer
from utils.dataset.utils import (
    DatasetSplitter,
    DatasetWarpper,
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

# ------------------------------------------
from chop.nn.snn import functional
from chop.nn.snn import modules as snn_modules
from chop.nn.snn.modules import neuron as snn_neuron
from chop.models.vision.snn import spikingResformer
from chop.nn.snn.modules import Conv1x1, Conv3x3, Linear, SpikingMatmul

# ------------------------------------------
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


def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    distributed, rank, world_size, local_rank = init_distributed(
        logger, args.distributed_init_mode
    )

    logger.info(str(args))

    # load data

    dataset_type = args.dataset
    one_hot = None
    if dataset_type == "CIFAR10":
        num_classes = 10
        input_size = (3, 32, 32)
    elif dataset_type == "CIFAR10DVS":
        one_hot = 10
        num_classes = 10
        input_size = (3, 64, 64)
    elif dataset_type == "DVS128Gesture":
        one_hot = 11
        num_classes = 11
        input_size = (3, 64, 64)
    elif dataset_type == "CIFAR100":
        num_classes = 100
        input_size = (3, 32, 32)
    elif dataset_type == "ImageNet":
        num_classes = 1000
        input_size = (3, 224, 224)
    elif dataset_type == "ImageNet100":
        num_classes = 100
        input_size = (3, 224, 224)
    else:
        raise ValueError(dataset_type)
    if len(args.input_size) != 0:
        input_size = args.input_size

    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path,
        args.batch_size,
        args.workers,
        dataset_type,
        distributed,
        args.augment,
        args.mixup,
        args.cutout,
        args.label_smoothing,
        args.T,
    )
    logger.info(
        "dataset_train: {}, dataset_test: {}".format(
            len(dataset_train), len(dataset_test)
        )
    )

    # model

    model = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size[-1],
    ).cuda()

    # optimzer

    optimizer = create_optimizer_v2(
        model,
        opt=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # loss_fn

    if args.mixup:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)
    criterion_eval = nn.CrossEntropyLoss()
    criterion_eval = CriterionWarpper(criterion_eval)

    # amp speed up

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # lr scheduler

    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=args.epochs,
        cooldown_epochs=10,
        min_lr=1e-5,
        warmup_lr=1e-5,
        warmup_epochs=3,
    )

    # Sync BN
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP

    model_without_ddp = model
    if distributed and not args.test_only:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # custom scheduler

    scheduler_per_iter = None
    scheduler_per_epoch = None

    # resume

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        max_acc1 = checkpoint["max_acc1"]
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        logger.info("Resume from epoch {}".format(start_epoch))
        start_epoch += 1
        # custom scheduler
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   Train
    ##################################################

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(
            os.path.join(args.output_dir, "tensorboard"), purge_step=start_epoch
        )

    logger.info("[Train]")
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)
        logger.info(
            "Epoch [{}] Start, lr {:.6f}".format(epoch, optimizer.param_groups[0]["lr"])
        )

        with Timer(" Train", logger):
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model,
                criterion,
                optimizer,
                data_loader_train,
                logger,
                args.print_freq,
                world_size,
                scheduler_per_iter,
                scaler,
                one_hot,
            )
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()

        with Timer(" Test", logger):
            test_loss, test_acc1, test_acc5 = evaluate(
                model,
                criterion_eval,
                data_loader_test,
                args.print_freq,
                logger,
                one_hot,
            )

        if is_main_process() and tb_writer is not None:
            tb_record(
                tb_writer,
                train_loss,
                train_acc1,
                train_acc5,
                test_loss,
                test_acc1,
                test_acc5,
                epoch,
            )

        logger.info(
            " Test loss: {:.5f}, Acc@1: {:.5f}, Acc@5: {:.5f}".format(
                test_loss, test_acc1, test_acc5
            )
        )

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "max_acc1": max_acc1,
        }
        if lr_scheduler is not None:
            checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        # custom scheduler

        if args.save_latest:
            save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint_latest.pth")
            )

        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint_max_acc1.pth")
            )

    logger.info("Training completed.")

    # ##################################################
    # #                   test
    # ##################################################

    # ##### reset utils #####

    # # reset model

    # del model, model_without_ddp

    # model = create_model(
    #     args.model,
    #     T=args.T,
    #     num_classes=num_classes,
    #     img_size=input_size[-1],
    # )

    # try:
    #     checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'),
    #                             map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    # except:
    #     logger.warning('Cannot load max acc1 model, skip test.')
    #     logger.warning('Exit.')
    #     return

    # # reload data

    # del dataset_train, dataset_test, data_loader_train, data_loader_test
    # _, _, _, data_loader_test = load_data(args.data_path, args.batch_size, args.workers, dataset_type, False,
    #                                       args.augment, args.mixup, args.cutout,
    #                                       args.label_smoothing, args.T)

    # ##### test #####

    # if is_main_process():
    #     test(model, data_loader_test, input_size, args, logger)
    # logger.info('All Done.')


if __name__ == "__main__":
    main()
