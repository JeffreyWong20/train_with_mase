import os
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

from timm.data import FastCollateMixup, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model


def load_data(
    dataset_dir: str,
    batch_size: int,
    workers: int,
    dataset_type: str,
    distributed: bool,
    augment: str,
    mixup: bool,
    cutout: bool,
    label_smoothing: float,
    T: int,
):
    if dataset_type == "MNIST":
        num_classes = 10
        input_size = (1, 28, 28)
    elif dataset_type == "CIFAR10":
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

    if dataset_type == "MNIST":
        dataset_train = torchvision.datasets.MNIST(
            root=os.path.join(dataset_dir),
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        dataset_test = torchvision.datasets.MNIST(
            root=os.path.join(dataset_dir),
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )

        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )

    elif dataset_type == "CIFAR10":
        dataset_train = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), train=True, download=True
        )
        dataset_test = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), train=False, download=True
        )
        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(
                dict(
                    color_jitter=0.0,
                    auto_augment=augment,
                )
            )
        if cutout:
            augment_args.update(
                dict(
                    re_prob=0.25,
                    re_mode="const",
                    re_count=1,
                    re_split=False,
                )
            )
        if mixup:
            augment_args.update(
                dict(
                    collate_fn=FastCollateMixup(
                        mixup_alpha=0.5,
                        cutmix_alpha=0.0,
                        cutmix_minmax=None,
                        prob=1.0,
                        switch_prob=0.5,
                        mode="batch",
                        label_smoothing=label_smoothing,
                        num_classes=num_classes,
                    )
                )
            )
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation="bicubic",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation="bicubic",
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )
    elif dataset_type == "CIFAR100":
        dataset_train = torchvision.datasets.CIFAR100(
            root=os.path.join(dataset_dir), train=True, download=True
        )
        dataset_test = torchvision.datasets.CIFAR100(
            root=os.path.join(dataset_dir), train=False, download=True
        )
        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(
                dict(
                    color_jitter=0.0,
                    auto_augment=augment,
                )
            )
        if cutout:
            augment_args.update(
                dict(
                    re_prob=0.25,
                    re_mode="const",
                    re_count=1,
                    re_split=False,
                )
            )
        if mixup:
            augment_args.update(
                dict(
                    collate_fn=FastCollateMixup(
                        mixup_alpha=0.5,
                        cutmix_alpha=0.0,
                        cutmix_minmax=None,
                        prob=1.0,
                        switch_prob=0.5,
                        mode="batch",
                        label_smoothing=label_smoothing,
                        num_classes=num_classes,
                    )
                )
            )
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation="bicubic",
            mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
            std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation="bicubic",
            mean=[n / 255.0 for n in [129.3, 124.1, 112.4]],
            std=[n / 255.0 for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )
        # elif dataset_type == 'CIFAR10DVS':
        #     from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        #     if augment:
        #         transform_train = DVStransform(transform=transforms.Compose([
        #             transforms.Resize(size=input_size[-2:], antialias=True),
        #             DVSAugment()]))
        #     else:
        #         transform_train = DVStransform(transform=transforms.Compose([
        #             transforms.Resize(size=input_size[-2:], antialias=True)]))
        #     transform_test = DVStransform(
        #         transform=transforms.Resize(size=input_size[-2:], antialias=True))

        #     dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        #     dataset_train, dataset_test = DatasetSplitter(dataset, 0.9,
        #                                                   True), DatasetSplitter(dataset, 0.1, False)
        #     dataset_train = DatasetWarpper(dataset_train, transform_train)
        #     dataset_test = DatasetWarpper(dataset_test, transform_test)
        #     if distributed:
        #         train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
        #             dataset_train)
        #         test_sampler = torch.utils.data.distributed.DistributedSampler(
        #             dataset_test)  # type:ignore
        #     else:
        #         train_sampler = torch.utils.data.RandomSampler(dataset_train)
        #         test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        #     data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
        #                                                     sampler=train_sampler, num_workers=workers,
        #                                                     pin_memory=True, drop_last=True)

        #     data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
        #                                                    sampler=test_sampler, num_workers=workers,
        #                                                    pin_memory=True, drop_last=False)
        # elif dataset_type == 'DVS128Gesture':
        #     from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        #     if augment:
        #         transform_train = DVStransform(transform=transforms.Compose([
        #             transforms.Resize(size=input_size[-2:], antialias=True),
        #             DVSAugment()]))
        #     else:
        #         transform_train = DVStransform(transform=transforms.Compose([
        #             transforms.Resize(size=input_size[-2:], antialias=True)]))
        #     transform_test = DVStransform(
        #         transform=transforms.Resize(size=input_size[-2:], antialias=True))

        #     dataset_train = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T,
        #                                   split_by='number')
        #     dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T,
        #                                  split_by='number')
        #     dataset_train = DatasetWarpper(dataset_train, transform_train)
        #     dataset_test = DatasetWarpper(dataset_test, transform_test)
        #     if distributed:
        #         train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
        #             dataset_train)
        #         test_sampler = torch.utils.data.distributed.DistributedSampler(
        #             dataset_test)  # type:ignore
        #     else:
        #         train_sampler = torch.utils.data.RandomSampler(dataset_train)
        #         test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        #     data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
        #                                                     sampler=train_sampler, num_workers=workers,
        #                                                     pin_memory=True, drop_last=True)

        #     data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
        #                                                    sampler=test_sampler, num_workers=workers,
        #                                                    pin_memory=True, drop_last=False)
        # elif dataset_type == 'ImageNet' or dataset_type == 'ImageNet100':
        traindir = os.path.join(dataset_dir, "train")
        valdir = os.path.join(dataset_dir, "val")
        dataset_train = torchvision.datasets.ImageFolder(traindir)
        dataset_test = torchvision.datasets.ImageFolder(valdir)
        augment_args = dict(
            scale=[0.08, 1.0],
            ratio=[3.0 / 4.0, 4.0 / 3.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(
                dict(
                    color_jitter=0.4,
                    auto_augment=augment,
                )
            )
        if cutout:
            augment_args.update(
                dict(
                    re_prob=0.25,
                    re_mode="const",
                    re_count=1,
                    re_split=False,
                )
            )
        if mixup:
            augment_args.update(
                dict(
                    collate_fn=FastCollateMixup(
                        mixup_alpha=0.2,
                        cutmix_alpha=1.0,
                        cutmix_minmax=None,
                        prob=1.0,
                        switch_prob=0.5,
                        mode="batch",
                        label_smoothing=label_smoothing,
                        num_classes=num_classes,
                    )
                )
            )
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation="bicubic",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation="bicubic",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            distributed=distributed,
            crop_pct=0.95,
            pin_memory=True,
        )
    else:
        raise ValueError(dataset_type)

    return dataset_train, dataset_test, data_loader_train, data_loader_test
