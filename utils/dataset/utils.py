import os
import time
from typing import Tuple, Union
from chop.nn.snn.modules.spiking_self_attention import SpikingMatmul
import torch
import torch.distributed
import torch.utils.data
import errno
import datetime
from torch import Tensor, nn
from math import nan
from torch.utils.tensorboard.writer import SummaryWriter

import sys

sys.path.append("..")


def is_distributed():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_rank():
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def tb_record(
    tb_writer: SummaryWriter,
    train_loss: float,
    train_acc1: float,
    train_acc5: float,
    test_loss: float,
    test_acc1: float,
    test_acc5: float,
    epoch: int,
):
    tb_writer.add_scalar("train/loss", train_loss, epoch)
    tb_writer.add_scalar("train/acc1", train_acc1, epoch)
    tb_writer.add_scalar("train/acc5", train_acc5, epoch)
    tb_writer.add_scalar("test/loss", test_loss, epoch)
    tb_writer.add_scalar("test/acc1", test_acc1, epoch)
    tb_writer.add_scalar("test/acc5", test_acc5, epoch)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class Record:
    r"""
    Synchronous record
    """

    def __init__(self, test: bool = False) -> None:
        self.value = torch.tensor([0], dtype=torch.float64, device="cuda")
        self.count = torch.tensor([0], dtype=torch.int64, device="cuda")
        self.global_value = 0.0
        self.global_count = 0
        self.test = test

    def sync(self) -> None:
        r"""
        reduce value and count, and update global ones
        """
        if is_distributed() and not self.test:
            torch.distributed.all_reduce(self.value, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.count, torch.distributed.ReduceOp.SUM)
        self.global_value += self.value.item()
        self.global_count += self.count.item()
        self.value[0] = 0.0
        self.count[0] = 0

    def update(self, value, count=1) -> None:
        r"""
        update local value and count
        """
        self.value[0] += value * count
        self.count[0] += count

    def reset(self) -> None:
        self.value[0] = 0.0
        self.count[0] = 0
        self.global_value = 0.0
        self.global_count = 0

    @property
    def ave(self):
        if self.global_count == 0:
            return nan
        return self.global_value / self.global_count


class RecordDict:
    r"""
    Synchronous record dict for distributed training
    """

    def __init__(self, dic: dict, test: bool = False) -> None:
        self.__inner_dict = dict()
        self.test = test
        for key in dic.keys():
            self.__inner_dict[key] = Record(test)

    def __getitem__(self, key) -> Record:
        return self.__inner_dict[key]

    def __setitem__(self, key, value) -> None:
        assert isinstance(value, Record)
        self.__inner_dict[key] = value

    def __str__(self) -> str:
        s = []
        for key, value in self.__inner_dict.items():
            s.append("{key}:{value}".format(key=key, value=value.ave))
        return ", ".join(s)

    def sync(self):
        for value in self.__inner_dict.values():
            value.sync()

    def reset(self):
        for value in self.__inner_dict.values():
            value.reset()

    def add_record(self, key):
        self.__inner_dict[key] = Record(self.test)


class Timer:
    def __init__(self, timer_name, logger):
        self.timer_name = timer_name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.logger.debug(
            "{} spent: {}.".format(
                self.timer_name, str(datetime.timedelta(seconds=int(self.interval)))
            )
        )

        # train_speed = train_samples / (train_time - start_time)
        # train_loss /= train_samples
        # train_acc /= train_samples


class GlobalTimer:
    r"""
    Global timer to time the whole training process
    """

    def __init__(self, timer_name, container):
        self.timer_name = timer_name
        self.container = container

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.container[0] += self.interval


class DatasetSplitter(torch.utils.data.Dataset):
    """To split CIFAR10DVS into training dataset and test dataset"""

    def __init__(self, parent_dataset, rate=0.1, train=True):

        self.parent_dataset = parent_dataset
        self.rate = rate
        self.train = train
        self.it_of_original = len(parent_dataset) // 10
        self.it_of_split = int(self.it_of_original * rate)

    def __len__(self):
        return int(len(self.parent_dataset) * self.rate)

    def __getitem__(self, index):
        base = (index // self.it_of_split) * self.it_of_original
        off = index % self.it_of_split
        if not self.train:
            off = self.it_of_original - off - 1
        item = self.parent_dataset[base + off]

        return item


class DatasetWarpper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.trasnform = transform

    def __getitem__(self, index):
        return self.trasnform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        img = img.view(shape)
        img3 = img.sum(dim=1, keepdim=True)
        img = torch.cat([img, img3], dim=1)
        return img


def unpack_for_conv(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x.flatten(0, 1)


def unpack_for_linear(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x.flatten(0, 1)


def unpack_for_matmul(
    x: Union[Tuple[torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor]:
    assert isinstance(x, tuple)
    assert x.__len__() == 2
    left, right = x
    return left.flatten(0, 1), right.flatten(0, 1)


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        for name, m in net.named_modules():
            if name in net.skip:  # type:ignore
                continue
            if isinstance(m, nn.Conv2d):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                self.hooks.append(
                    m.register_forward_hook(self.create_hook_conv(name))
                )  # type:ignore
            elif isinstance(m, nn.Linear):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                self.hooks.append(
                    m.register_forward_hook(self.create_hook_linear(name))
                )  # type:ignore
            elif isinstance(m, SpikingMatmul):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                self.hooks.append(
                    m.register_forward_hook(self.create_hook_matmul(name))
                )  # type:ignore

    def cal_sop_conv(self, x: Tensor, m: nn.Conv2d):
        with torch.no_grad():
            out = torch.nn.functional.conv2d(
                x,
                torch.ones_like(m.weight),
                None,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
            )
            return out.sum().unsqueeze(0)

    def create_hook_conv(self, name):
        def hook(m: nn.Conv2d, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_conv(unpack_for_conv(x).detach(), m))

        return hook

    def cal_sop_linear(self, x: Tensor, m: nn.Linear):
        with torch.no_grad():
            out = torch.nn.functional.linear(x, torch.ones_like(m.weight), None)
            return out.sum().unsqueeze(0)

    def create_hook_linear(self, name):
        def hook(m: nn.Conv2d, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.cal_sop_linear(unpack_for_linear(x).detach(), m)
                )

        return hook

    def cal_sop_matmul(self, left: Tensor, right: Tensor, m: SpikingMatmul):
        with torch.no_grad():
            if m.spike == "l":
                right = torch.ones_like(right)
            elif m.spike == "r":
                left = torch.ones_like(left)
            elif m.spike == "both":
                pass
            else:
                raise ValueError(m.spike)
            out = torch.matmul(left, right)
            return out.sum().unsqueeze(0)

    def create_hook_matmul(self, name):
        def hook(m: nn.Conv2d, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                left, right = unpack_for_matmul(x)
                self.records.append(
                    self.cal_sop_matmul(left.detach(), right.detach(), m)
                )

        return hook
