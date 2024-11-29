import logging
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from chop.nn.snn import functional


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
from utils.dataset.vision.augment import DVSAugment
from utils.scheduler import BaseSchedulerPerIter

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader_train: torch.utils.data.DataLoader,
    logger: logging.Logger,
    print_freq: int,
    factor: int,
    scheduler_per_iter: Optional[BaseSchedulerPerIter] = None,
    scaler: Optional[GradScaler] = None,
    one_hot: Optional[int] = None,
    encoder=None,
):
    """
    Train the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader_train (torch.utils.data.DataLoader): DataLoader for the training data.
        logger (logging.Logger): Logger for logging training progress.
        print_freq (int): Frequency of logging the training progress.
        factor (int): Factor for calculating iterations per second.
        scheduler_per_iter (Optional[BaseSchedulerPerIter], optional): Scheduler to update learning rate per iteration. Defaults to None.
        scaler (Optional[GradScaler], optional): Gradient scaler for mixed precision training. Defaults to None.
        one_hot (Optional[int], optional): If specified, convert targets to one-hot encoding with this number of classes. Defaults to None.
    Returns:
        Tuple[float, float, float]: Average loss, top-1 accuracy, and top-5 accuracy for the epoch.
    """

    model.train()
    metric_dict = RecordDict({"loss": None, "acc@1": None, "acc@5": None})
    timer_container = [0.0]

    optimizer.zero_grad()
    for idx, (image, target) in enumerate(data_loader_train):
        with GlobalTimer("iter", timer_container):
            image, target = image.cuda(), target.cuda()

            if one_hot:
                target = F.one_hot(target, one_hot).float()
            if encoder is not None:
                image = encoder(image)
            if scaler is not None:
                with autocast():
                    output = model(image)
                    loss = criterion(output.mean(0), target)
            else:
                output = model(image)
                loss = criterion(output.mean(0), target)

            metric_dict["loss"].update(loss.item())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                model.zero_grad()

            if scheduler_per_iter is not None:
                scheduler_per_iter.step()

            functional.reset_net(model)

            if target.dim() > 1:
                target = target.argmax(-1)
            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            acc1_s = acc1.item()
            acc5_s = acc5.item()

            batch_size = image.shape[0]
            metric_dict["acc@1"].update(acc1_s, batch_size)
            metric_dict["acc@5"].update(acc5_s, batch_size)

        if (
            print_freq != 0
            and ((idx + 1) % int(len(data_loader_train) / (print_freq))) == 0
        ):
            # torch.distributed.barrier()
            metric_dict.sync()
            logger.debug(
                " [{}/{}] it/s: {:.5f}, loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}".format(
                    idx + 1,
                    len(data_loader_train),
                    (idx + 1) * batch_size * factor / timer_container[0],
                    metric_dict["loss"].ave,
                    metric_dict["acc@1"].ave,
                    metric_dict["acc@5"].ave,
                )
            )

    # torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict["loss"].ave, metric_dict["acc@1"].ave, metric_dict["acc@5"].ave
