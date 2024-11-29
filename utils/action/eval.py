import logging

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
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter

logger = logging.getLogger(__name__)


def evaluate(
    model, criterion, data_loader, print_freq, logger, one_hot=None, encoder=None
):
    """
    Evaluate the performance of a model on a given dataset.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        criterion (callable): The loss function.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the dataset.
        print_freq (int): Frequency of logging the evaluation metrics.
        logger (logging.Logger): Logger for logging the evaluation metrics.
        one_hot (int, optional): Number of classes for one-hot encoding of targets. Defaults to None.
        encoder (callable, optional): Function to encode the input images. Defaults to None.
    Returns:
        tuple: A tuple containing the average loss, top-1 accuracy, and top-5 accuracy.
    """

    model.eval()
    metric_dict = RecordDict({"loss": None, "acc@1": None, "acc@5": None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image, target = image.cuda(), target.cuda()
            if one_hot:
                target = F.one_hot(target, one_hot).float()
            if encoder:
                image = encoder(image)
            output = model(image)
            loss = criterion(output, target)
            metric_dict["loss"].update(loss.item())
            functional.reset_net(model)

            if target.dim() > 1:
                target = target.argmax(-1)
            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_dict["acc@1"].update(acc1.item(), batch_size)
            metric_dict["acc@5"].update(acc5.item(), batch_size)

            if (
                print_freq != 0
                and ((idx + 1) % int(len(data_loader) / print_freq)) == 0
            ):
                # torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(
                    " [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}".format(
                        idx + 1,
                        len(data_loader),
                        metric_dict["loss"].ave,
                        metric_dict["acc@1"].ave,
                        metric_dict["acc@5"].ave,
                    )
                )

    # torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict["loss"].ave, metric_dict["acc@1"].ave, metric_dict["acc@5"].ave
