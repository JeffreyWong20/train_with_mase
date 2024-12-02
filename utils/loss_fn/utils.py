import os
import time
from typing import Tuple, Union
import torch
import torch.distributed
import torch.utils.data
import errno
import datetime
from torch import Tensor, nn
from math import nan
from torch.utils.tensorboard.writer import SummaryWriter


class CriterionWarpper(nn.Module):
    def __init__(self, criterion, TET=False, TET_phi=1.0, TET_lambda=0.0) -> None:
        super().__init__()
        self.criterion = criterion
        self.TET = TET
        self.TET_phi = TET_phi
        self.TET_lambda = TET_lambda
        self.mse = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.TET:
            loss = 0
            for t in range(output.shape[0]):
                loss = loss + (1.0 - self.TET_lambda) * self.criterion(
                    output[t], target
                )
            loss = loss / output.shape[0]
            if self.TET_lambda != 0:
                loss = loss + self.TET_lambda * self.mse(
                    output, torch.zeros_like(output).fill_(self.TET_phi)
                )
            return loss
        else:
            return self.criterion(output.mean(0), target)
