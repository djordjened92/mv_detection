import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.loss import FocalLoss


class GaussianMSE(nn.Module):

    def __init__(self):
        super().__init__()
        self.BCEcls = nn.BCEWithLogitsLoss(reduction='sum')
        self.focal = FocalLoss(self.BCEcls, 10., 0.999)

    def forward(self, x, target, kernel):
        #target = self._traget_transform(x, target, kernel)
        loss = self.focal(x, target)
        return loss

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target
