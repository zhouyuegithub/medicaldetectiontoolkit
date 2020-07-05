import torch
import torch.nn as nn
from torch.nn import functional as F

class DiceLoss(nn.Module):
    # Dice Loss
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pre, gt):
        num_classes = pre.shape[1]
        # get one hot ground truth
        gt = gt.float()
        pre = pre.float()

        # dice loss
        eps = 1e-6
        dims = (0,) + tuple(range(2, gt.ndimension())) # (0, 2, 3, 4)
        intersection = torch.sum(pre * gt_one_hot, dims)
        cardinality = torch.sum(pre + gt_one_hot, dims)
        dice = (2. * intersection / (cardinality + eps)).mean()
        loss = 1. - dice

        return loss