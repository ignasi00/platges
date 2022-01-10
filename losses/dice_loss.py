
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, reduce=None):
        super(DiceLoss, self).__init__()
        
        self.reduction = reduce or torch.mean

    def forward(self, input_, target):
        # input_ of size (#imgs, classes, h, w) and target of size (#imgs, h, w)

        # Estimate class probabilities (between 0 and 1)
        input_soft = F.softmax(input_, dim=1)
        
        # targets into input format
        target_hot = F.one_hot(target.to(torch.int64), num_classes=input_.shape[1]).permute(0, 3, 1, 2)

        # 2 * P * R / (P + R) = 2 * TP / (2 * TP + FP + FN) = 2 * input * target / (sum(input) + sum(target))
        dims = (1, 2, 3)
        dice_coeff = 2 * torch.sum(input_soft * target_hot, dims) / (torch.sum(input_soft + target_hot, dims))
        # if target has values, dice_coeff is not NaN; else NaN means a perfect "not" detection
        dice_coeff = torch.nan_to_num(dice_coeff, nan=1.0, posinf=1.0, neginf=1.0)
        # Nevertheless, at the moment this loss can not be negative

        return self.reduction(1 - dice_coeff)

