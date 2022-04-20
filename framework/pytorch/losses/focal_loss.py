
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=2 ,reduce=None):
        super(FocalLoss, self).__init__()
        
        self.gamma = gamma
        self.reduction = reduce or torch.mean

    def forward(self, input_, target):
        ce_loss = F.cross_entropy(input_, target, reduction='none') # logSoftmax (probs) and negative likelihood (not NaNs)
        pt = torch.exp(-ce_loss)                                    # correctly prediction probability (multiclass cross_entropy and focal_loss does not use error probabilities)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss             # focal loss definition

        return self.reduction(focal_loss)
        