
import numpy as np
import torch


# TODO: variant with a "presence on target" ponderation
def mIoU(seg_img, ground_truth):

    ground_truth = np.array(ground_truth).ravel()
    seg_img = np.array(seg_img).ravel()
    v_IoU = []

    # only targeded classes are used
    for cls_ in np.unique(ground_truth):
        ground_cls = (ground_truth == cls_)
        seg_cls = (seg_img == cls_)

        intersection = np.sum(seg_cls[ground_cls])
        union = np.sum(seg_cls) + np.sum(ground_cls) - intersection

        v_IoU.append(float(intersection) / float(union))
    
    return np.mean(v_IoU)

# TODO: variant using torch (GPU-friendly)
def torch_mIoU(seg_img, ground_truth):

    ground_truth = ground_truth.view(-1)
    #seg_img = seg_img.argmax(dim=1)
    seg_img = seg_img.view(-1)
    v_IoU = []

    # only targeded classes are used
    for cls_ in torch.unique(ground_truth):
        ground_cls = (ground_truth == cls_)
        seg_cls = (seg_img == cls_)

        intersection = seg_cls[ground_cls].long().sum().cpu()
        union = seg_cls.long().sum().cpu() + ground_cls.long().sum().cpu() - intersection

        v_IoU.append(float(intersection) / float(union))
    
    return np.mean(v_IoU)
