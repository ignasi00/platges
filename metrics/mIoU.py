
import numpy as np


def mIoU(ground_truth, seg_img):

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
