
import cv2
import numpy as np

from extern.SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend


def apply_net_eval(model, img, radi):
    # normalize image to "numpy float32 input image in range [0,1]".
    img = np.float32(img) / 255

    pts0, des = model.run(img)[:2]
    pts = tuple(cv2.KeyPoint(int(round(p[0])), int(round(p[1])), int(round(p[2] * radi))) for p in pts0.T)

    return pts, des.T
