
import cv2
import numpy as np
import os
from PIL import Image, ImageMath
from skimage import io, color
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.platges_homography_dataset import Platges_DronHomographyDataset

from extern.pyconvsegnet.model.pyconvsegnet import PyConvSegNet
from extern.pyconvsegnet.tool.test import scale_process, colorize


############# TODO: DOCOPT #############
DATA_PATH = '/mnt/c/Users/Ignasi/Downloads/4fotos' #'/mnt/c/Users/Ignasi/Downloads/PlatgesBCN/dron'
SAVE_PATH = '/mnt/c/Users/Ignasi/Downloads/4fotos_seg/'
CLUSTER = True
NUM_IMGS = None # int or None
DOWNSAMPLE = 4

PRETRAINED_PATH = '/home/ignasi/platges/extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth'
LAYERS = 152
CLASSES = 150
ZOOM_FACTOR = 8
BACKBONE_OUTPUT_STRIDE = 8
BACKBONE_NET = "pyconvresnet"

VALUE_SCALE = 255
MEAN = [0.485, 0.456, 0.406]
MEAN = [item * VALUE_SCALE for item in MEAN]
STD = [0.229, 0.224, 0.225]
STD = [item * VALUE_SCALE for item in STD]
CROP_H = 473
CROP_W = 473
BASE_SIZE = 512
SCALES = [1.0]
COLORS = "extern/pyconvsegnet/dataset/ade20k/ade20k_colors.txt"
ADE20K_TO_PLATGES = True

VERBOSE = True
PLOT = False
############# ##### ###### #############


def ade20k_to_platges(np_img):
    water = [9, 21, 26, 37, 109, 113, 128]
    sand = [0, 46, 81, 94] # and 13
    water_id = 21
    sand_id = 46
    for i in water:
        np_img[np_img == i] = water_id
    for i in sand:
        np_img[np_img == i] = sand_id
    return np_img

def apply_net(model, input_, classes, crop_h, crop_w, mean, std, base_size, scales, combine=False, colors=None):
    model.eval()
    
    input_ = input_.numpy()
    #input_ = np.squeeze(input_, axis=0)
    image = np.transpose(input_, (1, 2, 0))
    #h, w, _ = image.shape #TODO: shape is h, w, _ but the code below works well if assuming w, h, _ (output grey or colorize)
    w, h, _ = image.shape

    ########### to keep the same image size
    if base_size == 0:
        base_size = max(h, w)
    ###########

    prediction = np.zeros((h, w, classes), dtype=float)
    for scale in scales:
        long_size = round(scale * base_size)
        new_h = long_size
        new_w = long_size
        if h > w:
            new_w = round(long_size/float(h)*w)
        else:
            new_h = round(long_size/float(w)*h)
        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    
    prediction /= len(scales)
    prediction = np.argmax(prediction, axis=2)
    output = np.uint8(prediction)
    if combine: output = ade20k_to_platges(output)
    if colors is not None: output = colorize(output, colors)

    return output


if __name__ == "__main__":

    os.makedirs(SAVE_PATH, exist_ok = True)

    dataset = Platges_DronHomographyDataset(DATA_PATH, 
                                            to_tensor=True, 
                                            cluster=CLUSTER, 
                                            th_time=60, 
                                            longitude_bin_size=180, 
                                            latitude_bin_size=180, 
                                            min_per_bin=1,
                                            downsample=DOWNSAMPLE,
                                            read_flag=cv2.IMREAD_COLOR)
    def my_collate(x): return x # <- do not transform imgs to tensor here
    dataloader = DataLoader(dataset, collate_fn=my_collate)


    segmentation_net = PyConvSegNet(layers=LAYERS, classes=CLASSES, zoom_factor=ZOOM_FACTOR,
                                    pretrained=False, backbone_output_stride=BACKBONE_OUTPUT_STRIDE,
                                    backbone_net=BACKBONE_NET)

    checkpoint = torch.load(PRETRAINED_PATH)
    segmentation_net.load_state_dict(checkpoint, strict=False)

    colors = None
    if COLORS is not None: colors = np.loadtxt(COLORS).astype('uint8')

    for batch in dataloader:
        for imgs, metas in batch:
            if VERBOSE:
                print('\n'.join([f"{i} - {meta['path']}" for i, meta in enumerate(metas)]))
                print("\n---\n")

            if NUM_IMGS is not None:
                imgs = imgs[0:NUM_IMGS]

            for img, meta in zip(imgs, metas):
                save_path = f"{SAVE_PATH}seg_{os.path.basename(meta['path'])}"

                seg_img = apply_net(segmentation_net, img, CLASSES, CROP_H, CROP_W, MEAN, STD, BASE_SIZE, SCALES, combine=ADE20K_TO_PLATGES, colors=colors)

                if COLORS is None : cv2.imwrite(save_path, seg_img)
                else : seg_img.convert('RGB').save(save_path)

                if VERBOSE:
                    print("-", end='', flush=True)
            print()

