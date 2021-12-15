
import cv2
import numpy as np
import os
from PIL import Image, ImageMath
from skimage import io, color
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.platges_segmentation_dataset import Platges_ArgusNLDataset, LABELS

from extern.pyconvsegnet.model.pyconvsegnet import PyConvSegNet
from extern.pyconvsegnet.tool.test import scale_process, colorize


############# TODO: DOCOPT #############
DATA_PATH = '/mnt/c/Users/Ignasi/Downloads/ArgusNL'
SAVE_PATH = '/mnt/c/Users/Ignasi/Downloads/argus_saved/'
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

ALPHA = 0.5

WATER_ID = 21 # Indexes from ADE20K that does not colide with ArgusNL
SAND_ID = 46
OTHERS_ID = 0

VERBOSE = True
PLOT = False
############# ##### ###### #############


def argus_to_platges(np_img):
    water = [3, 7, 9]
    sand = [1, 4, 5, 8] # and 13
    for i in water:
        np_img[np_img == i] = WATER_ID
    for i in sand:
        np_img[np_img == i] = SAND_ID
    np_img[(np_img != WATER_ID) & (np_img != SAND_ID)] = OTHERS_ID
    return np_img

def ade20k_to_platges(np_img):
    water = [9, 21, 26, 37, 109, 113, 128]
    sand = [0, 46, 81, 94] # and 13
    for i in water:
        np_img[np_img == i] = WATER_ID
    for i in sand:
        np_img[np_img == i] = SAND_ID
    np_img[(np_img != WATER_ID) & (np_img != SAND_ID)] = OTHERS_ID
    return np_img

def apply_net(model, input_, classes, crop_h, crop_w, mean, std, base_size, scales, combine=False, colors=None):
    model.eval()
    
    input_ = input_.numpy()
    #input_ = np.squeeze(input_, axis=0)
    image = np.transpose(input_, (1, 2, 0))
    h, w, _ = image.shape

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


if __name__ == "__main__":

    os.makedirs(SAVE_PATH, exist_ok = True)

    dataset = Platges_ArgusNLDataset(   DATA_PATH,
                                        labels_map=LABELS,
                                        aug=None,
                                        to_tensor=True,
                                        downsample=DOWNSAMPLE,
                                        img_ext=None,
                                        seg_ext=None,
                                        cls_ext=None,
                                        default_value=-1)

    def my_collate(x): return x # <- do not transform imgs to tensor here
    dataloader = DataLoader(dataset, collate_fn=my_collate)


    segmentation_net = PyConvSegNet(layers=LAYERS, classes=CLASSES, zoom_factor=ZOOM_FACTOR,
                                    pretrained=False, backbone_output_stride=BACKBONE_OUTPUT_STRIDE,
                                    backbone_net=BACKBONE_NET)

    checkpoint = torch.load(PRETRAINED_PATH)
    segmentation_net.load_state_dict(checkpoint, strict=False)

    colors = None
    if COLORS is not None: colors = np.loadtxt(COLORS).astype('uint8')

    
    save_path_IoU = f"{SAVE_PATH}/argus_IoUs"
    v_mIoU = []

    for id_batch, batch in enumerate(dataloader):
        for img, segments, classes, img_path in batch:
            if VERBOSE:
                print(f"[{id_batch} / {len(dataset)}] - {img_path}")

            save_path_seg = f"{SAVE_PATH}/seg_{os.path.basename(img_path)}"
            save_path_ovr = f"{SAVE_PATH}/ovr_{os.path.basename(img_path)}"
            #save_path_grt = f"{SAVE_PATH}/grt_{os.path.basename(img_path)}"
            #save_path_gto = f"{SAVE_PATH}/gto_{os.path.basename(img_path)}"

            seg_img = apply_net(segmentation_net, img, CLASSES, CROP_H, CROP_W, MEAN, STD, BASE_SIZE, SCALES, combine=ADE20K_TO_PLATGES, colors=None)

            if colors is not None:
                seg_img_col = colorize(seg_img, colors)
                cv2.imwrite(save_path_seg, seg_img_col)
                img_with_seg = cv2.addWeighted(np.transpose(img.numpy(), (1, 2, 0)), 1-ALPHA, np.array(seg_img_col.convert('RGB')), ALPHA, 0.0)
                cv2.imwrite(save_path_ovr, img_with_seg)

            ground_truth = segments.clone()
            for gt_idx, gt_cls in enumerate(classes):
                ground_truth[segments == gt_idx] = gt_cls
            ground_truth = argus_to_platges(ground_truth)

            img_mIoU = mIoU(ground_truth, seg_img)
            print(f'img_mIoU =  {img_mIoU}')

            v_mIoU.append(img_mIoU)
    
    v_mIoU = np.array(v_mIoU)

    mean_mIoU = np.mean(v_mIoU)
    min_mIoU = np.min(v_mIoU)
    max_mIoU = np.max(v_mIoU)

    print(f'mean_mIou: {mean_mIoU}')
    print(f'min_mIou: {min_mIoU}')
    print(f'max_mIou: {max_mIoU}')

    v_mIoU.dump(f'{save_path_IoU}.npy')
    np.savetxt(f'{save_path_IoU}.txt', v_mIoU, fmt='%.3f')
