
import cv2
import numpy as np
import torch

from .crops_process import crops_process_numpy


def scales_process_numpy(model, input_, num_classes, crop_h, crop_w, mean, std, scales, base_size=0, stride_rate=2/3, device=None):
    # input_ shape is C, H, W; the image processing by crops function needs H, W, C; output is H, W (values are argmax(C))
    device = device or torch.device('cpu')

    if not isinstance(input_, np.ndarray) : input_ = np.asarray(input_)
    image = np.transpose(input_, (1, 2, 0)) # shape -> H, W, C
    h, w, _ = image.shape

    ########### to keep the same image size
    if base_size == 0:
        base_size = max(h, w)
    ###########

    prediction = np.zeros((h, w, num_classes), dtype=float)
    for scale in scales:

        long_size = round(scale * base_size)
        if h > w:
            new_h = long_size
            new_w = round(long_size / float(h) * w)
        else:
            new_h = round(long_size / float(w) * h)
            new_w = long_size
        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        prediction += crops_process_numpy(model, image_scaled, num_classes, crop_h, crop_w, h=h, w=w, mean=mean, std=std, stride_rate=stride_rate, device=device)
    
    prediction /= len(scales)
    prediction = np.argmax(prediction, axis=2)
    output = np.uint8(prediction)

    return output

def torch_batch_scales_process_numpy(model, batch_images, num_classes, crop_h, crop_w, mean, std, scales, base_size=0, stride_rate=2/3, device=None):
    # Batched images have (B, C, H, W) shape; outputs are (B, H, W)
    device = device or torch.device('cpu')

    outputs = []
    for i in range(batch_images.shape[0]):
        output = scales_process_numpy(model, batch_images[i], num_classes, crop_h, crop_w, mean, std, scales, base_size=base_size, stride_rate=stride_rate, device=device)
        outputs.append(output)
    
    return outputs

