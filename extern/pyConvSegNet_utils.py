
import cv2
import numpy as np

from extern.pyconvsegnet.tool.test import scale_process, colorize


def apply_net_cpu(model, input_, classes, crop_h, crop_w, mean, std, base_size, scales):
    # From pyConvSegNet code with small modifications
    model.eval()
    
    input_ = input_.numpy()
    #input_ = np.squeeze(input_, axis=0)
    image = np.transpose(input_, (1, 2, 0)) # shape = H, W, 3
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

    return output
