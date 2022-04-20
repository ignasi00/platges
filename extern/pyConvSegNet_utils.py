
import cv2
import numpy as np
import torch

from extern.pyconvsegnet.tool.test import scale_process, colorize


def apply_net_eval_cpu(model, input_, classes, crop_h, crop_w, mean, std, base_size, scales):
    # From pyConvSegNet test code with small modifications
    model.eval()
    
    if not isinstance(input_, np.ndarray) : input_ = input_.numpy()
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
        if h > w:
            new_h = long_size
            new_w = round(long_size / float(h) * w)
        else:
            new_h = round(long_size / float(w) * h)
            new_w = long_size

        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    
    prediction /= len(scales)
    prediction = np.argmax(prediction, axis=2)
    output = np.uint8(prediction)
    output = torch.Tensor(output).type(torch.uint8)

    return output

def apply_net_train(model, input_, train=True, cuda=False):
    # From pyConvSegNet train code with small modifications
    # TODO: model configuration should be applied as little as possible
    model.eval()
    if train:
        for m in model.modules(): # model.modules() yield all modules recursively (it goes inside submodules)
            if m.__class__.__name__.startswith('Dropout') or m.__class__.__name__.startswith('BatchNorm'):
                m.train()
    
    if cuda:
        model = model.cuda() # .to(device) seems more beautiful
        input_ = input_.cuda(non_blocking=True)
    
    output = model(input_)

    return output
