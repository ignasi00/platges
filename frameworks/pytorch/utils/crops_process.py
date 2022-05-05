
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def normalize(torch_image, mean, std=None):
    # Substract mean and divide by the standard deviation on each channel
    if std is None:
        for t, m in zip(torch_image, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(torch_image, mean, std):
            t.sub_(m).div_(s)

    return torch_image

def net_process(model, image, mean=0, std=None, flip=True, keep_size=True, return_numpy=True, device=None):
    # image is H, W, C; model input should be C, H, W. output is H, W, C
    device = device or torch.device('cpu')

    if isinstance(image, np.ndarray):
        torch_image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    else:
        torch_image = image.permute(2, 0, 1)

    if mean != 0 and std is not None : torch_image = normalize(torch_image, mean, std)

    torch_image = torch_image.unsqueeze(0).to(device)

    if flip : torch_image = torch.cat([torch_image, torch_image.flip(3)], 0) # Process the image as the mean of the original and a version with the x axis inverted

    with torch.no_grad():
        output = model(torch_image)
    
    if not isinstance(output, torch.Tensor):
        output = output[0]

    if keep_size:
        _, _, h_i, w_i = torch_image.shape
        _, _, h_o, w_o = output.shape
        
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)

    output = F.softmax(output, dim=1)

    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]

    if return_numpy:
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output
    return output.permute(1, 2, 0)

def pad_image_cpu(image, crop_h, crop_w, mean):
    ori_h, ori_w, _ = image.shape

    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    
    pad_h_up = int(pad_h / 2)
    pad_h_down = pad_h - pad_h_up
    pad_w_left = int(pad_w / 2)
    pad_w_right = pad_w - pad_w_left
    
    if pad_h > 0 or pad_w > 0 : image = cv2.copyMakeBorder(image, pad_h_up, pad_h_down, pad_w_left, pad_w_right, cv2.BORDER_CONSTANT, value=mean)
    
    return image, pad_h_up, pad_w_left

def crops_process_numpy(model, image, num_classes, crop_h, crop_w, h=None, w=None, mean=0, std=None, stride_rate=2/3, device=None):
    # Given an image (H, W, C), apply the model to cropped versions of it (in order to comply with the model input size, memory requirments, etc)
    device = device or torch.device('cpu')

    ori_h, ori_w, _ = image.shape
    h = h or ori_h
    w = w or ori_w

    # The image size has to be at least crop_h x crop_w
    image, pad_h_half, pad_w_half = pad_image_cpu(image, crop_h, crop_w, mean)
    new_h, new_w, _ = image.shape

    # Preparing for-loops
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    
    grid_h = int(np.ceil( float(new_h - crop_h) / stride_h ) + 1)
    grid_w = int(np.ceil( float(new_w - crop_w) / stride_w ) + 1)

    # Pre-allocating memory
    prediction_crop = np.zeros((new_h, new_w, num_classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)

    # for-loops
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            # vertical pixels from s_h to e_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            
            s_w = index_w * stride_w
            # horizontal pixels from s_w to e_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            
            # Input to be processed
            image_crop = image[s_h : e_h, s_w : e_w].copy()
            # Overlapped weigths are averaged
            count_crop[s_h : e_h, s_w : e_w] += 1
            # Process
            prediction_crop[s_h : e_h, s_w : e_w, :] += net_process(model, image_crop, mean, std, device=device)

    # Average overlapped weigths
    prediction_crop /= np.expand_dims(count_crop, 2)
    # Crop the added padding
    prediction_crop = prediction_crop[pad_h_half : pad_h_half + ori_h, pad_w_half : pad_w_half + ori_w]
    # Resize it if needed
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)

    return prediction

def torch_batch_crops_process_numpy(model, batch_images, num_classes, crop_h, crop_w, h=None, w=None, mean=0, std=None, stride_rate=2/3, device=None):
    # Batched images have (B, C, H, W) shape, crops_process_numpy needs (H, W, C); it will change into (C, H, W) just before the model input...
    device = device or torch.device('cpu')

    outputs = []
    for i in range(batch_images.shape[0]):
        if isinstance(batch_images, torch.Tensor):
            output = crops_process_numpy(model, batch_images[i].permute(1, 2, 0), num_classes, crop_h, crop_w, h=h, w=w, mean=mean, std=std, stride_rate=stride_rate, device=device)
        else:
            output = crops_process_numpy(model, batch_images[i].transpose(1, 2, 0), num_classes, crop_h, crop_w, h=h, w=w, mean=mean, std=std, stride_rate=stride_rate, device=device)
        outputs.append(output)
    
    return outputs
