
import cv2
import numpy as np


BGR_GRAY = (120, 120, 120)
BGR_RED = (0, 0, 255)
BGR_BLUE = (255, 0, 0)
BGR_YELLOW = (0, 255, 255)
BGR_HALF_GREEN = (0, 128, 0)

STR_COLOR = {
    'GRAY' : BGR_GRAY,
    'BLUE' : BGR_BLUE,
    'YELLOW' : BGR_YELLOW
}


def color_cls(mask, cls_, color):
    mask_b = (mask == cls_).astype(np.uint8) * 255
    mask_c = cv2.cvtColor(mask_b, cv2.COLOR_GRAY2BGR)
    mask_c[mask_b == 255] = color
    return mask_c

def save_rgb_seg(path, output, classes_color):
    
    blend_img = np.zeros(output.shape).astype(np.uint8)
    blend_img = cv2.cvtColor(blend_img, cv2.COLOR_GRAY2BGR)
    for cls_, name in classes_color.items():
        ch_mask = color_cls(output, cls_, STR_COLOR.get(name, (0, 0, 0)))
        blend_img = cv2.addWeighted(blend_img, 1, ch_mask, 1, 0)
    
    cv2.imwrite(path, blend_img)

def save_rgb_err(path, output, target):
    error_mask = (output != target).astype(np.uint8) * 255
    error_mask_color = cv2.cvtColor(error_mask, cv2.COLOR_GRAY2BGR)
    error_mask_color[error_mask == 255] = BGR_RED
    
    cv2.imwrite(path, error_mask_color)

def save_rgb_ovr_seg(path, input_, output, classes_color):
    blend_img = np.zeros(output.shape).astype(np.uint8)
    blend_img = cv2.cvtColor(blend_img, cv2.COLOR_GRAY2BGR)
    for cls_, name in classes_color.items():
        ch_mask = color_cls(output, cls_, STR_COLOR.get(name, (0, 0, 0)))
        blend_img = cv2.addWeighted(blend_img, 1, ch_mask, 1, 0)

    #input_ = cv2.cvtColor(input_, cv2.COLOR_RGB2BGR)
    input_ = cv2.cvtColor(input_, cv2.COLOR_RGB2GRAY)
    input_ = cv2.cvtColor(input_, cv2.COLOR_GRAY2BGR)
    blend_img = cv2.addWeighted(input_, 0.85, blend_img, 0.15, 0)

    cv2.imwrite(path, blend_img)

def save_rgb_ovr_err(path, input_, output, target, classes_color):
    error_mask = (output != target).astype(np.uint8) * 255
    error_mask_color = cv2.cvtColor(error_mask, cv2.COLOR_GRAY2BGR)
    error_mask_color[error_mask == 255] = BGR_RED

    #input_ = cv2.cvtColor(input_, cv2.COLOR_RGB2BGR)
    input_ = cv2.cvtColor(input_, cv2.COLOR_RGB2GRAY)
    input_ = cv2.cvtColor(input_, cv2.COLOR_GRAY2BGR)
    blend_img = cv2.addWeighted(input_, 0.7, error_mask_color, 0.3, 0)

    cv2.imwrite(path, blend_img)
