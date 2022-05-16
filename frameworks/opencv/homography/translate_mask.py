
import cv2

from .utils import compute_dimensions


def translate_mask(mask_orig, base_img, h):
    # The mask is cropped in order to fit the base_img window

    translated_mask = cv2.warpPerspective(mask_orig, h, base_img.shape[:2], flags=cv2.INTER_LINEAR)

    return translated_mask

def comparable_masks(mask_orig, mask_base, h):
    max_x, min_x, max_y, min_y = compute_dimensions(maks1, mask2, h)
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)

    translation_matrix = np.matrix([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]])
    ht = translation_matrix * h # * image

    mask_base_translated = cv2.warpPerspective(mask_base, translation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    mask_transformed = cv2.warpPerspective(mask_orig, ht, (width, height), flags=cv2.INTER_LINEAR)

    return mask_transformed, mask_base_translated
