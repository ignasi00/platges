
import cv2
import numpy as np
from PIL import Image, ImageDraw


def poly2mask(coordinates, width, height):
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(coordinates, outline=1, fill=1)
    mask = np.array(img)
    return mask

def compute_dimensions(composed_img, current_img, homography_matrix):
    h, w = composed_img.shape[:2] # It can be a color image: H W C
    composed_corners = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2).squeeze().astype(int)

    h, w = current_img.shape[:2]
    pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)

    next_corners = cv2.perspectiveTransform(pts, homography_matrix)
    next_corners = np.squeeze(next_corners) # Remove extra dimensions
    next_corners = np.round(next_corners).astype(int)   # Convert to integer

    x_coord = np.concatenate([composed_corners, next_corners]).squeeze()[:, 0]
    y_coord =  np.concatenate([composed_corners, next_corners]).squeeze()[:, 1]

    max_x = x_coord.max()
    min_x = x_coord.min()
    max_y = y_coord.max()
    min_y = y_coord.min()

    return max_x, min_x, max_y, min_y

def _image_idx_iterator(n_matches, base_idx):
    current_idxs = [base_idx]
    seen_pathes = [[base_idx]] * n_matches.shape[0]

    for _ in range(1, n_matches.shape[0]):

        search_mat = n_matches[current_idxs] # search on relevant rows
        search_mat[:, current_idxs] = -1 # discard seen columns

        next_idx = np.unravel_index(np.argmax(search_mat, axis=None), search_mat.shape)

        current_idx = next_idx[1] # matched column
        current_path = seen_pathes[current_idxs[next_idx[0]]] + [current_idx] # path of the matched row (from n_matches) plus the current column

        current_idxs.append(current_idx) # update filter
        current_idxs.sort()
        seen_pathes[current_idx] = current_path # "over"write path for img

        yield current_path

def compute_path_homography(current_idxs, H):
    # TODO: consider cache of sub pathes (current_idxs[:-1]) => recurrent function
    if current_idxs[-1] < current_idxs[-2]:
        if H[current_idxs[-1], current_idxs[-2]] is None : return None
        h = H[current_idxs[-1], current_idxs[-2]] # * image[-1] => image[-1 over -2]
    else:
        if H[current_idxs[-2], current_idxs[-1]] is None : return None
        h = np.linalg.inv(H[current_idxs[-2], current_idxs[-1]]) # * image[-1] => image[-1 over -2]

    for current_idx, step_idx in zip(current_idxs[-2:0:-1], current_idxs[-3::-1]):
        if current_idx < step_idx:
            if H[current_idx, step_idx] is None : return None # For inference is needed
            h = H[current_idx, step_idx] * h # * image[current] => image[current over step] => image[current over step1 over step2]
        else:
            if H[step_idx, current_idx] is None : return None # For inference is needed
            h = np.linalg.inv(H[step_idx, current_idx]) * h # * image[current] => image[current over step] => image[current over step1 over step2]

    return h
