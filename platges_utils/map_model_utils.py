
import cv2
from exif import Image
import numpy as np

from .map_model import MapModel, context_map_model


def unitary_square_geodesic_distance(x, y):
    # x, y are vectors of GPS coordinates (latitude, longitude)=(phi, lambda)
    deltas = x - y
    delta_phi = deltas[..., 0]
    delta_lambda = deltas[..., 1]

    phi_m = (x[..., 0] + y[..., 0]) / 2
    return delta_phi ** 2 + (np.cos(phi_m) * delta_lambda) ** 2

def gps_search_map_model(gps, map_model_filename_list, distance_funct=None, limit=0):
    # limit = minimum distance between map_models is good
    gps = np.asarray(gps) # shape 2 x 1
    assert gps.shape == (2, )

    distance_funct = distance_funct or unitary_square_geodesic_distance

    best = None
    best_score = np.inf
    for filename in map_model_filename_list:
        with context_map_model(filename) as map_model:
            model_gps = np.array(map_model.get_elements()[3]) # shape N x 2
            score = np.min(distance_funct(np.asarray(model_gps), np.asarray(gps)))
            
            if score < best_score:
                best_score = score
                best = filename
        
        if best_score <= limit : break
    return best

def gps_search_loaded_map_model(gps, map_model_list, distance_funct=None, limit=0):
    # limit = minimum distance between map_models is good
    gps = np.asarray(gps) # shape 2 x 1
    assert gps.shape == (2, )

    distance_funct = distance_funct or unitary_square_geodesic_distance

    best = None
    best_score = np.inf
    for map_model in map_model_list:
        
        model_gps = np.array(map_model.get_elements()[3]) # shape N x 2
        score = np.min(distance_funct(np.asarray(model_gps), np.asarray(gps)))
        
        if score < best_score:
            best_score = score
            best = map_model
        
        if best_score <= limit : break
    return best

def apply_map_model(image, img_gps, map_model, homography_estimator, mask_translator, image_mask=None, distance_funct=None, k_nearest=None, max_distance=None, undefined_class=0):
    # map_model already open or created by data on the scope

    elements = map_model.get_elements()
    #elements = zip(images, masks, homographies, gps)

    if distance_funct is not None:
        elements = list(zip(*sorted(zip(*elements), key=lambda elem : distance_funct(np.asarray(elem[3]), np.asarray(img_gps)), reverse=True)))
        elements = [e for e in elements[:k_nearest] if e < max_distance] # equivalent to the first ¿? elements (with ¿? <= k_nearest)
    
    map_images, map_masks, map_homographies, _ = elements

    imgs = [image]
    imgs.extend(map_images)

    masks = None
    if image_mask is not None:
        masks = [image_mask]
        masks.extend(map_masks)

    H, _, _, n_matches = homography_estimator(imgs, masks=masks)
    best_idx = np.argmax(n_matches[0])
    model_homography = map_homographies[best_idx - 1]
    h = np.matmul(np.matmul(model_homography, np.linalg.inv(H[best_idx])), H[0])

    _, map_masks, map_homographies, _ = map_model.get_elements()
    order = n_matches[0, 1:] # reverse=False
    _, map_masks, map_homographies = zip( *sorted( zip(order, map_masks, map_homographies), key=lambda elem : elem[0] ) )
    h_inv = np.linalg.inv(h)
    semisupervised_mask = np.ones(image.shape[:2]) * undefined_class
    for mask_orig, h_model in zip(map_masks, map_homographies):
        h_mask_translation = h_inv * h_model
        mask_component = mask_translator(mask_orig, semisupervised_mask, h_mask_translation)

        semisupervised_mask[semisupervised_mask == undefined_class] = mask_component[semisupervised_mask == undefined_class]

    name = map_model.get_name()
    return semisupervised_mask, h, name

def platja_list_applier(image_filename, map_model_filename_list, homography_estimator, mask_translator, image_mask=None, distance_funct=None, k_nearest=None, max_distance=None, undefined_class=0, nearness_limit=0, read_flag=cv2.IMREAD_COLOR):
    # masks allows to use the additional_mask to improve the homography (then start programs with semantic segmentation)
    with open(image_filename, 'rb') as src:
        img_exif = Image(src)
        gps_to_sec = lambda gps : gps[0] + gps[1] / 60 + gps[2] / 3600
        img_gps = (gps_to_sec(img_exif.gps_latitude), gps_to_sec(img_exif.gps_longitude))
    
    map_model_filename = gps_search_map_model(img_gps, map_model_filename_list, distance_funct=distance_funct, limit=nearness_limit)

    image = cv2.imread(image_filename, read_flag) # (#row, #col, #color) (shape = H, W, 3)
    if read_flag == cv2.IMREAD_COLOR : image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with context_map_model(map_model_filename) as map_model:
        semisupervised_mask, h, name = apply_map_model(image, img_gps, map_model, homography_estimator, mask_translator, image_mask, distance_funct, k_nearest, max_distance, undefined_class)
    
    return semisupervised_mask, h, name, image, map_model_filename

def normalized_correlation(mask1, mask2):
    return np.sum(mask1 * mask2) / (np.sum(mask1 ** 2) * np.sum(mask2 ** 2))

def refine_segmentation(base_mask, additional_mask, similarity_funct=None, undefined_classes=None):

    undefined_classes = undefined_classes or [0]
    
    mask = base_mask.copy()
    additional_mask = cv2.resize(additional_mask, (base_mask.shape[1], base_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    for undefined_class in undefined_classes:
        base_mask[base_mask == undefined_class] = additional_mask[base_mask == undefined_class]

    score = None
    if similarity_funct is not None: # A good similarity_funct is the normalized correlation
        # External Warning if bad score?
        score = similarity_funct(base_mask, additional_mask)
    
    return mask, score

