
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw
from tqdm import tqdm


def poly2mask(coordinates, width, height):
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(coordinates, outline=1, fill=1)
    mask = np.array(img)
    return mask

def batch_homographies(kp, des, matrix_finder, full=False, get_matches=False, VERBOSE=False):
    H = dict()
    n_matches = np.zeros((len(kp), len(kp)))
    matches = dict() # VERBOSE use

    if VERBOSE : tqdm_bar = tqdm(range( 2 * ( (len(kp) - 1) * len(kp) ) // 2 )) # leaving 2/2 because its 2 updates times num iterations
    for idx1, (kp1, des1) in enumerate(zip(kp, des)):
        if full : H.update( {(idx1, idx1) : np.eye(3)} )
        for idx2, (kp2, des2) in enumerate(zip(kp[idx1 + 1:], des[idx1 + 1:])):
            if VERBOSE : tqdm_bar.update(1)
            
            # find the homography matrixs and matches_info
            h, img_matches, inliers_mask = matrix_finder(kp1, des1, kp2, des2)
            H.update( {(idx1, idx2 + idx1 + 1) : np.matrix(h)} )

            if get_matches: matches.update({(idx1, idx2 + idx1 + 1) : img_matches}) # VERBOSE use
            if full:
                try:
                    H.update( {(idx2 + idx1 + 1, idx1) : np.linalg.inv(np.matrix(h))} )
                except:
                    H.update( {(idx2 + idx1 + 1, idx1) : None} )

            try:
                n_matches[idx1, idx2 + idx1 + 1] = sum(inliers_mask)
            except:
                n_matches[idx1, idx2 + idx1 + 1] = 0
            
            if VERBOSE : tqdm_bar.update(1)

    n_matches = n_matches + n_matches[::-1, ::-1]

    if get_matches : return H, n_matches, matches
    else : return H, n_matches

class BatchOverlapFinder():

    def __init__(self, point_finder_descriptor=None, homography_matrix_estimator=None, matrix_applier=None, training=False):
        """ point_finder_descriptor defines how to find and caracterize interest points. 
            homography_matrix_estimator defines how to estimate the homography matrix from the previous interest points.
            matrix_applier defines the output from applying the homography matrix (by instance, a point between 4 pixels may be rounded).
            
            kp, des = point_finder_descriptor(ima1, None)
            H12, matches, inliers_mask = homography_matrix_estimator(kp1, des1, kp2, des2) # if no H found, H == None
            img = homography_applier(img, h, size)
        """

        self.training = training

        self.descriptor = point_finder_descriptor
        if self.descriptor is None:
            # points of interest with histogram of gradient not direction dependent descriptors
            descriptor = cv2.SIFT_create()
            self.descriptor = lambda img, other : descriptor.detectAndCompute(img, other)
    
        self.matrix_finder = homography_matrix_estimator
        if self.matrix_finder is None:
            bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
            self.matrix_finder = RANSAC_MatrixFinder(point_matcher=bf, MIN_MATCH_COUNT=4, ransacReprojThreshold=5.0, maxIters=5)

        self.matrix_applier = matrix_applier
        if self.matrix_applier is None:
            self.matrix_applier = lambda img, h, size : cv2.warpPerspective(img, h, size, flags=cv2.INTER_LINEAR)

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def _image_idx_iterator(self, n_matches, base_idx, VERBOSE=False):
        current_idxs = [base_idx]
        seen_pathes = [[base_idx]] * n_matches.shape[0]

        if VERBOSE:
            print('\n********')
            print(n_matches, end='\n********\n')

        for _ in range(1, n_matches.shape[0]):

            search_mat = n_matches[current_idxs] # search on relevant rows
            search_mat[:, current_idxs] = -1 # discard seen columns

            next_idx = np.unravel_index(np.argmax(search_mat, axis=None), search_mat.shape)

            current_idx = next_idx[1] # matched column
            current_path = seen_pathes[current_idxs[next_idx[0]]] + [current_idx] # path of the matched row (from n_matches) plus the current column

            current_idxs.append(current_idx) # update filter
            current_idxs.sort()
            seen_pathes[current_idx] = current_path # "over"write path for img

            if VERBOSE:
                print('\n****')
                print(search_mat, end='\n****\n')
                print(current_path, end='\n¨¨¨¨¨\n')

            yield current_path
    
    def _compute_path_homography(self, current_idxs, H):
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

    def _compute_dimensions(self, composed_img, current_img, homography_matrix):
        h, w = composed_img.shape
        composed_corners = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2).squeeze().astype(int)

        h, w = current_img.shape
        pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        next_corners = cv2.perspectiveTransform(pts, homography_matrix)
        next_corners = np.squeeze(next_corners) # Remove extra dimensions
        next_corners = np.round(next_corners).astype(int)   # Convert to integer

        x_coord = np.concatenate([composed_corners, next_corners]).squeeze()[:,0]
        y_coord =  np.concatenate([composed_corners, next_corners]).squeeze()[:,1]

        max_x = x_coord.max()
        min_x = x_coord.min()
        max_y = y_coord.max()
        min_y = y_coord.min()

        return max_x, min_x, max_y, min_y

    def train_head(self, x, H, kp, base_idx, n_matches, matches=None, VERBOSE_TQDM=False, VERBOSE_MAT=False, VERBOSE_PLOT=False):
        # Stitch as much images over the base_idx image plane as posible
        # base_idx = len(x) - 1
        composed_img = x[base_idx].copy()

        # Translation of the base plane from previous stiched images
        translation = np.eye(3, 3)

        if VERBOSE_TQDM : tqdm_bar = tqdm(range(1, len(kp)))
        for current_idxs in self._image_idx_iterator(n_matches, base_idx, VERBOSE=VERBOSE_MAT):
            if VERBOSE_TQDM : tqdm_bar.update(1)

            current_img = x[current_idxs[-1]].copy()

            if VERBOSE_PLOT : self._plot_matches(x, kp, matches, current_idxs)

            # 1- compute the homography matrix for the current image (chained homographies with previous translations)
            homography_matrix = self._compute_path_homography(current_idxs, H)
            if homography_matrix is None : break
            homography_matrix = translation * homography_matrix

            # 2- compute the new dimensions
            max_x, min_x, max_y, min_y = self._compute_dimensions(composed_img, current_img, homography_matrix)
            width = int(max_x - min_x + 1)
            height = int(max_y - min_y + 1)

            # 3- compute the transformation matrix for the current image (homography matrix from step 1 with current trtanslation)
            translation_matrix = np.matrix([[1.0, 0.0, -min_x],[0.0, 1.0, -min_y],[0.0, 0.0, 1.0]])
            homography_matrix = translation_matrix * homography_matrix # * image

            # 4- save the current translation
            translation = translation_matrix * translation

            # 5- applying the transformation
            composed_img = self.matrix_applier(composed_img, translation_matrix, (width, height))
            current_img = self.matrix_applier(current_img, homography_matrix, (width, height))
            composed_img[composed_img == 0] = current_img[composed_img == 0]

        return composed_img

    def infere_head(self, x, H, n_matches, VERBOSE_TQDM=False, VERBOSE_MAT=False):
        # Idea: overlap of A & B in A can be used to get the overlap in B. Currently: computed individually using corners to define polygons

        paired_overlaps = dict()

        def shape_gen(x):
            for x_i in x: yield x_i.shape
        corner_coordinates_original = [((0, 0), (x_c - 1, 0), (x_c - 1, y_c - 1), (0, y_c - 1)) for x_c, y_c in shape_gen(x)]
        #corner_coordinates = {(i, i) : corner_coordinates_original[i] for i in len(x)}

        if VERBOSE_TQDM : tqdm_bar = tqdm(range(len(x) * (len(x) - 1)))
        for base_idx in range(len(x)):

            max_x, max_y = corner_coordinates_original[base_idx][2]
            paired_overlaps.update({ (base_idx, base_idx) : np.ones( (max_x + 1, max_y + 1) ) })

            for current_idxs in self._image_idx_iterator(n_matches, base_idx, VERBOSE=VERBOSE_MAT):
                if VERBOSE_TQDM : tqdm_bar.update(1)

                i = base_idx
                j = current_idxs[-1]

                h = self._compute_path_homography(current_idxs, H)
                if h is None:
                    #corner_coordinates.update({ (i, j) : corner_coordinates_original[i] + corner_coordinates_original[i][2] + 1 })
                    paired_overlaps.update({ (i, j) : np.zeros( (max_x + 1, max_y + 1) ) })
                else:
                    j_on_i_coordiantes = cv2.perspectiveTransform(np.float32(corner_coordinates_original[j]).reshape(-1,1,2), h)
                    j_on_i_coordiantes = np.squeeze(j_on_i_coordiantes) # Remove extra dimensions
                    j_on_i_coordiantes = np.round(j_on_i_coordiantes).astype(int)   # Convert to integer

                    j_on_i_coordiantes[ j_on_i_coordiantes < 0] = 0
                    j_on_i_coordiantes[ j_on_i_coordiantes[:, 0] > max_x , 0] = max_x
                    j_on_i_coordiantes[ j_on_i_coordiantes[:, 1] > max_y , 1] = max_y

                    #corner_coordinates.update({ (i, j) : j_on_i_coordiantes})
                    paired_overlaps.update({ (i, j) : poly2mask(j_on_i_coordiantes.flatten().tolist(), max_x + 1, max_y + 1) })
    
        return paired_overlaps

    def forward(self, x, VERBOSE_TQDM=False, VERBOSE_MAT=False, VERBOSE_PLOT=False, descriptor_arg2=None, base_idx=0):
        # find points of interestand its descriptors
        if hasattr(descriptor_arg2, '__len__') and len(x) == len(descriptor_arg2):
            y = zip(x, descriptor_arg2)
        else:
            y = zip(x, [descriptor_arg2] * len(x))
        kp, des = zip(*[self.descriptor(img, arg2) for img, arg2 in y])

        # find homography matrix and number of consistent matches
        H, n_matches, matches = batch_homographies(kp, des, matrix_finder=self.matrix_finder, get_matches=True, VERBOSE=VERBOSE_TQDM)

        if self.training:
            # We want to evaluate the ability of generating the original image given a base just cropped and others cropped and altered (how many?)
            composed_img = self.train_head(x, H, kp, base_idx, n_matches, matches, VERBOSE_TQDM=VERBOSE_TQDM, VERBOSE_MAT=VERBOSE_MAT, VERBOSE_PLOT=VERBOSE_PLOT)
            return composed_img, n_matches
        else:
            # We want to get masks of overlapped areas: overlap[img_to_be_masked_idx, img_that_overlaps_idx] = mask; img_seq is the 
            overlaps = self.infere_head(x, H, n_matches, VERBOSE_TQDM=VERBOSE_TQDM, VERBOSE_MAT=VERBOSE_MAT)
            return overlaps, n_matches

    __call__ = forward

    ###### VERBOSE functions ######

    def _plot_matches(self, x, kp, matches, current_idxs):
        import matplotlib.pyplot as plt

        try:
            matches12   = matches[current_idxs[-1], current_idxs[-2]] # may raise key error if not defined
            img1        = x[current_idxs[-1]]
            kp1         = kp[current_idxs[-1]]
            img2        = x[current_idxs[-2]]
            kp2         = kp[current_idxs[-2]]
        except:
            matches12   = matches[current_idxs[-2], current_idxs[-1]]
            img1        = x[current_idxs[-2]]
            kp1         = kp[current_idxs[-2]]
            img2        = x[current_idxs[-1]]
            kp2         = kp[current_idxs[-1]]

        img12 = cv2.drawMatches(img1, kp1, img2, kp2, matches12, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        def display_image(img, title='', size=None, show_axis=False):
            plt.gray()
            if not show_axis : plt.axis('off')
            h = plt.imshow(img, interpolation='none')
            if size:
                dpi = h.figure.get_dpi()/size
                h.figure.set_figwidth(img.shape[1] / dpi)
                h.figure.set_figheight(img.shape[0] / dpi)
                #h.figure.canvas.resize(img.shape[1] + 1, img.shape[0] + 1)
                h.axes.set_position([0, 0, 1, 1])
                if show_axis:
                    h.axes.set_xlim(-1, img.shape[1])
                    h.axes.set_ylim(img.shape[0], -1)
            plt.grid(False)
            plt.title(title)  
            plt.show()

        display_image(img12, 'matches', size=1)
        input()

class RANSAC_MatrixFinder():
        def __init__(self, point_matcher, MIN_MATCH_COUNT, ransacReprojThreshold=5.0, maxIters=5):
            # ransacReprojThreshold is inliear/outlier th, If kp1, kp2 measured in pixels: th usually somewhere between 1 and 10.
            self.point_matcher = point_matcher
            self.MIN_MATCH_COUNT = MIN_MATCH_COUNT
            self.ransacReprojThreshold = ransacReprojThreshold
            self.maxIters = maxIters

        def __call__(self, kp1, des1, kp2, des2):
            # kp is key_points; des is descriptors

            # least mean squeres matching
            matches = self.point_matcher.match(des1, des2)
            # RANSAC iterative adjust of matches
            if len(matches) > self.MIN_MATCH_COUNT:
                src_pts12 = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
                dst_pts12 = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts12, dst_pts12, cv2.RANSAC, self.ransacReprojThreshold, maxIters=self.maxIters)
                return H, matches, mask
            return None, matches, None
