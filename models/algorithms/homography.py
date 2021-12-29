
import cv2
import numpy as np


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

# TODO: Make a TorchOverlap class that compute overlaps through homography matrix and can implement nn.Module
class BasicStitching():

    def __init__(self, point_finder_descriptor=None, homography_matrix_estimator=None, matrix_applier=None):
        # kp, des = point_finder_descriptor(ima1, None)
        # H12, matches, inliers_mask = homography_matrix_estimator(kp1, des1, kp2, des2) # if no H found, H == None
        # img = homography_applier(img, h, size)

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

    def _find_all_homographies(self, kp, des, verbose=False):
        #H = np.zeros((len(kp), len(kp)), 3, 3)
        H = dict()
        n_matches = np.zeros((len(kp), len(kp)))
        matches = dict() # VERBOSE use

        for idx1, (kp1, des1) in enumerate(zip(kp, des)):
            if verbose : print("-", end='', flush=True)
            for idx2, (kp2, des2) in enumerate(zip(kp[idx1 + 1:], des[idx1 + 1:])):
                
                # find the homography matrixs and matches_info
                h, img_matches, inliers_mask = self.matrix_finder(kp1, des1, kp2, des2)

                matches.update({(idx1, idx2 + idx1 + 1) : img_matches}) # VERBOSE use
                
                H.update({(idx1, idx2 + idx1 + 1) : np.matrix(h)})
                #H[idx1, idx2 + idx1 + 1] = h
                try:
                    n_matches[idx1, idx2 + idx1 + 1] = sum(inliers_mask)
                except:
                    n_matches[idx1, idx2 + idx1 + 1] = 0
                
                if verbose : print("+-", end='', flush=True)
        if verbose : print("\n")
        n_matches = n_matches + n_matches[::-1, ::-1]

        return H, n_matches, matches

    def _image_idx_iterator(self, n_matches, base_idx, verbose=False):
        current_idxs = [base_idx]
        seen_pathes = [[base_idx]] * n_matches.shape[0]

        if verbose : print('\n********')
        if verbose : print(n_matches, end='\n********\n')

        for _ in range(1, n_matches.shape[0]):

            search_mat = n_matches[current_idxs] # search on relevant rows
            search_mat[:, current_idxs] = -1 # discard seen columns

            next_idx = np.unravel_index(np.argmax(search_mat, axis=None), search_mat.shape)

            current_idx = next_idx[1] # matched column
            current_path = seen_pathes[current_idxs[next_idx[0]]] + [current_idx] # path of the matched row (from n_matches) plus the current column

            current_idxs.append(current_idx) # update filter
            current_idxs.sort()
            seen_pathes[current_idx] = current_path # "over"write path for img

            if verbose : print('\n****')
            if verbose : print(search_mat, end='\n****\n')
            if verbose : print(current_path, end='\n¨¨¨¨¨\n')
            yield current_path
    
    def _compute_path_homography(self, current_idxs, H):
        if current_idxs[-1] < current_idxs[-2]:
            if H[current_idxs[-1], current_idxs[-2]] is None : return None
            h = H[current_idxs[-1], current_idxs[-2]] # * image[-1] => image[-1 over -2]
        else:
            if H[current_idxs[-2], current_idxs[-1]] is None : return None
            h = np.linalg.inv(H[current_idxs[-2], current_idxs[-1]]) # * image[-1] => image[-1 over -2]

        for current_idx, step_idx in zip(current_idxs[-2:0:-1], current_idxs[-3::-1]):
            if current_idx < step_idx:
                h = H[current_idx, step_idx] * h # * image[current] => image[current over step] => image[current over step1 over step2]
            else:
                h = np.linalg.inv(H[step_idx, current_idx]) * h # * image[current] => image[current over step] => image[current over step1 over step2]

        return h

    def forward(self, x, verbose=False, plot=False, descriptor_arg2=None):
        # find points of interestand its descriptors
        if hasattr(descriptor_arg2, '__len__') and len(x) == len(descriptor_arg2):
            y = zip(x, descriptor_arg2)
        else:
            y = zip(x, [descriptor_arg2] * len(x))
        kp, des = zip(*[self.descriptor(img, arg2) for img, arg2 in y])

        # find homography matrix and number of consistent matches
        H, n_matches, matches = self._find_all_homographies(kp, des, verbose=verbose)

        # Stitch as much images over the base_idx image plane as posible
        base_idx = len(x) - 1
        composed_img = x[base_idx].copy()

        # Translation of the base plane from previous stiched images
        translation = np.eye(3, 3)

        if verbose : print("-", end='', flush=True)
        for current_idxs in self._image_idx_iterator(n_matches, base_idx, verbose=verbose):
            current_img = x[current_idxs[-1]].copy()

            if verbose and plot : self._plot_matches(x, kp, matches, current_idxs)

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

            if verbose : print("+-", end='', flush=True)
        if verbose : print("\n")

        return composed_img

    __call__ = forward

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

