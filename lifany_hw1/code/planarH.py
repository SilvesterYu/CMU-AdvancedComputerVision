import numpy as np
import cv2
import random
import copy

def computeH(x1, x2):
    #Q2.2.1
    # TODO: Compute the homography between two sets of points
    # -- x_1i = Hx_2i, A_1h = 0, derive A which is (N*2) rows * 9 columns
    # -- here x_1 and x_2 are locs1 and locs2 after the match
    A = []
    for i in range(len(x1)):
      # -- append the two rows per x_2i
      A.append([x2[i][0], x2[i][1], 1, 0, 0, 0, -x2[i][0]*x1[i][0], -x2[i][1]*x1[i][0], -x1[i][0]])
      A.append([0, 0, 0, x2[i][0], x2[i][1], 1, -x2[i][0]*x1[i][1], -x2[i][1]*x1[i][1], -x1[i][1]])
    A = np.matrix(np.array(A))
    # -- h is the corresponding eigen-vector (column 9 of V) to the smallest eigenvalue of (A^T)A
    U, Sigma, VT = np.linalg.svd(A)
    V = VT.T
    h = V[:, -1]
    H2to1 = np.reshape(h, (3,3))
    return H2to1
    

def computeH_norm(x1, x2):
    #Q2.2.2
    # TODO: Compute the centroid of the points
    c1 = np.mean(x1, axis = 0)
    c2 = np.mean(x2, axis = 0)

    # TODO: Shift the origin of the points to the centroid
    # -- subtract the points with the centroid
    x1_shifted = x1 - c1
    x2_shifted = x2 - c2

    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_shifted_norm = np.linalg.norm(x1_shifted, axis = 1)
    x2_shifted_norm = np.linalg.norm(x2_shifted, axis = 1)
    max_norm1 = np.max(x1_shifted_norm)
    max_norm2 = np.max(x2_shifted_norm)
    scale1 = np.sqrt(2) / max_norm1
    scale2 = np.sqrt(2) / max_norm2
    x1_normalized = x1_shifted * scale1
    x2_normalized = x2_shifted * scale2

    # TODO: Similarity transform 1
    # -- T 3 by 3, with normalization and shifting
    T1 = np.array([
      [scale1, 0, -scale1*c1[0]],
      [0, scale1, -scale1*c1[1]],
      [0, 0, 1]
    ])

    # TODO: Similarity transform 2
    T2 = np.array([
      [scale2, 0, -scale2*c2[0]],
      [0, scale2, -scale2*c2[1]],
      [0, 0, 1]
    ])

    # TODO: Compute homography
    H = computeH(x1_normalized, x2_normalized)

    # TODO: Denormalization
    #H2to1 = np.matmul(np.linalg.inv(T1), np.matmul(H, T2))
    intermediate = np.matmul(H, T2)
    H2to1 = np.matmul(np.linalg.inv(T1), intermediate)
    return H2to1
    

def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    matches_count = len(locs1)
    inlier_count = 0
    # -- idx of best case inliers
    inliers = []
    bestH2to1 = 0

    for i in range(max_iters):
      # -- because we need a minimum of 4 points, we randomly select 4 indices
      selected_idx = random.sample(range(len(locs1)), 4)
      selected_points1 = np.array([locs1[i] for i in selected_idx])
      selected_points2 = np.array([locs2[i] for i in selected_idx])
      # -- exact H produced by the sampled 4 points
      H_selected = computeH_norm(selected_points1, selected_points2)
      # -- the points not selected
      others1 = np.delete(locs1, selected_idx, axis = 0)
      others2 = np.delete(locs2, selected_idx, axis = 0)
      # -- put 1 to augment the vectors
      ones = np.ones((others1.shape[0], 1))
      others2_extend = np.concatenate((others2, ones), axis = 1)
      # -- transform x2 to get predicted x1
      x2_transformed = np.stack([np.matmul(H_selected, others2_extend[j].T) for j in range(others2_extend.shape[0])])
      x2_transformed = (x2_transformed / x2_transformed[:,2])[:, :2]
      # -- compute inliers
      delta = x2_transformed - others1
      delta_norm = np.linalg.norm(delta, axis = 1)
      inliers_idx = []
      for k in range(len(delta_norm)):
        if delta_norm[k] < inlier_tol:
          inliers_idx.append(k)
      if (len(inliers_idx) > inlier_count):
        inlier_count = len(inliers_idx)
        inliers = inliers_idx
        bestH2to1 = H_selected

    if type(bestH2to1) is not int:
      return bestH2to1, inliers
    else:
      return H_selected, [0 for l in range(len(delta_norm))]


def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    
    # TODO: Create mask of same size as template
    temp_height = template.shape[0]
    temp_width = template.shape[1]
    mask = np.ones((temp_height, temp_width, 3))

    # TODO: Warp mask by appropriate homography
    H1to2 = np.linalg.inv(H2to1)
    height = img.shape[0]
    width = img.shape[1]
    dim = (width, height)
    mask_w = cv2.warpPerspective(mask, H1to2, dim)

    # TODO: Warp template by appropriate homography
    template_w = cv2.warpPerspective(template, H1to2, dim)

    # TODO: Use mask to combine the warped template and the image
    composite_img = np.where(mask_w, template_w, img)
    return composite_img









