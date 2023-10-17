
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ################### TODO Implement Lucas Kanade Affine ###################

    # initialize p0
    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # interpolate template & current image
    # allign its matrix indices with plotting (x, y) coordinates by transposing It
    Ith, Itw = It.shape[0], It.shape[1]
    yarr = np.arange(0, Ith)
    xarr = np.arange(0, Itw)
    D = Ith*Itw # total number of points
    It1_interp = RectBivariateSpline(xarr, yarr, It1.T)

    # get gradients of It1
    xgrid, ygrid = np.meshgrid(xarr, yarr)
    It1_xgrad = It1_interp.ev(xgrid, ygrid, dx = 1, dy = 0)
    It1_ygrad = It1_interp.ev(xgrid, ygrid, dx = 0, dy = 1)

    ### -- update p to minimize error, calculate A * delta_p = b
    delta_p = np.Inf
    it = 0
    while it < num_iters and np.linalg.norm(delta_p) > threshold:
      # I_t+1 of the warped points
      It1_warped = affine_transform(It1.T, M).T
      # the common area mask
      mask = np.where(It1_warped > 0, 1, 0)

      # construct matrix A
      # each element: multiply gradient w/ corresponding coordinate vals 
      Idxx = (np.where(mask, It1_xgrad, 0) * xgrid).reshape(D, 1)
      Idxy = (np.where(mask, It1_xgrad, 0) * ygrid).reshape(D, 1)
      Idyx = (np.where(mask, It1_ygrad, 0) * xgrid).reshape(D, 1)
      Idyy = (np.where(mask, It1_ygrad, 0) * ygrid).reshape(D, 1)
      It1_xgrad = np.where(mask, It1_xgrad, 0)
      It1_ygrad = np.where(mask, It1_ygrad, 0)
      It1_xgrad_flat = It1_xgrad.reshape(D, 1)
      It1_ygrad_flat = It1_ygrad.reshape(D, 1)
      A = np.hstack((Idxx, Idxy, It1_xgrad_flat, Idyx, Idyy, It1_ygrad_flat))

      # construct b
      b = It - It1_warped
      b = np.where(mask, b, 0)
      b = b.reshape(D, 1)

      # compute delta_p
      ATA_i = np.linalg.inv(np.matmul(A.T, A))
      ATb = np.matmul(A.T, b)
      delta_p = np.matmul(ATA_i, ATb)

      # update p0
      delta_p = delta_p.reshape((6, ))
      p += delta_p     

      # update M with p
      M = np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]], [0, 0, 1]])
      it += 1

    M = M[:2]
    return M