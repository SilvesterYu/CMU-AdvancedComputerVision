import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    ################### TODO Implement Inverse Composition Affine ###################
    # initialize p0
    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    delta_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # interpolate template & current image
    # allign its matrix indices with plotting (x, y) coordinates by transposing It
    Ith, Itw = It.shape[0], It.shape[1]
    yarr = np.arange(0, Ith)
    xarr = np.arange(0, Itw)
    D = Ith*Itw # total number of points
    It_interp = RectBivariateSpline(xarr, yarr, It.T)

    # get gradients of It
    xgrid, ygrid = np.meshgrid(xarr, yarr)
    It_xgrad = It_interp.ev(xgrid, ygrid, dx = 1, dy = 0)
    It_ygrad = It_interp.ev(xgrid, ygrid, dx = 0, dy = 1)

    # construct matrix A from It (Inversed the roles of It and It1)
    # each element: multiply gradient w/ corresponding coordinate vals 
    Idxx = (It_xgrad * xgrid).reshape(D, 1)
    Idxy = (It_xgrad * ygrid).reshape(D, 1)
    Idyx = (It_ygrad * xgrid).reshape(D, 1)
    Idyy = (It_ygrad * ygrid).reshape(D, 1)
    It_xgrad_flat = It_xgrad.reshape(D, 1)
    It_ygrad_flat = It_ygrad.reshape(D, 1)
    A = np.hstack((Idxx, Idxy, It_xgrad_flat, Idyx, Idyy, It_ygrad_flat))
    AT = A.T
    ATA_i = np.linalg.inv(np.matmul(AT, A))

    ### -- update p to minimize error, calculate A * delta_p = b
    delta_p = np.Inf
    it = 0
    while it < num_iters and np.linalg.norm(delta_p) > threshold:
      # find the points within the two images' common area
      It1_warped = affine_transform(It1.T, M).T
      # the common area mask
      mask = np.where(It1_warped > 0, 1, 0)

      # compute b
      b = np.where(mask, It1_warped - It, 0).reshape(D, 1)

      # comptue delta_p
      ATb = np.matmul(AT, b)
      delta_p = np.matmul(ATA_i, ATb)

      # update delta_M using delta_p
      delta_p = delta_p.reshape((6, ))
      delta_M = np.array([[1 + delta_p[0], delta_p[1], delta_p[2]], [delta_p[3], 1 + delta_p[4], delta_p[5]], [0, 0, 1]])
      
      # update M with M and delta_M using composite method
      M = np.matmul(M, np.linalg.inv(delta_M))
      it += 1

    M = M[:2]
    return M










