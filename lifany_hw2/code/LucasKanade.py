import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################

    # -- reminder: scipy ndimage.shift for fractional movement, rect bivariate spline
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.ev.html

    p = p0
    Ith, Itw = It.shape[0], It.shape[1]
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    
    # interpolate template & current image
    # allign its matrix indices with plotting (x, y) coordinates by transposing It
    yarr = np.arange(0, Ith, 1)
    xarr = np.arange(0, Itw, 1)
    It_interp = RectBivariateSpline(xarr, yarr, It.T)
    It1_interp = RectBivariateSpline(xarr, yarr, It1.T)

    # get points inside the rectangle
    It_xrect =  np.arange(x1, x2, 1)
    It_yrect =  np.arange(y1, y2, 1)
    D = len(It_xrect)*len(It_yrect)
    It_xgrid, It_ygrid =  np.meshgrid(It_xrect, It_yrect, indexing="ij")
    
    # interpolated values inside the template's rectangle
    It_interp_rect = It_interp.ev(It_xgrid, It_ygrid).reshape((D, 1))

    ### -- update p to minimize error
    # A * delta_p = b
    delta_p = np.Inf
    it = 0
    while it < num_iters and np.linalg.norm(delta_p) > threshold:

      # current image: points inside the rectangle into a meshgrid
      It1_xrect =  np.arange(x1, x2, 1) + p[0] # x coordinates
      It1_yrect =  np.arange(y1, y2, 1) + p[1] # y coordinates
      It1_xgrid, It1_ygrid =  np.meshgrid(It1_xrect, It1_yrect, indexing="ij")

      # compute matrix A from gradient of I over (x, y) in current image's rectangle
      It1_xgrad = It1_interp.ev(It1_xgrid, It1_ygrid, dx = 1, dy = 0).reshape((D, 1))
      It1_ygrad = It1_interp.ev(It1_xgrid, It1_ygrid, dx = 0, dy = 1).reshape((D, 1))
      A = np.hstack((It1_xgrad, It1_ygrad))

      # compute b
      It1_interp_rect = It1_interp.ev(It1_xgrid, It1_ygrid).reshape((D, 1))
      b = It_interp_rect - It1_interp_rect

      # compute delta_p
      ATA_i = np.linalg.inv(np.matmul(A.T, A))
      ATb = np.matmul(A.T, b)
      delta_p = np.matmul(ATA_i, ATb)

      # update p0
      delta_p = delta_p.reshape((2, ))
      p += delta_p
      it += 1

    return p
