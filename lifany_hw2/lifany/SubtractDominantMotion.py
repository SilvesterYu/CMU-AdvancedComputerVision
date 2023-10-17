import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

import matplotlib.pyplot as plt

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance, dataname, inverse_affine):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################
    if inverse_affine:
      M = InverseCompositionAffine(image1, image2, threshold, num_iters)
      M = np.linalg.inv(np.vstack((M, np.array([0, 0, 1]))))
    else:
      M = LucasKanadeAffine(image1, image2, threshold, num_iters)
      M = np.linalg.inv(np.vstack((M, np.array([0, 0, 1]))))

    image1_warped = affine_transform(image1.T, M).T
    sub = np.absolute(image2 - image1_warped)
    mask = np.where(sub > tolerance, 1, 0)

    # -- for aerial
    if "aerial" in dataname:
      mask = binary_erosion(mask).astype(mask.dtype)
      mask = binary_dilation(mask).astype(mask.dtype)
      mask = binary_dilation(mask).astype(mask.dtype)
      mask = binary_erosion(mask).astype(mask.dtype)

    # -- for ant
    elif "ant" in dataname:
      mask = binary_dilation(mask).astype(mask.dtype)
      mask = binary_erosion(mask).astype(mask.dtype)
      mask = binary_dilation(mask).astype(mask.dtype)

    return mask.astype(bool)
