import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
from matchPics import *
from planarH import *
from helper import *
from displayMatch import *

# Q2.2.4
def warpImage(image1, image2, template, fname, opts):

    # -- compute Homography
    matches, locs1, locs2 = matchPics(image1, image2, opts)
    locs1[:, 0], locs1[:, 1] = locs1[:, 1], locs1[:, 0].copy()
    locs2[:, 0], locs2[:, 1] = locs2[:, 1], locs2[:, 0].copy()
    matched1 = np.array([locs1[item[0]] for item in matches])
    matched2 = np.array([locs2[item[1]] for item in matches])
    bestH2to1, _ = computeH_ransac(matched1, matched2, opts)

    # -- resize template
    h1 = image1.shape[0]
    w1 = image1.shape[1]
    dim = (w1, h1)
    temp_resized = cv2.resize(template, dim)

    # -- composite the images
    composite_img = compositeH(bestH2to1, temp_resized, image2)
    cv2.imwrite(fname, composite_img)
    return composite_img

if __name__ == "__main__":
    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')
    cover = cv2.imread('../data/hp_cover.jpg')
    warpImage(image1, image2, cover, '2-2-4-result.png', opts)


