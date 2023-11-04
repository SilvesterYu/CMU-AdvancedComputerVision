import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

from skimage.io import imsave
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, area_opening, binary_closing
from skimage.color import label2rgb


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    # reference: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
    sigma_est = skimage.restoration.estimate_sigma(image, channel_axis=-1, average_sigmas=True)
    print("image noise", sigma_est)
    denoised_image = skimage.restoration.denoise_bilateral(image, channel_axis=-1)
    grey_image = skimage.color.rgb2gray(denoised_image)

    thresh = threshold_otsu(grey_image)
    bw = binary_closing(grey_image < thresh, square(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label image regions
    label_image = label(cleared)

    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        print("region area", region.area)
        if region.area >= 400 and region.area < 20000:
            # draw rectangle
            print("bbox", region.bbox)
            minr, minc, maxr, maxc = region.bbox
            print("locs", minr, minc, maxr, maxc)
            bboxes.append([minr, minc, maxr, maxc])

    return bboxes, bw