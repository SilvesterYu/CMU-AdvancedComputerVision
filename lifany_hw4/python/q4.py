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
    thresh = skimage.filters.threshold_otsu(grey_image)
    print("denoised image", denoised_image.shape)
    bw = skimage.morphology.closing(grey_image > thresh, skimage.morphology.square(3))
    imsave("bw.png", bw)
    print("bw", bw.shape)
    # cleared = skimage.segmentation.clear_border(bw)
    # imsave("bwcleared.png", cleared)
    label_image = skimage.measure.label(bw)
    # image_label_overlay = skimage.color.label2rgb(label_image, image=image, bg_label=0)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)
    # breakpoint()
    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        print("region area", region.area)
        if region.area >= 100:
            # draw rectangle
            print("bbox", region.bbox)
            minr, minc, maxr, maxc = region.bbox
            print("locs", minr, minc, maxr, maxc)
            bboxes.append([minc, minr, maxc, maxr])
    print(bboxes)

    return bboxes, bw
