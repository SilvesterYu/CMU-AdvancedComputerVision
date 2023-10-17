import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts

from scipy import ndimage, misc
import matplotlib.pyplot as plt
from displayMatch import *

#Q2.1.6
def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    I = cv2.imread('../data/cv_cover.jpg')
    degrees = []
    counts = []

    for i in range(36):

        # TODO: Rotate Image
        d = 10*i + 10
        print("rotation: ", d)
        I_rotate = ndimage.rotate(I, d, mode = 'constant')

        # TODO: Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(I, I_rotate, opts)
        match_count = len(matches)
    
        # TODO: Update histogram
        degrees.append(d)
        counts.append(match_count)
        displayMatched(opts, I, I_rotate, str(i)+"rot.png")

    # TODO: Display histogram
    fig, ax = plt.subplots()
    ax.set_xlabel('rotation', fontweight ='bold')
    ax.set_ylabel('number of matches', fontweight ='bold')
    ax.bar([str(item) for item in degrees], counts)
    plt.savefig("2_1_6_result.png")
    plt.show()

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
    
