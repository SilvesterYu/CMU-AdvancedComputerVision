import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

from planarH import *
import subprocess

def displayMatched(opts, image1, image2, fname):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image1, image2: Source images
    """
    matches, locs1, locs2 = matchPics(image1, image2, opts)

    matched1 = np.array([locs1[item[0]] for item in matches])
    matched2 = np.array([locs2[item[1]] for item in matches])

    #display matched features
    plotMatches(image1, image2, matches, locs1, locs2, fname)

if __name__ == "__main__":

    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    displayMatched(opts, image1, image2, "2_1_4_result.png")


















