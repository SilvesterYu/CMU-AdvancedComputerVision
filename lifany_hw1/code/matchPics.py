import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4
def matchPics(I1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        
        # TODO: Convert Images to GrayScale
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        
        # TODO: Detect Features in Both Images
        I1_corners = corner_detection(I1, sigma)
        I2_corners = corner_detection(I2, sigma)
        
        # TODO: Obtain descriptors for the computed feature locations
        I1_descriptor, locs1 = computeBrief(I1, I1_corners)
        I2_descriptor, locs2 = computeBrief(I2, I2_corners)

        # TODO: Match features using the descriptors
        matches = briefMatch(I1_descriptor, I2_descriptor, ratio)
        print("Number of matches: ", len(matches))
        
        return matches, locs1, locs2




















