import numpy as np
import cv2
# Import necessary functions
from matchPics import *
from planarH import *
from helper import *
from displayMatch import *

# Q4
def pano(image1, image2, fname, ops):

    # -- create bigger canvas
    blank = np.zeros((image2.shape[0],int(0.5*image2.shape[1]),3), np.uint8)
    canvas = cv2.hconcat([blank, image2])

    # -- compute image 1's transformation onto canvas
    matches, locs1, locs2 = matchPics(image1, canvas, opts)
    plotMatches(image1, canvas, matches, locs1, locs2, "4-match.jpg")
    locs1[:, 0], locs1[:, 1] = locs1[:, 1], locs1[:, 0].copy()
    locs2[:, 0], locs2[:, 1] = locs2[:, 1], locs2[:, 0].copy()
    matched1 = np.array([locs1[item[0]] for item in matches])
    matched2 = np.array([locs2[item[1]] for item in matches])
    H, _ = computeH_ransac(matched2, matched1, opts)

    # -- warp image1 onto canvas
    image1_pred = cv2.warpPerspective(image1, H, (canvas.shape[1], canvas.shape[0]))
    mask = np.ones((image2.shape[0], image2.shape[1], 3))
    mask_w = cv2.warpPerspective(mask, H, (canvas.shape[1], canvas.shape[0]))
    dst = np.where(mask_w, image1_pred, canvas)

    # -- canvas cropping
    stiched_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(stiched_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    max_x = np.max(approx[:, 0, 0])
    max_y = np.max(approx[:, 0, 1])
    min_x = np.min(approx[:, 0, 0])
    min_y = np.min(approx[:, 0, 1])
    output = dst[min_y:max_y,min_x:max_x]
    cv2.imwrite("cropped-"+fname, output)
    

if __name__ == "__main__":
    opts = get_opts()
    image1 = cv2.imread('../data/left6.jpg')
    image2 = cv2.imread('../data/right6.jpg')
    pano(image1, image2, "4-pano-test6.jpg", opts)
    


