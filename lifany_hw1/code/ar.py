import numpy as np
import cv2

#Import necessary functions
from opts import get_opts
from matchPics import *
from planarH import *
from helper import *
from displayMatch import *
from HarryPotterize import *

def incorporate_vid(src_path, dst_path, obj_path, save_f, opts):

    # -- retrieve frames
    frames_src = loadVid(src_path) # the panda frames
    frames_dst = loadVid(dst_path) # the book frames
    obj = cv2.imread(obj_path) # the cv book cover pic
    frames = [] # to store all the composite frames

    # -- match and warp frame by frame
    min_len = min(len(frames_src), len(frames_dst))
    for i in range(min_len):
      img = frames_src[i] # panda frame
      img2 = frames_dst[i] # book frame
      positions = np.nonzero(img)
      top, bottom = 55, 320 # remove the top and bottom dark spaces
      left, right = int(img.shape[1]/3), int(img.shape[1]*(2/3))
      img = img[top:bottom, left:right]

      # -- warp resized panda to book frame
      try:
        composite_img = warpImage(obj, img2, img, str(i) + 'result.png', opts)
        frames.append(composite_img)
        print("processed frame ", i)
      except:
        print("failed on frame ", i)
        continue

    # -- save to video
    out = cv2.VideoWriter("4-output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (img2.shape[1], img2.shape[0]))
    for frame in frames:
        out.write(frame) # frame is a numpy.ndarray with shape (length, width, 3)
    out.release()


#Write script for Q3.1
if __name__ == "__main__":
    opts = get_opts()
    incorporate_vid("../data/ar_source.mov", "../data/book.mov", "../data/cv_cover.jpg", "3-1-result.mov", opts)










