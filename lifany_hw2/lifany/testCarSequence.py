import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import scipy
import matplotlib
from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

# -- my test code
# plot image with rectangle
def save_plot(im, r, fname):
  fig, ax = plt.subplots(figsize=plt.figaspect(im))
  ax.set_axis_off()
  plt.gray()
  ax.imshow(im)
  x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
  w = x2 - x1
  h = y2 - y1
  rect_patch = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
  ax.add_patch(rect_patch)
  plt.savefig(fname, bbox_inches='tight', pad_inches=0)
  
# list of all the rectangles
rects = [rect]

# iterate through each image, get delta p and plot rectangles
img_count = seq.shape[2]
for i in range(1, img_count):
  It = seq[:, :, i-1]
  It1 = seq[:, :, i]
  p = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))
  x1, y1, x2, y2  = rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]
  rect = [x1, y1, x2, y2]
  rects.append(rect)
  if i == 1 or i % 100 == 0:
    save_plot(It1, rect, '1-3-car-iter' + str(i) + '.png')

# save the rectangles as .npy file
np.save('carseqrects.npy', rects)
