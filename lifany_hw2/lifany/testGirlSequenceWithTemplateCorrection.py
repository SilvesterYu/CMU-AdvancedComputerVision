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
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

# -- my test code

# plot image with rectangle
def save_plot(im, r, rp, fname):
  fig, ax = plt.subplots(figsize=plt.figaspect(im))
  ax.set_axis_off()
  plt.gray()
  ax.imshow(im)
  x1, y1, x2, y2, x1p, y1p, x2p, y2p = r[0], r[1], r[2], r[3], rp[0], rp[1], rp[2], rp[3]
  w, h = x2 - x1, y2 - y1
  rect_patch_p = patches.Rectangle((x1p, y1p), w, h, linewidth=1, edgecolor='b', facecolor='none')
  rect_patch = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
  ax.add_patch(rect_patch_p)
  ax.add_patch(rect_patch)
  plt.savefig(fname, bbox_inches='tight', pad_inches=0)

# list of all the rectangles
rects = [rect]

# -- additional variables for template correction
# the initial template image
I0 = seq[:, :, 0]
# the initial rectangle
rect0 = [item for item in rect]
# the template image at each step
It = seq[:, :, 0]
# for plotting
rects_prev = np.load("girlseqrects.npy")
# --

# iterate through each image, get delta p and plot rectangles
img_count = seq.shape[2]
for i in range(1, img_count):
  It1 = seq[:, :, i]
  # -- template correction steps
  # the total shift so far from image0 to imagen
  p_sofar = np.array([rect[0] - rect0[0], rect[1] - rect0[1]])
  # pn at current step based on current template
  pn = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))
  # the total shift so far + the newly calculate pn: p_{p=p_n} W(x;p)
  p0_n = p_sofar + pn
  # pn_star is a shift at current step, so we subtract the total shift so far
  pn_star = LucasKanade(I0, It1, rect0, threshold, num_iters, p0=p0_n) - p_sofar
  # update template
  if np.linalg.norm(pn_star - pn, ord=1) <= template_threshold:
    p = pn_star
    It = It1
  # --

  x1, y1, x2, y2  = rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]
  rect = [x1, y1, x2, y2]
  rects.append(rect)

  if i == 1 or i % 20 == 0:
    rect_prev = rects_prev[i]
    save_plot(It1, rect, rect_prev, '1-3-girl-wcrt-iter' + str(i) + '.png')

# save the rectangles as .npy file
np.save('girlseqrects-wcrt.npy', rects)



