import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanadeAffine import *
from SubtractDominantMotion import *

# write your script here, we recommend all or some of the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq',
    default='../data/aerialseq.npy',
)
parser.add_argument(
  '--inverse',
  default = False
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file_path = args.seq
inverse_affine = args.inverse

seq = np.load(seq_file_path)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''
# -- my test code
# plot image with mask
def save_plot(im, mask, fname, inverse_affine):
  fig, ax = plt.subplots(figsize=plt.figaspect(im))
  ax.set_axis_off()
  ax.imshow(im, cmap='gray')
  zeros = np.zeros(im.shape)
  I = np.dstack([zeros, zeros, zeros, zeros])
  x, y = np.where(mask == True)
  if inverse_affine:
    for i in range(len(x)):
      I[x[i], y[i], :] = [0, 128, 0, im[x[i]][y[i]]]
    fname = "i-" + fname
  else:
    for i in range(len(x)):
      I[x[i], y[i], :] = [0, 0, 255, im[x[i]][y[i]]]
  ax.imshow(I)
  plt.savefig(fname, bbox_inches='tight', pad_inches=0)

# iterate through each image, get delta p and plot rectangles
img_count = seq.shape[2]
for i in range(1, img_count):
  It = seq[:, :, i-1]
  It1 = seq[:, :, i]
  mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance, seq_file_path, inverse_affine)
  if i == 1 or i % 30 == 0:
    save_plot(It1, mask, "2-3-aerial-iter-" + str(i) + ".png", inverse_affine)


