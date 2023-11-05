import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

for img in os.listdir("../images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("../images", img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    # list of center points corresponding to the returned bboxes
    points = np.array([(bbox[0] + bbox[2])/2 for bbox in bboxes])
    points_h = [(i, (bboxes[i][1] + bboxes[i][3])/2) for i in range(len(bboxes))]
    print(points)
    lines = []
    # clustering parameter
    eps = 50
    curr_point = points[0]
    curr_cluster = [points_h[0]]
    for i in range(1, len(points)):
        point = points[i]
        if point <= curr_point + eps:
            curr_cluster.append(points_h[i])
        else:
            lines.append(curr_cluster)
            curr_cluster = [points_h[i]]
        curr_point = point      
    lines.append(curr_cluster)
    sorted_lines = []
    for i in range(len(lines)):
        line = lines[i]
        line.sort(key = lambda x: x[1])
        sorted_lines.append([item[0] for item in line])
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    for sl in sorted_lines:
        print("\n", sl)
        for idx in sl:
            bbox = bboxes[i]
            print((bbox[0] + bbox[2])*(bboxes[i][1] + bboxes[i][3]))

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    params = pickle.load(open("q3_weights.pickle", "rb"))
    ##########################
    ##### your code here #####
    ##########################
    # the new images


    # initialize layers
    initialize_weights(train_x.shape[1], hidden_size, params, "layer1")
    initialize_weights(hidden_size, train_y.shape[1], params, "output")

    h1 = forward(train_x, params, "layer1")
    probs = forward(h1, params, "output", softmax)
    loss, acc = compute_loss_and_acc(train_y, probs)



