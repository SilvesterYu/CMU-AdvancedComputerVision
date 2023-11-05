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
from skimage.transform import resize

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

# --
hidden_size = 64
new_max, new_min = 1, 0 # for enhancing image contrast
d = np.ones((11, 10)) # for letter dilation
t1 = 'TODOLIST1MAKEATODOLIST2CHECKOFFTHEFIRSTTHINGONTODOLIST3REALIZEYOUHAVEALREADYCOMPLETED2THINGS4REWARDYOURSELFWITHANAP'
t2 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
t3 = 'HAIKUSAREEASYBUTSOMETIMESTHEYDONTMAKESENSEREFRIGERATOR'
t4 = 'DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING'
# --

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
    N = len(points)
    # clustering parameter
    eps = 50
    curr_point = points[0]
    curr_cluster = [points_h[0]]
    lines = []
    for i in range(1, N):
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
    # preprocess the mini images and populate the X matrix for model input
    myX = np.zeros((N, 1024))
    xidx = 0
    for sl in sorted_lines:
        for idx in sl:
            # retrieve the mini images using bounding boxes and resize
            bbox = bboxes[idx]
            y1, x1, y2, x2 = bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5
            im = bw[y1:y2, x1:x2]
            im = np.pad(im, (20,18), 'maximum') 
            dilated_im = erosion(im, d)
            resized_im = resize(dilated_im, (32, 32))
            resized_im = resized_im.T.reshape(1, 1024)
            # increase each mini image's contrast
            minimum, maximum = np.min(resized_im), np.max(resized_im)
            m = (new_max - new_min) / (maximum - minimum)
            b = new_min - m * minimum
            resized_im = m * resized_im + b
            # populate the X matrix
            myX[xidx][:] = resized_im
            xidx += 1

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
    # Create y groundtruth data
    indices = [i for i in range(36)]
    D = dict(zip(letters, indices))
    D1 = dict(zip(indices, letters))
    if "01" in img:
        myt = t1
    elif "02" in img:
        myt = t2
    elif "03" in img:
        myt = t3
    elif "04" in img:
        myt = t4
    myY = np.zeros((len(myt), 36))
    for i in range(len(myt)):
        myY[i][D[myt[i]]] = 1

    # Model predictions
    h1 = forward(myX, params, "layer1")
    probs = forward(h1, params, "output", softmax)
    loss, acc = compute_loss_and_acc(myY, probs)

    # The raw extracted text
    txt = "".join([D1[probs[idx].argmax()] for idx in range(myY.shape[0])])
    print("img name: ", img, " loss: ", loss, " accuracy: ", acc)

    # Print the extracted text in rows
    txt_rows = ''
    a, b = 0, 0
    for line in sorted_lines:
        b += len(line)
        txt_rows += txt[a:b] + "\n"
        a = b
    print(txt_rows)