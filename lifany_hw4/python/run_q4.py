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
from skimage.transform import resize

# --
hidden_size = 64
new_max, new_min = 1, 0
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
    print(points)
    lines = []
    N = len(points)
    # clustering parameter
    eps = 50
    curr_point = points[0]
    curr_cluster = [points_h[0]]
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
    myX = np.zeros((N, 1024))
    xidx = 0
    for sl in sorted_lines:
        print("\n", sl)
        for idx in sl:
            bbox = bboxes[idx]
            y1, x1, y2, x2 = bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5
            im = bw[y1:y2, x1:x2]
            im = np.pad(im, (20,18), 'maximum') 
            d = np.ones((11, 10))
            dilated_im = erosion(im, d)
            resized_im = resize(dilated_im, (32, 32))
            resized_im = resized_im.T.reshape(1, 1024)
            resized_im = (resized_im - 0.5)*10
            minimum, maximum = np.min(resized_im), np.max(resized_im)
            m = (new_max - new_min) / (maximum - minimum)
            b = new_min - m * minimum
            resized_im = m * resized_im + b

            myX[xidx][:] = resized_im
            xidx += 1
    print(myX)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    print("letters", letters)
    params = pickle.load(open("q3_weights.pickle", "rb"))
    ##########################
    ##### your code here #####
    ##########################
    indices = [i for i in range(36)]
    D = dict(zip(letters, indices))
    D1 = dict(zip(indices, letters))
    print(D)
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
    
    # print(myY)
    # print(myX)
    # print(myX[0])
    # im = myX[0].reshape(32, 32).T
    # print(im)
    # # plt.imsave("imMy.png", im)
    # print(myY)
    # breakpoint()
    # initialize layers
    # initialize_weights(myX.shape[1], hidden_size, params, "layer1")
    # initialize_weights(hidden_size, myY.shape[1], params, "output")

    h1 = forward(myX, params, "layer1")
    probs = forward(h1, params, "output", softmax)
    loss, acc = compute_loss_and_acc(myY, probs)
    
    extracted_txt = []
    for idx in range(myY.shape[0]):
        print("Y", myY[idx].argmax(), probs[idx].argmax())
        extracted_txt.append(D1[probs[idx].argmax()])
    txt = "".join(extracted_txt) + " "
    print("loss", loss, "acc", acc)
    print("text", "".join(extracted_txt) + " ")
    each_line = [len(line) for line in sorted_lines]
    splt = []
    for l in each_line:
        splt.append(txt[:l])
        txt = txt[l:]
    print("\n".join(splt))



