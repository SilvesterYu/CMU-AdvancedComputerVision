import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles, displayEpipolarF

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break

        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            print("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, "ro", markersize=8, linewidth=2)
        plt.draw()


"""
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

"""

# (3) Use guassian weighting to weight the pixel simlairty
def gaussian_weighing_err(window1, window2):
    err = np.sum(np.absolute(window1 - window2))
    print("err", err)
    return err

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    # (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    point1 = np.array([x1, y1, 1])
    epi_coords = np.matmul(F, point1)
    
    a, b, c = epi_coords[0], epi_coords[1], epi_coords[2]
    print("epipolar line a, b, c coordinates for ax + by + c = 0", a, b, c)

    # (2) Search along this line to check nearby pixel intensity (you can define a search window) to  find the best matches
    imW, imH = im1.shape[1], im1.shape[0]
    W = 5 # for convenience, define 0.5*window size in number of pixels as W
    xstart1, xend1, ystart1, yend1 = x1 - W, x1 + W, y1 - W, y1 + W # check window in x1, y1's surroundings
    window1 = im1[ystart1:yend1, xstart1:xend1] # original window in image 1
    #print("window1", xstart1, xend1, ystart1, yend1, window1.shape, window1)
    x2, y2 = 0, 0
    minErr = np.Inf
    # slide along the correct axis to avoid sliding out if the image and getting zero intensity values
    slideX = False
    if b != 0:
        if np.absolute(a/b) < 1:
            slideX = True
    if slideX:
        # slide along x axis only if the absolute slope is relatively small
        minX, maxX = max(0, xstart1-W), min(xend1+W, im2.shape[0])
        for myX in range(minX, maxX):
            myY = int((-a/b)*myX - c/b)
            if myY > W and myY < imH - W:
                xstart2, xend2, ystart2, yend2 = myX - W, myX + W, myY - W, myY + W
                window2 = im2[ystart2:yend2, xstart2:xend2]
                #print("window2 ", xstart2, xend2, ystart2, yend2, window2.shape)
                err = gaussian_weighing_err(window1, window2)
                if err < minErr:
                    x2, y2, minErr = myX, myY, err
    else:
        # or we slide along y axis
        minY, maxY = max(0, ystart1-W), min(yend1+W, im2.shape[0])
        for myY in range(minY, maxY):
            myX = int((-b/a)*myY - c/a)
            if myX > W and myX < imW - W:
                xstart2, xend2, ystart2, yend2 = myX - W, myX + W, myY - W, myY + W
                window2 = im2[ystart2:yend2, xstart2:xend2]
                #print("else window2 ", xstart2, xend2, ystart2, yend2, window2.shape, window2)
                err = gaussian_weighing_err(window1, window2)
                if err < minErr:
                    x2, y2, minErr = myX, myY, err
    print("im W H", imW, imH)
    return x2, y2


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    #
    #displayEpipolarF(im1, im2, F)
    epipolarCorrespondence(im1, im2, F, 119, 217)
    #
    np.savez("q4_1.npz", F, pts1, pts2)
    epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    print("x2, y2 final", x2, y2)
    assert np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10
