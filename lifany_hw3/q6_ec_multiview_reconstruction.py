import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here

"""
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""


def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=500):
    # TODO: Replace pass by your implementation
    N = pts1.shape[0]
    P = np.zeros((N, 3))
    err = 0
    print("pts1", pts1)
    print("pts2", pts2)
    print("pts3", pts3)
    mask1, mask2, mask3 = np.where(pts1[:, -1]<Thres, 0, 1), np.where(pts2[:, -1]<Thres, 0, 1), np.where(pts3[:, -1]<Thres, 0, 1)
    out_idx = np.where(mask1 + mask2 + mask3 < 1)[0]
    print(out_idx)
    for i in out_idx:
        pass

    

    '''
    # (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    # Math reference: https://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    p11, p21, p31 = C1[0, :], C1[1, :], C1[2, :]
    p12, p22, p32 = C2[0, :], C2[1, :], C2[2, :]
    for i in range(pts1.shape[0]):
        xi1, yi1, xi2, yi2 = pts1[i][0], pts1[i][1], pts2[i][0], pts2[i][1]
        Ai = np.array([
            yi2 * p32 - p22,
            p12 - xi2 * p32,
            yi1 * p31 - p21,
            p11 - xi1 * p31
        ])

        # (2) Solve for the least square solution using np.linalg.svd
        U, Sigma, VT = np.linalg.svd(Ai, 0)
        Xi = VT[-1, :]
        Xi = Xi/Xi[-1]

        # (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        #     homogeneous coordinates to non-homogeneous ones)
        proji1, proji2 = np.matmul(C1, Xi.reshape((4, 1))), np.matmul(C2, Xi.reshape(4, 1))
        proji1, proji2 = (proji1/proji1[-1][0])[:-1], (proji2/proji2[-1][0])[:-1]

        # (4) Keep track of the 3D points and projection error, and continue to next point 
        P[i] = Xi[:-1]
        xiyi1, xiyi2 = pts1[i].reshape(2, 1), pts2[i].reshape(2, 1)
        err1, err2 = np.linalg.norm(xiyi1 - proji1), np.linalg.norm(xiyi2 - proji2)
        err += err1**2 + err2**2

    return P, err
    '''
    TODO


"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""


def plot_3d_keypoint_video(pts_3d_video):
    # TODO: Replace pass by your implementation
    pass


# Extra Credit
if __name__ == "__main__":
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further
        img = visualize_keypoints(im2, pts2)

        # TODO: YOUR CODE HERE
        C1, C2, C3 = np.matmul(K1, M1), np.matmul(K2, M2), np.matmul(K3, M3)
        thisP, thisErr = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)