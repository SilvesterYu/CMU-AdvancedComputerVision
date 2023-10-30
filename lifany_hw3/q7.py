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


def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=300):
    # TODO: Replace pass by your implementation
    P = np.zeros((pts1.shape[0], 3))
    totalErr = 0
    in1 = np.where(pts1[:, -1]<Thres, 0, 1)
    in2 = np.where(pts2[:, -1]<Thres, 0, 1)
    in3 = np.where(pts3[:, -1]<Thres, 0, 1)

    p11, p21, p31 = C1[0, :], C1[1, :], C1[2, :]
    p12, p22, p32 = C2[0, :], C2[1, :], C2[2, :]
    p13, p23, p33 = C3[0, :], C3[1, :], C3[2, :]

    for i in range(pts1.shape[0]):
        Amatrix = np.array([])
        if in1[i] and in2[i]:
            Amatrix = np.append(Amatrix, np.array([pts1[i][1] * p31 - p21, p11 - pts1[i][0] * p31])).reshape(-1, 4)
            Amatrix = np.append(Amatrix, np.array([pts2[i][1] * p32 - p22, p12 - pts2[i][0] * p32])).reshape(-1, 4)
        elif in2[i] and in3[i]:
            Amatrix = np.append(Amatrix, np.array([pts2[i][1] * p32 - p22, p12 - pts2[i][0] * p32])).reshape(-1, 4)
            Amatrix = np.append(Amatrix, np.array([pts3[i][1] * p33 - p23, p13 - pts3[i][0] * p33])).reshape(-1, 4)
        elif in3[i] and in1[i]:
            Amatrix = np.append(Amatrix, np.array([pts1[i][1] * p31 - p21, p11 - pts1[i][0] * p31])).reshape(-1, 4)
            Amatrix = np.append(Amatrix, np.array([pts3[i][1] * p33 - p23, p13 - pts3[i][0] * p33])).reshape(-1, 4)
        elif in1[i] and in2[i] and in3[i]:
            Amatrix = np.append(Amatrix, np.array([pts1[i][1] * p31 - p21, p11 - pts1[i][0] * p31])).reshape(-1, 4)
            Amatrix = np.append(Amatrix, np.array([pts2[i][1] * p32 - p22, p12 - pts2[i][0] * p32])).reshape(-1, 4)
            Amatrix = np.append(Amatrix, np.array([pts3[i][1] * p33 - p23, p13 - pts3[i][0] * p33])).reshape(-1, 4)

        _, _, VT = np.linalg.svd(Amatrix, 0)
        X = VT[-1, :]
        X = X/X[-1]
        XT = X.reshape(4, 1)
        P[i] = X[:-1]
        errCurr = 0
    
        if in1[i]:
            p1 = np.matmul(C1, XT)
            p1 = (p1/p1[-1][0])[:-1]
            xiyi1 = pts1[i][:-1].reshape(2, 1)
            e1 = np.linalg.norm(xiyi1 - p1)    
        if in2[i]:
            p2 = np.matmul(C2, XT)
            p2 = (p2/p2[-1][0])[:-1]
            xiyi2 = pts2[i][:-1].reshape(2, 1)
            e2 = np.linalg.norm(xiyi2 - p2)
        if in3[i]:
            p3 = np.matmul(C3, XT)
            p3 = (p3/p3[-1][0])[:-1]
            xiyi3 = pts3[i][:-1].reshape(2, 1)
            e3 = np.linalg.norm(xiyi3 - p3)

        if in1[i] and in2[i]:
            errCurr = (e1 + e2)/2
        elif in2[i] and in3[i]:
            errCurr = (e2 + e3)/2
        elif in3[i] and in1[i]:
            errCurr = (e3 + e1)/2
        elif in1[i] and in2[i] and in3[i]:
            errCurr = (e3 + e1 + e2)/3

        totalErr += errCurr
    return P, totalErr


"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""


def plot_3d_keypoint_video(pts_3d_video):
    # TODO: Replace pass by your implementation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    N = len(pts_3d_video)
    for i in range(N):
        pts_3d = pts_3d_video[i]
        num_points = pts_3d.shape[0]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
            yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
            zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j], alpha = i/N)

    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()
    



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
        #img = visualize_keypoints(im2, pts2)

        # TODO: YOUR CODE HERE
        C1, C2, C3 = np.matmul(K1, M1), np.matmul(K2, M2), np.matmul(K3, M3)
        thisP, thisErr = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)
        #plot_3d_keypoint(thisP)
        pts_3d_video.append(thisP)

    plot_3d_keypoint_video(pts_3d_video)

    #np.savez("q6_1.npz", np.array(pts_3d_video))