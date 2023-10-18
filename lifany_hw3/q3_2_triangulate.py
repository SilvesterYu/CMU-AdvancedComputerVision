import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


"""
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
"""


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    P = np.zeros((pts1.shape[0], 3))
    err = 0
    
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


"""
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
"""


def findM2(F, pts1, pts2, intrinsics, filename="q3_3.npz"):
    """
    Q2.2: Function to find camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of the projection error through best_error and retain the best one.
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    """
    # ----- TODO -----
    # YOUR CODE HERE
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))

    minErr = np.Inf
    
    for i in range(M2s.shape[-1]):
        thisM2 = M2s[:, :, i]
        C1, C2 = np.matmul(K1, M1), np.matmul(K2, thisM2)
        thisP, thisErr = triangulate(C1, pts1, C2, pts2)
        if thisErr < minErr:
            minErr = thisErr
            P = thisP
            M2 = thisM2

    C2 = np.matmul(K2, M2)
    np.savez("q3_3.npz", M2, C2, P)

    return M2, C2, P


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert err < 500
