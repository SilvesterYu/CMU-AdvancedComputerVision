import numpy as np
import matplotlib.pyplot as plt

from q2_1_eightpoint import eightpoint

# Insert your package here


"""
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
"""


def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE

    pass


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    E = essentialMatrix(F, K1, K2)
    np.savez("q3_1.npz", E, F)

    # Simple Tests to verify your implementation:
    assert np.linalg.matrix_rank(E) == 2
