from pickle import FALSE
import numpy as np
import matplotlib.pyplot as plt
from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here

"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
"""

def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE

    # (1) Normalize the input pts1 and pts2 using the matrix T.
    npts1, npts2 = pts1/M, pts2/M

    # (2) Setup the eight point algorithm's equation, which is AF(:) = 0
    N = pts1.shape[0]
    A = np.zeros(shape=(N, 9))
    npts11, npts21 = np.append(npts1, np.ones((N, 1)), axis=1), np.append(npts2, np.ones((N, 1)), axis=1)
    for i in range(N):
      row = np.matmul(npts11[i].reshape((3, 1)), npts21[i].reshape((1, 3))).reshape((9, ))
      A[i] = row

    # (3) Solve for the least square solution using SVD. 
    U, Sigma, VT = np.linalg.svd(A)
    # Take last row of VT in special case of SVD when Ax = 0
    # Math reference: www.cse.unr.edu/~bebis/CS791E/Notes/SVD.pdf
    F = VT[-1, :].reshape((3, 3))

    # (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    F = _singularize(F)

    # (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
    F = refineF(F, npts1, npts2)
    t = np.diag([1/M, 1/M, 1])
    f2 = np.matmul(t.T, (F @ t))
    f2 = f2 / f2[2, 2]

    # (6) Unscale the fundamental matrix

    print(F)
    print(f2)

    return f2


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
