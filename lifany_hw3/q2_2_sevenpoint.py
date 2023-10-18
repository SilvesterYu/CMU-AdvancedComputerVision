import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here


"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""


def sevenpoint(pts1, pts2, M):
    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
    # (1) Normalize the input pts1 and pts2 scale paramter M.
    N = pts1.shape[0]
    pts11 = np.append(pts1, np.ones((N, 1)), axis=1)
    pts21 = np.append(pts2, np.ones((N, 1)), axis=1)
    T = np.array([[1/M, 0, 0],
                  [0, 1/M, 0],
                  [0, 0, 1]])
    npts1, npts2 = np.matmul(pts11, T), np.matmul(pts21, T)

    # (2) Setup the seven point algorithm's equation.
    A = np.zeros(shape=(N, 9))
    for i in range(N):
      row = np.matmul(npts2[i].reshape((3, 1)), npts1[i].reshape((1, 3))).reshape((9, ))
      A[i] = row

    # (3) Solve for the least square solution using SVD. 
    U, Sigma, VT = np.linalg.svd(A)  

    # (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    f1 = VT[-1, :].reshape((3, 3))
    f2 = VT[-2, :].reshape((3, 3))

    # (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
    #     det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
    #     Use np.polynomial.polynomial.polyroots to solve for the roots
    def det_func(a):
        return np.linalg.det(a*f1 + (1-a)*f2) 

    # solve [c3, c2, c1, c0]] using:
    # [a_0**3, a_0**2, a_0, 1]  [c3  [ f(a0)
    # [a_1**3, a_1**2, a_1, 1] * c2 =  f(a1)
    # [a_2**3, a_2**2, a_2, 1]   c1    f(a2)
    # [a_3**3, a_3**2, a_3, 1]   c0]   f(a3) ]
    avals = [0, 1/4, 1/2, 3/4]
    amat = np.zeros((4, 4))
    for i in range(4):
        amat[i] = np.array([avals[i]**3, avals[i]**2, avals[i], 1])
    bmat = np.array([det_func(avals[i]) for i in range(4)]).reshape((4, 1))
    res = np.ravel(np.linalg.solve(amat, bmat))
    c3, c2, c1, c0 = res[0], res[1], res[2], res[3]
    roots = np.polynomial.polynomial.polyroots([c3, c2, c1, c0])
    idx, = np.where(np.iscomplex(roots) == False)
    roots = roots[list(idx)]

    # (6) Unscale the fundamental matrixes and return as Farray
    Farray = []
    for r in roots:
        F = r*f1 + (1-r)*f2
        F = _singularize(F)
        F = refineF(F, npts1[:, :-1], npts2[:, :-1])
        F = np.matmul(T, np.matmul(F, T))
        F = F/F[-1][-1]
        Farray.append(F)

    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]

    print("F", F)

    np.savez("q2_2.npz", F, M)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        if i % 100 == 0:
            print(i, "-"*100)
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])

    print("final F", F)

    np.savez("q2_2_final.npz", F, M)

    displayEpipolarF(im1, im2, F)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
