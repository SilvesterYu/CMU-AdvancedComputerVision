import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy

# Insert your package here
import random


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        try:
            x, y = plt.ginput(1, mouse_stop=2)[0]
            plt.draw()
        except:
            break


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""


def ransacF(pts1, pts2, M, nIters=1000, tol=5):
    # TODO: Replace pass by your implementation
    N = pts1.shape[0]
    x = 8
    it, maxnumInliers, minErr = 0, 0, np.Inf
    while it < nIters:
        idx = random.sample(range(0, N), x)
        sampled_points1, sampled_points2 = pts1[idx], pts2[idx]
        myF = eightpoint(sampled_points1, sampled_points2, M)
        pts1_homo, pts2_homo = np.hstack((pts1, np.ones((N, 1)))), np.hstack((pts2, np.ones((N, 1))))
        err = calc_epi_error(pts1_homo, pts2_homo, myF)
        in_idx = np.where(err < tol)[0]
        numInliers = len(in_idx)
        if numInliers > maxnumInliers:
            F, inliers = myF, np.where(err < tol, True, False).reshape((N, 1))
            minErr, maxnumInliers = err[in_idx].mean(), numInliers, 
        it += 1
    print("minErr", minErr, "maxnumInliers", maxnumInliers)
    return F, inliers

"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""

# reference: https://courses.cs.duke.edu//fall13/compsci527/notes/rodrigues.pdf
def rodrigues(r):
    # TODO: Replace pass by your implementation
    theta = np.linalg.norm(r)
    r = r.reshape((3, 1))
    u = r/theta
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ux = np.array([[0, -u[2][0], u[1][0]], [u[2][0], 0, -u[0][0]], [-u[1][0], u[0][0], 0]])
    R = I * np.cos(theta) + (1 - np.cos(theta)) * np.matmul(u, u.T) + ux * np.sin(theta)
    return R


"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""

# reference: https://courses.cs.duke.edu//fall13/compsci527/notes/rodrigues.pdf
def invRodrigues(R):
    # TODO: Replace pass by your implementation
    A = 0.5 * (R - R.T)
    rho = np.array([A[2][1], A[0][2], A[1][0]])
    s = np.linalg.norm(rho)
    c = 0.5 * (R[0][0] + R[1][1] + R[2][2] - 1)

    if s == 0 and c == 1:
        r = np.array([0, 0, 0])
    elif s == 0 and c == -1:
        RI = R + np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        normL = [np.linalg.norm(RI[:, 0]), np.linalg.norm(RI[:, 1]), np.linalg.norm(RI[:, 2])]
        idx = normL.index(max(normL))
        v = RI[:, idx]
        u = v/np.linalg.norm(v)
        u = u.reshape((0, 3))
        r = -u * np.pi
    else:
        u = rho / s
        theta = np.arctan2(s, c)
        r = u * theta

    return r


"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""

def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    # (1) recover M2 from (R2, t2) in x
    t2, r2, P = x[-3:], x[-6:-3], x[:-6]
    R2 = rodrigues(r2)
    P = P.reshape((int(len(P)/3), 3))
    M2 = np.hstack((R2, t2.reshape((3, 1))))

    # (2) calculate projected points in both images
    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)
    P_homo = np.hstack((P, np.ones((P.shape[0], 1)))) # N x 4
    proj1, proj2 = np.matmul(C1, P_homo.T), np.matmul(C2, P_homo.T) # 3 x 4 by 4 x N = 3 x N
    proj1, proj2 = (proj1/proj1[-1])[:-1].T, (proj2/proj2[-1])[:-1].T # Normalise and reshape into N x 2

    # (3) calculate residuals
    residual = np.append((p1 - proj1).flatten(), (p2 - proj2).flatten())
    return residual


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""

def myFunc(x, K1, M1, p1, K2, p2):
    # objective function, the reprojection error
    return np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x))**2

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    # (1) Calculate initial residual
    R2_init, t2_init = M2_init[:, :-1], M2_init[:, -1]
    r2_init= invRodrigues(R2_init)
    x_init = np.append(P_init.flatten(), r2_init)
    x_init = np.append(x_init, t2_init)
    obj_start = rodriguesResidual(K1, M1, p1, K2, p2, x_init)

    # (2) Minimize objective function using scipy.optimize.minimize the rodrigues residual
    x_op = scipy.optimize.minimize(myFunc, x_init, args=(K1, M1, p1, K2, p2), method="Powell").x

    # (3) Collect the optimized results
    obj_end = rodriguesResidual(K1, M1, p1, K2, p2, x_op)
    t2, r2, P = x_op[-3:], x_op[-6:-3], x_op[:-6]
    R2 = rodrigues(r2)
    P = P.reshape((int(len(P)/3), 3))
    M2 = np.hstack((R2, t2.reshape((3, 1))))

    return M2, P, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    '''
    # -- UNCOMMENT THIS PART FOR EIGHTPOINT & RANSAC COMPARISON!!
    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    inliers = np.ravel(inliers)
    print("F", F)
    print("inliers", inliers.sum())
    print("in", inliers)
    M = np.max([*im1.shape, *im2.shape])
    in_pts1, in_pts2 = noisy_pts1[inliers, :], noisy_pts1[inliers, :]
    F_all = eightpoint(noisy_pts1, noisy_pts2, M)
    N = noisy_pts1.shape[0]
    N_in = in_pts1.shape[0]
    all_pts1_homo, all_pts2_homo = np.hstack((noisy_pts1, np.ones((N, 1)))), np.hstack((noisy_pts2, np.ones((N, 1))))
    in_pts1_homo, in_pts2_homo = np.hstack((in_pts1, np.ones((N_in, 1)))), np.hstack((in_pts2, np.ones((N_in, 1))))
    err_all = calc_epi_error(all_pts1_homo, all_pts2_homo, F_all)
    err_in = calc_epi_error(in_pts1_homo, in_pts2_homo, F_all)
    print("err all", err_all.mean(), "err_in", err_in.mean())
    breakpoint()
    '''

    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(
        noisy_pts2
    )

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization:
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    # (1) Call the ransacF function to find the fundamental matrix
    F, inliers = ransacF(pts1, pts2, M)
    inliers = np.ravel(inliers)
    in_pts1, in_pts2 = pts1[inliers, :], pts2[inliers, :]

    # (2) Call the findM2 function to find the extrinsics of the second camera
    M2_init, C2, P_init = findM2(F, in_pts1, in_pts2, intrinsics)
    print("M2 start", M2_init)

    # (3) Call the bundleAdjustment function to optimize the extrinsics and 3D points
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    M2, P, obj_start, obj_end = bundleAdjustment(K1, M1, in_pts1, K2, M2_init, in_pts2, P_init)

    # (4) Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    plot_3D_dual(P_init, P)
    print("M2 end", M2)
    print("reprojection error start", np.linalg.norm(obj_start)**2)
    print("reprojection error end", np.linalg.norm(obj_end)**2)  