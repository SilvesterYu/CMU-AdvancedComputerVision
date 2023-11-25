# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# 
# Nov, 2023
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import (
    loadData,
    estimateAlbedosNormals,
    displayAlbedosNormals,
    estimateShape,
)
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface, integrateFrankot

def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    # Your code here
    U, Sigma, VT = np.linalg.svd(I, full_matrices=False)
    U = U[:, :3]
    VT = VT[:3, :]
    Sigma_sqrt = np.diag(np.sqrt(Sigma[:3]))
    # L = np.matmul(U, Sigma_sqrt).T
    # B = np.matmul(Sigma_sqrt, VT)
    L = np.matmul(U, np.diag(Sigma[:3])).T
    B = VT
    return B, L


def plotBasRelief(B, mu, nu, lam, s):
    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    # Your code here
    G = np.array([[1, 0, 0], 
                  [0, 1, 0],
                  [mu, nu, lam]])
    Bt = enforceIntegrability(B, s)
    B_bas = np.matmul(np.linalg.inv(G).T, Bt)
    albedos2, normals2 = estimateAlbedosNormals(B_bas)
    albedoIm, normalIm = displayAlbedosNormals(albedos2, normals2, s)
    # repeat (d)
    surface2 = estimateShape(normals2, s)
    plotSurface(surface2)

if __name__ == "__main__":

    # Part 2 (b)
    # Your code here
    I, L, s = loadData("../data/")
    print("original L ", L)
    B1, L1 = estimatePseudonormalsUncalibrated(I)

    albedos, normals = estimateAlbedosNormals(B1)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("2a-a.png", albedoIm, cmap="gray")
    plt.imsave("2a-b.png", normalIm, cmap="rainbow")
    print("estimated L ", L1)

    # Part 2 (d)
    # Your code here
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (e)
    # Your code here
    Bt = enforceIntegrability(B1, s)
    # repeat (b)
    albedos1, normals1 = estimateAlbedosNormals(Bt)
    # repeat (d)
    surface_f = estimateShape(normals1, s)
    plotSurface(surface_f)

    # Part 2 (f)
    # Your code here
    mus = [1, 0, 1, -1, 0, -1, 5, 0, 1]
    nus = [0, 1, 1, 0, -1, -1, 0, 5, 1]
    lams = [1, 1, 1, -1, -1, -1, 1, 1, 10]
    for i in range(len(mus)):
        plotBasRelief(B1, mus[i], nus[i], lams[i], s)