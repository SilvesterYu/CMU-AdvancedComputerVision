# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# Nov, 2023
###################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import plotSurface
import skimage
import scipy


def renderNDotLSphere(center, rad, light, pxSize, res):
    # https://stackoverflow.com/questions/67867707/make-a-sphere-from-a-circle-of-dots
    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0] / 2) * pxSize * 1.0e-4
    Y = (Y - res[1] / 2) * pxSize * 1.0e-4
    Z = np.sqrt(rad**2 + 0j - X**2 - Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)

    # Your code here
    # im = plt.figure()
    # ax = im.add_subplot(projection='3d')
    # ax.plot_surface(X, Y, Z)
    # plt.show()

    w, h = res[0], res[1]
    XYZ = np.stack((X, Y, Z), axis=2).reshape((w * h, -1))
    image = np.dot(XYZ, light).reshape((h, w))
    image = np.clip(image, 0, None)

    return image


def loadData(path="../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.npy.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    # Your code here
    I = []
    for i in range(1, 8):
        im_name = "input_" + str(i) + ".tif"
        im0 = skimage.io.imread(fname = path + im_name)
        print(im0)
        im = skimage.color.rgb2gray(im0).flatten()
        I.append(im)
        print(im)
    I = np.array(I).astype(np.int16)

    L = np.load(path + 'sources.npy').T

    s = im0.shape

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    # Your code here
    return B


def estimateAlbedosNormals(B):
    """
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    """

    albedos = None
    normals = None
    # Your code here
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None
    # Your code here
    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    # Your code here
    return surface


if __name__ == "__main__":
    # Part 1(b)
    radius = 0.75  # cm
    center = np.asarray([0, 0, 0])  # cm
    pxSize = 7  # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.show()
    plt.imsave("1b-a.png", image, cmap="gray")

    light = np.asarray([1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.show()
    plt.imsave("1b-b.png", image, cmap="gray")

    light = np.asarray([-1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.show()
    plt.imsave("1b-c.png", image, cmap="gray")

    # Part 1(c)
    I, L, s = loadData("../data/")

    # Part 1(d)
    # Your code here
    print(I)
    U, Sigma, VT = np.linalg.svd(I, full_matrices=False)
    print(Sigma)
    print(np.linalg.matrix_rank(I))


    '''

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("1f-a.png", albedoIm, cmap="gray")
    plt.imsave("1f-b.png", normalIm, cmap="rainbow")

    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
    '''
