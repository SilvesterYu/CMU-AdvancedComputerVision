# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# Nov, 2023
###################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import plotSurface, integrateFrankot
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

    # im = plt.figure()
    # ax = im.add_subplot(projection='3d')
    # ax.plot_surface(X, Y, Z)
    # plt.show()

    # w, h = res[0], res[1]
    # XYZ = np.stack((X, -Y, Z), axis = 2).reshape((h*w, -1))
    # image = np.dot(XYZ, light).reshape((h, w))

    # Your code here
    image = X * light[0] - Y * light[1] + Z * light[2]
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
        # preserve bit depth while reading, datatype should be uint16
        im0 = skimage.io.imread(fname = path + im_name).astype(np.uint16)
        # Y channel (channel 1) is the luminance
        im = skimage.color.rgb2xyz(im0)[:,:,1].flatten()
        I.append(im)
    I = np.array(I)

    L = np.load(path + 'sources.npy').T

    s = im0.shape[:2]

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

    B = np.linalg.lstsq(L.T, I)[0]
    # Your code here
    return B


def estimateAlbedosNormals(B):
    """
    Question 1 (f)

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

    albedos = np.linalg.norm(B, axis = 0)
    normals = B / albedos
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
    # Your code here
    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape(s[0], s[1], 3)
    scale = np.max(normalIm) - np.min(normalIm)
    normalIm = (normalIm - np.min(normalIm)) / scale
    #normalIm = np.clip(normalIm, 0, 1)
    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (i)

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
    # Your code here
    fx = - normals[0, :] / normals[2, :]
    fy = - normals[1, :] / normals[2, :]
    fx = fx.reshape(s)
    fy = fy.reshape(s)
    surface = integrateFrankot(fx, fy)
    
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
    I, L, s = loadData()
    U, Sigma, VT = np.linalg.svd(I, full_matrices=False)
    print("1(d) Sigma: ", Sigma)
    

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)
    print("1(d) estimate B using least squares, B = ", B)


    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("1f-a.png", albedoIm, cmap="gray")
    plt.imsave("1f-b.png", normalIm, cmap="rainbow")


    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface, "q1-i")