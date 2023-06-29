__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

# from Scripts.Scripts import *
import datetime
from scipy import signal
import cv2 as cv
import numpy as np
import os
import sys
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def get_active_area_mask(mat, patch_size=20, threshold=20, show = 0):
    patch = np.ones((patch_size, patch_size))
    active_area_mask = signal.convolve2d(np.abs(mat), patch, boundary='symm', mode='same')

    active_area_mask[active_area_mask < threshold] = 0
    active_area_mask[active_area_mask >= threshold] = 1
    if show:
        cv.imshow('mask',active_area_mask)
        cv.waitKey(0)
    return active_area_mask

def show_optical_flow_on_img(p0, delta_p0, img, magnification = 0.0, show = 1):
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    for (point, delta) in zip(p0, delta_p0):
        x = int(point[0, 0])
        y = int(point[0, 1])
        delta_x = delta[0, 0]
        delta_y = delta[0, 1]

        x2 = int(x + (1+magnification) * delta_x)
        y2 = int(y + (1+magnification) * delta_y)
        if x >= img.shape[1] or y >= img.shape[0]:
            print('Problem (x, y) =',x, y)
            continue
        img[y, x, 2] = 255
        img = cv.arrowedLine(img, (x, y), (x2, y2), (255, 0, 0), 1)



    # fig = plt.figure(figsize=(15,10))
    plt.figure(figsize=(13, 8))
    print('Points =', len(p0))
    if len(img.shape) == 2:
        plt.imshow(img ,cmap = 'gray')
    else:
        plt.imshow(img)

    if show:
        plt.show()
    # cv.namedWindow('img', cv.WINDOW_NORMAL)
    # cv.resizeWindow('img', img.shape[1] // 2, img.shape[0] // 2)
    # cv.imshow('img', img)
    # cv.waitKey(0)

    return img





def filter_image_by_spectral_kernel(img, spectral_kernel, show = 0):
    import numpy as np
    from matplotlib import pyplot as plt

    spectral_img = np.fft.fft2(img)
    spectral_img_shift = np.fft.fftshift(spectral_img)
    crow, ccol = img.shape[0] / 2, img.shape[1] / 2
    kcrow, kccol = spectral_kernel.shape[0] / 2, spectral_kernel.shape[1] / 2

    img_mask = spectral_img_shift[ int(crow - kcrow): int(crow + kcrow), int(ccol - kccol) : int(ccol + kccol) ]
    spectral_img_shift[ int(crow - kcrow): int(crow + kcrow), int(ccol - kccol): int(ccol + kccol)] = np.multiply(spectral_kernel, img_mask)

    f_ishift = np.fft.ifftshift(spectral_img_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    if show:
        plt.subplot(131), plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(spectral_kernel, cmap = 'gray')
        plt.title('Filter'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(img_back, cmap = 'gray')
        plt.title('Output Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    return img_back

def add_diagonals_to_img(img, distance = 11, density = 5):
    height = img.shape[0]
    width = img.shape[1]
    mask_pattern = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            if np.abs(i-j)%distance <= density and np.abs(i - (width - j))%distance <= density:
                mask_pattern[i, j] = 1.0
    return np.multiply( img, mask_pattern)


def add_lines_to_img(img, orientation = 'H', distance = 11, density = 5):
    mask_pattern = np.zeros_like(img)
    if orientation == 'V':
        width = img.shape[1]
        for j in range(width):
            if j%distance <= density:
                mask_pattern[:, j] = 1.0
    else:
        height = img.shape[1]
        for i in range(height):
            if i%distance <= density:
                mask_pattern[i,:] = 1.0
    return np.multiply( img, mask_pattern)