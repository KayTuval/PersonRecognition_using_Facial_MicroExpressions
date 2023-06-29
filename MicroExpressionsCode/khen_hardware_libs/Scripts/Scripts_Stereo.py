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



###############################
######   Stereo scripts  ######
###############################


def get_matching_points(img1,img2,show = 0):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # print(len(matches))
    # print(matches[0].distance)
    # Draw first matches.
    if show == 1:
        fig1 = plt.figure(figsize=(15, 8))
        # print('Distance range: [{:.1f},{:.1f}]'.format(np.abs(kp1[matches[-1].queryIdx].pt[0]-kp2[matches[-1].trainIdx].pt[0]-img1.shape[0]),\
        #                                        np.abs(kp1[matches[0].queryIdx].pt[0]-kp2[matches[0].trainIdx].pt[0])-img1.shape[0]))
        img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3), plt.show()
    return kp1, kp2, matches


def display_two_imgs_rectified(left_img, right_img, stereo_system):

    fig1 = plt.figure(figsize=(15, 8))
    rect_left, rect_right = stereo_system.rectify_two_images(left_img, right_img)
    ax = plt.subplot(2, 2, 1)
    ax.minorticks_on()
    plt.imshow(cv.cvtColor(left_img, cv.COLOR_BGR2RGB))
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='blue', axis='y')
    # ax.grid(which='minor', linestyle='-', linewidth='0.5', color='red',axis = 'y')
    ax = plt.subplot(2, 2, 2)
    ax.minorticks_on()
    plt.imshow(cv.cvtColor(right_img, cv.COLOR_BGR2RGB))
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='blue', axis='y')
    # ax.grid(which='minor', linestyle='-', linewidth='0.5', color='red',axis = 'y')
    ax = plt.subplot(2, 2, 3)
    ax.minorticks_on()

    plt.imshow(cv.cvtColor(rect_left, cv.COLOR_BGR2RGB))
    # find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    # is_found, corners = cv2.findChessboardCorners(rect_left, pattern_size, flags=find_chessboard_flags)
    # cv2.drawChessboardCorners(rect_left, pattern_size, corners, is_found)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='blue', axis='y')
    # ax.grid(which='minor', linestyle='-', linewidth='0.5', color='red',axis = 'y')
    ax = plt.subplot(2, 2, 4)
    ax.minorticks_on()

    plt.imshow(cv.cvtColor(rect_right, cv.COLOR_BGR2RGB))
    # is_found, corners = cv2.findChessboardCorners(rect_right, pattern_size, flags=find_chessboard_flags)
    # cv2.drawChessboardCorners(rect_right, pattern_size, corners, is_found)
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='blue', axis='y')
    # ax.grid(which='minor', linestyle='-', linewidth='0.5', color='red',axis = 'y')
    plt.show()
    return rect_left, rect_right

def get_sgm_disparity_map(imgL, imgR, max_disparity, blockSize =16, min_disparity = 10, show = 0):
    # disparity range is tuned for 'aloe' image pair
    window_size = 5
    num_disp = max_disparity - min_disparity
    stereo = cv.StereoSGBM_create(minDisparity=min_disparity,
                                   numDisparities=num_disp,
                                   blockSize=blockSize,
                                   P1=1,#8 * 3 * window_size ** 2,
                                   P2=1,#32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=150,
                                   speckleRange=32)

    # stereo = cv.StereoBM_create(numDisparities=16, blockSize=16)

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    if show == 1:
        fig1 = plt.figure(figsize=(15, 8))
        ax = plt.subplot(1, 2, 1)
        # plt.imshow(cv.cvtColor(imgL, cv.COLOR_BGR2RGB))
        plt.imshow(imgL,cmap = 'gray')
        ax = plt.subplot(1, 2, 2)
        plt.imshow((disp - min_disparity) / num_disp,cmap = 'gray')
        plt.show()
    return (disp - min_disparity) / num_disp



def get_mixed_images_disparity(left_img, right_img, show = 0):
    mixed_images = np.concatenate( (left_img[:, :, np.newaxis], right_img[:, :, np.newaxis]), axis = 2)
    mixed_images = np.abs(mixed_images)
    mixed_images = mixed_images / np.max(mixed_images)
    mixed_images = np.concatenate((mixed_images, np.zeros((left_img.shape[0], left_img.shape[1], 1))), axis=2)
    if show == 1:
        plt.imshow(mixed_images)
        plt.show()
    return mixed_images



def get_active_area_mask(mat, patch_size=20, threshold=20, show = 0):
    patch = np.ones((patch_size, patch_size))
    active_area_mask = signal.convolve2d(np.abs(mat), patch, boundary='symm', mode='same')

    active_area_mask[active_area_mask < threshold] = 0
    active_area_mask[active_area_mask >= threshold] = 1
    if show:
        cv.imshow('mask',active_area_mask)
        cv.waitKey(0)
    return active_area_mask


from scipy.interpolate import griddata
def interpulate_zeros_in_image(mat, method = 'linear'):
    height, width = mat.shape[0], mat.shape[1]
    cols_vec, rows_vec = np.meshgrid(np.arange(width), np.arange(height))
    points = np.where(mat != 0)
    values = mat[points]
    points = np.transpose(points)
    mat = griddata(points, values, (rows_vec, cols_vec), method=method, fill_value = 0)
    return mat


def interpulate_masked_matrix(mat, mask, method = 'linear', filter = None, show = 0):
    # mask is 0-1 matrix, index with 1 means to interpulate there 0 means don't interpulate there
    height, width = mat.shape[0], mat.shape[1]
    cols_vec, rows_vec = np.meshgrid(np.arange(width), np.arange(height))
    points = np.where(mat != 0)
    values = mat[points]
    points = np.transpose(points)

    grid_z0 = griddata(points, values, (rows_vec, cols_vec), method=method)

    # if show:    # Test!!
    #     plt.imshow(grid_z0, cmap='gray')
    #     plt.title(method)
    #     plt.show()

    if filter == 'median3':
        grid_z0 = cv.medianBlur(grid_z0.astype(np.float32), ksize = 3)
    elif filter == 'median5':
        grid_z0 = cv.medianBlur(grid_z0.astype(np.float32), ksize = 5)
    elif filter == 'median3gaussian3':
        grid_z0 = cv.medianBlur(grid_z0.astype(np.float32), ksize = 3)
        grid_z0 = cv.GaussianBlur(grid_z0, ksize=(0, 0), sigmaX=3.0)  # Test

    # if show:
    #     plt.imshow(grid_z0, cmap='gray')
    #     plt.title(method)
    #     plt.show()

    grid_z0[mask == 0] = mat[mask == 0]

    if show:
        plt.imshow(grid_z0, cmap='gray')
        plt.title(method)
        plt.show()
    return grid_z0.astype(np.float32)


def min_max_avg_disparity_estimator(left_img, right_img):
    # This function estimates the min_disparity and max_disparity
    height, width = left_img.shape[0], left_img.shape[1]
    min_disparity = width
    max_disparity = 0
    sum_disp = 0
    for row in range(height):
        conv_rows = signal.fftconvolve(left_img[row,:], right_img[row,:], axes=0, mode='full')
        disp = np.argmax(conv_rows) - width
        sum_disp += disp
        if disp < min_disparity:
            min_disparity = disp
        if disp > max_disparity:
            max_disparity = disp

    return min_disparity, max_disparity, sum_disp//height




